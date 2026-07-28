"""HTTP control surface — spec §10.

    /health /kill /approve /positions /pnl

The kill switch has three access paths (spec §FR-MON-03: UI, CLI, HTTP); this
module is the HTTP one. Every handler is a thin translation from an HTTP verb
to a :class:`~atrader.app.runtime.Runtime` method — no trading logic lives
here, only wire-format shaping, so nothing in this file needs its own
safety-property tests beyond "does it call the right ``Runtime`` method".

Two endpoints beyond the five named in the spec: ``/metrics`` (Prometheus
scraping, spec §9.2 — a metrics module with nothing serving it is not
"exposed") and ``/reconcile``/``/audit`` (the HTTP side of the CLI's
``reconcile``/``verify-audit`` commands, which need to reach a *running*
process's state rather than start a fresh one).
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from atrader.app.runtime import Runtime
from atrader.audit.hashchain import records_to_json_bytes
from atrader.core.models import Position
from atrader.risk.approval import ApprovalRequest
from atrader.risk.killswitch import KillSwitchSource

__all__ = ["create_app"]


class HealthOut(BaseModel):
    running: bool
    environment: str
    started_at_ns: int | None
    ticks_processed: int
    bars_processed: int
    orders_submitted: int
    kill_switch_engaged: bool
    system_state: str
    breaker_level: str
    reconciliation_clean: bool
    equity: Decimal
    margin_status: str


class KillRequest(BaseModel):
    reason: str
    actor: str = "operator"
    liquidate: bool = False


class ReleaseRequest(BaseModel):
    reason: str
    actor: str = "operator"


class KillOut(BaseModel):
    engaged: bool
    reason: str
    source: str


class ApprovalDecisionRequest(BaseModel):
    by: str
    note: str = ""


class ApprovalOut(BaseModel):
    request_id: UUID
    kind: str
    summary: str
    status: str
    requested_at_ns: int
    expires_at_ns: int


class PositionOut(BaseModel):
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


class PnlOut(BaseModel):
    realized: Decimal
    unrealized: Decimal
    equity: Decimal


class ReconcileRequest(BaseModel):
    inject_break: bool = False


class ReconcileOut(BaseModel):
    clean: bool
    breaks: list[str]
    positions_checked: int
    orders_checked: int


def _approval_out(request: ApprovalRequest) -> ApprovalOut:
    return ApprovalOut(
        request_id=request.request_id,
        kind=request.kind.value,
        summary=request.summary,
        status=request.status.value,
        requested_at_ns=request.requested_at_ns,
        expires_at_ns=request.expires_at_ns,
    )


def _position_out(position: Position) -> PositionOut:
    return PositionOut(
        symbol=position.symbol,
        quantity=position.quantity,
        avg_price=position.avg_price,
        unrealized_pnl=position.unrealized_pnl,
        realized_pnl=position.realized_pnl,
    )


def create_app(runtime: Runtime) -> FastAPI:
    """Build the API bound to one running :class:`~atrader.app.runtime.Runtime`.

    A fresh app per ``Runtime`` rather than a module-level singleton — the
    CLI constructs both together, and each test gets a fully isolated pair.
    """
    app = FastAPI(title="atrader", version="0.1.0")

    @app.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        status = runtime.status()
        return HealthOut(**{field: getattr(status, field) for field in HealthOut.model_fields})

    @app.post("/kill", response_model=KillOut, status_code=202)
    async def kill(request: KillRequest) -> KillOut:
        event = await runtime.kill(
            reason=request.reason,
            source=KillSwitchSource.HTTP,
            actor=request.actor,
            liquidate=request.liquidate,
        )
        return KillOut(engaged=event.engaged, reason=event.reason, source=event.source.value)

    @app.delete("/kill", response_model=KillOut, status_code=202)
    async def release_kill(request: ReleaseRequest) -> KillOut:
        event = await runtime.release(
            reason=request.reason, source=KillSwitchSource.HTTP, actor=request.actor
        )
        return KillOut(engaged=event.engaged, reason=event.reason, source=event.source.value)

    @app.get("/approve", response_model=list[ApprovalOut])
    def list_pending_approvals() -> list[ApprovalOut]:
        return [_approval_out(r) for r in runtime.approvals.pending()]

    @app.post("/approve/{request_id}", response_model=ApprovalOut)
    def grant_approval(request_id: UUID, request: ApprovalDecisionRequest) -> ApprovalOut:
        try:
            granted = runtime.approvals.grant(request_id, by=request.by, note=request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _approval_out(granted)

    @app.delete("/approve/{request_id}", response_model=ApprovalOut)
    def deny_approval(request_id: UUID, request: ApprovalDecisionRequest) -> ApprovalOut:
        try:
            denied = runtime.approvals.deny(request_id, by=request.by, note=request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _approval_out(denied)

    @app.get("/positions", response_model=list[PositionOut])
    def positions() -> list[PositionOut]:
        return [_position_out(p) for p in runtime.positions_snapshot()]

    @app.get("/pnl", response_model=PnlOut)
    def pnl() -> PnlOut:
        return PnlOut(**runtime.pnl_snapshot())

    @app.post("/reconcile", response_model=ReconcileOut)
    async def reconcile(request: ReconcileRequest | None = None) -> ReconcileOut:
        if request is not None and request.inject_break:
            # A synthetic, unmistakable break for smoke-testing the path
            # itself (`atrader reconcile --inject-break`, acceptance
            # criterion #6) — a local position the broker has never heard of.
            runtime.storage.positions.upsert(
                Position(symbol="__diagnostic_break__", quantity=Decimal("1"))
            )
        report = await runtime.reconciler.reconcile()
        return ReconcileOut(
            clean=report.is_clean,
            breaks=[b.message for b in report.breaks],
            positions_checked=report.positions_checked,
            orders_checked=report.orders_checked,
        )

    @app.get("/audit")
    def audit_records() -> Response:
        body = records_to_json_bytes(runtime.storage.audit.read_all())
        return Response(content=body, media_type="application/json")

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=runtime.metrics.render(), media_type="text/plain; version=0.0.4")

    return app
