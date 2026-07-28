"""HTTP control surface — spec §10, "/health /kill /approve /positions /pnl"."""

from __future__ import annotations

from decimal import Decimal

from fastapi.testclient import TestClient

from atrader.app.api import create_app
from atrader.app.runtime import Runtime
from atrader.audit.hashchain import records_from_json_bytes
from atrader.config.schema import (
    AppConfig,
    InstrumentSpec,
    MarketDataConfig,
    RiskConfig,
    UniverseConfig,
)
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import Position
from atrader.features.engine import FeatureEngine
from atrader.features.registry import FeatureRegistry
from atrader.features.store import FeatureStore
from atrader.marketdata.feeds.replay import ReplayFeed
from atrader.marketdata.models import Tick
from atrader.risk.approval import ApprovalKind
from atrader.strategy.base import Strategy

BASE_NS = 1_700_000_000 * NS_PER_SECOND


class _NeverBuy(Strategy):
    def on_bar(self, bar, context):  # type: ignore[no-untyped-def]
        return []


def make_config(**overrides: object) -> AppConfig:
    return AppConfig(
        account_equity=Decimal("100000"),
        risk=RiskConfig(**overrides) if overrides else RiskConfig(),
        universe=UniverseConfig(symbols=("AAPL",), sectors={"AAPL": "TECHNOLOGY"}),
        instruments=(
            InstrumentSpec(
                symbol="AAPL",
                sector="TECHNOLOGY",
                market_open_utc="00:00",
                market_close_utc="23:59",
            ),
        ),
        market_data=MarketDataConfig(bar_intervals=("1s",)),
    )


def make_tick(*, offset_seconds: int, price: str = "100", seq: int = 1) -> Tick:
    ts = BASE_NS + offset_seconds * NS_PER_SECOND
    return Tick(
        symbol="AAPL",
        exchange_ts=ts,
        ingest_ts=ts + 1_000_000,
        last=Decimal(price),
        last_size=Decimal("1000"),
        seq=seq,
    )


def make_client(**config_overrides: object) -> tuple[TestClient, Runtime]:
    clock = SimulatedClock(start_ns=BASE_NS)
    ids = DeterministicIdGenerator(clock, seed=1)
    feature_engine = FeatureEngine(store=FeatureStore(), registry=FeatureRegistry(), specs=())
    runtime = Runtime(
        make_config(**config_overrides),
        [_NeverBuy("idle")],
        feature_engine,
        clock=clock,
        ids=ids,
        feed=ReplayFeed([]),
    )
    app = create_app(runtime)
    return TestClient(app), runtime


class TestHealth:
    def test_reports_not_running_before_start(self) -> None:
        client, _ = make_client()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["running"] is False

    def test_reports_running_after_start(self) -> None:
        import asyncio

        client, runtime = make_client()
        asyncio.run(runtime.start())
        response = client.get("/health")
        assert response.json()["system_state"] == "RUNNING"
        assert Decimal(str(response.json()["equity"])) == Decimal("100000")


class TestKill:
    def test_engaging_is_reflected_in_health(self) -> None:
        client, _ = make_client()
        response = client.post("/kill", json={"reason": "test", "actor": "tester"})
        assert response.status_code == 202
        assert response.json()["engaged"] is True

        health = client.get("/health").json()
        assert health["kill_switch_engaged"] is True

    def test_release_reopens_the_gate(self) -> None:
        client, _ = make_client()
        client.post("/kill", json={"reason": "test", "actor": "tester"})
        response = client.request("DELETE", "/kill", json={"reason": "resolved", "actor": "tester"})
        assert response.status_code == 202
        assert response.json()["engaged"] is False
        assert client.get("/health").json()["kill_switch_engaged"] is False


class TestApprovals:
    def test_pending_is_empty_by_default(self) -> None:
        client, _ = make_client()
        assert client.get("/approve").json() == []

    def test_granting_an_unknown_request_is_a_404(self) -> None:
        client, _ = make_client()
        response = client.post(
            "/approve/00000000-0000-0000-0000-000000000000", json={"by": "a_human"}
        )
        assert response.status_code == 404

    def test_a_real_pending_request_can_be_granted(self) -> None:
        client, runtime = make_client()
        request = runtime.approvals.request(ApprovalKind.LARGE_ORDER, "buy 10 AAPL")
        response = client.post(f"/approve/{request.request_id}", json={"by": "a_human"})
        assert response.status_code == 200
        assert response.json()["status"] == "granted"


class TestPositionsAndPnl:
    def test_positions_starts_empty(self) -> None:
        client, _ = make_client()
        assert client.get("/positions").json() == []

    def test_pnl_has_the_expected_shape(self) -> None:
        client, _ = make_client()
        response = client.get("/pnl").json()
        assert set(response) == {"realized", "unrealized", "equity"}

    def test_positions_reflects_storage(self) -> None:
        client, runtime = make_client()
        runtime.storage.positions.upsert(Position(symbol="AAPL", quantity=Decimal("5")))
        response = client.get("/positions").json()
        assert len(response) == 1
        assert response[0]["symbol"] == "AAPL"


class TestReconcile:
    def test_a_clean_book_reports_clean(self) -> None:
        client, _ = make_client()
        response = client.post("/reconcile", json={})
        assert response.status_code == 200
        assert response.json()["clean"] is True

    def test_inject_break_produces_a_dirty_report(self) -> None:
        client, _ = make_client()
        response = client.post("/reconcile", json={"inject_break": True})
        assert response.json()["clean"] is False
        assert len(response.json()["breaks"]) >= 1


class TestAuditAndMetrics:
    def test_audit_round_trips_through_the_hashchain_helpers(self) -> None:
        client, runtime = make_client()
        runtime.audit.append("test.event", payload={"k": "v"})
        response = client.get("/audit")
        assert response.status_code == 200
        records = records_from_json_bytes(response.content)
        assert len(records) == 1
        assert records[0].event_type == "test.event"

    def test_metrics_is_prometheus_text_format(self) -> None:
        client, _ = make_client()
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"atrader_" in response.content
