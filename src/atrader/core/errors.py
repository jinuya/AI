"""Exception hierarchy.

Broker errors are split into the three classes from spec §6.4, because they
demand genuinely different handling:

* **transient** — retry with backoff.
* **permanent** — fail immediately; retrying changes nothing.
* **ambiguous** — the request went out and no response came back. *Never*
  auto-resend. Query for the actual state; if that fails, lock the symbol and
  call a human. This is the classic cause of duplicate fills.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ATraderError",
    "AmbiguousBrokerError",
    "ApprovalRequiredError",
    "ApprovalTimeoutError",
    "BrokerError",
    "ConfigError",
    "DataQualityError",
    "FailClosedError",
    "IllegalStateTransitionError",
    "KillSwitchEngagedError",
    "LLMError",
    "LLMRefusalError",
    "LLMSchemaError",
    "PermanentBrokerError",
    "ReconciliationBreakError",
    "RiskRejectionError",
    "TransientBrokerError",
    "UnsupportedByBrokerError",
]


class ATraderError(Exception):
    """Base class for every error raised by this package."""


# --------------------------------------------------------------------------
# Configuration and startup
# --------------------------------------------------------------------------


class ConfigError(ATraderError):
    """Configuration is missing, malformed, or fails validation."""


class FailClosedError(ATraderError):
    """A safety component is unavailable, so the system refuses to trade.

    Spec §7.1: if the risk engine does not answer, orders are rejected; if risk
    configuration cannot be loaded, the system refuses to boot.
    """


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------


class DataQualityError(ATraderError):
    """Incoming market data failed validation (spec §FR-MD-03)."""

    def __init__(self, symbol: str, reason: str) -> None:
        super().__init__(f"{symbol}: {reason}")
        self.symbol = symbol
        self.reason = reason


# --------------------------------------------------------------------------
# Order lifecycle
# --------------------------------------------------------------------------


class IllegalStateTransitionError(ATraderError):
    """An undefined order state transition was attempted (spec §FR-EXE-01)."""

    def __init__(self, order_id: str, from_status: str, to_status: str) -> None:
        super().__init__(f"order {order_id}: illegal transition {from_status} -> {to_status}")
        self.order_id = order_id
        self.from_status = from_status
        self.to_status = to_status


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


class RiskRejectionError(ATraderError):
    """The risk engine refused an intent."""

    def __init__(self, check_name: str, reason: str, *, intent_id: str | None = None) -> None:
        prefix = f"intent {intent_id}: " if intent_id else ""
        super().__init__(f"{prefix}rejected by {check_name}: {reason}")
        self.check_name = check_name
        self.reason = reason
        self.intent_id = intent_id


class KillSwitchEngagedError(ATraderError):
    """The kill switch is active; no new orders may be sent (spec §FR-MON-03)."""


class ApprovalRequiredError(ATraderError):
    """A human must approve this action before it proceeds (spec §7.6)."""

    def __init__(self, request_id: str, reason: str) -> None:
        super().__init__(f"approval {request_id} required: {reason}")
        self.request_id = request_id
        self.reason = reason


class ApprovalTimeoutError(ATraderError):
    """No answer within the approval window, so the request was auto-denied.

    Spec §7.6: *"응답이 없는데 나중에 실행되는 게 더 위험하다."*
    """


# --------------------------------------------------------------------------
# Broker (spec §6.4)
# --------------------------------------------------------------------------


class BrokerError(ATraderError):
    """Base class for broker adapter failures."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class TransientBrokerError(BrokerError):
    """Timeout, 429, 5xx, dropped connection. Safe to retry with backoff."""


class PermanentBrokerError(BrokerError):
    """400, insufficient funds, invalid symbol, trading halted. Do not retry."""


class AmbiguousBrokerError(BrokerError):
    """The request was sent but no response arrived — the dangerous case.

    Never auto-resend on this. Query the broker for the order's real state; if
    that cannot be determined, lock the symbol and escalate to a human.
    """

    def __init__(
        self, message: str, *, client_order_id: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details=details)
        self.client_order_id = client_order_id


class UnsupportedByBrokerError(PermanentBrokerError):
    """The adapter rejected the request up front via ``BrokerCapabilities``.

    Spec §6.1: failing here beats a round trip to the exchange that comes back
    rejected anyway.
    """


class ReconciliationBreakError(ATraderError):
    """Local state disagrees with the broker (spec §FR-EXE-05).

    The broker is the source of truth. New orders stop until a human resolves
    it; the system never auto-corrects.
    """

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"reconciliation break ({kind}): {detail}")
        self.kind = kind
        self.detail = detail


# --------------------------------------------------------------------------
# LLM strategy (spec §7.7)
# --------------------------------------------------------------------------


class LLMError(ATraderError):
    """Base class for LLM strategy failures."""


class LLMSchemaError(LLMError):
    """The model's output failed schema validation.

    Spec §7.7 allows exactly one retry, then the cycle is skipped. Attempting to
    salvage the text by parsing it is forbidden.
    """


class LLMRefusalError(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"model refused (category={category}): {explanation}")
        self.category = category
        self.explanation = explanation
