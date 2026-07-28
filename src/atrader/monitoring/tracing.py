"""Trace propagation, signal to fill — spec §9.2.

    시그널 발생부터 체결까지 하나의 trace_id로 엮여야 사고 조사 때 "이 주문이
    왜 나갔는가"를 로그 grep 하나로 재구성할 수 있다.

The actual propagation mechanism — an ambient value inherited by nested calls
without threading it through every function signature, the same shape
OpenTelemetry's own context propagation takes — already lives in
:mod:`atrader.monitoring.logging` (``trace_context``, backed by
``contextvars``) because every log line needs it, not just tracing. This
module is the entry point for the trading pipeline specifically:
:func:`start_trace` mints the id one signal-to-fill chain gets.
"""

from __future__ import annotations

from atrader.core.ids import IdGenerator
from atrader.monitoring.logging import bind_trace_id, current_trace_id, trace_context

__all__ = ["bind_trace_id", "current_trace_id", "start_trace", "trace_context"]


def start_trace(ids: IdGenerator) -> str:
    """Mint a fresh trace id for one strategy decision cycle.

    Call once per cycle (e.g. once per ``on_bar``), before any risk check,
    order, or fill that traces back to it — wrap the rest of the cycle in
    ``with trace_context(start_trace(ids)):`` so every log line and audit
    record produced along the way carries it. Uses the same
    :class:`~atrader.core.ids.IdGenerator` every other id in the system comes
    from, so a backtest or replay run's trace ids are exactly as reproducible
    as its order ids — a live run's are just as unpredictable, by the same
    design (spec §2.2).
    """
    return str(ids.new_id())
