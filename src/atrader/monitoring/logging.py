"""Structured logging.

Spec §9.3 fixes the shape of every log line — ``timestamp`` (UTC nanoseconds),
``level``, ``component``, ``trace_id``, ``event_type``, ``payload`` — and it says
plainly why: *"grep 가능한 평문 로그는 나중에 반드시 후회한다."* When you are
reconstructing an incident you want to filter on a trace id, not write a regex.

Secret masking runs as a processor rather than being left to call sites. Spec
§6.2/§9.3 asks for *"로거 레벨에서 필터를 걸어 실수를 구조적으로 방지"* — a rule
that says "remember not to log the API key" is a rule that gets broken once.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, Final

import structlog
from structlog.typing import EventDict, WrappedLogger

from atrader.config.secrets import Secret, is_sensitive_key, redact
from atrader.core.clock import Clock, SystemClock

__all__ = [
    "bind_trace_id",
    "configure_logging",
    "current_trace_id",
    "get_logger",
    "mask_secrets",
    "trace_context",
]

#: Propagated from signal generation all the way to the fill (spec §9.2), so a
#: single order can be followed across every component.
_trace_id: ContextVar[str | None] = ContextVar("atrader_trace_id", default=None)

#: Keys that stay at the top level; everything else is nested under ``payload``.
_RESERVED_KEYS: Final[frozenset[str]] = frozenset(
    {"timestamp", "level", "component", "trace_id", "event_type", "exception", "logger"}
)

_clock: Clock = SystemClock()
_configured = False


def current_trace_id() -> str | None:
    """The trace id bound to the current context, if any."""
    return _trace_id.get()


def bind_trace_id(trace_id: str | None) -> None:
    """Bind a trace id for the current context."""
    _trace_id.set(trace_id)


class trace_context:  # noqa: N801 — used as a context manager, reads as one
    """Bind a trace id for the duration of a block."""

    __slots__ = ("_token", "_trace_id")

    def __init__(self, trace_id: str) -> None:
        self._trace_id = trace_id
        self._token: Any = None

    def __enter__(self) -> str:
        self._token = _trace_id.set(self._trace_id)
        return self._trace_id

    def __exit__(self, *exc_info: object) -> None:
        _trace_id.reset(self._token)


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


def _add_timestamp(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    event_dict["timestamp"] = _clock.now_ns()
    return event_dict


def _add_trace_id(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    event_dict.setdefault("trace_id", _trace_id.get())
    return event_dict


def _rename_event_to_event_type(
    _logger: WrappedLogger, _name: str, event_dict: EventDict
) -> EventDict:
    if "event" in event_dict:
        event_dict["event_type"] = event_dict.pop("event")
    return event_dict


def mask_secrets(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """Redact credentials before anything is written.

    Handles three shapes: :class:`Secret` wrappers, values under a
    credential-looking key, and raw strings that *look* like a key wherever they
    appear (an API key pasted into a free-text error message still gets caught).
    """
    return _mask_mapping(event_dict)


def _mask_mapping(mapping: EventDict) -> EventDict:
    for key, value in list(mapping.items()):
        mapping[key] = _mask_value(key, value)
    return mapping


def _mask_value(key: str, value: object) -> object:
    if isinstance(value, Secret):
        return repr(value)
    if isinstance(value, str):
        return "***REDACTED***" if is_sensitive_key(key) else redact(value)
    if isinstance(value, dict):
        return {str(k): _mask_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        masked = [_mask_value(key, item) for item in value]
        return type(value)(masked) if isinstance(value, tuple) else masked
    return value


def _nest_payload(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """Collect non-reserved keys under ``payload`` (spec §9.3 field list)."""
    payload = {k: v for k, v in event_dict.items() if k not in _RESERVED_KEYS}
    for key in payload:
        del event_dict[key]
    event_dict["payload"] = payload
    return event_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = True,
    clock: Clock | None = None,
    force: bool = False,
) -> None:
    """Install the logging pipeline. Idempotent unless ``force`` is set.

    Pass a :class:`~atrader.core.clock.SimulatedClock` in tests and replay so log
    timestamps are reproducible along with everything else.
    """
    global _clock, _configured
    if _configured and not force:
        return
    if clock is not None:
        _clock = clock

    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        _add_timestamp,
        _add_trace_id,
        _rename_event_to_event_type,
        mask_secrets,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _nest_payload,
    ]
    processors.append(
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to *component*.

    Component names match the spec §4.2 boxes — ``market_data``, ``risk``,
    ``execution``, ``portfolio`` — so a filter on one component gives you one
    subsystem's story.
    """
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger().bind(component=component)
    return logger
