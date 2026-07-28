"""The LLM's only channel to the outside world — spec §7.7.

Nothing in this module ever reaches a broker (see ``strategy/base.py`` and the
import-linter contracts in ``pyproject.toml``: ``atrader.strategy`` cannot
import ``atrader.brokers`` or ``atrader.execution`` at all). A model call goes
in, a schema-validated :class:`LLMTradingResponse` comes out, and everything
downstream — :mod:`atrader.strategy.llm.guards`, then the risk engine — treats
that response exactly as untrusted as a news feed.

Three implementations of :class:`LLMClient`:

* :class:`AnthropicLLMClient` — the real thing. Structured output is forced
  via ``client.messages.parse(output_format=LLMTradingResponse)`` rather than
  asking the model for JSON and parsing a string, and ``stop_reason ==
  "refusal"`` is checked before any content is read.
* :class:`RecordedLLMClient` — record/replay by request hash. Spec §2.2 and
  §7.7 ask for ``temperature=0`` determinism; current Claude models reject
  that parameter outright (see ``docs/llm-determinism.md``), so this is the
  mechanism that actually makes a backtest or replay byte-reproducible.
* :class:`FakeLLMClient` — canned responses for unit tests. No network access,
  ever.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from atrader.core.clock import Clock
from atrader.core.errors import LLMError, LLMRefusalError, LLMSchemaError

__all__ = [
    "AnthropicLLMClient",
    "FakeLLMClient",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMTradingDecision",
    "LLMTradingResponse",
    "LLMUsage",
    "RecordedLLMClient",
]


# ---------------------------------------------------------------------------
# The structured output schema. Numeric fields are decimal strings, not
# ``float`` or JSON Schema's native ``number`` — this project treats float as
# categorically unsafe for money and quantities, and there is no reason the
# one boundary where an LLM invents numbers should be the exception. Strings
# also sidestep the JSON Schema limitation that numeric constraints
# (minimum/maximum) are not enforced server-side; guards.py is the actual
# enforcement point regardless.
# ---------------------------------------------------------------------------


class LLMTradingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    action: Literal["buy", "sell", "hold"]
    quantity: str | None = None
    """Decimal string, whole shares. Required unless ``action == "hold"``."""
    confidence: str
    """Decimal string in ``[0, 1]``. The model's genuine belief, not a knob to
    inflate — a downstream guard scales position size by this number."""
    rationale: str
    stop_loss: str | None = None
    take_profit: str | None = None


class LLMTradingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: tuple[LLMTradingDecision, ...] = ()


# ---------------------------------------------------------------------------
# Request / response envelopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system_prompt: str
    user_content: str
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_tokens: int

    def cache_key(self) -> str:
        """Stable hash of everything that determines the response.

        This is :class:`RecordedLLMClient`'s lookup key, so it must hash
        identically across processes and Python versions — hence sorted-key
        JSON rather than ``repr``/``hash()``, neither of which makes that
        promise.
        """
        payload = {
            "system_prompt": self.system_prompt,
            "user_content": self.user_content,
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    decisions: LLMTradingResponse
    model: str
    """The model that actually answered. Compare to the requested model to
    catch alias drift or a server-side refusal fallback (spec §7.7)."""
    stop_reason: str
    usage: LLMUsage
    latency_ms: int
    request_id: str | None
    raw_text: str
    """The parsed decisions re-serialised to JSON, kept for the audit log
    (spec §9.3) — never re-parsed by this system."""


@runtime_checkable
class LLMClient(Protocol):
    """Source of trading decisions from a language model."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send *request* and return a schema-validated response.

        Raises :class:`~atrader.core.errors.LLMRefusalError` when the model
        declines (checked before any content is read) and
        :class:`~atrader.core.errors.LLMSchemaError` when no valid
        :class:`LLMTradingResponse` could be extracted — truncation at
        ``max_tokens`` included, since truncated JSON fails schema
        validation the same way malformed JSON would.
        """
        ...


# ---------------------------------------------------------------------------
# Live Anthropic client
# ---------------------------------------------------------------------------

#: Beta header for the ``fallbacks: "default"`` scalar form (distinct from the
#: array form's ``server-side-fallback-2026-06-01``).
_REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMClient:
    """Real Anthropic API calls.

    Never used directly for a backtest or a replay run — those go through
    :class:`RecordedLLMClient` instead, which is the actual determinism
    mechanism (see module docstring and ``docs/llm-determinism.md``).
    """

    def __init__(
        self,
        *,
        api_key: str,
        clock: Clock,
        enable_refusal_fallback: bool = False,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._clock = clock
        self._enable_refusal_fallback = enable_refusal_fallback

    def complete(self, request: LLMRequest) -> LLMResponse:
        import pydantic

        start_ns = self._clock.monotonic_ns()
        # Typed as Any: these mirror the Anthropic SDK's own TypedDict shapes,
        # but the SDK is an optional dependency (pyproject `llm` extra) and
        # imported lazily above — pinning mypy to its exact param types here
        # would make this file fail to type-check in an environment where the
        # extra is intentionally not installed.
        system: Any = [
            {
                "type": "text",
                "text": request.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages: Any = [{"role": "user", "content": request.user_content}]
        output_config: Any = {"effort": request.effort}

        try:
            response: Any
            if self._enable_refusal_fallback:
                response = self._client.beta.messages.parse(
                    betas=[_REFUSAL_FALLBACK_BETA],
                    fallbacks="default",
                    model=request.model,
                    max_tokens=request.max_tokens,
                    system=system,
                    messages=messages,
                    output_format=LLMTradingResponse,
                    output_config=output_config,
                )
            else:
                response = self._client.messages.parse(
                    model=request.model,
                    max_tokens=request.max_tokens,
                    system=system,
                    messages=messages,
                    output_format=LLMTradingResponse,
                    output_config=output_config,
                )
        except pydantic.ValidationError as exc:
            raise LLMSchemaError(f"response failed schema validation: {exc}") from exc

        latency_ms = (self._clock.monotonic_ns() - start_ns) // 1_000_000
        stop_reason = str(response.stop_reason)

        # Checked before reading any content — spec §7.7.
        if stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details else None
            explanation = details.explanation if details else None
            raise LLMRefusalError(category, explanation)

        parsed = response.parsed_output
        if parsed is None:
            raise LLMSchemaError(
                f"no parsable structured output in response (stop_reason={stop_reason!r})"
            )

        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
        )
        return LLMResponse(
            decisions=parsed,
            model=str(response.model),
            stop_reason=stop_reason,
            usage=usage,
            latency_ms=int(latency_ms),
            request_id=getattr(response, "_request_id", None),
            raw_text=parsed.model_dump_json(),
        )


# ---------------------------------------------------------------------------
# Record/replay client — the actual determinism mechanism (spec §2.2, §8.3)
# ---------------------------------------------------------------------------


def _usage_to_dict(usage: LLMUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
    }


def _response_to_dict(request_hash: str, response: LLMResponse) -> dict[str, object]:
    return {
        "request_hash": request_hash,
        "decisions": response.decisions.model_dump(mode="json"),
        "model": response.model,
        "stop_reason": response.stop_reason,
        "usage": _usage_to_dict(response.usage),
        "latency_ms": response.latency_ms,
        "request_id": response.request_id,
        "raw_text": response.raw_text,
    }


def _response_from_dict(data: Mapping[str, object]) -> LLMResponse:
    usage_data = data["usage"]
    assert isinstance(usage_data, Mapping)
    latency_ms = data["latency_ms"]
    assert isinstance(latency_ms, int)
    request_id = data["request_id"]
    assert request_id is None or isinstance(request_id, str)
    return LLMResponse(
        decisions=LLMTradingResponse.model_validate(data["decisions"]),
        model=str(data["model"]),
        stop_reason=str(data["stop_reason"]),
        usage=LLMUsage(
            input_tokens=int(usage_data["input_tokens"]),
            output_tokens=int(usage_data["output_tokens"]),
            cache_read_input_tokens=int(usage_data["cache_read_input_tokens"]),
            cache_creation_input_tokens=int(usage_data["cache_creation_input_tokens"]),
        ),
        latency_ms=latency_ms,
        request_id=request_id,
        raw_text=str(data["raw_text"]),
    )


class RecordedLLMClient:
    """Record/replay by a hash of the request.

    In *replay* mode (``upstream=None``) every call must hit a recording made
    earlier — that byte-for-byte reproducibility is what lets
    :mod:`atrader.backtest.replay` compare two runs and is what spec §7.7's
    "output variance monitoring" is actually monitoring for drift against. In
    *record* mode (an ``upstream`` client supplied), a live call is made once
    per distinct request and the result is cached for every future replay.
    """

    def __init__(
        self,
        *,
        recordings: Mapping[str, LLMResponse] | None = None,
        upstream: LLMClient | None = None,
        on_record: Callable[[str, LLMResponse], None] | None = None,
    ) -> None:
        self._recordings: dict[str, LLMResponse] = dict(recordings or {})
        self._upstream = upstream
        self._on_record = on_record

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()
        cached = self._recordings.get(key)
        if cached is not None:
            return cached
        if self._upstream is None:
            raise LLMError(
                f"no recording for request hash {key} and no upstream client configured "
                "— this client is in replay-only mode (spec §8.3 determinism)"
            )
        response = self._upstream.complete(request)
        self._recordings[key] = response
        if self._on_record is not None:
            self._on_record(key, response)
        return response

    def __len__(self) -> int:
        return len(self._recordings)

    def to_jsonl(self, path: Path) -> None:
        """Persist every recording as one JSON object per line."""
        with path.open("w", encoding="utf-8") as handle:
            for key, response in self._recordings.items():
                handle.write(json.dumps(_response_to_dict(key, response), ensure_ascii=False))
                handle.write("\n")

    @classmethod
    def from_jsonl(cls, path: Path, *, upstream: LLMClient | None = None) -> RecordedLLMClient:
        """Load recordings written by :meth:`to_jsonl`."""
        recordings: dict[str, LLMResponse] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                recordings[str(data["request_hash"])] = _response_from_dict(data)
        return cls(recordings=recordings, upstream=upstream)


# ---------------------------------------------------------------------------
# Fake client for unit tests — no network access, ever.
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMClient:
    """Scripted responses for tests.

    Pass ``responses`` for a fixed sequence consumed in order (one per call),
    or ``default`` as a fallback once the queue is empty, or ``raises`` to
    make every call fail the same way (e.g. simulate a refusal). Every request
    is recorded on :attr:`requests` so a test can assert on exactly what
    :mod:`atrader.strategy.llm.prompt` built — the untrusted-data tagging in
    particular.
    """

    responses: Iterable[LLMResponse] = ()
    default: LLMResponse | None = None
    raises: Exception | None = None
    requests: list[LLMRequest] = field(default_factory=list)
    _queue: deque[LLMResponse] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = deque(self.responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self._queue:
            return self._queue.popleft()
        if self.default is not None:
            return self.default
        raise LLMError("FakeLLMClient has no more scripted responses")
