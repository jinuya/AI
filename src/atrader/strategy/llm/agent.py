"""The LLM agent strategy — spec §7.7.

Ties :mod:`atrader.strategy.llm.client`, :mod:`atrader.strategy.llm.prompt`,
and :mod:`atrader.strategy.llm.guards` together into a
:class:`~atrader.strategy.base.Strategy`. Structurally it is no different
from :class:`~atrader.strategy.rules.sma_crossover.SmaCrossoverStrategy` —
same base class, same ``on_bar`` signature, same inability to import
``atrader.brokers`` or ``atrader.execution`` — because spec §2.2 requires
strategy code to be identical in backtest and live trading, and an LLM
strategy that needed a special code path would violate that the moment it
mattered.

Every model call, response, and guard rejection is written to the audit log
(spec §7.7, §9.2) — this is the strategy where "AI 모델은 절대 브로커 API를 직접
호출하지 않는다" (the model never calls the broker) stops being an abstract
principle and becomes the concrete fact that this class returns
:class:`~atrader.core.models.TradingIntent` objects and nothing else, exactly
like every other strategy, leaving the risk engine as the only thing that can
turn one into an order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from atrader.audit.logger import AuditEvent, AuditLogger
from atrader.config.schema import LLMConfig
from atrader.core.errors import LLMRefusalError, LLMSchemaError
from atrader.core.models import TradingIntent
from atrader.marketdata.models import Bar
from atrader.strategy.base import Strategy, StrategyContext
from atrader.strategy.llm.client import LLMClient, LLMRequest, LLMResponse
from atrader.strategy.llm.guards import apply_guards
from atrader.strategy.llm.prompt import build_system_prompt, build_user_content

__all__ = ["LLMAgentStrategy"]

#: Untrusted external context for one symbol: (source label, raw text).
NewsProvider = Callable[[str], Sequence[tuple[str, str]]]


class LLMAgentStrategy(Strategy):
    """Trading decisions from a language model, gated by deterministic guards.

    ``decision_interval_bars`` controls how often the model is actually
    called per symbol (default: every bar). A larger interval trades reaction
    speed for cost — the model call is the expensive step, everything else in
    this class is nearly free.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: Iterable[str],
        *,
        client: LLMClient,
        config: LLMConfig,
        audit: AuditLogger | None = None,
        decision_interval_bars: int = 1,
        news_provider: NewsProvider | None = None,
    ) -> None:
        super().__init__(strategy_id)
        if decision_interval_bars < 1:
            raise ValueError(f"decision_interval_bars must be >= 1, got {decision_interval_bars}")
        self._client = client
        self._config = config
        self._audit = audit
        self._decision_interval_bars = decision_interval_bars
        self._news_provider = news_provider
        self._symbols = frozenset(symbols)
        self._system_prompt = build_system_prompt(symbols=sorted(self._symbols))
        self._bar_count: dict[str, int] = {}
        self._call_count = 0
        self._rejection_count = 0

    def on_bar(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        if bar.symbol not in self._symbols:
            return []

        count = self._bar_count.get(bar.symbol, 0) + 1
        self._bar_count[bar.symbol] = count
        if count % self._decision_interval_bars != 0:
            return []

        return self._decide(bar, context)

    def snapshot(self) -> dict[str, Any]:
        return {
            "bar_count": dict(self._bar_count),
            "call_count": self._call_count,
            "rejection_count": self._rejection_count,
        }

    # -- internals ----------------------------------------------------

    def _decide(self, bar: Bar, context: StrategyContext) -> list[TradingIntent]:
        position = context.position_of(bar.symbol)
        untrusted = tuple(self._news_provider(bar.symbol)) if self._news_provider else ()
        user_content = build_user_content(
            as_of_ns=context.now_ns,
            market_summary=(
                f"{bar.symbol} {bar.interval} bar: open={bar.open} high={bar.high} "
                f"low={bar.low} close={bar.close} volume={bar.volume}"
            ),
            positions_summary=(
                f"{bar.symbol}: quantity={position.quantity} avg_price={position.avg_price} "
                f"unrealized_pnl={position.unrealized_pnl}"
            ),
            account_summary=(
                f"cash={context.account.cash} equity={context.account.equity} "
                f"buying_power={context.account.buying_power}"
            ),
            untrusted_sections=untrusted,
        )
        request = LLMRequest(
            system_prompt=self._system_prompt,
            user_content=user_content,
            model=self._config.model,
            effort=self._config.effort,
            max_tokens=self._config.max_tokens,
        )

        response = self._call_with_retry(request)
        if response is None:
            return []

        if response.model != self._config.model:
            self._log(
                AuditEvent.LLM_RESPONSE,
                payload={
                    "warning": "response model differs from the requested model",
                    "requested_model": self._config.model,
                    "response_model": response.model,
                },
            )

        guard_result = apply_guards(
            response.decisions,
            context=context,
            strategy_id=self.strategy_id,
            symbol_whitelist=self._symbols,
            min_confidence=self._config.min_confidence,
            require_symbol_whitelist=self._config.require_symbol_whitelist,
            reference_prices={bar.symbol: bar.close},
        )
        for rejection in guard_result.rejections:
            self._rejection_count += 1
            self._log(
                AuditEvent.LLM_REJECTED,
                payload={
                    "symbol": rejection.symbol,
                    "reason": rejection.reason,
                    "critical": rejection.critical,
                },
            )
        return list(guard_result.intents)

    def _call_with_retry(self, request: LLMRequest) -> LLMResponse | None:
        attempts = self._config.schema_retry + 1
        for attempt in range(attempts):
            self._call_count += 1
            self._log(
                AuditEvent.LLM_REQUEST,
                payload={
                    "attempt": attempt,
                    "model": request.model,
                    "effort": request.effort,
                    "max_tokens": request.max_tokens,
                    "system_prompt": request.system_prompt
                    if self._config.log_full_prompts
                    else None,
                    "user_content": request.user_content if self._config.log_full_prompts else None,
                },
            )
            try:
                response = self._client.complete(request)
            except LLMRefusalError as exc:
                self._log(
                    AuditEvent.LLM_REJECTED,
                    payload={
                        "reason": "refusal",
                        "category": exc.category,
                        "explanation": exc.explanation,
                    },
                )
                return None
            except LLMSchemaError as exc:
                self._log(
                    AuditEvent.LLM_REJECTED,
                    payload={"reason": "schema_error", "detail": str(exc), "attempt": attempt},
                )
                continue

            self._log(
                AuditEvent.LLM_RESPONSE,
                payload={
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                        "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                    },
                    "latency_ms": response.latency_ms,
                    "request_id": response.request_id,
                    "raw_text": response.raw_text,
                },
            )
            return response

        # Schema retries exhausted — spec §7.7: skip the cycle, never salvage
        # the text by parsing it ourselves.
        return None

    def _log(self, event_type: str, *, payload: dict[str, Any]) -> None:
        if self._audit is not None:
            self._audit.append(event_type, actor=f"strategy:{self.strategy_id}", payload=payload)
