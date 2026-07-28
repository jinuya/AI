"""Configuration schema.

Spec §4.6: strategy parameters and risk limits live in version-controlled YAML,
never in code, and changing a limit requires a reviewed PR. *"런타임에 임의로
한도를 올릴 수 있으면 그건 한도가 아니다."*

Two decisions worth flagging:

**Every model forbids extra keys.** A typo'd risk limit that silently falls back
to a default is exactly the failure mode this system cannot afford, so
``extra="forbid"`` turns it into a boot-time error instead.

**Every model is frozen.** Nothing can mutate a limit after load. Combined with
fail-closed loading (spec §7.1), that means the limits in force are always the
limits that were reviewed.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atrader.core.types import AssetClass, CostBasisMethod, OrderType, SizingMethod

__all__ = [
    "AccountLimits",
    "AppConfig",
    "CircuitBreakerConfig",
    "DataQualityConfig",
    "ExecutionConfig",
    "InstrumentSpec",
    "LLMConfig",
    "MarketDataConfig",
    "OrderLimits",
    "PositionLimits",
    "RiskConfig",
    "SizingConfig",
    "StopsConfig",
    "StrategyConfig",
    "UniverseConfig",
]

Pct = Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(100))]
PositiveDecimal = Annotated[Decimal, Field(gt=Decimal(0))]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal(0))]


class _Base(BaseModel):
    """Frozen, extra-forbidding base for every config model."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)


# ---------------------------------------------------------------------------
# Risk limits (spec §13)
# ---------------------------------------------------------------------------


class AccountLimits(_Base):
    max_leverage: PositiveDecimal = Decimal("1.0")
    """1.0 means cash trading — no borrowed exposure."""
    daily_loss_limit_pct: Pct = Decimal("2.0")
    max_drawdown_pct: Pct = Decimal("10.0")
    min_cash_buffer_pct: Pct = Decimal("5.0")


class PositionLimits(_Base):
    max_position_pct: Pct = Decimal("10.0")
    max_sector_pct: Pct = Decimal("30.0")
    max_correlated_exposure_pct: Pct = Decimal("25.0")
    correlation_threshold: Annotated[Decimal, Field(ge=Decimal(-1), le=Decimal(1))] = Decimal("0.7")
    """Above this correlation, adding a name is really just growing the same bet."""


class OrderLimits(_Base):
    max_order_notional_pct: Pct = Decimal("2.0")
    max_adv_participation_pct: Pct = Decimal("5.0")
    limit_price_deviation_pct: Pct = Decimal("5.0")
    """Fat-finger guard: reject a limit price more than this far from the market."""
    max_orders_per_minute: Annotated[int, Field(gt=0)] = 30
    duplicate_window_seconds: Annotated[int, Field(ge=0)] = 5
    human_approval_threshold_pct: Pct = Decimal("10.0")
    """Single-order notional above this share of the account needs a human (§7.6)."""
    adv_split_threshold_pct: Pct = Decimal("1.0")
    """Above this share of ADV, the order must be worked by an algo (§FR-EXE-03)."""
    allow_market_orders: bool = False
    """Spec §FR-EXE-02: market orders are off by default. In a thin name the
    slippage is unbounded; an aggressive limit is the recommended default."""
    aggressive_limit_ticks: Annotated[int, Field(ge=0)] = 2
    """How many ticks through the touch an 'aggressive limit' prices."""


class SizingConfig(_Base):
    method: SizingMethod = SizingMethod.VOLATILITY_TARGET
    risk_per_trade_pct: Pct = Decimal("0.5")
    """Designed maximum loss on any single trade, as a share of equity."""
    atr_multiple: PositiveDecimal = Decimal("2.0")
    kelly_fraction: Annotated[Decimal, Field(gt=Decimal(0), le=Decimal("0.25"))] = Decimal("0.25")
    """Capped at quarter-Kelly. Full Kelly is optimal only if your edge estimate
    is exact; it is not, and the ruin probability is brutal (spec §7.3)."""


class StopsConfig(_Base):
    hard_stop_atr_multiple: PositiveDecimal = Decimal("2.0")
    trailing_stop_atr_multiple: PositiveDecimal = Decimal("3.0")
    time_stop_bars: Annotated[int, Field(gt=0)] = 20
    place_at_broker: bool = True
    """Resting the stop at the broker keeps protection alive if we crash."""


class CircuitBreakerConfig(_Base):
    l1_daily_loss_pct: Pct = Decimal("1.0")
    l1_cooldown_minutes: Annotated[int, Field(gt=0)] = 30
    l1_auto_recover: bool = True
    l1_consecutive_losses: Annotated[int, Field(gt=0)] = 3
    l1_consecutive_window_minutes: Annotated[int, Field(gt=0)] = 5

    l2_daily_loss_pct: Pct = Decimal("2.0")
    l2_max_drawdown_pct: Pct = Decimal("7.0")
    l2_auto_recover: bool = False
    """Deliberately false. L2 means something went wrong; a human looks at it."""

    l3_daily_loss_pct: Pct = Decimal("3.0")
    l3_max_drawdown_pct: Pct = Decimal("10.0")
    l3_liquidate_all: bool = True

    anomaly_order_rate_multiple: PositiveDecimal = Decimal("5.0")
    """Orders per minute this far above baseline usually means a loop bug."""
    anomaly_roundtrip_count: Annotated[int, Field(gt=0)] = 4
    """Repeated buy/sell round trips in one symbol — another loop signature."""

    @model_validator(mode="after")
    def _levels_must_escalate(self) -> CircuitBreakerConfig:
        if not (self.l1_daily_loss_pct < self.l2_daily_loss_pct < self.l3_daily_loss_pct):
            raise ValueError(
                "circuit breaker daily-loss thresholds must increase L1 < L2 < L3, got "
                f"{self.l1_daily_loss_pct} / {self.l2_daily_loss_pct} / {self.l3_daily_loss_pct}. "
                "Out-of-order thresholds mean a level can never fire."
            )
        if not self.l2_max_drawdown_pct < self.l3_max_drawdown_pct:
            raise ValueError(
                "circuit breaker drawdown thresholds must increase L2 < L3, got "
                f"{self.l2_max_drawdown_pct} / {self.l3_max_drawdown_pct}"
            )
        return self


# ---------------------------------------------------------------------------
# LLM strategy (spec §7.7)
# ---------------------------------------------------------------------------

#: Model IDs this system is allowed to pin to.
#:
#: Spec §7.7 asks for "날짜가 포함된 정확한 모델 버전" so an alias cannot drift
#: under us. Current Claude IDs carry no date suffix — appending one 404s — so
#: the equivalent guarantee is an explicit allowlist: an unrecognised string is
#: rejected at boot rather than discovered in production.
KNOWN_MODEL_IDS: Final[frozenset[str]] = frozenset(
    {
        "claude-opus-5",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    }
)

#: Models that reject ``temperature`` / ``top_p`` / ``top_k`` with HTTP 400.
SAMPLING_FREE_MODELS: Final[frozenset[str]] = frozenset(
    {
        "claude-opus-5",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
    }
)


class LLMConfig(_Base):
    """Configuration for the LLM agent strategy.

    See ``docs/llm-determinism.md`` for why ``temperature`` is not simply set to
    zero the way spec §13 asks: on current Claude models the parameter has been
    removed and sending it returns a 400.
    """

    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    """Replaces ``temperature`` as the depth/cost control on current models."""
    max_tokens: Annotated[int, Field(gt=0, le=128_000)] = 8_000
    min_confidence: Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(1))] = Decimal("0.6")
    """Below this, size down proportionally or skip the intent entirely."""
    schema_retry: Annotated[int, Field(ge=0, le=3)] = 1
    """Spec §7.7 allows one retry, then skip the cycle. Never salvage by parsing."""
    require_symbol_whitelist: bool = True
    log_full_prompts: bool = True
    enable_refusal_fallback: bool = False
    """Opt in to server-side fallbacks so a safety refusal is re-served by
    another model rather than silently skipping the cycle."""
    variance_alert_threshold: Annotated[Decimal, Field(ge=Decimal(0), le=Decimal(1))] = Decimal(
        "0.2"
    )
    """Fraction of differing responses to identical input that triggers a WARN.
    Spec §7.7: rising variance usually means the model version moved."""
    allow_unknown_model: bool = False
    """Escape hatch for a model released after this allowlist was written.
    Turning it on is a deliberate, reviewable act."""

    temperature: Decimal | None = None
    """Deprecated. Present only because spec §13 names it. Must stay unset on
    any model in :data:`SAMPLING_FREE_MODELS` — see the validator below."""
    top_p: Decimal | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_model_pin(self) -> LLMConfig:
        if not self.allow_unknown_model and self.model not in KNOWN_MODEL_IDS:
            raise ValueError(
                f"model {self.model!r} is not in the pinned allowlist. "
                "Spec §7.7 forbids an unpinned model because the strategy's character "
                "would change the day the alias moves. Add the exact ID to "
                "KNOWN_MODEL_IDS in a reviewed change, or set allow_unknown_model: true "
                "if you accept that risk deliberately."
            )
        return self

    @model_validator(mode="after")
    def _reject_removed_sampling_params(self) -> LLMConfig:
        if self.model not in SAMPLING_FREE_MODELS:
            return self
        offenders = [
            name
            for name, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("seed", self.seed),
            )
            if value is not None
        ]
        if offenders:
            raise ValueError(
                f"{', '.join(offenders)} cannot be used with model {self.model!r}: "
                "these parameters were removed from the Claude API and sending them "
                "returns HTTP 400. Spec §13 asks for temperature=0 to get determinism; "
                "on current models that is achieved with recorded/replayed responses "
                "instead — see docs/llm-determinism.md. Control depth with `effort`."
            )
        return self


# ---------------------------------------------------------------------------
# Data quality (spec §5.3, §FR-MD-03)
# ---------------------------------------------------------------------------


class DataQualityConfig(_Base):
    max_feed_lag_ms: Annotated[int, Field(gt=0)] = 1_000
    stale_threshold_seconds: Annotated[int, Field(gt=0)] = 30
    price_jump_threshold_pct: Pct = Decimal("20.0")
    dual_source_divergence_pct: Pct = Decimal("0.5")
    """Two sources disagreeing by more than this stops new orders. Trading while
    you do not know which price is right is the worst of the options."""
    max_sequence_gap: Annotated[int, Field(ge=0)] = 0
    """Tolerated gap in feed sequence numbers before the symbol is DEGRADED."""
    clock_drift_alert_ms: Annotated[int, Field(gt=0)] = 100


class RiskConfig(_Base):
    """The ``risk:`` block of ``config/risk.yaml`` (spec §13)."""

    account: AccountLimits = AccountLimits()
    position: PositionLimits = PositionLimits()
    order: OrderLimits = OrderLimits()
    sizing: SizingConfig = SizingConfig()
    stops: StopsConfig = StopsConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    llm: LLMConfig = LLMConfig()
    data_quality: DataQualityConfig = DataQualityConfig()

    @model_validator(mode="after")
    def _breaker_must_not_exceed_account_limits(self) -> RiskConfig:
        # An L2 that fires later than the hard daily loss limit is dead code:
        # the pre-trade check would already have rejected everything.
        if self.circuit_breaker.l2_daily_loss_pct > self.account.daily_loss_limit_pct:
            raise ValueError(
                f"circuit breaker L2 ({self.circuit_breaker.l2_daily_loss_pct}%) fires after "
                f"the account daily loss limit ({self.account.daily_loss_limit_pct}%), so it "
                "could never trigger. Lower L2 or raise the account limit."
            )
        if self.circuit_breaker.l3_max_drawdown_pct > self.account.max_drawdown_pct:
            raise ValueError(
                f"circuit breaker L3 drawdown ({self.circuit_breaker.l3_max_drawdown_pct}%) "
                f"exceeds the account max drawdown ({self.account.max_drawdown_pct}%)"
            )
        return self


# ---------------------------------------------------------------------------
# Universe, instruments, strategies
# ---------------------------------------------------------------------------


class InstrumentSpec(_Base):
    """Spec §FR-PF-05: per-asset-class trading rules live in a master table."""

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    tick_size: PositiveDecimal = Decimal("0.01")
    lot_size: PositiveDecimal = Decimal("1")
    contract_multiplier: PositiveDecimal = Decimal("1")
    currency: str = "USD"
    sector: str = "UNKNOWN"
    exchange: str = "UNKNOWN"
    settlement_days: Annotated[int, Field(ge=0)] = 2
    market_open_utc: str = "14:30"
    market_close_utc: str = "21:00"
    shortable: bool = True
    commission_bps: NonNegativeDecimal = Decimal("0")
    tax_bps: NonNegativeDecimal = Decimal("0")
    borrow_bps_annual: NonNegativeDecimal = Decimal("0")


class UniverseConfig(_Base):
    """The tradable whitelist. Anything outside it is rejected (spec §7.2 #2)."""

    symbols: tuple[str, ...] = ()
    sectors: dict[str, str] = Field(default_factory=dict)
    """symbol -> sector, for the sector concentration check."""

    @model_validator(mode="after")
    def _sectors_must_reference_known_symbols(self) -> UniverseConfig:
        unknown = sorted(set(self.sectors) - set(self.symbols))
        if unknown:
            raise ValueError(
                f"sector map references symbols outside the universe: {unknown}. "
                "A stale entry here silently mis-buckets sector exposure."
            )
        return self


class StrategyConfig(_Base):
    strategy_id: str
    version: str = "1"
    enabled: bool = False
    """Off by default. Spec §FR-STR-05: a strategy is promoted deliberately."""
    universe: tuple[str, ...] = ()
    capital_allocation_pct: Pct = Decimal("0")
    params: dict[str, str | int | Decimal | bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Execution, market data, infrastructure
# ---------------------------------------------------------------------------


class ExecutionConfig(_Base):
    default_order_type: OrderType = OrderType.LIMIT
    reconciliation_interval_seconds: Annotated[int, Field(gt=0)] = 30
    deadman_timeout_seconds: Annotated[int, Field(gt=0)] = 60
    cancel_before_close_minutes: Annotated[int, Field(ge=0)] = 10
    rate_limit_per_minute: Annotated[int, Field(gt=0)] = 200
    rate_limit_burst: Annotated[int, Field(gt=0)] = 20
    rate_limit_reserve_pct: Pct = Decimal("20")
    """Spec §6.3: reserve headroom so an emergency cancel is never rate-limited."""
    max_retries: Annotated[int, Field(ge=0, le=10)] = 5
    circuit_breaker_failures: Annotated[int, Field(gt=0)] = 5
    circuit_breaker_open_seconds: Annotated[int, Field(gt=0)] = 30
    cost_basis_method: CostBasisMethod = CostBasisMethod.FIFO
    rebalance_deadband_pct: Pct = Decimal("0.5")
    default_algo: Literal["twap", "vwap", "pov"] = "twap"
    """Which execution algorithm splits an order once it crosses
    ``order.adv_split_threshold_pct``. TWAP is the default because it needs
    nothing but a duration — no volume curve, no live tape."""
    algo_slice_count: Annotated[int, Field(gt=0)] = 10
    algo_duration_seconds: Annotated[int, Field(gt=0)] = 1800
    """How long a TWAP/VWAP schedule is worked over, by default: 30 minutes."""
    pov_participation_rate_pct: Pct = Decimal("10")
    """Share of *incremental* observed volume a POV slice claims. Distinct from
    ``order.max_adv_participation_pct`` (§7.2 #7), which bounds the order's
    total size against the *day's* ADV — this bounds the rate of one slice
    against volume as it prints."""


class MarketDataConfig(_Base):
    primary_source: str = "simulated"
    backup_source: str | None = None
    bar_intervals: tuple[str, ...] = ("1m", "5m", "1h", "1d")
    reconnect_base_seconds: Annotated[int, Field(gt=0)] = 1
    reconnect_max_seconds: Annotated[int, Field(gt=0)] = 60
    reconnect_jitter_pct: Pct = Decimal("20")
    """Spec §FR-MD-01: without jitter, every instance reconnects in lockstep and
    the venue throttles all of them."""
    backfill_on_reconnect: bool = True


class StorageConfig(_Base):
    dsn: str = "sqlite+pysqlite:///:memory:"
    echo_sql: bool = False


class BusConfig(_Base):
    backend: Literal["memory", "redis"] = "memory"
    url: str | None = None
    consumer_group: str = "atrader"


class MonitoringConfig(_Base):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    metrics_port: Annotated[int, Field(gt=0, lt=65536)] = 9090
    heartbeat_interval_seconds: Annotated[int, Field(gt=0)] = 10
    approval_timeout_seconds: Annotated[int, Field(gt=0)] = 300
    """Spec §7.6: an unanswered approval auto-denies. Executing later is worse."""


class AppConfig(_Base):
    """Everything the runtime needs, assembled from ``config/*.yaml``."""

    environment: Literal["dev", "staging", "production"] = "dev"
    account_equity: PositiveDecimal = Decimal("100000")
    base_currency: str = "USD"
    risk: RiskConfig = RiskConfig()
    universe: UniverseConfig = UniverseConfig()
    instruments: tuple[InstrumentSpec, ...] = ()
    strategies: tuple[StrategyConfig, ...] = ()
    execution: ExecutionConfig = ExecutionConfig()
    market_data: MarketDataConfig = MarketDataConfig()
    storage: StorageConfig = StorageConfig()
    bus: BusConfig = BusConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @model_validator(mode="after")
    def _instruments_cover_the_universe(self) -> AppConfig:
        specified = {spec.symbol for spec in self.instruments}
        missing = sorted(set(self.universe.symbols) - specified)
        if missing:
            raise ValueError(
                f"no InstrumentSpec for universe symbols {missing}. Tick size, lot size "
                "and trading hours are needed before an order can be priced correctly."
            )
        return self

    @model_validator(mode="after")
    def _strategy_universes_are_subsets(self) -> AppConfig:
        for strategy in self.strategies:
            outside = sorted(set(strategy.universe) - set(self.universe.symbols))
            if outside:
                raise ValueError(
                    f"strategy {strategy.strategy_id!r} lists symbols outside the global "
                    f"universe: {outside}. The risk engine would reject every one of them."
                )
        return self

    @model_validator(mode="after")
    def _allocations_do_not_exceed_capital(self) -> AppConfig:
        total = sum((s.capital_allocation_pct for s in self.strategies if s.enabled), Decimal(0))
        if total > Decimal(100):
            raise ValueError(
                f"enabled strategies allocate {total}% of capital, which is more than exists"
            )
        return self

    def instrument(self, symbol: str) -> InstrumentSpec | None:
        for spec in self.instruments:
            if spec.symbol == symbol:
                return spec
        return None
