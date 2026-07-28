"""Property-based risk invariants — spec §8.3.

    속성 기반 테스트(property-based testing). Hypothesis로 불변식을 검증한다.
    예: [...] "리스크 한도를 초과하는 주문은 어떤 경로로도 통과할 수 없다".

Example-based tests check the cases someone thought of. These check the
*property*, across thousands of generated combinations of equity, prices,
existing positions and intent shapes — including the ones nobody thought of,
which is where the expensive bugs live.

Each invariant below is stated as "for all inputs, if the engine approved an
order, then <limit> holds". A counterexample is a way for a limit to be crossed.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from atrader.config.schema import (
    AccountLimits,
    AppConfig,
    InstrumentSpec,
    OrderLimits,
    PositionLimits,
    RiskConfig,
    UniverseConfig,
)
from atrader.core.clock import NS_PER_SECOND, SimulatedClock
from atrader.core.ids import DeterministicIdGenerator
from atrader.core.models import AccountState, Position, TradingIntent
from atrader.core.types import (
    BreakerLevel,
    DataQuality,
    RiskAction,
    Side,
    SystemState,
    TargetType,
)
from atrader.risk.engine import RiskEngine
from atrader.risk.killswitch import KillSwitch
from atrader.risk.state import RiskSnapshot

BASE_NS = 1_700_000_000 * NS_PER_SECOND
SYMBOLS = ("AAPL", "MSFT", "JPM")

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# --- strategies -------------------------------------------------------------

equities = st.decimals(min_value=Decimal("10000"), max_value=Decimal("10000000"), places=2)
prices = st.decimals(min_value=Decimal("1"), max_value=Decimal("5000"), places=2)
quantities = st.decimals(min_value=Decimal("1"), max_value=Decimal("100000"), places=0)
pcts = st.decimals(min_value=Decimal("0.1"), max_value=Decimal("50"), places=1)
signed_quantities = st.decimals(min_value=Decimal("-10000"), max_value=Decimal("10000"), places=0)


def build_engine(
    *,
    max_order_pct: Decimal = Decimal("2"),
    max_position_pct: Decimal = Decimal("10"),
    max_sector_pct: Decimal = Decimal("30"),
    max_leverage: Decimal = Decimal("1.0"),
    daily_loss_pct: Decimal = Decimal("2"),
    drawdown_pct: Decimal = Decimal("10"),
) -> RiskEngine:
    clock = SimulatedClock(start_ns=BASE_NS)
    config = AppConfig(
        account_equity=Decimal("100000"),
        risk=RiskConfig(
            account=AccountLimits(
                max_leverage=max_leverage,
                daily_loss_limit_pct=daily_loss_pct,
                max_drawdown_pct=drawdown_pct,
            ),
            position=PositionLimits(
                max_position_pct=max_position_pct, max_sector_pct=max_sector_pct
            ),
            order=OrderLimits(
                max_order_notional_pct=max_order_pct,
                human_approval_threshold_pct=Decimal("100"),
                duplicate_window_seconds=0,
            ),
        ),
        universe=UniverseConfig(
            symbols=SYMBOLS,
            sectors={"AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "JPM": "FINANCIALS"},
        ),
        instruments=tuple(InstrumentSpec(symbol=s) for s in SYMBOLS),
    )
    return RiskEngine(
        config=config,
        clock=clock,
        ids=DeterministicIdGenerator(clock),
        kill_switch=KillSwitch(clock=clock),
    )


def build_snapshot(
    *,
    equity: Decimal,
    price: Decimal,
    held: Decimal = Decimal(0),
    daily_pnl_pct: Decimal = Decimal(0),
    drawdown_pct: Decimal = Decimal(0),
    **overrides: Any,
) -> RiskSnapshot:
    positions = {}
    if held != 0:
        positions["AAPL"] = Position(
            symbol="AAPL", quantity=held, avg_price=price, last_price=price
        )
    defaults: dict[str, Any] = {
        "now_ns": BASE_NS,
        "system_state": SystemState.RUNNING,
        "breaker_level": BreakerLevel.NONE,
        "account": AccountState(cash=equity, equity=equity, buying_power=equity),
        "positions": positions,
        "universe": frozenset(SYMBOLS),
        "sectors": {"AAPL": "TECHNOLOGY", "MSFT": "TECHNOLOGY", "JPM": "FINANCIALS"},
        "data_quality": dict.fromkeys(SYMBOLS, DataQuality.OK),
        "prices": {"AAPL": price, "MSFT": price, "JPM": price},
        "is_market_open": dict.fromkeys(SYMBOLS, True),
        "daily_pnl_pct": daily_pnl_pct,
        "drawdown_pct": drawdown_pct,
    }
    return RiskSnapshot(**{**defaults, **overrides})


def buy_intent(quantity: Decimal, symbol: str = "AAPL") -> TradingIntent:
    return TradingIntent(
        intent_id=uuid4(),
        strategy_id="prop",
        symbol=symbol,
        side=Side.BUY,
        target_type=TargetType.SHARES,
        target_value=quantity,
        created_at_ns=BASE_NS,
    )


# ---------------------------------------------------------------------------
# Invariant 1 — nothing over a limit can pass
# ---------------------------------------------------------------------------


class TestApprovedOrdersRespectEveryLimit:
    @SETTINGS
    @given(equity=equities, price=prices, quantity=quantities, limit_pct=pcts)
    def test_single_order_notional_never_exceeds_its_cap(
        self, equity: Decimal, price: Decimal, quantity: Decimal, limit_pct: Decimal
    ) -> None:
        engine = build_engine(max_order_pct=limit_pct, max_position_pct=Decimal("100"))
        decision = engine.evaluate(buy_intent(quantity), build_snapshot(equity=equity, price=price))

        if decision.order is not None:
            notional = decision.order.quantity * price
            allowed = equity * limit_pct / Decimal(100)
            # One share of tolerance: quantity is floored to a whole lot, so the
            # largest permissible order can never be expressed exactly.
            assert notional <= allowed + price

    @SETTINGS
    @given(equity=equities, price=prices, quantity=quantities, held=signed_quantities)
    def test_post_fill_position_concentration_never_exceeds_its_cap(
        self, equity: Decimal, price: Decimal, quantity: Decimal, held: Decimal
    ) -> None:
        cap = Decimal("10")
        engine = build_engine(max_order_pct=Decimal("100"), max_position_pct=cap)
        snapshot = build_snapshot(equity=equity, price=price, held=held)
        decision = engine.evaluate(buy_intent(quantity), snapshot)

        if decision.order is not None:
            resulting = abs(held + decision.order.quantity) * price
            assert resulting <= equity * cap / Decimal(100) + price

    @SETTINGS
    @given(equity=equities, price=prices, quantity=quantities, held=signed_quantities)
    def test_post_fill_leverage_never_exceeds_its_cap(
        self, equity: Decimal, price: Decimal, quantity: Decimal, held: Decimal
    ) -> None:
        engine = build_engine(
            max_order_pct=Decimal("100"),
            max_position_pct=Decimal("100"),
            max_sector_pct=Decimal("100"),
            max_leverage=Decimal("1.0"),
        )
        snapshot = build_snapshot(equity=equity, price=price, held=held)
        decision = engine.evaluate(buy_intent(quantity), snapshot)

        if decision.order is not None:
            gross = abs(held + decision.order.quantity) * price
            assert gross <= equity + price

    @SETTINGS
    @given(equity=equities, price=prices, quantity=quantities, loss_pct=pcts)
    def test_no_new_exposure_once_the_daily_loss_limit_is_hit(
        self, equity: Decimal, price: Decimal, quantity: Decimal, loss_pct: Decimal
    ) -> None:
        limit = Decimal("2")
        assume(loss_pct >= limit)
        engine = build_engine(daily_loss_pct=limit)
        snapshot = build_snapshot(equity=equity, price=price, daily_pnl_pct=-loss_pct)

        assert engine.evaluate(buy_intent(quantity), snapshot).order is None

    @SETTINGS
    @given(equity=equities, price=prices, quantity=quantities, dd_pct=pcts)
    def test_no_new_exposure_once_the_drawdown_limit_is_hit(
        self, equity: Decimal, price: Decimal, quantity: Decimal, dd_pct: Decimal
    ) -> None:
        limit = Decimal("10")
        assume(dd_pct >= limit)
        engine = build_engine(drawdown_pct=limit)
        snapshot = build_snapshot(equity=equity, price=price, drawdown_pct=dd_pct)

        assert engine.evaluate(buy_intent(quantity), snapshot).order is None


# ---------------------------------------------------------------------------
# Invariant 2 — nothing that reduces risk can be blocked
# ---------------------------------------------------------------------------


class TestRiskReducingOrdersAreNeverBlocked:
    """Spec §7.4, the mirror image of invariant 1.

    A limit that stops the stop-loss is worse than no limit at all: it removes
    protection at exactly the moment the protection was designed for.
    """

    @SETTINGS
    @given(
        equity=equities,
        price=prices,
        held=st.decimals(min_value=Decimal("1"), max_value=Decimal("10000"), places=0),
        loss_pct=st.decimals(min_value=Decimal("0"), max_value=Decimal("99"), places=1),
        dd_pct=st.decimals(min_value=Decimal("0"), max_value=Decimal("99"), places=1),
    )
    def test_a_liquidation_is_approved_however_bad_the_losses(
        self,
        equity: Decimal,
        price: Decimal,
        held: Decimal,
        loss_pct: Decimal,
        dd_pct: Decimal,
    ) -> None:
        engine = build_engine()
        snapshot = build_snapshot(
            equity=equity,
            price=price,
            held=held,
            daily_pnl_pct=-loss_pct,
            drawdown_pct=dd_pct,
        )
        sell = TradingIntent(
            intent_id=uuid4(),
            strategy_id="stop_loss",
            symbol="AAPL",
            side=Side.SELL,
            target_type=TargetType.SHARES,
            target_value=held,
            created_at_ns=BASE_NS,
        )
        decision = engine.evaluate(sell, snapshot, reduce_only=True)
        assert decision.approved, f"stop-loss blocked: {decision.reason}"
        assert decision.order is not None
        assert decision.order.quantity == held

    @SETTINGS
    @given(
        equity=equities,
        price=prices,
        held=st.decimals(min_value=Decimal("1"), max_value=Decimal("10000"), places=0),
    )
    def test_a_liquidation_is_approved_in_every_blocked_state(
        self, equity: Decimal, price: Decimal, held: Decimal
    ) -> None:
        for state, level in (
            (SystemState.THROTTLED, BreakerLevel.L1),
            (SystemState.BLOCKED, BreakerLevel.L2),
            (SystemState.LIQUIDATING, BreakerLevel.L3),
        ):
            engine = build_engine()
            snapshot = build_snapshot(
                equity=equity,
                price=price,
                held=held,
                system_state=state,
                breaker_level=level,
            )
            sell = TradingIntent(
                intent_id=uuid4(),
                strategy_id="liquidator",
                symbol="AAPL",
                side=Side.SELL,
                target_type=TargetType.SHARES,
                target_value=held,
                created_at_ns=BASE_NS,
            )
            decision = engine.evaluate(sell, snapshot, reduce_only=True)
            assert decision.approved, f"{state.value}: {decision.reason}"


# ---------------------------------------------------------------------------
# Invariant 3 — the kill switch admits nothing
# ---------------------------------------------------------------------------


class TestKillSwitchIsAbsolute:
    @SETTINGS
    @given(equity=equities, price=prices, quantity=quantities)
    def test_no_new_order_survives_the_kill_switch(
        self, equity: Decimal, price: Decimal, quantity: Decimal
    ) -> None:
        engine = build_engine()
        engine.kill_switch.engage(reason="property test")
        decision = engine.evaluate(buy_intent(quantity), build_snapshot(equity=equity, price=price))
        assert decision.order is None
        assert decision.action is RiskAction.REJECT


# ---------------------------------------------------------------------------
# Invariant 4 — evaluation is total
# ---------------------------------------------------------------------------


class TestEvaluationIsTotal:
    @SETTINGS
    @given(
        equity=st.decimals(min_value=Decimal("0"), max_value=Decimal("10000000"), places=2),
        price=st.decimals(min_value=Decimal("0"), max_value=Decimal("5000"), places=2),
        quantity=st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=0),
        held=signed_quantities,
        symbol=st.sampled_from([*SYMBOLS, "FAKECO", ""]),
    )
    def test_evaluate_never_raises_whatever_it_is_given(
        self,
        equity: Decimal,
        price: Decimal,
        quantity: Decimal,
        held: Decimal,
        symbol: str,
    ) -> None:
        """The engine must always answer, even on nonsense.

        An exception escaping here would propagate into the order path, and a
        risk engine that crashes is a risk engine that is not checking anything.
        """
        engine = build_engine()
        snapshot = build_snapshot(equity=equity, price=price, held=held)
        try:
            intent = buy_intent(quantity, symbol=symbol or "AAPL")
        except Exception:
            return

        decision = engine.evaluate(intent, snapshot)
        assert decision.action in set(RiskAction)
        if decision.order is not None:
            assert decision.order.quantity > 0
            assert decision.order.risk_check_id == decision.risk_check_id
