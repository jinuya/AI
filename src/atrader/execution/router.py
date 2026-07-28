"""Intent-to-order routing — spec §FR-EXE-02, §FR-EXE-03.

    ADV의 1%를 초과하는 주문은 반드시 분할 실행해야 한다.

By the time an :class:`~atrader.core.models.Order` reaches this module, the
risk engine has already built it — order type, aggressive-limit pricing, and
final size are all settled (spec §FR-EXE-02, see
:meth:`atrader.risk.engine.RiskEngine._build_order`). What is left to decide is
purely a question of *how* to work it: send it whole, or split it.

That split is sized against ADV, which is a coarser number than the pre-trade
ADV-participation check already ran (§7.2 #7, capped at 5% of ADV — see
:func:`atrader.risk.checks.pretrade.check_adv_participation`). The threshold
here is lower (1% by default) and answers a different question: not "is this
order too large to be safe," which the risk engine already settled, but "is
this order large enough that sending it as one clip would move the price
against itself." A 3%-of-ADV order can clear the risk check and still deserve
to be worked over ten minutes rather than dropped on the book at once.

Splitting never changes the total quantity the risk engine approved — every
child order sums back to exactly the parent's size (spec: each
:class:`~atrader.execution.algos.SliceSchedule` enforces this itself). What
changes is only when the shares reach the broker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from atrader.config.schema import ExecutionConfig, OrderLimits
from atrader.core.ids import IdGenerator
from atrader.core.models import Order
from atrader.core.money import ZERO, as_pct
from atrader.execution.algos import Slice, SliceSchedule
from atrader.execution.algos.pov import POVExecutor
from atrader.execution.algos.twap import plan_twap
from atrader.execution.algos.vwap import plan_vwap

__all__ = ["AlgoChoice", "Router", "RoutingPlan", "needs_split"]


class AlgoChoice(StrEnum):
    DIRECT = "direct"
    """Small enough relative to ADV to send as one order."""
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"


def needs_split(quantity: Decimal, adv: Decimal | None, *, threshold_pct: Decimal) -> bool:
    """Spec §FR-EXE-03: above this share of ADV, the order must be worked.

    A missing ADV estimate means no basis to decide, so it does not split —
    consistent with how the ADV-participation risk check itself treats a
    missing estimate (``allow``, not ``reject``): a newly listed name should
    not be unexecutable just because there is no volume history for it yet.
    """
    if adv is None or adv <= ZERO:
        return False
    return as_pct(abs(quantity), adv) > threshold_pct


@dataclass(frozen=True, slots=True)
class RoutingPlan:
    """The outcome of routing one risk-approved order.

    For :attr:`AlgoChoice.DIRECT`, :attr:`AlgoChoice.TWAP` and
    :attr:`AlgoChoice.VWAP` this fully determines what to submit and when —
    ``children`` (or ``(parent,)`` for direct) paired with each slice's
    ``offset_seconds`` from :attr:`schedule`. For :attr:`AlgoChoice.POV` there
    is nothing to submit yet: :attr:`pov` is a live executor that
    :meth:`Router.pov_child` feeds volume into as it prints, producing child
    orders one at a time.
    """

    parent: Order
    algo: AlgoChoice
    children: tuple[Order, ...] = ()
    schedule: SliceSchedule | None = None
    pov: POVExecutor | None = None

    @property
    def is_split(self) -> bool:
        return self.algo is not AlgoChoice.DIRECT

    @property
    def orders_to_submit_now(self) -> tuple[Order, ...]:
        """What can be sent immediately.

        ``DIRECT`` is the whole order. ``TWAP``/``VWAP`` is only the slices
        whose schedule offset is zero — the rest wait on the clock, which is
        the caller's job to drive (this module has no timer of its own; see
        :mod:`atrader.core.clock`). ``POV`` is always empty here.
        """
        if self.algo is AlgoChoice.DIRECT:
            return (self.parent,)
        if self.algo is AlgoChoice.POV or self.schedule is None:
            return ()
        pairs = zip(self.children, self.schedule.slices, strict=True)
        return tuple(child for child, sl in pairs if sl.offset_seconds == 0.0)


@dataclass
class Router:
    """Decides direct-vs-algo and builds the child orders for a split."""

    ids: IdGenerator
    order_limits: OrderLimits
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def route(
        self,
        order: Order,
        *,
        adv: Decimal | None,
        algo: AlgoChoice | None = None,
        volume_curve: Sequence[Decimal] | None = None,
        lot_size: Decimal | None = None,
    ) -> RoutingPlan:
        """Decide how *order* should reach the broker.

        *algo* overrides the configured default (an urgent liquidation, say,
        may want ``DIRECT`` regardless of ADV — that choice is the caller's,
        this method only supplies the default). *volume_curve*, if given,
        selects VWAP shaping even when the configured default is TWAP, since a
        curve is exactly what VWAP needs and TWAP has no use for.
        """
        if not needs_split(
            order.quantity, adv, threshold_pct=self.order_limits.adv_split_threshold_pct
        ):
            return RoutingPlan(parent=order, algo=AlgoChoice.DIRECT)

        chosen = algo or (
            AlgoChoice.VWAP if volume_curve is not None else AlgoChoice(self.execution.default_algo)
        )

        if chosen is AlgoChoice.POV:
            executor = POVExecutor(
                total_quantity=order.quantity,
                participation_rate_pct=self.execution.pov_participation_rate_pct,
                lot_size=lot_size or Decimal(1),
            )
            return RoutingPlan(parent=order, algo=AlgoChoice.POV, pov=executor)

        if chosen is AlgoChoice.VWAP:
            if not volume_curve:
                raise ValueError("VWAP routing requires a volume_curve")
            schedule = plan_vwap(
                order.quantity,
                volume_curve=volume_curve,
                bucket_seconds=self.execution.algo_duration_seconds / max(len(volume_curve), 1),
                lot_size=lot_size,
            )
        else:
            schedule = plan_twap(
                order.quantity,
                duration_seconds=self.execution.algo_duration_seconds,
                slice_count=self.execution.algo_slice_count,
                lot_size=lot_size,
            )

        children = tuple(self._child_order(order, sl) for sl in schedule.slices)
        return RoutingPlan(parent=order, algo=chosen, children=children, schedule=schedule)

    def pov_child(self, plan: RoutingPlan, traded_quantity: Decimal) -> Order | None:
        """Feed one volume observation into a POV plan; get a child order or ``None``.

        The parent order is never submitted for a POV plan — only children,
        which is why :attr:`RoutingPlan.orders_to_submit_now` is empty for it.
        """
        if plan.algo is not AlgoChoice.POV or plan.pov is None:
            raise ValueError("pov_child called on a non-POV routing plan")
        quantity = plan.pov.on_market_volume(traded_quantity)
        if quantity <= ZERO:
            return None
        return self._child_order(plan.parent, Slice(0, 0.0, quantity))

    def pov_finish(self, plan: RoutingPlan) -> Order | None:
        """Force out whatever a POV plan has left. See
        :meth:`~atrader.execution.algos.pov.POVExecutor.finish_remaining`."""
        if plan.algo is not AlgoChoice.POV or plan.pov is None:
            raise ValueError("pov_finish called on a non-POV routing plan")
        quantity = plan.pov.finish_remaining()
        if quantity <= ZERO:
            return None
        return self._child_order(plan.parent, Slice(0, 0.0, quantity))

    def _child_order(self, parent: Order, sl: Slice) -> Order:
        """A slice of *parent*, as its own order.

        Carries the same ``risk_check_id`` and ``parent_intent_id`` as the
        parent — the risk engine approved the *total* quantity, and every
        child traces back to that one decision rather than each minting a
        fresh (and fictitious) approval.
        """
        new_id = self.ids.new_id()
        return parent.model_copy(
            update={
                "order_id": new_id,
                "client_order_id": str(new_id),
                "quantity": sl.quantity,
                "filled_quantity": Decimal(0),
                "status": parent.status,
                "broker_order_id": None,
                "avg_fill_price": None,
            }
        )
