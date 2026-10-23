"""Position management rules (EX-04..06, EX-08), evaluated every cycle."""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Optional

from engines.rule_engine.base import PositionAction, PositionContext, PositionRule
from engines.rule_engine.market_time import SessionWindow, active_window


class TimeStopRule(PositionRule):
    """EX-04 — close a position held longer than N bars."""
    rule_id = "EX-04"

    def __init__(self, max_bars: int) -> None:
        self._max_bars = max_bars

    def evaluate(self, ctx: PositionContext) -> Optional[PositionAction]:
        held = ctx.now - ctx.position.entry_ts
        limit = timedelta(seconds=self._max_bars * ctx.timeframe_s)
        if held >= limit:
            p = ctx.position
            return PositionAction("close", p.ticket, p.symbol, self.rule_id,
                                  f"time stop: held {held.seconds // 60} min ≥ {self._max_bars} bars")
        return None


class SessionCloseRule(PositionRule):
    """EX-08 — flatten before the authorised session ends (and outside it)."""
    rule_id = "EX-08"

    def __init__(self, windows_for: Callable[[str], list[SessionWindow]], buffer_min: int) -> None:
        self._windows_for = windows_for
        self._buffer = buffer_min

    def evaluate(self, ctx: PositionContext) -> Optional[PositionAction]:
        p = ctx.position
        window = active_window(self._windows_for(p.symbol), ctx.now)
        if window is None:
            return PositionAction("close", p.ticket, p.symbol, self.rule_id, "outside authorised session")
        left = window.minutes_to_end(ctx.now.time())
        if left <= self._buffer:
            return PositionAction("close", p.ticket, p.symbol, self.rule_id,
                                  f"session ends in {left:.0f} min")
        return None


class BreakevenRule(PositionRule):
    """EX-05 — once the trade is k × ATR in profit, move SL to entry + costs."""
    rule_id = "EX-05"

    def __init__(self, trigger_atr: float) -> None:
        self._trigger = trigger_atr

    def evaluate(self, ctx: PositionContext) -> Optional[PositionAction]:
        p = ctx.position
        if p.entry_atr <= 0 or p.favourable_move < self._trigger * p.entry_atr:
            return None
        new_sl = p.open_price + p.direction * p.round_trip_cost
        return PositionAction("modify_sl", p.ticket, p.symbol, self.rule_id,
                              "breakeven + costs", new_sl=new_sl)


class AtrTrailingRule(PositionRule):
    """EX-06 — ATR trailing stop: active after k × ATR of profit, stays
    `step` × ATR behind the current price."""
    rule_id = "EX-06"

    def __init__(self, activation_atr: float, step_atr: float) -> None:
        self._activation = activation_atr
        self._step = step_atr

    def evaluate(self, ctx: PositionContext) -> Optional[PositionAction]:
        p = ctx.position
        if p.entry_atr <= 0 or p.favourable_move < self._activation * p.entry_atr:
            return None
        new_sl = p.current_price - p.direction * self._step * p.entry_atr
        return PositionAction("modify_sl", p.ticket, p.symbol, self.rule_id,
                              "ATR trailing stop", new_sl=new_sl)
