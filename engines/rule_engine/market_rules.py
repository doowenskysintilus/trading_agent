"""Stage 2 — market conditions, checked before any signal is computed."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np

from engines.rule_engine.base import EntryContext, EntryRule, RuleResult, Stage
from engines.rule_engine.indicators import atr_series
from engines.rule_engine.market_time import SessionWindow, active_window, parse_hhmm


class SessionRule(EntryRule):
    """MK-01 / MK-02 — authorised UTC sessions (FX/metals vs crypto).
    MK-03 — no FX entry on the weekend, after the Friday cutoff, or in the
    first minutes after the Sunday open."""
    rule_id = "MK-01/02/03"
    stage = Stage.MARKET

    def __init__(
        self,
        fx_sessions: list[str],
        crypto_sessions: list[str],
        crypto_symbols: list[str],
        friday_cutoff: str = "19:00",
        sunday_open: str = "22:00",
        sunday_open_wait_min: int = 30,
    ) -> None:
        self._fx = [SessionWindow.parse(s) for s in fx_sessions]
        self._crypto = [SessionWindow.parse(s) for s in crypto_sessions]
        self._crypto_symbols = {s.upper() for s in crypto_symbols}
        self._friday_cutoff = parse_hhmm(friday_cutoff)
        self._sunday_open = parse_hhmm(sunday_open)
        self._sunday_wait = timedelta(minutes=sunday_open_wait_min)

    def is_crypto(self, symbol: str) -> bool:
        return symbol.upper() in self._crypto_symbols

    def windows_for(self, symbol: str) -> list[SessionWindow]:
        return self._crypto if self.is_crypto(symbol) else self._fx

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        now = ctx.now
        if not self.is_crypto(ctx.symbol):
            weekday = now.weekday()           # Mon=0 … Sun=6
            if weekday == 5:
                return RuleResult.reject("MK-03", "weekend — FX market closed")
            if weekday == 4 and now.time() >= self._friday_cutoff:
                return RuleResult.reject("MK-03", f"Friday after {self._friday_cutoff:%H:%M} UTC")
            if weekday == 6:
                opens = now.replace(hour=self._sunday_open.hour, minute=self._sunday_open.minute,
                                    second=0, microsecond=0)
                if now < opens + self._sunday_wait:
                    return RuleResult.reject("MK-03", "Sunday open not yet settled")
        window = active_window(self.windows_for(ctx.symbol), now)
        if window is None:
            rule = "MK-02" if self.is_crypto(ctx.symbol) else "MK-01"
            return RuleResult.reject(rule, f"outside authorised sessions ({now:%H:%M} UTC)")
        return RuleResult.ok(self.rule_id)


class SpreadRule(EntryRule):
    """MK-04 — spread ≤ ratio × ATR and ≤ a fixed cap per symbol (points)."""
    rule_id = "MK-04"
    stage = Stage.MARKET

    def __init__(self, max_atr_ratio: float, max_points: dict[str, float]) -> None:
        self._ratio = max_atr_ratio
        self._caps = {k.upper(): v for k, v in max_points.items()}

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        snap = ctx.snapshot
        if snap is None:
            return RuleResult.reject(self.rule_id, "no live quote — spread unknown")
        spread_pts = snap.spread_points
        metrics = {"spread_points": round(spread_pts, 2)}
        cap = self._caps.get(ctx.symbol.upper())
        if cap is not None and spread_pts > cap:
            return RuleResult.reject(self.rule_id, f"spread {spread_pts:.1f} pts > cap {cap:g} pts", **metrics)
        if ctx.atr > 0 and snap.spread_price > self._ratio * ctx.atr:
            return RuleResult.reject(
                self.rule_id,
                f"spread {snap.spread_price:.5f} > {self._ratio:g} × ATR {ctx.atr:.5f}", **metrics,
            )
        return RuleResult.ok(self.rule_id, **metrics)


class VolatilityRule(EntryRule):
    """MK-05 — ATR ≥ k × round-trip cost (enough room to pay the costs).
    MK-06 — ATR ≤ k × median ATR of the recent window (no news spikes)."""
    rule_id = "MK-05/06"
    stage = Stage.MARKET

    def __init__(self, min_cost_ratio: float, max_median_ratio: float, median_window: int) -> None:
        self._min_cost_ratio = min_cost_ratio
        self._max_median_ratio = max_median_ratio
        self._window = median_window

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if ctx.atr <= 0:
            return RuleResult.reject("MK-05", "ATR unavailable")
        metrics = {"atr": ctx.atr}
        if ctx.round_trip_cost > 0 and ctx.atr < self._min_cost_ratio * ctx.round_trip_cost:
            return RuleResult.reject(
                "MK-05",
                f"ATR {ctx.atr:.5f} < {self._min_cost_ratio:g} × cost {ctx.round_trip_cost:.5f}",
                **metrics,
            )
        median = self._median_atr(ctx)
        if median and ctx.atr > self._max_median_ratio * median:
            return RuleResult.reject(
                "MK-06", f"ATR {ctx.atr:.5f} > {self._max_median_ratio:g} × median {median:.5f}",
                **metrics,
            )
        return RuleResult.ok(self.rule_id, **metrics)

    def _median_atr(self, ctx: EntryContext) -> Optional[float]:
        if ctx.bars is None or len(ctx.bars) < 30:
            return None
        series = atr_series(ctx.bars.tail(self._window + 14)).dropna()
        if series.empty:
            return None
        median = float(np.median(series.values))
        return median if median > 0 else None


class EventWindowRule(EntryRule):
    """MK-07 — no entry within ±N minutes of a HIGH-impact event."""
    rule_id = "MK-07"
    stage = Stage.MARKET

    def __init__(self, calendar_provider, window_minutes: int) -> None:
        self._calendar = calendar_provider
        self._window = window_minutes

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if self._calendar is None:
            return RuleResult.ok(self.rule_id)
        try:
            active = self._calendar.event_is_active(ctx.symbol, window_minutes=self._window)
        except Exception as exc:          # calendar outage must not block trading
            return RuleResult.ok(self.rule_id, calendar_error=str(exc))
        if active:
            return RuleResult.reject(self.rule_id, f"high-impact event within ±{self._window} min")
        return RuleResult.ok(self.rule_id)


class DataFreshnessRule(EntryRule):
    """MK-08 — the last closed bar must be recent. Compared with the broker
    clock of the live quote (bars carry broker time, not UTC)."""
    rule_id = "MK-08"
    stage = Stage.MARKET

    def __init__(self, max_age_bars: float) -> None:
        self._max_age_bars = max_age_bars

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if ctx.bars is None or len(ctx.bars) == 0:
            return RuleResult.reject(self.rule_id, "no bars")
        if ctx.snapshot is None or ctx.snapshot.server_time is None:
            return RuleResult.reject(self.rule_id, "no broker clock — cannot check freshness")
        last_bar = ctx.bars.index[-1].to_pydatetime()
        # A closed bar opened at T is complete at T + timeframe.
        age_s = (ctx.snapshot.server_time - last_bar).total_seconds() - ctx.timeframe_s
        limit_s = self._max_age_bars * ctx.timeframe_s
        if age_s > limit_s:
            return RuleResult.reject(
                self.rule_id, f"last closed bar is {age_s:.0f}s old (max {limit_s:.0f}s)",
                data_age_s=round(age_s),
            )
        return RuleResult.ok(self.rule_id, data_age_s=round(age_s))
