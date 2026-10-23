"""Stage 1 — trade frequency and capital-protection rules."""

from __future__ import annotations

from datetime import timedelta

from engines.rule_engine.base import EntryContext, EntryRule, RuleResult, Stage
from engines.rule_engine.journal import TradeJournal


class SinglePositionRule(EntryRule):
    """SC-04 / SC-05 — at most one open position per symbol. Forbids by
    construction averaging down, martingale and same-symbol hedging."""
    rule_id = "SC-04"
    stage = Stage.FREQUENCY

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if ctx.symbol in ctx.open_symbols:
            return RuleResult.reject(self.rule_id, f"a position is already open on {ctx.symbol}")
        return RuleResult.ok(self.rule_id)


class TradeFrequencyRule(EntryRule):
    """FQ-01 / FQ-02 — max entries per symbol per hour and per UTC day."""
    rule_id = "FQ-01/02"
    stage = Stage.FREQUENCY

    def __init__(self, journal: TradeJournal, max_per_symbol_hour: int, max_per_day: int) -> None:
        self._journal = journal
        self._per_hour = max_per_symbol_hour
        self._per_day = max_per_day

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        hour_count = self._journal.count_entries(ctx.now - timedelta(hours=1), ctx.symbol)
        if hour_count >= self._per_hour:
            return RuleResult.reject(
                "FQ-01", f"{hour_count} trades on {ctx.symbol} in the last hour (max {self._per_hour})",
                trades_last_hour=hour_count,
            )
        day_start = ctx.now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_count = self._journal.count_entries(day_start)
        if day_count >= self._per_day:
            return RuleResult.reject(
                "FQ-02", f"{day_count} trades today (max {self._per_day})", trades_today=day_count,
            )
        return RuleResult.ok(self.rule_id, trades_last_hour=hour_count, trades_today=day_count)


class CooldownRule(EntryRule):
    """FQ-03 — pause after a loss on a symbol.
    FQ-04 — global pause after N consecutive losses."""
    rule_id = "FQ-03/04"
    stage = Stage.FREQUENCY

    def __init__(
        self,
        journal: TradeJournal,
        symbol_cooldown_min: int,
        pause_after_losses: int,
        pause_min: int,
    ) -> None:
        self._journal = journal
        self._symbol_cooldown = timedelta(minutes=symbol_cooldown_min)
        self._pause_losses = pause_after_losses
        self._pause = timedelta(minutes=pause_min)

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        streak, last_ts = self._journal.loss_streak()
        if streak >= self._pause_losses and last_ts and ctx.now - last_ts < self._pause:
            remaining = self._pause - (ctx.now - last_ts)
            return RuleResult.reject(
                "FQ-04", f"{streak} consecutive losses — global pause, {remaining.seconds // 60} min left",
                loss_streak=streak,
            )
        last_loss = self._journal.last_loss_ts(ctx.symbol)
        if last_loss and ctx.now - last_loss < self._symbol_cooldown:
            remaining = self._symbol_cooldown - (ctx.now - last_loss)
            return RuleResult.reject(
                "FQ-03", f"cooldown after a loss on {ctx.symbol}, {remaining.seconds}s left",
            )
        return RuleResult.ok(self.rule_id, loss_streak=streak)


class DailyTargetRule(EntryRule):
    """FQ-08 — stop opening trades once the daily profit target is reached."""
    rule_id = "FQ-08"
    stage = Stage.FREQUENCY

    def __init__(self, target_pct: float) -> None:
        self._target = target_pct

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        ps = ctx.portfolio_state
        if ps is None or ps.daily_start_equity <= 0:
            return RuleResult.ok(self.rule_id)
        day_pnl_pct = (ps.equity - ps.daily_start_equity) / ps.daily_start_equity
        if day_pnl_pct >= self._target:
            return RuleResult.reject(
                self.rule_id,
                f"daily target reached (+{day_pnl_pct:.2%} ≥ +{self._target:.2%})",
                day_pnl_pct=round(day_pnl_pct, 5),
            )
        return RuleResult.ok(self.rule_id, day_pnl_pct=round(day_pnl_pct, 5))
