"""Stage 0 — hard stops evaluated before anything else."""

from __future__ import annotations

from typing import Callable

from engines.rule_engine.base import EntryContext, EntryRule, RuleResult, Stage


class RiskHaltRule(EntryRule):
    """FQ-05..07 / FQ-09 — kill-switch, drawdown and FTMO limits.

    Delegates to RiskEngine.check_halt() so the limits are defined once
    (RiskConfig) and simply evaluated earlier, before any computation.
    """
    rule_id = "FQ-05..07"
    stage = Stage.HALT

    def __init__(self, risk_engine) -> None:
        self._risk = risk_engine

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if ctx.portfolio_state is None:
            return RuleResult.ok(self.rule_id)
        check = self._risk.check_halt(ctx.portfolio_state)
        if check is not None:
            reason = check.reason.value if check.reason else check.name
            return RuleResult.reject(self.rule_id, f"{reason}: {check.message}")
        return RuleResult.ok(self.rule_id)


class PersistenceRule(EntryRule):
    """DB-04 — no new trade unless decisions can be journaled
    (MySQL, or its JSONL fallback)."""
    rule_id = "DB-04"
    stage = Stage.HALT

    def __init__(self, is_persistence_ok: Callable[[], bool]) -> None:
        self._is_ok = is_persistence_ok

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if self._is_ok():
            return RuleResult.ok(self.rule_id)
        return RuleResult.reject(
            self.rule_id, "decision journal unavailable (MySQL and JSONL fallback both down)",
        )
