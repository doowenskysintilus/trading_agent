"""
Live-readiness gate (SCALPING_RULES.md §8, VA-02..VA-06)
=========================================================
A rule-set version may move from demo to a real account only if its
walk-forward results pass every criterion below.

    report = check_live_readiness(walk_forward_result)
    if not report.ready:
        print(report.failures)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.walk_forward import WalkForwardResult


@dataclass(frozen=True)
class LiveReadinessCriteria:
    min_folds: int            = 4      # VA-02
    min_profitable_folds: int = 3      # VA-02
    min_trades: int           = 300    # VA-03 (out-of-sample)
    min_profit_factor: float  = 1.3    # VA-04 (net of costs)
    max_drawdown: float       = 0.05   # VA-05 (fraction)
    max_demo_win_rate_gap: float = 0.10  # VA-06


@dataclass
class ReadinessReport:
    n_folds: int = 0
    profitable_folds: int = 0
    n_trades: int = 0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.failures


def check_live_readiness(
    wf: WalkForwardResult,
    criteria: LiveReadinessCriteria = LiveReadinessCriteria(),
) -> ReadinessReport:
    """VA-02..VA-05 on the portfolio-level out-of-sample folds."""
    report = ReadinessReport(n_folds=len(wf.folds))
    pnls: list[float] = []
    for fold in wf.folds:
        pm = fold.result.portfolio_metrics
        if pm is not None:
            if pm.total_return_pct > 0:
                report.profitable_folds += 1
            report.max_drawdown = max(report.max_drawdown, abs(pm.max_drawdown_pct))
        log = fold.result.trade_log
        if isinstance(log, pd.DataFrame) and "pnl" in log.columns:
            pnls.extend(float(p) for p in log["pnl"])

    report.n_trades = len(pnls)
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    report.profit_factor = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)

    c = criteria
    if report.n_folds < c.min_folds:
        report.failures.append(f"VA-02: {report.n_folds} folds < {c.min_folds}")
    if report.profitable_folds < c.min_profitable_folds:
        report.failures.append(
            f"VA-02: {report.profitable_folds} profitable folds < {c.min_profitable_folds}")
    if report.n_trades < c.min_trades:
        report.failures.append(f"VA-03: {report.n_trades} out-of-sample trades < {c.min_trades}")
    if report.profit_factor < c.min_profit_factor:
        report.failures.append(
            f"VA-04: profit factor {report.profit_factor:.2f} < {c.min_profit_factor}")
    if report.max_drawdown > c.max_drawdown:
        report.failures.append(
            f"VA-05: max drawdown {report.max_drawdown:.2%} > {c.max_drawdown:.0%}")
    return report


def demo_matches_backtest(
    demo_win_rate: float,
    backtest_win_rate: float,
    criteria: LiveReadinessCriteria = LiveReadinessCriteria(),
) -> bool:
    """VA-06 — demo results consistent with the backtest (win-rate gap)."""
    return abs(demo_win_rate - backtest_win_rate) <= criteria.max_demo_win_rate_gap
