"""
WalkForwardEngine
==================
Chronological robustness check on top of BacktestEngine.

BacktestEngine.run() only does a single in-sample pass over the whole
period — a strategy can look great purely because the dataset happens to
contain one lucky trending regime, and there was previously no way to tell
the difference between "real edge" and "curve-fit to this exact period".

This splits `data` into N contiguous, non-overlapping folds (strictly
chronological — later folds never leak into earlier ones) and runs
BacktestEngine.run() independently on each, then reports the per-fold
metrics plus their dispersion. Consistent metrics across folds is real
evidence of a robust strategy; wide swings (e.g. Sharpe strongly positive
in most folds but deeply negative in one or two) is a red flag that
performance was concentrated in a specific historical regime rather than a
durable edge.

The strategies here (momentum, mean_reversion, calendar_alpha) are
rule-based and don't "fit" per fold, so there's no separate train/test
split within a fold — each fold IS an out-of-sample period relative to the
others. RLAlpha/WinClassifier-backed strategies, if included, are treated
the same way (evaluated as-is per fold, not retrained per fold).

Known limitation — read before trusting these numbers
-------------------------------------------------------
BacktestEngine is reused exactly as-is and does NOT apply RiskEngine
(position sizing / exposure / leverage caps) in its simulation loop (see
backtest_engine.py::_StrategySim — RiskEngine is imported but never
called). These walk-forward numbers reflect raw strategy signal quality
across time, not what the live risk-managed system would actually produce.

Usage
-----
    from backtesting.walk_forward import WalkForwardEngine, WalkForwardConfig

    engine = WalkForwardEngine(WalkForwardConfig(n_folds=6))
    result = engine.run(strategies=[MomentumAlpha("EURUSD")], data=df)
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from research.alpha_models.base import AlphaModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    n_folds: int = 6
    min_fold_bars: int = 100   # matches BacktestEngine._validate()'s own floor
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_index: int
    start: str
    end: str
    n_bars: int
    result: BacktestResult


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """One row per (strategy or 'portfolio') x metric: mean/std/min/max
        across folds. A large std relative to the mean (or a sign flip
        between min and max) is the signal to look for — it means the
        strategy's edge is regime-dependent, not durable.
        """
        rows = []
        names = list(self.strategies) + ["portfolio"]
        for name in names:
            for metric in ("sharpe", "cagr", "max_drawdown_pct", "win_rate", "profit_factor"):
                if name == "portfolio":
                    metrics_objs = [f.result.portfolio_metrics for f in self.folds
                                     if f.result.portfolio_metrics is not None]
                else:
                    metrics_objs = [f.result.metrics[name] for f in self.folds
                                     if name in f.result.metrics]
                values = [getattr(m, metric) for m in metrics_objs]
                values = [v for v in values if np.isfinite(v)]
                if not values:
                    continue
                rows.append({
                    "strategy": name,
                    "metric":   metric,
                    "n_folds":  len(values),
                    "mean":     float(np.mean(values)),
                    "std":      float(np.std(values)),
                    "min":      float(np.min(values)),
                    "max":      float(np.max(values)),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """Runs BacktestEngine independently over N contiguous chronological
    folds of `data`. Does not modify BacktestEngine in any way."""

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        self.config = config or WalkForwardConfig()

    def run(self, strategies: list[AlphaModel], data: pd.DataFrame) -> WalkForwardResult:
        cfg = self.config
        n = len(data)
        fold_size = n // cfg.n_folds
        if fold_size < cfg.min_fold_bars:
            raise ValueError(
                f"Not enough data for {cfg.n_folds} folds of >= {cfg.min_fold_bars} "
                f"bars each: {n} total bars -> {fold_size} bars/fold. Reduce "
                f"n_folds or provide more history."
            )

        result = WalkForwardResult(strategies=[s.name for s in strategies])
        engine = BacktestEngine(cfg.backtest_config)

        for i in range(cfg.n_folds):
            start_idx = i * fold_size
            # Last fold absorbs any remainder so no trailing bars are dropped.
            end_idx = n if i == cfg.n_folds - 1 else start_idx + fold_size
            fold_data = data.iloc[start_idx:end_idx]

            # Stateful strategies (e.g. RLAlpha's LSTM hidden state) must not
            # carry context across folds — each fold is evaluated as an
            # independent out-of-sample period.
            for s in strategies:
                reset = getattr(s, "reset_state", None)
                if callable(reset):
                    reset()

            logger.info(
                "Walk-forward fold %d/%d — bars [%d:%d) (%d bars)",
                i + 1, cfg.n_folds, start_idx, end_idx, len(fold_data),
            )
            bt_result = engine.run(strategies, fold_data, seed=42 + i)

            result.folds.append(FoldResult(
                fold_index=i,
                start=str(fold_data.index[0]),
                end=str(fold_data.index[-1]),
                n_bars=len(fold_data),
                result=bt_result,
            ))

        return result
