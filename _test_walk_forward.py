#!/usr/bin/env python3
"""
Smoke test for backtesting/walk_forward.py (Phase D).

No raw OHLCV dataset is available offline in this environment — the only
local parquet (data/storage/features/v1/EURUSD_H1.parquet) is the
FeatureStore output, which by design strips OHLCV columns (see
FeatureEngineer.compute()'s docstring: "No raw price columns"). Since
BacktestEngine._StrategySim needs both engineered features (for the
strategies' compute()) AND raw OHLCV (for _validate/_check_sl_tp/fill
prices) in the SAME DataFrame, this generates a synthetic OHLCV series and
runs it through the real FeatureEngineer before concatenating — exercising
the exact same code path a real dataset would need to go through.

This verifies WalkForwardEngine's own orchestration logic (fold splitting,
per-fold BacktestEngine invocation, summary aggregation) — not whether any
real edge exists in EURUSD, which requires real market data the user
supplies separately.
"""

import numpy as np
import pandas as pd

from research.feature_store.feature_engineer import FeatureEngineer, FeatureConfig
from strategies.momentum.momentum_alpha import MomentumAlpha
from strategies.mean_reversion.mean_reversion_alpha import MeanReversionAlpha
from backtesting.walk_forward import WalkForwardEngine, WalkForwardConfig
from backtesting.backtest_engine import BacktestConfig

PASS = "✓"
FAIL = "✗"
all_ok = True

rng = np.random.default_rng(11)
n_bars = 3000
close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n_bars))
high = close + np.abs(rng.normal(0, 0.0003, n_bars))
low = close - np.abs(rng.normal(0, 0.0003, n_bars))
open_ = close + rng.normal(0, 0.0001, n_bars)
volume = rng.integers(100, 2000, n_bars).astype(float)
idx = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")

ohlcv = pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
    index=idx,
)

print("=" * 70)
print("Building combined OHLCV + engineered-features dataset")
print("=" * 70)
features = FeatureEngineer(FeatureConfig()).compute(ohlcv)
combined = pd.concat([ohlcv, features], axis=1).dropna()
print(f"  OHLCV bars: {len(ohlcv)} -> combined (post-dropna): {len(combined)} bars, "
      f"{combined.shape[1]} columns")

strategies = [MomentumAlpha(name="momentum"), MeanReversionAlpha(name="mean_reversion")]

print("\n" + "=" * 70)
print("Running WalkForwardEngine (6 folds)")
print("=" * 70)
wf_cfg = WalkForwardConfig(n_folds=6, backtest_config=BacktestConfig(initial_balance=100_000.0))
engine = WalkForwardEngine(wf_cfg)
result = engine.run(strategies, combined)

ok = len(result.folds) == 6
print(f"  {PASS if ok else FAIL} got {len(result.folds)} folds (expected 6)")
all_ok &= ok

# Folds must be strictly chronological and non-overlapping.
prev_end = None
chrono_ok = True
for f in result.folds:
    if prev_end is not None and pd.Timestamp(f.start) < pd.Timestamp(prev_end):
        chrono_ok = False
    prev_end = f.end
print(f"  {PASS if chrono_ok else FAIL} folds are strictly chronological / non-overlapping")
all_ok &= chrono_ok

total_fold_bars = sum(f.n_bars for f in result.folds)
ok2 = total_fold_bars == len(combined)
print(f"  {PASS if ok2 else FAIL} fold bars sum to full dataset: {total_fold_bars} == {len(combined)}")
all_ok &= ok2

for f in result.folds:
    has_metrics = "momentum" in f.result.metrics and "mean_reversion" in f.result.metrics
    print(f"    fold {f.fold_index}: [{f.start} -> {f.end}] n_bars={f.n_bars} "
          f"has_metrics={has_metrics}")
    all_ok &= has_metrics

print("\n" + "=" * 70)
print("Summary table (mean/std/min/max per strategy x metric across folds)")
print("=" * 70)
summary = result.summary()
print(summary.to_string(index=False))
ok3 = not summary.empty and {"strategy", "metric", "mean", "std"}.issubset(summary.columns)
print(f"\n  {PASS if ok3 else FAIL} summary() returned a non-empty DataFrame with expected columns")
all_ok &= ok3

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All walk-forward smoke checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
