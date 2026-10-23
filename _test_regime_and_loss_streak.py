#!/usr/bin/env python3
"""
Test script for:
  1. Loss-streak size taper (LiveTrader._build_trade_order)
  2. Market regime detection (LiveTrader._detect_regime, Kaufman ER)
  3. Regime-based dynamic strategy reweighting (_apply_regime_weighting) —
     including that repeated calls do NOT compound (always rescales from
     the original registered weight, not the aggregator's current one).
"""

import numpy as np
import pandas as pd

from live_trading.live_trader import LiveTrader, LiveTraderConfig
from engines.signal_engine.signal_aggregator import AggregatedDecision
from research.alpha_models.base import AlphaModel, AlphaSignal, SignalType

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True


class _DummyStrategy(AlphaModel):
    def __init__(self, name):
        super().__init__(name=name)

    def compute(self, data):
        return self._hold()


def make_ohlcv(closes, n_extra=0):
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes, "high": closes * 1.0005, "low": closes * 0.9995,
        "close": closes, "volume": 100.0,
    }, index=idx)


# ---------------------------------------------------------------------------
print("=" * 70)
print("TEST 1: loss-streak size taper")
print("=" * 70)
cfg = LiveTraderConfig(symbols=["EURUSD"], min_confidence=0.45)
trader = LiveTrader(cfg)
trader._data_cache["EURUSD"] = make_ohlcv(np.full(50, 1.10))
trader._portfolio_state.equity = 10_000.0


def size_at_losses(n_losses: int) -> float:
    trader._portfolio_state.consecutive_losses = n_losses
    decision = AggregatedDecision(
        signal=SignalType.BUY, confidence=1.0,
        buy_score=1.0, sell_score=0.0, net_score=1.0,
    )
    return trader._build_trade_order("EURUSD", decision).size


s0 = size_at_losses(0)
s2 = size_at_losses(2)
s4 = size_at_losses(4)
s10 = size_at_losses(10)  # beyond floor
print(f"  size @0 losses={s0}, @2={s2}, @4={s4}, @10={s10}")
ok1 = s0 > s2 > s4 and abs(s4 - s10) < 1e-6  # floors out by 4 losses (1 - 0.15*4 = 0.4)
print(f"  {PASS if ok1 else FAIL} strictly decreasing then floors at 0.4x")
all_ok &= ok1
ok2 = abs(s4 / s0 - 0.4) < 0.02
print(f"  {PASS if ok2 else FAIL} floor ratio is ~0.4x baseline (ratio={s4/s0:.4f})")
all_ok &= ok2
trader._portfolio_state.consecutive_losses = 0  # reset for later tests

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 2: regime detection (Kaufman Efficiency Ratio)")
print("=" * 70)

# Strong monotonic uptrend -> ER close to 1 -> "trending"
trending_closes = 1.10 + np.linspace(0, 0.01, 30)
trader._data_cache["EURUSD"] = make_ohlcv(trending_closes)
regime_trend = trader._detect_regime("EURUSD")
ok3 = regime_trend == "trending"
print(f"  {PASS if ok3 else FAIL} monotonic uptrend -> '{regime_trend}' (expected 'trending')")
all_ok &= ok3

# Pure oscillation back to start -> ER == 0 -> "ranging"
n = 30
choppy_closes = 1.10 + 0.005 * np.sin(np.linspace(0, 6 * np.pi, n))
choppy_closes[-1] = choppy_closes[0]  # ensure net_change ~ 0
trader._data_cache["EURUSD"] = make_ohlcv(choppy_closes)
regime_range = trader._detect_regime("EURUSD")
ok4 = regime_range == "ranging"
print(f"  {PASS if ok4 else FAIL} oscillation back to start -> '{regime_range}' (expected 'ranging')")
all_ok &= ok4

# Not enough data -> "neutral"
trader._data_cache["EURUSD"] = make_ohlcv(np.full(5, 1.10))
regime_short = trader._detect_regime("EURUSD")
ok5 = regime_short == "neutral"
print(f"  {PASS if ok5 else FAIL} insufficient data -> '{regime_short}' (expected 'neutral')")
all_ok &= ok5

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 3: regime-based reweighting — boosts/cuts, and does not compound")
print("=" * 70)
trader2 = LiveTrader(LiveTraderConfig(symbols=["EURUSD"]))
trader2.register_strategy(_DummyStrategy("momentum"), weight=1.0)
trader2.register_strategy(_DummyStrategy("mean_reversion"), weight=1.0)

trader2._data_cache["EURUSD"] = make_ohlcv(trending_closes)  # trending
trader2._apply_regime_weighting("EURUSD")
mom_w1 = trader2._aggregator.effective_weight("momentum")
mr_w1  = trader2._aggregator.effective_weight("mean_reversion")
print(f"  after 1st trending call: momentum={mom_w1}, mean_reversion={mr_w1}")
ok6 = mom_w1 > 1.0 and mr_w1 < 1.0
print(f"  {PASS if ok6 else FAIL} momentum boosted, mean_reversion cut in a trending regime")
all_ok &= ok6

# Call again with the SAME trending regime — must NOT compound (should be
# identical to the first call, not boosted further).
trader2._apply_regime_weighting("EURUSD")
mom_w2 = trader2._aggregator.effective_weight("momentum")
ok7 = abs(mom_w2 - mom_w1) < 1e-9
print(f"  {PASS if ok7 else FAIL} repeated call does not compound: {mom_w1} == {mom_w2}")
all_ok &= ok7

# Now switch to a ranging regime -> mean_reversion should now be boosted,
# momentum cut, scaled from the ORIGINAL weight (1.0), not from mom_w1.
trader2._data_cache["EURUSD"] = make_ohlcv(choppy_closes)
trader2._apply_regime_weighting("EURUSD")
mom_w3 = trader2._aggregator.effective_weight("momentum")
mr_w3  = trader2._aggregator.effective_weight("mean_reversion")
print(f"  after switching to ranging: momentum={mom_w3}, mean_reversion={mr_w3}")
ok8 = mr_w3 > 1.0 and mom_w3 < 1.0
print(f"  {PASS if ok8 else FAIL} mean_reversion boosted, momentum cut in a ranging regime")
all_ok &= ok8

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All regime / loss-streak checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
