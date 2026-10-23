#!/usr/bin/env python3
"""
Test script for conviction-scaled position sizing (Phase: pro-trader sizing).

Before this fix, LiveTrader._build_trade_order() sized every trade the same
(risk_per_trade_pct x equity / ATR-based SL distance) regardless of how
confident the aggregated signal was, as long as it cleared min_confidence.
This verifies size now scales from 0.5x (at min_confidence) to 1.0x
(at confidence=1.0) roughly linearly, and never falls outside that band.
"""

import numpy as np
import pandas as pd

from live_trading.live_trader import LiveTrader, LiveTraderConfig
from engines.signal_engine.signal_aggregator import AggregatedDecision
from research.alpha_models.base import SignalType

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True

cfg = LiveTraderConfig(symbols=["EURUSD"], min_confidence=0.45)
trader = LiveTrader(cfg)

n = 50
idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
close = np.full(n, 1.1000)
trader._data_cache["EURUSD"] = pd.DataFrame({
    "open": close, "high": close * 1.001, "low": close * 0.999,
    "close": close, "volume": 100.0,
}, index=idx)
trader._portfolio_state.equity = 10_000.0


def size_at(confidence: float) -> float:
    decision = AggregatedDecision(
        signal=SignalType.BUY, confidence=confidence,
        buy_score=confidence, sell_score=0.0, net_score=confidence,
    )
    order = trader._build_trade_order("EURUSD", decision)
    return order.size


print("=" * 70)
print("TEST: size scales with conviction (min_confidence=0.45)")
print("=" * 70)

size_at_min   = size_at(0.45)   # should be ~0.5x baseline
size_at_mid   = size_at(0.725)  # halfway -> ~0.75x baseline
size_at_max   = size_at(1.0)    # should be ~1.0x baseline (max size)

print(f"  size at confidence=0.45 (min):  {size_at_min}")
print(f"  size at confidence=0.725 (mid): {size_at_mid}")
print(f"  size at confidence=1.00 (max):  {size_at_max}")

ok1 = size_at_max > size_at_mid > size_at_min > 0
print(f"  {PASS if ok1 else FAIL} strictly increasing with conviction")
all_ok &= ok1

ratio = size_at_min / size_at_max
ok2 = abs(ratio - 0.5) < 0.02
print(f"  {PASS if ok2 else FAIL} min-confidence size is ~0.5x max-confidence size (ratio={ratio:.4f})")
all_ok &= ok2

# A very low, out-of-range confidence (shouldn't normally reach this method —
# already filtered by min_confidence upstream — but the formula must still
# clip sanely rather than producing a negative or oversized order).
size_floor = size_at(0.0)
ok3 = abs(size_floor - size_at_min) < 1e-6  # clipped to the same 0.5x floor
print(f"  {PASS if ok3 else FAIL} confidence below min_confidence still clips to the 0.5x floor "
      f"(size={size_floor})")
all_ok &= ok3

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All conviction-sizing checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
