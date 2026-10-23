#!/usr/bin/env python3
"""
Test script for the returns_matrix wiring (Phase: correlation risk).

Before this fix, RiskEngine._check_correlation() always early-returned
APPROVE because PortfolioState.returns_matrix was never populated anywhere
in live_trading/live_trader.py. This verifies:
  1. LiveTrader._update_returns_matrix() builds a real DataFrame from the
     per-symbol OHLCV cache.
  2. With that matrix wired in, RiskEngine Layer 5 actually detects a
     correlated cluster and reduces the proposed size — proving the layer
     is no longer a permanent no-op.
"""

import numpy as np
import pandas as pd

from live_trading.live_trader import LiveTrader, LiveTraderConfig
from engines.risk_engine.risk_engine import TradeOrder, OpenPosition, RiskVerdict

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True

trader = LiveTrader(LiveTraderConfig(symbols=["EURUSD", "GBPUSD", "XAUUSD"]))

rng = np.random.default_rng(3)
n = 200
common_factor = rng.normal(0, 0.0005, n)  # shared USD-side driver

idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")


def make_ohlcv(base_returns):
    close = 1.10 * np.exp(np.cumsum(base_returns))
    return pd.DataFrame({
        "open": close, "high": close * 1.0002, "low": close * 0.9998,
        "close": close, "volume": 100.0,
    }, index=idx)


# EURUSD and GBPUSD driven by the same USD factor -> should end up highly
# correlated. XAUUSD gets independent noise -> should NOT cluster with them.
trader._data_cache["EURUSD"] = make_ohlcv(common_factor + rng.normal(0, 0.0001, n))
trader._data_cache["GBPUSD"] = make_ohlcv(common_factor + rng.normal(0, 0.0001, n))
trader._data_cache["XAUUSD"] = make_ohlcv(rng.normal(0, 0.0005, n))

print("=" * 70)
print("TEST 1: _update_returns_matrix() populates PortfolioState.returns_matrix")
print("=" * 70)
assert trader._portfolio_state.returns_matrix is None, "should start unset"
trader._update_returns_matrix()
matrix = trader._portfolio_state.returns_matrix
ok = matrix is not None and set(matrix.columns) == {"EURUSD", "GBPUSD", "XAUUSD"} and len(matrix) > 20
print(f"  {PASS if ok else FAIL} matrix built: shape={None if matrix is None else matrix.shape}, "
      f"columns={None if matrix is None else list(matrix.columns)}")
all_ok &= ok

corr = matrix.corr()
eur_gbp_corr = corr.loc["EURUSD", "GBPUSD"]
eur_xau_corr = abs(corr.loc["EURUSD", "XAUUSD"])
ok2 = eur_gbp_corr > 0.7 and eur_xau_corr < 0.7
print(f"  {PASS if ok2 else FAIL} EURUSD/GBPUSD correlation={eur_gbp_corr:.3f} (>0.7 expected), "
      f"EURUSD/XAUUSD={eur_xau_corr:.3f} (<0.7 expected)")
all_ok &= ok2

print("\n" + "=" * 70)
print("TEST 2: RiskEngine Layer 5 now actually detects the correlated cluster")
print("=" * 70)
ps = trader._portfolio_state
ps.equity = 10_000.0
ps.balance = 10_000.0
ps.peak_equity = 10_000.0
ps.daily_start_equity = 10_000.0
# Already holding a big GBPUSD position — notional 5500 / equity 10000 = 55%,
# above max_correlated_exposure_pct (default 50%) on its own.
ps.open_positions = [
    OpenPosition(symbol="GBPUSD", strategy="momentum", direction=1,
                 size=5000.0, entry_price=1.10, current_price=1.10),
]

# Propose a new EURUSD trade that would push the correlated USD-cluster
# (GBPUSD + EURUSD) past max_correlated_exposure_pct (default 50%).
trade = TradeOrder(
    symbol="EURUSD", strategy="momentum", direction=1,
    size=3636.0, entry_price=1.10, atr=0.0005,
)
check = trader._risk_engine._check_correlation(trade, ps)
ok3 = not check.passed and check.verdict in (RiskVerdict.REDUCE, RiskVerdict.BLOCK)
print(f"  {PASS if ok3 else FAIL} correlation check: passed={check.passed} verdict={check.verdict} "
      f"message={check.message}")
all_ok &= ok3

print("\n" + "=" * 70)
print("TEST 3: an uncorrelated symbol (XAUUSD) is NOT flagged by the same cluster")
print("=" * 70)
trade_xau = TradeOrder(
    symbol="XAUUSD", strategy="momentum", direction=1,
    size=3636.0, entry_price=1.10, atr=0.0005,
)
check_xau = trader._risk_engine._check_correlation(trade_xau, ps)
ok4 = check_xau.passed
print(f"  {PASS if ok4 else FAIL} XAUUSD correlation check: passed={check_xau.passed} "
      f"message={check_xau.message}")
all_ok &= ok4

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All returns-matrix / correlation-risk checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
