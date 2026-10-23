#!/usr/bin/env python3
"""
Test script for the "one full analysis pass per symbol per bar" gate
(Phase: fix duplicate-entry stacking + redundant RL state stepping).

Root cause confirmed live: with TRADING_CYCLE_SECONDS=3 and
TRADING_TIMEFRAME=M1, the same M1 bar gets re-analysed ~20x before it
closes. An unchanged signal on that unchanged bar re-executed a brand new
position every single cycle — 14 stacked SELL positions on 2 symbols
within 7 cycles / 21 seconds in the live account, all at ~the same price.

The gate lives right after data fetch (stage 1b), BEFORE feature/signal
computation — not just before execution — because rl_agent's LSTM hidden
state steps forward on every compute() call. Gating only at the execution
boundary would still let the RL agent "see" (and advance its temporal
state on) the same unchanged bar ~20x per minute.

This bypasses MT5 entirely by monkeypatching _fetch_data / _run_alpha_models
/ _execute_trade so _run_cycle can run end-to-end without a live terminal,
and verifies:
  1. Same bar, called twice -> _run_alpha_models (where RL steps) is only
     invoked ONCE, and only the first call reaches execution.
  2. A new (later) bar -> analysis + execution are allowed again.
  3. A HOLD-yielding bar is ALSO gated on repeat (marking happens
     unconditionally at fetch time now, not only for actionable decisions).
"""

import numpy as np
import pandas as pd

from live_trading.live_trader import LiveTrader, LiveTraderConfig, CycleStatus
from research.alpha_models.base import AlphaSignal, SignalType

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True

cfg = LiveTraderConfig(
    symbols=["EURUSD"],
    htf_enabled=False,          # isolate the bar-dedup gate specifically
    ml_filter_enabled=False,
    min_confidence=0.1,
    warmup_bars=30,
)
trader = LiveTrader(cfg)

execute_calls = []
alpha_model_calls = []


def fake_execute_trade(decision, trade_order, risk_decision):
    execute_calls.append((trade_order.symbol, decision.signal.name))
    return []  # no real ExecutionResult needed for this test


def fake_apply_risk(order):
    from engines.risk_engine.risk_engine import RiskDecision, RiskVerdict
    return RiskDecision(verdict=RiskVerdict.APPROVE, approved_trades=[order])


def make_bar_data(bar_index: int, n: int = 30):
    idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC") + pd.Timedelta(minutes=bar_index)
    close = np.full(n, 1.10)
    return pd.DataFrame({
        "open": close, "high": close * 1.0005, "low": close * 0.9995,
        "close": close, "volume": 100.0,
        "ema_diff": 0.001, "rsi_norm": 0.6,  # harmless extra feature cols
    }, index=idx)


signal_to_emit = {"value": SignalType.SELL}


def fake_run_alpha_models(features):
    alpha_model_calls.append(signal_to_emit["value"])
    return [AlphaSignal(signal=signal_to_emit["value"], confidence=0.9, strategy_name="momentum")]


current_bar_index = {"value": 0}


def fake_fetch_data(symbol):
    df = make_bar_data(current_bar_index["value"])
    trader._data_cache[symbol] = df   # real _fetch_data's side effect _get_last_price relies on
    return df


trader._fetch_data          = fake_fetch_data
trader._compute_features    = lambda data: data
trader._run_alpha_models    = fake_run_alpha_models
trader._apply_risk          = fake_apply_risk
trader._execute_trade       = fake_execute_trade
trader._allocate_capital    = lambda: type("A", (), {"__repr__": lambda s: "alloc"})()

print("=" * 70)
print("TEST 1: same bar, called twice -> alpha models run ONCE, execute ONCE")
print("=" * 70)
r1 = trader._run_cycle("EURUSD")
r2 = trader._run_cycle("EURUSD")

print(f"  cycle 1: status={r1.status.name}")
print(f"  cycle 2: status={r2.status.name}")
print(f"  _run_alpha_models calls: {len(alpha_model_calls)}")
print(f"  _execute_trade calls: {len(execute_calls)} -> {execute_calls}")

ok1 = r1.status == CycleStatus.OK
print(f"  {PASS if ok1 else FAIL} first cycle on a fresh bar executes (status=OK)")
all_ok &= ok1

ok2 = r2.status == CycleStatus.BAR_ALREADY_TRADED
print(f"  {PASS if ok2 else FAIL} second cycle on the SAME bar is gated at fetch time (BAR_ALREADY_TRADED)")
all_ok &= ok2

ok3 = len(alpha_model_calls) == 1
print(f"  {PASS if ok3 else FAIL} _run_alpha_models (where RL steps its LSTM state) called ONCE, not twice")
all_ok &= ok3

ok4 = len(execute_calls) == 1
print(f"  {PASS if ok4 else FAIL} _execute_trade called exactly once")
all_ok &= ok4

print("\n" + "=" * 70)
print("TEST 2: a genuinely NEW bar is analysed and executed again")
print("=" * 70)
current_bar_index["value"] = 1  # advance one bar
r3 = trader._run_cycle("EURUSD")
print(f"  cycle 3 (new bar): status={r3.status.name}")
print(f"  _run_alpha_models calls total: {len(alpha_model_calls)}")
print(f"  _execute_trade calls total: {len(execute_calls)}")

ok5 = r3.status == CycleStatus.OK
print(f"  {PASS if ok5 else FAIL} new bar executes again (status=OK)")
all_ok &= ok5

ok6 = len(alpha_model_calls) == 2 and len(execute_calls) == 2
print(f"  {PASS if ok6 else FAIL} both alpha models and execution ran again for the new bar")
all_ok &= ok6

print("\n" + "=" * 70)
print("TEST 3: a HOLD-yielding bar is ALSO gated on repeat (not just actionable ones)")
print("=" * 70)
current_bar_index["value"] = 2  # yet another new bar
signal_to_emit["value"] = SignalType.HOLD
r4 = trader._run_cycle("EURUSD")   # first look at bar 2 -> HOLD, no execution
r5 = trader._run_cycle("EURUSD")   # same bar again -> must be gated too

print(f"  cycle 4 (new bar, HOLD): status={r4.status.name}")
print(f"  cycle 5 (same bar again): status={r5.status.name}")
print(f"  _run_alpha_models calls total: {len(alpha_model_calls)}")

ok7 = r4.status == CycleStatus.NO_SIGNAL
print(f"  {PASS if ok7 else FAIL} first look at the HOLD bar -> NO_SIGNAL")
all_ok &= ok7

ok8 = r5.status == CycleStatus.BAR_ALREADY_TRADED
print(f"  {PASS if ok8 else FAIL} repeat on the SAME HOLD bar -> gated too (BAR_ALREADY_TRADED)")
all_ok &= ok8

ok9 = len(alpha_model_calls) == 3   # only +1 since bar 2, not +2
print(f"  {PASS if ok9 else FAIL} _run_alpha_models NOT called again for the repeat "
      f"(total={len(alpha_model_calls)}, expected 3)")
all_ok &= ok9

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All bar-dedup checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
