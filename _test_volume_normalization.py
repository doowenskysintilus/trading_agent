#!/usr/bin/env python3
"""
Test script for MT5ExecutionEngine._normalize_volume() (Phase: fix
retcode 10014 INVALID_VOLUME).

Root cause: LiveTrader._build_trade_order() sizes positions in raw
base-currency UNITS (risk_amount / sl_distance -> e.g. 166,666 "units" of
EURUSD), but MT5ExecutionEngine.execute() sent that number directly as the
order's `volume` field, which MT5 expects in LOTS (1.0 lot = symbol's
trade_contract_size units, typically 100,000 for FX). A sizing result of
166,666 was submitted as a request for 166,666 LOTS -> always rejected.

Connects read-only to whatever MT5 terminal is already logged in (no fresh
login, doesn't disturb a live session) to validate against REAL broker
volume_min/max/step constraints, not guessed values.
"""

import MetaTrader5 as mt5

from engines.execution_engine.execution_engine import MT5ExecutionEngine, ExecutionConfig

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True

engine = MT5ExecutionEngine(ExecutionConfig())
connected = engine.connect()
print(f"MT5 connect(): {connected}")
if not connected:
    print("Cannot reach MT5 terminal in this environment — skipping live checks.")
    raise SystemExit(0)

try:
    print("=" * 70)
    print("TEST 1: realistic raw-unit sizing converts to a sane lot size")
    print("=" * 70)
    # ~1.67 standard lots of EURUSD if contract_size=100,000 (166,666 / 100,000)
    for symbol, raw_units in [("EURUSD", 166_666.0), ("XAUUSD", 500.0), ("BTCUSD", 0.5)]:
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"  (skip {symbol} — not available on this account)")
            continue
        lots = engine._normalize_volume(symbol, raw_units)
        contract = getattr(info, "trade_contract_size", 100_000)
        ok = info.volume_min <= lots <= info.volume_max
        print(f"  {symbol}: raw_units={raw_units} contract_size={contract} -> "
              f"lots={lots} (broker min={info.volume_min} max={info.volume_max} step={info.volume_step})")
        print(f"  {PASS if ok else FAIL} within broker [volume_min, volume_max]")
        all_ok &= ok

    print("\n" + "=" * 70)
    print("TEST 2: an oversized raw request clips to volume_max, not rejected")
    print("=" * 70)
    info = mt5.symbol_info("EURUSD")
    if info is not None:
        huge_units = 50_000_000.0  # would be 500 lots at 100k contract size — likely past max
        lots = engine._normalize_volume("EURUSD", huge_units)
        ok = lots <= info.volume_max and lots > 0
        print(f"  huge raw_units={huge_units} -> lots={lots} (max={info.volume_max})")
        print(f"  {PASS if ok else FAIL} clipped to volume_max, not left oversized")
        all_ok &= ok

    print("\n" + "=" * 70)
    print("TEST 3: an undersized raw request floors to volume_min, not zero")
    print("=" * 70)
    if info is not None:
        tiny_units = 1.0
        lots = engine._normalize_volume("EURUSD", tiny_units)
        ok = lots == info.volume_min
        print(f"  tiny raw_units={tiny_units} -> lots={lots} (min={info.volume_min})")
        print(f"  {PASS if ok else FAIL} floored to volume_min")
        all_ok &= ok

    print("\n" + "=" * 70)
    print("TEST 4: result is always an exact multiple of volume_step")
    print("=" * 70)
    if info is not None:
        lots = engine._normalize_volume("EURUSD", 234_567.0)
        step = info.volume_step
        remainder = round(lots / step) * step - lots
        ok = abs(remainder) < 1e-6
        print(f"  lots={lots} step={step} -> remainder from nearest step={remainder:.8f}")
        print(f"  {PASS if ok else FAIL} exact multiple of volume_step")
        all_ok &= ok

    print("\n" + "=" * 70)
    print("TEST 5: unknown symbol returns 0 (caller rejects cleanly) not a crash")
    print("=" * 70)
    lots = engine._normalize_volume("NOT_A_REAL_SYMBOL_XYZ", 100_000.0)
    ok = lots == 0.0
    print(f"  {PASS if ok else FAIL} unknown symbol -> {lots}")
    all_ok &= ok

finally:
    engine.disconnect()

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All volume-normalization checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
