#!/usr/bin/env python3
"""
Test script for TRADE_OPENED / TRADE_CLOSED Telegram-style alerts
(monitoring/monitor.py). Verifies:
  1. A cycle with a FILLED execution fires TRADE_OPENED with symbol,
     direction, price, size, confidence.
  2. record_trade() (called by LiveTrader on trade close) fires
     TRADE_CLOSED with pnl/outcome.
  3. A cycle with no executions fires neither.

Uses a captured alert hook instead of a real Telegram call.
"""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from monitoring.monitor import TradingMonitor, MonitorConfig
from live_trading.live_trader import TradingCycleResult, CycleStatus
from engines.execution_engine.execution_engine import ExecutionResult, ExecutionStatus
from engines.signal_engine.signal_aggregator import AggregatedDecision
from engines.risk_engine.risk_engine import PortfolioState
from research.alpha_models.base import SignalType

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True

tmpdir = Path(tempfile.mkdtemp(prefix="alert_test_"))
try:
    captured: list[tuple[str, dict]] = []

    monitor = TradingMonitor(MonitorConfig(log_dir=str(tmpdir)))
    monitor.register_alert(lambda t, d: captured.append((t, d)))

    ps = PortfolioState(equity=10000.0, balance=10000.0, peak_equity=10000.0, daily_start_equity=10000.0)

    decision = AggregatedDecision(
        signal=SignalType.BUY, confidence=0.82,
        buy_score=0.82, sell_score=0.0, net_score=0.82,
    )
    execution = ExecutionResult(
        status=ExecutionStatus.FILLED, ticket=555,
        requested_price=1.0950, fill_price=1.0951, volume=0.5,
    )

    result = TradingCycleResult(
        cycle_id=1, symbol="EURUSD", timestamp=datetime.now(timezone.utc),
        status=CycleStatus.OK, decision=decision, executions=[execution],
        portfolio_state=ps,
    )

    print("=" * 70)
    print("TEST 1: cycle with a FILLED execution -> TRADE_OPENED")
    print("=" * 70)
    monitor(result)
    opened = [d for t, d in captured if t == "TRADE_OPENED"]
    ok = (
        len(opened) == 1
        and opened[0]["symbol"] == "EURUSD"
        and opened[0]["direction"] == "BUY"
        and opened[0]["price"] == 1.0951
        and opened[0]["size"] == 0.5
        and opened[0]["confidence"] == 0.82
        and opened[0]["ticket"] == 555
        # No live MT5 connection in this test -> falls back to lots x price.
        and opened[0]["invested_value"] == round(0.5 * 1.0951, 2)
    )
    print(f"  {PASS if ok else FAIL} TRADE_OPENED fired with correct data: {opened}")
    all_ok &= ok

    print("\n" + "=" * 70)
    print("TEST 2: record_trade() (trade close) -> TRADE_CLOSED")
    print("=" * 70)
    captured.clear()
    monitor.record_trade(
        ts=datetime.now(timezone.utc), symbol="EURUSD", strategy="momentum",
        direction="BUY", size=0.5, entry_price=1.0951, exit_price=1.0975,
        pnl=12.0, duration_s=300, exit_reason="take-profit", ticket="555",
    )
    closed = [d for t, d in captured if t == "TRADE_CLOSED"]
    ok2 = (
        len(closed) == 1
        and closed[0]["pnl"] == 12.0
        and closed[0]["outcome"] == "WIN"
        and closed[0]["symbol"] == "EURUSD"
        and closed[0]["exit_reason"] == "take-profit"
        and closed[0]["invested_value"] == round(0.5 * 1.0951, 2)
    )
    print(f"  {PASS if ok2 else FAIL} TRADE_CLOSED fired with correct data: {closed}")
    all_ok &= ok2

    print("\n" + "=" * 70)
    print("TEST 3: cycle with NO executions -> no TRADE_OPENED")
    print("=" * 70)
    captured.clear()
    result_no_exec = TradingCycleResult(
        cycle_id=2, symbol="GBPUSD", timestamp=datetime.now(timezone.utc),
        status=CycleStatus.NO_SIGNAL, executions=None, portfolio_state=ps,
    )
    monitor(result_no_exec)
    ok3 = not any(t == "TRADE_OPENED" for t, d in captured)
    print(f"  {PASS if ok3 else FAIL} no TRADE_OPENED fired: {captured}")
    all_ok &= ok3

    print("\n" + "=" * 70)
    print("TEST 4: a REJECTED execution does not fire TRADE_OPENED")
    print("=" * 70)
    captured.clear()
    rejected = ExecutionResult(status=ExecutionStatus.REJECTED, ticket=0, message="no money")
    result_rejected = TradingCycleResult(
        cycle_id=3, symbol="EURUSD", timestamp=datetime.now(timezone.utc),
        status=CycleStatus.OK, decision=decision, executions=[rejected], portfolio_state=ps,
    )
    monitor(result_rejected)
    ok4 = not any(t == "TRADE_OPENED" for t, d in captured)
    print(f"  {PASS if ok4 else FAIL} no TRADE_OPENED fired for a rejected order: {captured}")
    all_ok &= ok4

    print("\n" + "=" * 70)
    print("TEST 5: Telegram message text shows 'Valeur investie' and 'Bénéfice'")
    print("=" * 70)
    from monitoring.alerts import TelegramAlertSender

    open_text = TelegramAlertSender._format("TRADE_OPENED", opened[0])
    close_text = TelegramAlertSender._format("TRADE_CLOSED", closed[0])
    print(f"  TRADE_OPENED text:\n{open_text}\n")
    print(f"  TRADE_CLOSED text:\n{close_text}\n")

    ok5 = "Valeur investie" in open_text and "USD" in open_text
    print(f"  {PASS if ok5 else FAIL} TRADE_OPENED message shows invested value")
    all_ok &= ok5

    ok6 = "Valeur investie" in close_text and "Bénéfice" in close_text
    print(f"  {PASS if ok6 else FAIL} TRADE_CLOSED message shows both invested value and profit")
    all_ok &= ok6

    ok7 = "Raison: take-profit" in close_text
    print(f"  {PASS if ok7 else FAIL} TRADE_CLOSED message shows the real exit reason (not generic 'closed')")
    all_ok &= ok7

    print("\n" + "=" * 70)
    print("TEST 6: MT5 DEAL_REASON_* -> human label mapping")
    print("=" * 70)
    from engines.execution_engine.execution_engine import _DEAL_REASON_LABELS

    expected = {0: "manual (desktop)", 4: "stop-loss", 5: "take-profit"}
    ok8 = all(_DEAL_REASON_LABELS.get(k) == v for k, v in expected.items())
    print(f"  {PASS if ok8 else FAIL} reason codes map correctly: {_DEAL_REASON_LABELS}")
    all_ok &= ok8

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All trade-alert checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
