#!/usr/bin/env python3
"""
Test script for monitoring.account_history.AccountHistory:
  1. Deals are paired into closed positions (entry + exit), net of fees
  2. Open / partially closed positions and cash flows are not trades
  3. Bot vs manual attribution by magic number
  4. Broker clock → UTC conversion
  5. P&L buckets per day / month on the viewer's local calendar
  6. Summary totals (since opening, deposits, win rate, profit factor)
No MT5 terminal needed — deals are plain dicts.
"""

from datetime import datetime, timezone

from monitoring.account_history import AccountHistory

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True


def check(cond, label):
    global all_ok
    print(f"  {PASS if cond else FAIL} {label}")
    all_ok &= bool(cond)


BOT = 20260524
OFFSET = 3 * 3600          # broker clock = UTC+3


def broker_ts(*args):
    """Epoch seconds as the BROKER stamps them (its wall clock read as UTC)."""
    return int(datetime(*args, tzinfo=timezone.utc).timestamp()) + OFFSET


def deal(ticket, pos, ts, typ, entry, price, vol=0.1, profit=0.0, commission=0.0,
         swap=0.0, magic=BOT, reason=3, symbol="EURUSD"):
    return {"ticket": ticket, "position_id": pos, "time": ts, "time_msc": ts * 1000, "type": typ,
            "entry": entry, "price": price, "volume": vol, "profit": profit,
            "commission": commission, "swap": swap, "fee": 0.0, "magic": magic,
            "reason": reason, "symbol": symbol, "comment": ""}


DEALS = [
    {"ticket": 1, "position_id": 0, "time": broker_ts(2026, 8, 6, 18), "type": 2, "entry": 0,
     "price": 0, "volume": 0, "profit": 10_000.0, "commission": 0, "swap": 0, "fee": 0,
     "magic": 0, "reason": 0, "symbol": "", "comment": "deposit"},
    # bot win, 1 Oct 2026 10:00 → 10:05 UTC
    deal(2, 100, broker_ts(2026, 10, 1, 10, 0), 0, 0, 1.1000),
    deal(3, 100, broker_ts(2026, 10, 1, 10, 5), 1, 1, 1.1010, profit=10.0, commission=-0.7, reason=5),
    # manual loss, 2 Oct 23:30 UTC (= 3 Oct 01:30 for a UTC+2 viewer)
    deal(4, 101, broker_ts(2026, 10, 2, 23, 0), 1, 0, 1.1050, magic=0, reason=0),
    deal(5, 101, broker_ts(2026, 10, 2, 23, 30), 0, 1, 1.1070, profit=-20.0, commission=-0.7,
         magic=0, reason=4),
    # still open
    deal(6, 102, broker_ts(2026, 10, 3, 9), 0, 0, 1.1100),
    # partially closed (0.1 in, 0.05 out)
    deal(7, 103, broker_ts(2026, 10, 3, 9), 0, 0, 1.1100),
    deal(8, 103, broker_ts(2026, 10, 3, 10), 1, 1, 1.1120, vol=0.05, profit=1.0),
]

h = AccountHistory(lambda a, b: DEALS, lambda: OFFSET, BOT)

print("=" * 70)
print("TEST 1-3: pairing, open/partial positions, attribution")
print("=" * 70)
trades = h.closed_trades()
check(len(trades) == 2, f"2 closed positions (open, partial and deposit ignored) → {len(trades)}")
by_id = {t.position_id: t for t in trades}
win, loss = by_id.get(100), by_id.get(101)
check(win and win.net == 9.3 and win.direction == "BUY", "bot win: net = profit + commission = +9.30")
check(win and win.source == "bot" and loss and loss.source == "manual", "bot / manual attribution by magic")
check(loss and loss.exit_reason == "stop-loss" and loss.direction == "SELL", "manual SELL closed by stop-loss")
check(trades[0].position_id == 101, "newest close first")

print("\n" + "=" * 70)
print("TEST 4: broker clock → UTC")
print("=" * 70)
check(win and win.close_time.startswith("2026-10-01T10:05:00"), f"close time in UTC: {win and win.close_time}")
check(win and win.duration_s == 300, "duration 5 min")

print("\n" + "=" * 70)
print("TEST 5: P&L buckets")
print("=" * 70)
days_utc = h.pnl_series("day", tz_offset_min=0)
check([(p["period"][:10], p["net"]) for p in days_utc] == [("2026-10-01", 9.3), ("2026-10-02", -20.7)],
      f"UTC days: {[(p['period'][:10], p['net']) for p in days_utc]}")
days_paris = h.pnl_series("day", tz_offset_min=120)
check(days_paris[-1]["period"][:10] == "2026-10-03" and days_paris[-1]["net"] == -20.7,
      "UTC+2 viewer: the 23:30 UTC loss falls on 3 Oct local")
check(len(days_paris) == 3 and days_paris[1]["net"] == 0, "empty day kept as 0")
check(days_paris[-1]["cumulative"] == -11.4, "cumulative curve")
months = h.pnl_series("month")
check(len(months) == 1 and months[0]["trades"] == 2 and months[0]["wins"] == 1, "1 month, 2 trades, 1 win")
check(h.pnl_series("day", source="bot")[0]["net"] == 9.3 and len(h.pnl_series("day", source="manual")) == 1,
      "source filter")
ranged = h.pnl_series("day", start=datetime(2026, 9, 28, tzinfo=timezone.utc),
                      end=datetime(2026, 10, 6, tzinfo=timezone.utc))
check([p["period"][:10] for p in ranged][0] == "2026-09-28" and len(ranged) == 8
      and sum(p["trades"] for p in ranged) == 2,
      f"requested range fully covered (28 Sep → 5 Oct, {len(ranged)} days, 2 trades)")
start = datetime(2026, 10, 2, tzinfo=timezone.utc)
check([t["position_id"] for t in h.trades(start=start)] == [101], "date filter on close time")

print("\n" + "=" * 70)
print("TEST 5b: commission rates per symbol")
print("=" * 70)
# 0.1 lot × 100 000 × 1.1000 = 11 000 $ notional; 0.70 $ round trip → 0.35 $/side
rates = h.commission_rates(lambda s: 100_000.0, min_trades=1)
eur = rates.get("EURUSD")
check(eur is not None and abs(eur["rate_per_side"] - 0.70 / 2 / 11_000) < 1e-9,
      f"EURUSD rate per side = 0.35 $ / 11 000 $ → {eur and eur['rate_per_side']:.8f}")
check(eur is not None and eur["per_lot_round_trip"] == 7.0, "7 $ per lot round trip (0.70 $ for 0.1 lot)")
check(h.commission_rates(lambda s: 100_000.0) == {}, "fewer than 3 trades → no measured rate (fallback)")

print("\n" + "=" * 70)
print("TEST 6: summary")
print("=" * 70)
s = h.summary()
check(s["all"] == {"net": -11.4, "trades": 2}, f"since opening: {s['all']}")
check(s["deposits"] == 10_000.0 and s["return_pct"] == -0.114, "deposits and return %")
check(s["win_rate"] == 0.5 and s["profit_factor"] == round(9.3 / 20.7, 3), "win rate / profit factor")
check(s["account_opened"].startswith("2026-08-06T18:00"), "account opening date (first deal)")

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All account history checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
