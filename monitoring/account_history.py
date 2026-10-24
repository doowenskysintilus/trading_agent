"""
AccountHistory
==============
Read-only view of the trading account's full MT5 history — every position
ever closed on the account (by the bot or by hand), deposits/withdrawals,
and realised P&L grouped by hour / day / week / month / year.

Works on plain deal dicts (MT5 `history_deals_get` rows) supplied by an
injected callable, so it can be tested without a terminal.

Times: MT5 stamps deals with the broker's clock; `utc_offset_s` converts
them to UTC, and every public method takes the viewer's UTC offset
(`tz_offset_min`) so periods are cut at the viewer's local midnight.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

# MT5 enums (MetaTrader5 package constants, inlined to stay importable
# without the terminal).
DEAL_TYPE_BUY, DEAL_TYPE_SELL = 0, 1
ENTRY_IN, ENTRY_OUT, ENTRY_INOUT, ENTRY_OUT_BY = 0, 1, 2, 3
_TRADE_TYPES = {DEAL_TYPE_BUY, DEAL_TYPE_SELL}

_REASON_LABELS = {
    0: "manuel (PC)", 1: "manuel (mobile)", 2: "manuel (web)", 3: "bot / API",
    4: "stop-loss", 5: "take-profit", 6: "stop-out", 7: "rollover",
}

GROUP_RULES = {"hour": "h", "day": "D", "week": "W-MON", "month": "MS", "year": "YS"}

ACCOUNT_START = datetime(2000, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ClosedTrade:
    position_id: int
    symbol: str
    direction: str            # BUY / SELL
    volume: float
    open_time: str            # ISO, UTC
    close_time: str           # ISO, UTC
    open_price: float
    close_price: float
    profit: float
    commission: float
    swap: float
    net: float                # profit + commission + swap + fees
    exit_reason: str
    source: str               # bot / manual
    duration_s: int

    def as_dict(self) -> dict:
        return asdict(self)


class AccountHistory:
    def __init__(
        self,
        fetch_deals: Callable[[datetime, datetime], list[dict]],
        utc_offset_s: Callable[[], int],
        bot_magic: int,
        cache_ttl_s: float = 30.0,
    ) -> None:
        self._fetch = fetch_deals
        self._offset = utc_offset_s
        self._magic = bot_magic
        self._ttl = cache_ttl_s
        self._cache: Optional[tuple[float, list[dict]]] = None

    # ------------------------------------------------------------------
    # Deals
    # ------------------------------------------------------------------

    def _deals(self) -> list[dict]:
        """All deals since the account was opened, cached a few seconds
        (the dashboard asks for several views at once)."""
        now = time.monotonic()
        if self._cache and now - self._cache[0] < self._ttl:
            return self._cache[1]
        offset = self._offset()
        # +1 day: the broker clock may run ahead of UTC.
        raw = self._fetch(ACCOUNT_START, datetime.now(timezone.utc) + timedelta(days=1))
        deals = []
        for d in raw:
            broker_ts = (d.get("time_msc") or d.get("time", 0) * 1000) / 1000.0
            deals.append({**d, "utc": datetime.fromtimestamp(broker_ts - offset, tz=timezone.utc)})
        deals.sort(key=lambda d: d["utc"])
        self._cache = (now, deals)
        return deals

    def closed_trades(self) -> list[ClosedTrade]:
        """One record per position fully closed, newest first."""
        by_position: dict[int, list[dict]] = {}
        for d in self._deals():
            if d.get("type") in _TRADE_TYPES and d.get("position_id"):
                by_position.setdefault(d["position_id"], []).append(d)

        trades: list[ClosedTrade] = []
        for pid, deals in by_position.items():
            ins = [d for d in deals if d.get("entry") in (ENTRY_IN, ENTRY_INOUT)]
            outs = [d for d in deals if d.get("entry") in (ENTRY_OUT, ENTRY_OUT_BY, ENTRY_INOUT)]
            if not ins or not outs:
                continue                       # still open (or partial history)
            vol_in = sum(d.get("volume", 0.0) for d in ins)
            vol_out = sum(d.get("volume", 0.0) for d in outs if d.get("entry") != ENTRY_INOUT)
            if outs[-1].get("entry") != ENTRY_INOUT and vol_out + 1e-9 < vol_in:
                continue                       # partially closed — not finished yet
            first, last = ins[0], outs[-1]
            profit = sum(d.get("profit", 0.0) for d in deals)
            commission = sum(d.get("commission", 0.0) for d in deals)
            swap = sum(d.get("swap", 0.0) for d in deals)
            fee = sum(d.get("fee", 0.0) for d in deals)
            trades.append(ClosedTrade(
                position_id = int(pid),
                symbol      = first.get("symbol", ""),
                direction   = "BUY" if first.get("type") == DEAL_TYPE_BUY else "SELL",
                volume      = round(vol_in, 4),
                open_time   = first["utc"].isoformat(),
                close_time  = last["utc"].isoformat(),
                open_price  = _vwap(ins),
                close_price = _vwap(outs),
                profit      = round(profit, 2),
                commission  = round(commission, 2),
                swap        = round(swap, 2),
                net         = round(profit + commission + swap + fee, 2),
                exit_reason = _REASON_LABELS.get(last.get("reason"), "autre"),
                source      = "bot" if first.get("magic") == self._magic else "manual",
                duration_s  = int((last["utc"] - first["utc"]).total_seconds()),
            ))
        trades.sort(key=lambda t: t.close_time, reverse=True)
        return trades

    def commission_rates(
        self,
        contract_size: Callable[[str], float],
        last_n: int = 50,
        min_trades: int = 3,
    ) -> dict[str, dict]:
        """Real commission per symbol, as a fraction of traded notional per
        side (the unit the scalping cost model uses), measured on the last
        `last_n` closed trades of each symbol. Symbols with fewer than
        `min_trades` trades are left out (caller keeps its default).

        Notional is in the account currency (USD): units × price for xxxUSD
        symbols, units for USDxxx symbols (the base is already USD).
        """
        samples: dict[str, list[tuple[float, float]]] = {}
        for t in self.closed_trades():              # newest first
            rows = samples.setdefault(t.symbol, [])
            if len(rows) >= last_n or not t.volume or not t.commission:
                continue
            size = contract_size(t.symbol) or 0.0
            units = t.volume * size
            notional = units if t.symbol.upper().startswith("USD") else units * t.open_price
            if notional > 0:
                rows.append((abs(t.commission) / (2.0 * notional), abs(t.commission) / t.volume))
        out = {}
        for symbol, rows in samples.items():
            if len(rows) < min_trades:
                continue
            rates = sorted(r for r, _ in rows)
            per_lot = sorted(p for _, p in rows)
            out[symbol] = {
                "rate_per_side":         rates[len(rates) // 2],          # median
                "per_lot_round_trip":    round(per_lot[len(per_lot) // 2], 2),
                "trades":                len(rows),
            }
        return out

    def cash_flows(self) -> list[dict]:
        """Deposits, withdrawals, credits — every non-trading deal with money."""
        return [{
            "time": d["utc"].isoformat(), "amount": round(d.get("profit", 0.0), 2),
            "comment": d.get("comment", ""),
        } for d in self._deals() if d.get("type") not in _TRADE_TYPES and d.get("profit")]

    # ------------------------------------------------------------------
    # Views used by the API
    # ------------------------------------------------------------------

    def trades(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        source: str = "all",
        symbol: Optional[str] = None,
    ) -> list[dict]:
        out = []
        for t in self.closed_trades():
            ct = datetime.fromisoformat(t.close_time)
            if start and ct < start or end and ct >= end:
                continue
            if source in ("bot", "manual") and t.source != source:
                continue
            if symbol and t.symbol.upper() != symbol.upper():
                continue
            out.append(t.as_dict())
        return out

    def pnl_series(
        self,
        group: str = "day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        source: str = "all",
        tz_offset_min: int = 0,
    ) -> list[dict]:
        """Net P&L per period, cut on the viewer's local calendar. Empty
        periods are kept (as 0) except for hourly grouping, and cover the
        whole requested range — not just the span between the first and
        last trade."""
        rule = GROUP_RULES.get(group, "D")
        rows = self.trades(start, end, source)
        if not rows:
            return []
        shift = pd.Timedelta(minutes=tz_offset_min)
        df = pd.DataFrame({
            "local": pd.to_datetime([r["close_time"] for r in rows], utc=True).tz_localize(None) + shift,
            "net":   [r["net"] for r in rows],
            "trade": 1,
            "win":   [int(r["net"] > 0) for r in rows],
        })
        # Zero rows at the range bounds stretch the resampled index over the
        # full requested range (they add no trade, no win, no P&L).
        bounds = [b for b in (
            start and start.replace(tzinfo=None) + shift,
            end and end.replace(tzinfo=None) + shift - pd.Timedelta(seconds=1),
        ) if b]
        if bounds:
            df = pd.concat([df, pd.DataFrame({"local": bounds, "net": 0.0, "trade": 0, "win": 0})])
        sums = df.set_index("local").sort_index().resample(rule).sum()
        table = pd.DataFrame({"net": sums["net"], "trades": sums["trade"], "wins": sums["win"]})
        if group == "hour":
            table = table[table["trades"] > 0]
        table["cumulative"] = table["net"].cumsum()
        return [{
            "period":     idx.isoformat(),
            "net":        round(float(r.net), 2),
            "trades":     int(r.trades),
            "wins":       int(r.wins),
            "cumulative": round(float(r.cumulative), 2),
        } for idx, r in table.iterrows()]

    def summary(self, source: str = "all", tz_offset_min: int = 0) -> dict:
        """Totals since account opening and for the current local periods."""
        rows = self.trades(source=source)
        flows = self.cash_flows()
        local_now = datetime.now(timezone.utc) + timedelta(minutes=tz_offset_min)
        day = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        starts = {
            "today": day,
            "week":  day - timedelta(days=day.weekday()),
            "month": day.replace(day=1),
            "year":  day.replace(month=1, day=1),
        }

        def total_since(local_start: datetime) -> dict:
            utc_start = local_start - timedelta(minutes=tz_offset_min)
            sel = [r["net"] for r in rows if datetime.fromisoformat(r["close_time"]) >= utc_start]
            return {"net": round(sum(sel), 2), "trades": len(sel)}

        nets = [r["net"] for r in rows]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n < 0]
        deposits = sum(f["amount"] for f in flows if f["amount"] > 0)
        first_deal = self._deals()[0]["utc"].isoformat() if self._deals() else None
        return {
            "account_opened": first_deal,
            "deposits":       round(deposits, 2),
            "withdrawals":    round(sum(f["amount"] for f in flows if f["amount"] < 0), 2),
            "all":            {"net": round(sum(nets), 2), "trades": len(nets)},
            **{name: total_since(s) for name, s in starts.items()},
            "win_rate":       round(len(wins) / len(nets), 4) if nets else None,
            "profit_factor":  round(sum(wins) / abs(sum(losses)), 3) if losses else None,
            "best_trade":     max(nets) if nets else None,
            "worst_trade":    min(nets) if nets else None,
            "return_pct":     round(100 * sum(nets) / deposits, 3) if deposits else None,
        }


def _vwap(deals: list[dict]) -> float:
    vol = sum(d.get("volume", 0.0) for d in deals)
    if vol <= 0:
        return float(deals[-1].get("price", 0.0))
    return round(sum(d.get("price", 0.0) * d.get("volume", 0.0) for d in deals) / vol, 6)
