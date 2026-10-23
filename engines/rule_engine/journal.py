"""
TradeJournal
============
In-memory history of entries and closed trades that the frequency rules
(FQ-01..04) read. Rebuilt from MySQL at startup (DB-03) so a restart does
not reset trade counts, cooldowns or the global pause.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional


@dataclass(frozen=True)
class JournalEntry:
    symbol: str
    ts: datetime


@dataclass(frozen=True)
class JournalClose:
    symbol: str
    ts: datetime
    pnl: float


class TradeJournal:
    def __init__(self, keep: timedelta = timedelta(days=2)) -> None:
        self._keep = keep
        self._entries: list[JournalEntry] = []
        self._closes: list[JournalClose] = []
        self._lock = threading.Lock()

    # ---- writes ----------------------------------------------------------

    def record_entry(self, symbol: str, ts: datetime) -> None:
        with self._lock:
            self._entries.append(JournalEntry(symbol, ts))
            self._prune(ts)

    def record_close(self, symbol: str, ts: datetime, pnl: float) -> None:
        with self._lock:
            self._closes.append(JournalClose(symbol, ts, pnl))
            self._closes.sort(key=lambda c: c.ts)
            self._prune(ts)

    def load(self, entries: Iterable[JournalEntry], closes: Iterable[JournalClose]) -> None:
        with self._lock:
            self._entries = sorted(entries, key=lambda e: e.ts)
            self._closes = sorted(closes, key=lambda c: c.ts)

    # ---- reads -----------------------------------------------------------

    def count_entries(self, since: datetime, symbol: Optional[str] = None) -> int:
        with self._lock:
            return sum(
                1 for e in self._entries
                if e.ts >= since and (symbol is None or e.symbol == symbol)
            )

    def last_loss_ts(self, symbol: Optional[str] = None) -> Optional[datetime]:
        with self._lock:
            for c in reversed(self._closes):
                if c.pnl < 0 and (symbol is None or c.symbol == symbol):
                    return c.ts
        return None

    def loss_streak(self) -> tuple[int, Optional[datetime]]:
        """Consecutive losing trades at the end of the history (all symbols)
        and the close time of the last one."""
        with self._lock:
            streak, last_ts = 0, None
            for c in reversed(self._closes):
                if c.pnl >= 0:
                    break
                streak += 1
                last_ts = last_ts or c.ts
            return streak, last_ts

    # ---- internals -------------------------------------------------------

    def _prune(self, now: datetime) -> None:
        cutoff = now - self._keep
        self._entries = [e for e in self._entries if e.ts >= cutoff]
        self._closes = [c for c in self._closes if c.ts >= cutoff]
