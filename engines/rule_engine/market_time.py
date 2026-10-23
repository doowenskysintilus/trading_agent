"""
Market-time helpers: trading-session windows and timeframe lengths.
All session times are UTC ("HH:MM-HH:MM", end exclusive).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
}


def timeframe_seconds(timeframe: str) -> int:
    return TIMEFRAME_SECONDS.get(timeframe.upper(), 60)


def parse_hhmm(value: str) -> time:
    hours, _, minutes = value.strip().partition(":")
    return time(int(hours), int(minutes or 0))


@dataclass(frozen=True)
class SessionWindow:
    start: time
    end: time

    @classmethod
    def parse(cls, spec: str) -> "SessionWindow":
        start, _, end = spec.partition("-")
        return cls(parse_hhmm(start), parse_hhmm(end))

    def contains(self, t: time) -> bool:
        if self.start <= self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end   # window crossing midnight

    def minutes_to_end(self, t: time) -> float:
        now_m = t.hour * 60 + t.minute + t.second / 60
        end_m = self.end.hour * 60 + self.end.minute
        if end_m <= now_m:
            end_m += 24 * 60
        return end_m - now_m


def active_window(windows: list[SessionWindow], now: datetime) -> Optional[SessionWindow]:
    t = now.time()
    return next((w for w in windows if w.contains(t)), None)
