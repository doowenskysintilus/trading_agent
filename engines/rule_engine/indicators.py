"""Indicators used by the rules (computed on closed OHLC bars)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (simple moving average of the true range)."""
    high  = bars["high"].astype(float)
    low   = bars["low"].astype(float)
    prev  = bars["close"].astype(float).shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def last_atr(bars: pd.DataFrame, period: int = 14) -> float:
    if bars is None or len(bars) < period + 1:
        return 0.0
    value = atr_series(bars, period).iloc[-1]
    return float(value) if np.isfinite(value) else 0.0
