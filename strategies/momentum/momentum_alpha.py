"""
MomentumAlpha
=============
Signal based on short-term vs long-term EMA crossover + RSI filter.

BUY  when fast EMA crosses above slow EMA and RSI < overbought threshold.
SELL when fast EMA crosses below slow EMA and RSI > oversold threshold.
HOLD otherwise.

Confidence is derived from the normalised gap between the two EMAs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.alpha_models.base import AlphaModel, AlphaSignal


class MomentumAlpha(AlphaModel):
    """
    EMA-crossover momentum strategy with RSI confirmation.

    Parameters
    ----------
    fast_period : int
        Period of the fast EMA (default 12).
    slow_period : int
        Period of the slow EMA (default 26).
    rsi_period  : int
        Period for the RSI calculation (default 14).
    rsi_overbought : float
        RSI level above which BUY signals are suppressed (default 70).
    rsi_oversold   : float
        RSI level below which SELL signals are suppressed (default 30).
    """

    def __init__(
        self,
        name: str = "momentum",
        fast_period: int = 12,
        slow_period: int = 26,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self.fast_period    = fast_period
        self.slow_period    = slow_period
        self.rsi_period     = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold   = rsi_oversold

    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        close = data["close"].astype(float)

        min_rows = self.slow_period + self.rsi_period + 2
        if len(close) < min_rows:
            return self._hold(reason="insufficient_data", rows=len(close))

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        rsi      = self._compute_rsi(close)

        gap_now  = fast_ema.iloc[-1] - slow_ema.iloc[-1]
        gap_prev = fast_ema.iloc[-2] - slow_ema.iloc[-2]
        rsi_now  = rsi.iloc[-1]

        # Normalised confidence: EMA gap as a fraction of price
        confidence = float(np.clip(abs(gap_now) / close.iloc[-1], 0.0, 1.0))

        meta = {
            "fast_ema": round(fast_ema.iloc[-1], 5),
            "slow_ema": round(slow_ema.iloc[-1], 5),
            "ema_gap":  round(gap_now, 5),
            "rsi":      round(rsi_now, 2),
        }

        # Bullish crossover
        if gap_prev <= 0 and gap_now > 0 and rsi_now < self.rsi_overbought:
            return self._buy(confidence, **meta)

        # Bearish crossover
        if gap_prev >= 0 and gap_now < 0 and rsi_now > self.rsi_oversold:
            return self._sell(confidence, **meta)

        return self._hold(**meta)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta  = close.diff()
        gain   = delta.clip(lower=0).ewm(com=self.rsi_period - 1, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs     = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
