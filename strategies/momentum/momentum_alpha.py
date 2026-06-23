"""
MomentumAlpha
=============
Signal based on EMA crossover + RSI filter, using pre-computed feature columns.

BUY  when ema_diff crosses from negative to positive AND rsi_norm < overbought.
SELL when ema_diff crosses from positive to negative AND rsi_norm > oversold.
HOLD otherwise.

Uses FeatureEngineer output columns (no raw close needed):
  ema_diff  = (ema_fast - ema_slow) / close   — sign change = crossover
  ema_ratio = ema_fast / ema_slow             — > 1 bullish, < 1 bearish
  rsi_norm  = rsi / 100                       — 0–1 range
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
    rsi_overbought : float   RSI level (0–100 scale) above which BUY is suppressed.
    rsi_oversold   : float   RSI level (0–100 scale) below which SELL is suppressed.
    min_diff       : float   Minimum |ema_diff| to consider a crossover real (noise filter).
    """

    def __init__(
        self,
        name: str = "momentum",
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        min_diff: float = 0.00005,   # ~0.5 pip on FX: ignore micro-crossovers
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self.rsi_overbought = rsi_overbought / 100.0   # normalise to [0-1]
        self.rsi_oversold   = rsi_oversold   / 100.0
        self.min_diff       = min_diff

    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        missing = {"ema_diff", "rsi_norm"} - set(data.columns)
        if missing:
            return self._hold(reason=f"missing_features:{','.join(missing)}")

        if len(data) < 2:
            return self._hold(reason="insufficient_data", rows=len(data))

        diff_now  = float(data["ema_diff"].iloc[-1])
        diff_prev = float(data["ema_diff"].iloc[-2])
        rsi_now   = float(data["rsi_norm"].iloc[-1])

        # Confidence: how wide the gap is relative to a 1% move (cap at 1.0)
        confidence = float(np.clip(abs(diff_now) / 0.01, 0.0, 1.0))

        meta = {
            "ema_diff": round(diff_now,  6),
            "rsi":      round(rsi_now * 100, 1),
        }

        # Bullish crossover: diff crosses zero upward + not overbought
        if (diff_prev <= 0 and diff_now > self.min_diff
                and rsi_now < self.rsi_overbought):
            return self._buy(confidence, **meta)

        # Bearish crossover: diff crosses zero downward + not oversold
        if (diff_prev >= 0 and diff_now < -self.min_diff
                and rsi_now > self.rsi_oversold):
            return self._sell(confidence, **meta)

        return self._hold(**meta)
