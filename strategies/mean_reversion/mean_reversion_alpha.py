"""
MeanReversionAlpha
==================
Signal based on Bollinger Bands + Z-score of price deviation.

BUY  when price touches or breaches the lower band (oversold).
SELL when price touches or breaches the upper band (overbought).
HOLD otherwise.

Confidence is the normalised Z-score of the deviation from the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.alpha_models.base import AlphaModel, AlphaSignal


class MeanReversionAlpha(AlphaModel):
    """
    Bollinger Band mean-reversion strategy.

    Parameters
    ----------
    window : int
        Rolling window for the moving average and std (default 20).
    n_std  : float
        Number of standard deviations for the bands (default 2.0).
    zscore_threshold : float
        |Z-score| above which a signal is generated (default 1.5).
    """

    def __init__(
        self,
        name: str = "mean_reversion",
        window: int = 20,
        n_std: float = 2.0,
        zscore_threshold: float = 1.5,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self.window           = window
        self.n_std            = n_std
        self.zscore_threshold = zscore_threshold

    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        close = data["close"].astype(float)

        if len(close) < self.window + 1:
            return self._hold(reason="insufficient_data", rows=len(close))

        rolling_mean = close.rolling(self.window).mean()
        rolling_std  = close.rolling(self.window).std(ddof=0)

        upper = rolling_mean + self.n_std * rolling_std
        lower = rolling_mean - self.n_std * rolling_std

        price   = close.iloc[-1]
        mean    = rolling_mean.iloc[-1]
        std     = rolling_std.iloc[-1]
        upper_b = upper.iloc[-1]
        lower_b = lower.iloc[-1]

        if std < 1e-10:
            return self._hold(reason="zero_std")

        zscore = (price - mean) / std

        # Confidence: how far past the threshold (capped at 1.0)
        confidence = float(np.clip(
            (abs(zscore) - self.zscore_threshold) / self.zscore_threshold,
            0.0, 1.0,
        ))

        meta = {
            "price":   round(price, 5),
            "mean":    round(mean, 5),
            "upper":   round(upper_b, 5),
            "lower":   round(lower_b, 5),
            "zscore":  round(zscore, 3),
        }

        # Price at or below lower band → mean-reversion BUY
        if zscore <= -self.zscore_threshold:
            return self._buy(confidence, **meta)

        # Price at or above upper band → mean-reversion SELL
        if zscore >= self.zscore_threshold:
            return self._sell(confidence, **meta)

        return self._hold(**meta)
