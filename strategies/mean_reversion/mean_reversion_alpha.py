"""
MeanReversionAlpha
==================
Signal based on Bollinger Band position, using pre-computed feature columns.

BUY  when price is near the lower Bollinger Band (bb_pct ≈ 0 — oversold).
SELL when price is near the upper Bollinger Band (bb_pct ≈ 1 — overbought).
HOLD otherwise.

Uses FeatureEngineer output columns (no raw close needed):
  bb_pct   = (close - bb_lower) / (bb_upper - bb_lower)  — 0=lower, 1=upper
  bb_width = (bb_upper - bb_lower) / bb_mid               — normalised bandwidth

bb_pct threshold derivation from classic Z-score parameters:
  zscore = n_std × (2 × bb_pct − 1)
  bb_pct for zscore = +threshold : (threshold/n_std + 1) / 2  → e.g. 0.875
  bb_pct for zscore = -threshold : (1 - threshold/n_std) / 2  → e.g. 0.125
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
    n_std             : float  Std multiplier used in FeatureEngineer (must match).
    zscore_threshold  : float  |Z-score| equivalent threshold for signal generation.
    min_width         : float  Minimum bb_width to avoid signals in flat markets.
    """

    def __init__(
        self,
        name: str = "mean_reversion",
        n_std: float = 2.0,
        zscore_threshold: float = 1.5,
        min_width: float = 0.0003,   # ignore signals when bands are too tight
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self.n_std            = n_std
        self.zscore_threshold = zscore_threshold
        self.min_width        = min_width

        # Pre-compute bb_pct thresholds from Z-score parameters
        # bb_pct = (zscore/n_std + 1) / 2
        self._pct_upper = (zscore_threshold / n_std + 1.0) / 2.0   # e.g. 0.875
        self._pct_lower = (1.0 - zscore_threshold / n_std) / 2.0   # e.g. 0.125

    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        missing = {"bb_pct", "bb_width"} - set(data.columns)
        if missing:
            return self._hold(reason=f"missing_features:{','.join(missing)}")

        bb_pct   = float(data["bb_pct"].iloc[-1])
        bb_width = float(data["bb_width"].iloc[-1])

        if bb_width < self.min_width:
            return self._hold(reason="bands_too_tight", bb_width=round(bb_width, 6))

        meta = {
            "bb_pct":   round(bb_pct, 3),
            "bb_width": round(bb_width, 5),
        }

        # Price near lower band → mean-reversion BUY
        if bb_pct <= self._pct_lower:
            # Confidence: how far below the lower threshold (capped at 1.0)
            conf = float(np.clip(
                (self._pct_lower - bb_pct) / max(self._pct_lower, 1e-6),
                0.0, 1.0,
            ))
            return self._buy(conf, **meta)

        # Price near upper band → mean-reversion SELL
        if bb_pct >= self._pct_upper:
            conf = float(np.clip(
                (bb_pct - self._pct_upper) / max(1.0 - self._pct_upper, 1e-6),
                0.0, 1.0,
            ))
            return self._sell(conf, **meta)

        return self._hold(**meta)
