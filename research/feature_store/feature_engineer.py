"""
FeatureEngineer
===============
Computes technical indicators and statistical features from raw OHLCV data.

Supported features
------------------
- Returns        : log_return, pct_return
- Volatility     : rolling_volatility, parkinson_volatility
- Trend          : ema_fast, ema_slow, ema_diff
- Momentum       : rsi, macd_line, macd_signal, macd_hist
- Volatility ATR : atr, natr (normalised)
- Volume         : volume_change, volume_zscore
- Bollinger      : bb_upper, bb_lower, bb_width, bb_pct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Controls which feature groups are computed and their parameters."""

    # Returns
    returns_periods: list[int] = field(default_factory=lambda: [1, 5, 10])

    # Volatility
    volatility_window: int = 20
    parkinson: bool = True

    # EMA
    ema_fast: int = 12
    ema_slow: int = 26

    # RSI
    rsi_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR
    atr_period: int = 14

    # Bollinger
    bb_window: int = 20
    bb_std: float = 2.0

    # Volume
    volume_window: int = 20

    # Drop NaN rows created by long-period indicators
    dropna: bool = True


# ---------------------------------------------------------------------------
# Engineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms a raw OHLCV DataFrame into a feature matrix ready for ML.

    Parameters
    ----------
    config : FeatureConfig
        Controls which indicators are computed and with which parameters.

    Usage
    -----
        eng = FeatureEngineer()
        features = eng.compute(df)           # pd.DataFrame
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from a raw OHLCV DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain lowercase columns: open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed like `data`. NaN rows are dropped if
            config.dropna is True.
        """
        df = data.copy()
        self._validate(df)

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_  = df["open"].astype(float)

        features = pd.DataFrame(index=df.index)

        # ---- Returns -----------------------------------------------------
        log_ret = np.log(close / close.shift(1))
        features["log_return"] = log_ret
        features["pct_return"] = close.pct_change()

        for p in self.config.returns_periods:
            features[f"return_{p}b"] = np.log(close / close.shift(p))

        # ---- Rolling volatility (annualised) -----------------------------
        cfg = self.config
        features["volatility"] = (
            log_ret.rolling(cfg.volatility_window).std() * np.sqrt(252)
        )

        if cfg.parkinson:
            ln_hl  = np.log(high / low) ** 2
            features["parkinson_vol"] = (
                np.sqrt(ln_hl.rolling(cfg.volatility_window).mean() / (4 * np.log(2)))
                * np.sqrt(252)
            )

        # ---- EMA trend ---------------------------------------------------
        ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()

        features["ema_fast"]  = ema_fast
        features["ema_slow"]  = ema_slow
        features["ema_diff"]  = (ema_fast - ema_slow) / close       # normalised
        features["ema_ratio"] = ema_fast / ema_slow

        # ---- RSI ---------------------------------------------------------
        features["rsi"] = self._rsi(close, cfg.rsi_period)

        # ---- MACD --------------------------------------------------------
        macd_line, macd_signal, macd_hist = self._macd(
            close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        features["macd_line"]   = macd_line / close    # normalised
        features["macd_signal"] = macd_signal / close
        features["macd_hist"]   = macd_hist / close

        # ---- ATR ---------------------------------------------------------
        atr = self._atr(high, low, close, cfg.atr_period)
        features["atr"]  = atr
        features["natr"] = atr / close                 # normalised ATR

        # ---- Bollinger Bands ---------------------------------------------
        bb_mid   = close.rolling(cfg.bb_window).mean()
        bb_std   = close.rolling(cfg.bb_window).std(ddof=0)
        bb_upper = bb_mid + cfg.bb_std * bb_std
        bb_lower = bb_mid - cfg.bb_std * bb_std
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        bb_pct   = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        features["bb_upper"] = bb_upper
        features["bb_lower"] = bb_lower
        features["bb_width"] = bb_width
        features["bb_pct"]   = bb_pct

        # ---- Volume features ---------------------------------------------
        vol_change  = volume.pct_change()
        vol_rolling = volume.rolling(cfg.volume_window)
        vol_zscore  = (volume - vol_rolling.mean()) / (vol_rolling.std(ddof=0) + 1e-10)

        features["volume_change"] = vol_change
        features["volume_zscore"] = vol_zscore

        # ---- OHLCV pass-through (raw prices useful for some models) ------
        for col in ["open", "high", "low", "close", "volume"]:
            features[col] = df[col].astype(float)

        if cfg.dropna:
            features = features.dropna()

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return the list of output feature column names (requires one compute call)."""
        return list(self.compute(
            pd.DataFrame({
                "open":   np.random.rand(100) + 100,
                "high":   np.random.rand(100) + 101,
                "low":    np.random.rand(100) + 99,
                "close":  np.random.rand(100) + 100,
                "volume": np.random.rand(100) * 1000,
            })
        ).columns)

    # ------------------------------------------------------------------
    # Indicator implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs    = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(
        close: pd.Series,
        fast: int,
        slow: int,
        signal_period: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast   = close.ewm(span=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        macd_sig   = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_hist  = macd_line - macd_sig
        return macd_line, macd_sig, macd_hist

    @staticmethod
    def _atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty.")
