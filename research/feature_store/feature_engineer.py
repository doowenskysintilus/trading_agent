"""
FeatureEngineer
===============
Computes technical indicators and statistical features from raw OHLCV data.

All output features are NORMALISED (no raw prices, no non-stationary columns).
This is critical for ML/RL models to generalise across price levels and time.

Output features (all bounded / stationary)
-------------------------------------------
Returns      : log_return, return_5b, return_10b
Volatility   : volatility, parkinson_vol
Bar structure: high_low_range, close_position, gap
EMA (norm.)  : ema_diff, ema_ratio
RSI (norm.)  : rsi_norm  (0–1)
MACD (norm.) : macd_line, macd_signal, macd_hist  (divided by close)
ATR (norm.)  : natr
Bollinger    : bb_width, bb_pct
Volume       : volume_zscore, volume_change_clipped
Time (cyclic): hour_sin, hour_cos, dow_sin, dow_cos
Calendar     : event_is_active, event_hours_until, event_vol_expectation (opt.)

Removed (non-stationary / raw prices):
  ema_fast, ema_slow, bb_upper, bb_lower, atr (absolute),
  pct_return (≈ log_return), open, high, low, close, volume (raw)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .calendar_provider import CalendarProvider, EventImportance
    _CALENDAR_AVAILABLE = True
except ImportError:
    _CALENDAR_AVAILABLE = False
    CalendarProvider = None
    EventImportance = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Controls which feature groups are computed and their parameters."""

    # Returns (multi-period log returns)
    returns_periods: list[int] = field(default_factory=lambda: [5, 10])

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

    # Time-of-day / day-of-week cyclic encoding
    # Requires a DatetimeIndex on the input DataFrame.
    include_time_features: bool = True

    # Economic Calendar
    # WARNING: only enable in LIVE inference, NOT in historical training.
    # Historical rows will receive the current event snapshot, not historical ones.
    include_calendar_features: bool = False
    calendar_source: str = "mock"

    # Drop NaN rows created by long-period indicators
    dropna: bool = True


# ---------------------------------------------------------------------------
# Engineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms a raw OHLCV DataFrame into a normalised feature matrix.

    All output columns are stationary and bounded so that ML / RL models
    trained on one date range generalise to other date ranges.
    Raw prices (EMA levels, Bollinger levels, ATR in price units, OHLCV) are
    NEVER included in the output.

    Parameters
    ----------
    config : FeatureConfig

    Usage
    -----
        eng  = FeatureEngineer()
        feat = eng.compute(df)   # pd.DataFrame, all columns normalised
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        calendar_provider: Optional[CalendarProvider] = None,
    ) -> pd.DataFrame:
        """
        Compute all features from a raw OHLCV DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain lowercase columns: open, high, low, close, volume.
            A DatetimeIndex is required when include_time_features=True.

        Returns
        -------
        pd.DataFrame
            Normalised feature matrix. No raw price columns.
        """
        df = data.copy()
        self._validate(df)

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_  = df["open"].astype(float)

        features = pd.DataFrame(index=df.index)
        cfg      = self.config

        # ----------------------------------------------------------------
        # 1. Returns (stationary, small magnitude)
        # ----------------------------------------------------------------
        log_ret = np.log(close / close.shift(1))
        features["log_return"] = log_ret

        for p in cfg.returns_periods:
            features[f"return_{p}b"] = np.log(close / close.shift(p))

        # ----------------------------------------------------------------
        # 2. Volatility (annualised — bounded, positive)
        # ----------------------------------------------------------------
        features["volatility"] = (
            log_ret.rolling(cfg.volatility_window).std() * np.sqrt(252)
        )

        if cfg.parkinson:
            ln_hl  = np.log(high / low) ** 2
            features["parkinson_vol"] = (
                np.sqrt(ln_hl.rolling(cfg.volatility_window).mean() / (4 * np.log(2)))
                * np.sqrt(252)
            )

        # ----------------------------------------------------------------
        # 3. Bar structure (all relative to bar range / close)
        # ----------------------------------------------------------------
        bar_range = high - low
        features["high_low_range"]  = bar_range / (close + 1e-10)   # range / price
        features["close_position"]  = (close - low) / (bar_range + 1e-10)  # 0–1
        features["gap"]             = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)

        # ----------------------------------------------------------------
        # 4. EMA — only normalised derivatives (no raw price levels)
        # ----------------------------------------------------------------
        ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()

        features["ema_diff"]  = (ema_fast - ema_slow) / (close + 1e-10)  # ≈ ±0.003
        features["ema_ratio"] = ema_fast / (ema_slow + 1e-10)             # ≈ 1.000 ± small

        # ----------------------------------------------------------------
        # 5. RSI — normalised to [0, 1]
        # ----------------------------------------------------------------
        features["rsi_norm"] = self._rsi(close, cfg.rsi_period) / 100.0

        # ----------------------------------------------------------------
        # 6. MACD — divided by close price (removes price level)
        # ----------------------------------------------------------------
        macd_line, macd_sig, macd_hist = self._macd(
            close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        features["macd_line"]   = macd_line   / (close + 1e-10)
        features["macd_signal"] = macd_sig    / (close + 1e-10)
        features["macd_hist"]   = macd_hist   / (close + 1e-10)

        # ----------------------------------------------------------------
        # 7. ATR — only normalised (natr = atr / close)
        # ----------------------------------------------------------------
        atr = self._atr(high, low, close, cfg.atr_period)
        features["natr"] = atr / (close + 1e-10)   # ≈ 0.001–0.010 for FX

        # ----------------------------------------------------------------
        # 8. Bollinger Bands — only normalised derivatives
        # ----------------------------------------------------------------
        bb_mid   = close.rolling(cfg.bb_window).mean()
        bb_std_s = close.rolling(cfg.bb_window).std(ddof=0)
        bb_upper = bb_mid + cfg.bb_std * bb_std_s
        bb_lower = bb_mid - cfg.bb_std * bb_std_s

        features["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)  # relative width
        features["bb_pct"]   = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)  # 0–1

        # ----------------------------------------------------------------
        # 9. Volume (z-score + clipped log change)
        # ----------------------------------------------------------------
        vol_rolling = volume.rolling(cfg.volume_window)
        features["volume_zscore"] = (
            (volume - vol_rolling.mean()) / (vol_rolling.std(ddof=0) + 1e-10)
        ).clip(-4, 4)

        # Log-signed volume change avoids extreme outliers from volume spikes
        vol_chg = volume.pct_change().clip(-1, 10)
        features["volume_change"] = np.sign(vol_chg) * np.log1p(vol_chg.abs())

        # ----------------------------------------------------------------
        # 10. Time features — cyclic sin/cos encoding
        # Encodes session (London/NY/Asian) and day-of-week patterns.
        # Requires DatetimeIndex; silently skipped if index has no time info.
        # ----------------------------------------------------------------
        if cfg.include_time_features:
            try:
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex):
                    idx = pd.to_datetime(idx, utc=True)
                hour = idx.hour + idx.minute / 60.0          # 0–24
                dow  = idx.dayofweek.astype(float)            # 0 Mon … 6 Sun
                features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
                features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
                features["dow_sin"]  = np.sin(2 * np.pi * dow  /  7.0)
                features["dow_cos"]  = np.cos(2 * np.pi * dow  /  7.0)
            except Exception:
                pass   # Index has no usable time info — skip silently

        # ----------------------------------------------------------------
        # 11. Economic Calendar features (LIVE inference only)
        # ----------------------------------------------------------------
        if cfg.include_calendar_features and _CALENDAR_AVAILABLE:
            if calendar_provider is None and symbol:
                calendar_provider = CalendarProvider(source=cfg.calendar_source)
            if calendar_provider and symbol:
                features = self._add_calendar_features(
                    features, calendar_provider, symbol, df
                )

        if cfg.dropna:
            features = features.dropna()

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return the column names produced by compute()."""
        dummy = pd.DataFrame({
            "open":   np.linspace(1.08, 1.10, 200),
            "high":   np.linspace(1.09, 1.11, 200),
            "low":    np.linspace(1.07, 1.09, 200),
            "close":  np.linspace(1.08, 1.10, 200),
            "volume": np.random.rand(200) * 1000 + 100,
        }, index=pd.date_range("2023-01-02 00:00", periods=200, freq="1h", tz="UTC"))
        return list(self.compute(dummy).columns)

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
        ema_fast  = close.ewm(span=fast,          adjust=False).mean()
        ema_slow  = close.ewm(span=slow,          adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_sig  = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, macd_sig, macd_line - macd_sig

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
    # Calendar features (live only — do NOT use in historical training)
    # ------------------------------------------------------------------

    def _add_calendar_features(
        self,
        features: pd.DataFrame,
        calendar_provider,
        symbol: str,
        original_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add economic calendar snapshot features for LIVE inference.

        WARNING: This method uses the CURRENT timestamp, so it is only
        correct when called in live trading (not historical back-testing).
        """
        try:
            is_active   = calendar_provider.event_is_active(symbol, window_minutes=30)
            hours_until = calendar_provider.hours_until_next_event(
                symbol, min_importance=EventImportance.MEDIUM
            ) or 999.0
            vol_mult    = calendar_provider.expected_volatility_multiplier(
                symbol, window_hours=2.0
            )
            features["event_is_active"]       = float(is_active)
            features["event_hours_until"]     = float(min(hours_until, 999.0))
            features["event_vol_expectation"] = float(vol_mult)
        except Exception:
            features["event_is_active"]       = 0.0
            features["event_hours_until"]     = 999.0
            features["event_vol_expectation"] = 1.0
        return features

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")
        if df.empty:
            raise ValueError("DataFrame is empty.")
