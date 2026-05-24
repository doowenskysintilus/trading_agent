"""
DataFeed
========
MT5 market data fetcher for the live trading system.

Provides two modes:
  - Historical bulk pull  (backfill / warmup)
  - Tick / rate streaming  (continuous live feed via background thread)

Handles:
  - Connection retries
  - Symbol info validation
  - Timeframe normalization
  - On-bar event callbacks (called when a new candle closes)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MT5 optional import
# ---------------------------------------------------------------------------

try:
    import MetaTrader5 as mt5
    _MT5_OK = True
except ImportError:
    _MT5_OK = False
    mt5 = None  # type: ignore

# ---------------------------------------------------------------------------
# Timeframe constants
# ---------------------------------------------------------------------------

TF_SECONDS: dict[str, int] = {
    "M1":  60,
    "M5":  300,
    "M15": 900,
    "M30": 1800,
    "H1":  3600,
    "H4":  14400,
    "D1":  86400,
    "W1":  604800,
}

def _tf_to_mt5(tf: str) -> int:
    if not _MT5_OK:
        return 0
    mapping = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }
    val = mapping.get(tf.upper())
    if val is None:
        raise ValueError(f"Unknown timeframe: {tf!r}. Valid: {list(mapping)}")
    return val


# ---------------------------------------------------------------------------
# Bar / Tick dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OHLCVBar:
    timestamp:  datetime
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    spread:     float = 0.0

    def to_dict(self) -> dict:
        return {
            "time":   self.timestamp,
            "open":   self.open,
            "high":   self.high,
            "low":    self.low,
            "close":  self.close,
            "volume": self.volume,
            "spread": self.spread,
        }


@dataclass
class TickData:
    timestamp: datetime
    bid:       float
    ask:       float
    last:      float
    volume:    float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

OnBarCallback  = Callable[[str, OHLCVBar, pd.DataFrame], None]
OnTickCallback = Callable[[str, TickData], None]


# ---------------------------------------------------------------------------
# DataFeed configuration
# ---------------------------------------------------------------------------

@dataclass
class DataFeedConfig:
    login:    int  = 0
    password: str  = ""
    server:   str  = ""
    path:     str  = ""

    # Fetch
    default_bars: int   = 200
    retry_count:  int   = 3
    retry_delay:  float = 2.0

    # Streaming
    poll_interval_ms: int = 200   # tick poll frequency (milliseconds)
    bar_lookahead_s:  int = 5     # seconds before bar close to trigger on_bar


# ---------------------------------------------------------------------------
# Main DataFeed class
# ---------------------------------------------------------------------------

class DataFeed:
    """
    MT5 market data feed.

    Usage
    -----
    >>> feed = DataFeed(config)
    >>> feed.connect()
    >>> df = feed.fetch_ohlcv("EURUSD", "H1", n_bars=200)
    >>> feed.subscribe("EURUSD", "H1", on_bar=my_callback)
    >>> feed.start_streaming()
    """

    def __init__(self, config: DataFeedConfig | None = None) -> None:
        self.cfg            = config or DataFeedConfig()
        self._connected     = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event    = threading.Event()

        # symbol → (timeframe, on_bar, on_tick, last_bar_time)
        self._subscriptions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to MT5 terminal.
        Returns True on success, False if MT5 is not available.
        """
        if not _MT5_OK:
            logger.warning("MetaTrader5 package not installed — DataFeed in simulation mode.")
            return False

        kwargs: dict = {}
        if self.cfg.path:
            kwargs["path"] = self.cfg.path
        if self.cfg.login:
            kwargs.update(
                login    = self.cfg.login,
                password = self.cfg.password,
                server   = self.cfg.server,
            )

        for attempt in range(1, self.cfg.retry_count + 1):
            if mt5.initialize(**kwargs):
                info = mt5.terminal_info()
                logger.info(
                    "MT5 connected — build=%s company=%s",
                    getattr(info, "build", "?"),
                    getattr(info, "company", "?"),
                )
                self._connected = True
                return True
            logger.warning(
                "MT5 connect attempt %d/%d failed: %s",
                attempt, self.cfg.retry_count, mt5.last_error(),
            )
            time.sleep(self.cfg.retry_delay)

        logger.error("MT5 connection failed after %d attempts.", self.cfg.retry_count)
        return False

    def disconnect(self) -> None:
        self.stop_streaming()
        if _MT5_OK and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def __enter__(self) -> "DataFeed":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Historical fetch
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:     str,
        timeframe:  str,
        n_bars:     int  | None = None,
        start:      datetime | None = None,
        end:        datetime | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars from MT5.

        Parameters
        ----------
        symbol    : trading symbol (e.g. "EURUSD")
        timeframe : "M1", "M5", "M15", "H1", "H4", "D1", etc.
        n_bars    : number of most recent bars (mutually exclusive with start/end)
        start     : UTC start datetime
        end       : UTC end datetime

        Returns
        -------
        pd.DataFrame with columns: open, high, low, close, volume, spread
        Index: UTC datetime
        """
        if not _MT5_OK:
            logger.debug("MT5 unavailable — returning None from fetch_ohlcv")
            return None

        tf_mt5 = _tf_to_mt5(timeframe)
        n      = n_bars or self.cfg.default_bars

        for attempt in range(1, self.cfg.retry_count + 1):
            try:
                if start and end:
                    rates = mt5.copy_rates_range(symbol, tf_mt5, start, end)
                elif start:
                    rates = mt5.copy_rates_from(symbol, tf_mt5, start, n)
                else:
                    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n)

                if rates is not None and len(rates) > 0:
                    return self._rates_to_df(rates)

                err = mt5.last_error()
                logger.warning(
                    "[%s] Fetch attempt %d/%d: %s",
                    symbol, attempt, self.cfg.retry_count, err,
                )
            except Exception as exc:
                logger.error("[%s] fetch_ohlcv error: %s", symbol, exc)

            time.sleep(self.cfg.retry_delay)

        logger.error("[%s] fetch_ohlcv failed after %d retries.", symbol, self.cfg.retry_count)
        return None

    def fetch_tick(self, symbol: str) -> Optional[TickData]:
        """Fetch the latest tick for a symbol."""
        if not _MT5_OK or not self._connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return TickData(
            timestamp = datetime.fromtimestamp(tick.time, tz=timezone.utc),
            bid       = float(tick.bid),
            ask       = float(tick.ask),
            last      = float(tick.last),
            volume    = float(tick.volume),
        )

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Return basic symbol properties (point, digits, spread, etc.)."""
        if not _MT5_OK:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "symbol":      info.name,
            "digits":      info.digits,
            "point":       info.point,
            "spread":      info.spread,
            "trade_mode":  info.trade_mode,
            "volume_min":  info.volume_min,
            "volume_max":  info.volume_max,
            "volume_step": info.volume_step,
            "currency_base":   info.currency_base,
            "currency_profit": info.currency_profit,
        }

    def validate_symbol(self, symbol: str) -> bool:
        """Return True if the symbol is tradeable."""
        info = self.get_symbol_info(symbol)
        return info is not None and info.get("trade_mode", 0) != 0

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def subscribe(
        self,
        symbol:    str,
        timeframe: str,
        on_bar:    Optional[OnBarCallback]  = None,
        on_tick:   Optional[OnTickCallback] = None,
    ) -> "DataFeed":
        """
        Subscribe to bar/tick events for a symbol.

        on_bar(symbol, bar, full_df)   — called when a new candle closes
        on_tick(symbol, tick)          — called on every new tick
        """
        self._subscriptions[symbol] = {
            "timeframe":     timeframe,
            "on_bar":        on_bar,
            "on_tick":       on_tick,
            "last_bar_time": None,
            "df_cache":      None,
        }
        logger.info("Subscribed %s @ %s", symbol, timeframe)
        return self

    def start_streaming(self) -> threading.Thread:
        """Start the background polling thread."""
        if self._stream_thread and self._stream_thread.is_alive():
            logger.warning("Streaming already running.")
            return self._stream_thread

        self._stop_event.clear()
        t = threading.Thread(
            target = self._stream_loop,
            daemon = True,
            name   = "DataFeed-stream",
        )
        t.start()
        self._stream_thread = t
        logger.info("DataFeed streaming started — %d subscriptions", len(self._subscriptions))
        return t

    def stop_streaming(self) -> None:
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
        logger.info("DataFeed streaming stopped.")

    def _stream_loop(self) -> None:
        interval = self.cfg.poll_interval_ms / 1000.0
        while not self._stop_event.is_set():
            for symbol, sub in self._subscriptions.items():
                try:
                    self._process_subscription(symbol, sub)
                except Exception as exc:
                    logger.error("[%s] stream error: %s", symbol, exc)
            time.sleep(interval)

    def _process_subscription(self, symbol: str, sub: dict) -> None:
        tf = sub["timeframe"]

        # Tick callback
        if sub["on_tick"]:
            tick = self.fetch_tick(symbol)
            if tick:
                sub["on_tick"](symbol, tick)

        # Bar close callback
        if sub["on_bar"]:
            df = self.fetch_ohlcv(symbol, tf, n_bars=2)
            if df is None or len(df) < 1:
                return

            last_bar_time = df.index[-1]
            if sub["last_bar_time"] is None:
                sub["last_bar_time"] = last_bar_time
                return

            if last_bar_time > sub["last_bar_time"]:
                # A new bar has closed
                sub["last_bar_time"] = last_bar_time
                # Fetch full history for the callback
                full_df = self.fetch_ohlcv(symbol, tf, n_bars=200)
                if full_df is not None:
                    sub["df_cache"] = full_df
                    closed_bar = OHLCVBar(
                        timestamp = full_df.index[-1].to_pydatetime(),
                        open      = float(full_df["open"].iloc[-1]),
                        high      = float(full_df["high"].iloc[-1]),
                        low       = float(full_df["low"].iloc[-1]),
                        close     = float(full_df["close"].iloc[-1]),
                        volume    = float(full_df["volume"].iloc[-1]),
                        spread    = float(full_df.get("spread", pd.Series([0])).iloc[-1]),
                    )
                    sub["on_bar"](symbol, closed_bar, full_df)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rates_to_df(rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.columns = [c.lower() for c in df.columns]
        rename_map = {"tick_volume": "volume", "real_volume": "real_volume"}
        df = df.rename(columns=rename_map)
        if "volume" not in df.columns and "real_volume" in df.columns:
            df["volume"] = df["real_volume"]
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
        return df[["open", "high", "low", "close", "volume", "spread"]
                  if "spread" in df.columns
                  else ["open", "high", "low", "close", "volume"]]
