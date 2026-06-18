"""
CalendarAlpha
=============
Economic calendar-aware strategy that trades around economic releases.

Strategy logic
--------------
1. Identify upcoming HIGH-impact economic events (NFP, ECB rate, CPI, etc.)
2. HOLD during event windows (volatility too high, timing uncertain)
3. BUY when consensus is beaten (actual > forecast)
4. SELL when consensus is missed (actual < forecast)
5. Adjust confidence based on surprise magnitude

This strategy exploits post-event reversals and mean-reversion patterns
after data-driven price shock.

Configuration
--------------
- forecast_weight: how much to trust forecasts vs. historical surprises (0-1)
- event_lookahead_hours: how far ahead to plan (typically 24h)
- min_event_importance: only trade HIGH-importance events (LOW/MEDIUM/HIGH)
- post_event_bars: how many bars to wait after event before trading (default: none)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from research.alpha_models.base import AlphaModel, AlphaSignal, SignalType

try:
    from research.feature_store.calendar_provider import (
        CalendarProvider, EventImportance
    )
    _CALENDAR_AVAILABLE = True
except ImportError:
    _CALENDAR_AVAILABLE = False
    CalendarProvider = None
    EventImportance = None

logger = logging.getLogger(__name__)


class CalendarAlpha(AlphaModel):
    """
    Economic calendar-aware alpha strategy.

    Parameters
    ----------
    name : str
        Strategy identifier (default "calendar_alpha")
    calendar_provider : CalendarProvider, optional
        Provider for economic event data. If None, will try to instantiate
        with default source ("mock").
    forecast_weight : float
        How much to trust forecasts vs. historical patterns (0.0-1.0).
        Higher = rely more on forecast hits/misses. (default: 0.7)
    event_lookahead_hours : int
        How far ahead to look for events (default: 24)
    min_importance : EventImportance, optional
        Minimum importance level for events to trade (default: HIGH)
    post_event_wait_minutes : int
        Minutes to wait after event before entering (default: 0)
    enabled : bool
        Whether strategy is enabled (default: True)
    """

    def __init__(
        self,
        name: str = "calendar_alpha",
        calendar_provider: Optional[CalendarProvider] = None,
        forecast_weight: float = 0.7,
        event_lookahead_hours: int = 24,
        min_importance: Optional[EventImportance] = None,
        post_event_wait_minutes: int = 0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        
        if not _CALENDAR_AVAILABLE:
            logger.warning("CalendarAlpha: calendar_provider not available")
        
        self.calendar_provider = calendar_provider
        if calendar_provider is None and _CALENDAR_AVAILABLE:
            try:
                self.calendar_provider = CalendarProvider(source="mock")
            except Exception as e:
                logger.warning(f"CalendarAlpha: failed to create default provider: {e}")
        
        self.forecast_weight = float(np.clip(forecast_weight, 0.0, 1.0))
        self.event_lookahead_hours = int(event_lookahead_hours)
        self.min_importance = min_importance or (
            EventImportance.HIGH if _CALENDAR_AVAILABLE else None
        )
        self.post_event_wait_minutes = int(post_event_wait_minutes)
        
        self._last_event_symbol = None
        self._last_event_time = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        """
        Generate signal based on economic calendar events and features.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV + features (including calendar features if available).
            Expected columns: close, event_is_active, event_hours_until, etc.

        Returns
        -------
        AlphaSignal
            BUY/SELL/HOLD with confidence and metadata
        """
        if not _CALENDAR_AVAILABLE or self.calendar_provider is None:
            return self._hold(reason="calendar_unavailable")
        
        if len(data) < 2:
            return self._hold(reason="insufficient_data", rows=len(data))
        
        try:
            # Extract symbol from data if available
            symbol = self._infer_symbol(data)
            
            # Get the current event status for this symbol
            upcoming_events = self.calendar_provider.get_events_for_symbol(
                symbol or "EURUSD",
                next_n_hours=self.event_lookahead_hours,
                min_importance=self.min_importance,
            )
            
            if not upcoming_events:
                return self._hold(reason="no_upcoming_events")
            
            next_event = upcoming_events[0]
            
            # ---- Decision logic ----
            
            # 1. Check if we're in the event window (hold or post-event reversal)
            is_active = next_event.is_active(window_minutes=30)
            hours_until = (next_event.timestamp - datetime.now(timezone.utc)).total_seconds() / 3600.0
            
            # If event is happening NOW → HOLD
            if is_active:
                return self._hold(
                    reason="event_active_now",
                    event=next_event.name,
                    country=next_event.country,
                )
            
            # If event is coming very soon (within 1 hour) → HOLD
            if 0 < hours_until <= 1.0:
                return self._hold(
                    reason="event_upcoming_soon",
                    event=next_event.name,
                    hours_until=round(hours_until, 2),
                )
            
            # 2. If event has a surprise (actual published) → post-event reversal
            if next_event.actual is not None and next_event.forecast is not None:
                surprise = next_event.surprise
                
                if surprise is None:
                    return self._hold(reason="surprise_calculation_failed")
                
                # Normalise surprise to [-1, +1] range
                norm_surprise = float(np.clip(surprise / (abs(next_event.forecast) + 1e-10), -1.0, 1.0))
                
                # Confidence based on surprise magnitude
                confidence = abs(norm_surprise) * self.forecast_weight + 0.3 * (1 - self.forecast_weight)
                confidence = float(np.clip(confidence, 0.0, 1.0))
                
                meta = {
                    "event": next_event.name,
                    "country": next_event.country,
                    "forecast": next_event.forecast,
                    "actual": next_event.actual,
                    "surprise": round(norm_surprise, 3),
                    "confidence_basis": "surprise_magnitude",
                }
                
                # Post-event reversal: if beat expectations, revert upward; if miss, revert downward
                if norm_surprise > 0.1:
                    # Beat consensus → BUY (expect continuation/reversal bounce)
                    return AlphaSignal(
                        signal=SignalType.BUY,
                        confidence=confidence,
                        strategy_name=self._name,
                        metadata=meta,
                    )
                elif norm_surprise < -0.1:
                    # Missed consensus → SELL (expect downside reversal)
                    return AlphaSignal(
                        signal=SignalType.SELL,
                        confidence=confidence,
                        strategy_name=self._name,
                        metadata=meta,
                    )
                else:
                    # Close to forecast → neutral
                    return self._hold(
                        reason="surprise_close_to_forecast",
                        event=next_event.name,
                        surprise=round(norm_surprise, 3),
                    )
            
            # 3. If event is far in future → no signal (pre-event preparation)
            if hours_until > 12.0:
                return self._hold(
                    reason="event_too_far_ahead",
                    event=next_event.name,
                    hours_until=round(hours_until, 2),
                )
            
            # 4. Event coming within 1-12 hours: prep phase (low confidence)
            # Weak signal based on historical forecast accuracy
            historical_beats = self._estimate_beat_probability(next_event)
            
            if historical_beats > 0.55:
                confidence = 0.3  # Weak pre-event signal
                return AlphaSignal(
                    signal=SignalType.BUY,
                    confidence=confidence,
                    strategy_name=self._name,
                    metadata={
                        "event": next_event.name,
                        "country": next_event.country,
                        "hours_until": round(hours_until, 2),
                        "forecast": next_event.forecast,
                        "historical_beat_prob": round(historical_beats, 2),
                        "confidence_basis": "pre_event_historical",
                    },
                )
            
            return self._hold(
                reason="event_prep_no_signal",
                event=next_event.name,
                hours_until=round(hours_until, 2),
            )
        
        except Exception as e:
            logger.warning(f"CalendarAlpha.compute failed: {e}")
            return self._hold(reason="compute_error", error=str(e))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_symbol(self, data: pd.DataFrame) -> Optional[str]:
        """Try to extract symbol from data index or columns."""
        # Fallback: check if there's a symbol column
        if "symbol" in data.columns:
            return str(data["symbol"].iloc[0])
        # Could also check data.index.name or attrs
        return None

    def _estimate_beat_probability(self, event) -> float:
        """
        Estimate P(consensus will be beaten) based on historical event patterns.
        
        For MVP, use a simple heuristic:
        - If event is typically bullish-biased → higher beat probability
        - Currently just returns 0.5 (neutral) as baseline
        
        Future: integrate with historical surprise database
        """
        # Placeholder: all events are 50/50 in the MVP
        return 0.5

    def _hold(self, **meta) -> AlphaSignal:
        """Return HOLD signal with optional metadata."""
        return AlphaSignal(
            signal=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self._name,
            metadata=meta,
        )


# Backward-compatible helper
def _get_base() -> type:
    """Return the base class for import compatibility."""
    return AlphaModel
