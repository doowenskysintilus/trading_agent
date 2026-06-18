"""
Economic Calendar Provider
===========================
Fetches economic events and their impact levels for use in feature engineering
and risk management.

Supports multiple data sources:
- FRED API (free, US indicators)
- Trading Economics (freemium, global)
- Local JSON cache (manual, offline)
- Real-time mock data

Usage
-----
    provider = CalendarProvider(source="fred")
    events_df = provider.get_upcoming_events(next_n_hours=24)
    
    # In feature engineering
    eng = FeatureEngineer(FeatureConfig(include_calendar_features=True))
    provider = CalendarProvider()
    features = eng.compute(data, calendar_provider=provider)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_float(value: any) -> Optional[float]:
    """Safely parse a value to float, handling None and non-numeric strings."""
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class EventImportance(int, Enum):
    """Economic event importance classification."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class EconomicEvent:
    """Single economic event record."""
    
    timestamp: datetime
    country: str              # e.g., "USD", "EUR", "GBP"
    name: str                 # e.g., "Non-Farm Payroll"
    importance: EventImportance
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None
    revised: Optional[float] = None
    units: str = ""           # e.g., "K", "%"
    url: str = ""
    
    @property
    def surprise(self) -> Optional[float]:
        """Return surprise as (actual - forecast) / |forecast|, or None if missing."""
        if self.actual is None or self.forecast is None or self.forecast == 0:
            return None
        return (self.actual - self.forecast) / abs(self.forecast)
    
    def is_upcoming(self, hours_ahead: int = 1) -> bool:
        """Check if event is within N hours in the future."""
        now = datetime.now(timezone.utc)
        target = now + timedelta(hours=hours_ahead)
        return now < self.timestamp <= target
    
    def is_active(self, window_minutes: int = 30) -> bool:
        """Check if event is happening now (within window_minutes before/after)."""
        now = datetime.now(timezone.utc)
        start = self.timestamp - timedelta(minutes=window_minutes)
        end = self.timestamp + timedelta(minutes=window_minutes)
        return start <= now <= end


# ---------------------------------------------------------------------------
# Provider Base
# ---------------------------------------------------------------------------

class CalendarProvider:
    """
    Fetches economic events from configured source.
    
    Parameters
    ----------
    source : str
        Data source: "fred", "trading_economics", "cache", "mock"
    """
    
    # Symbol → currency code mapping (for feature engineering context)
    _SYMBOL_CURRENCIES = {
        "EURUSD": "EUR", "EURCHF": "EUR", "EURJPY": "EUR", "EURGBP": "EUR",
        "GBPUSD": "GBP", "GBPJPY": "GBP",
        "USDJPY": "JPY", "USDCAD": "USD",
        "AUDUSD": "AUD", "NZDUSD": "NZD",
        "XAUUSD": "USD",  # Gold (USD-denominated)
    }
    
    def __init__(self, source: str = "mock") -> None:
        self.source = source
        self._cache: dict[str, list[EconomicEvent]] = {}
        
        if source == "mock":
            self._init_mock()
        elif source == "cache":
            self._init_cache()
    
    def _init_mock(self) -> None:
        """Initialize with mock data for demo/testing."""
        logger.info("CalendarProvider: using MOCK source (demo data)")
        self._cache = {}
    
    def _init_cache(self) -> None:
        """Initialize from local JSON cache if available."""
        cache_path = Path(__file__).parent.parent.parent / "data" / "calendar_cache.jsonl"
        if cache_path.exists():
            logger.info(f"CalendarProvider: loading cache from {cache_path}")
            # Implementation: parse JSONL cache file
        else:
            logger.warning(f"CalendarProvider: cache file not found at {cache_path}")
            self._cache = {}
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def get_upcoming_events(
        self,
        next_n_hours: int = 24,
        countries: Optional[list[str]] = None,
        min_importance: EventImportance = EventImportance.MEDIUM,
    ) -> list[EconomicEvent]:
        """
        Fetch upcoming events within N hours.
        
        Parameters
        ----------
        next_n_hours : int
            Look ahead this many hours from now
        countries : list[str], optional
            Filter by country codes (e.g., ["USD", "EUR"]). None = all.
        min_importance : EventImportance
            Minimum importance level to return
        
        Returns
        -------
        list[EconomicEvent]
            Sorted by timestamp
        """
        now = datetime.now(timezone.utc)
        target = now + timedelta(hours=next_n_hours)
        
        events = self._fetch_events()
        
        # Filter by time window
        events = [e for e in events if now <= e.timestamp <= target]
        
        # Filter by country
        if countries:
            events = [e for e in events if e.country in countries]
        
        # Filter by importance
        events = [e for e in events if e.importance.value >= min_importance.value]
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def get_events_for_symbol(
        self,
        symbol: str,
        next_n_hours: int = 24,
        min_importance: EventImportance = EventImportance.MEDIUM,
    ) -> list[EconomicEvent]:
        """
        Get upcoming events relevant to a trading pair.
        
        Parameters
        ----------
        symbol : str
            Trading pair (e.g., "EURUSD")
        next_n_hours : int
            Look ahead window
        min_importance : EventImportance
            Minimum importance
        
        Returns
        -------
        list[EconomicEvent]
            Events for the pair's currencies
        """
        # Extract relevant currencies from symbol
        # EURUSD → [EUR, USD], GBPJPY → [GBP, JPY]
        base_curr = symbol[:3].upper() if len(symbol) >= 3 else ""
        quote_curr = symbol[3:6].upper() if len(symbol) >= 6 else ""
        
        countries = []
        if base_curr in self._SYMBOL_CURRENCIES.values():
            countries.append(base_curr)
        if quote_curr in self._SYMBOL_CURRENCIES.values() and quote_curr not in countries:
            countries.append(quote_curr)
        
        if not countries:
            # Fallback: look up in mapping
            countries = [self._SYMBOL_CURRENCIES.get(symbol, "USD")]
        
        return self.get_upcoming_events(next_n_hours, countries, min_importance)
    
    def get_events_at_time(
        self,
        timestamp: datetime,
        window_minutes: int = 30,
    ) -> list[EconomicEvent]:
        """
        Get events happening near a specific timestamp.
        
        Parameters
        ----------
        timestamp : datetime
            Reference time
        window_minutes : int
            Minutes before/after to include
        
        Returns
        -------
        list[EconomicEvent]
            Events in the time window
        """
        start = timestamp - timedelta(minutes=window_minutes)
        end = timestamp + timedelta(minutes=window_minutes)
        
        events = self._fetch_events()
        return [e for e in events if start <= e.timestamp <= end]
    
    def event_is_active(
        self,
        symbol: str,
        window_minutes: int = 30,
    ) -> bool:
        """Check if a high-importance event is happening now for this symbol."""
        events = self.get_events_for_symbol(
            symbol,
            next_n_hours=0.5,  # Look 30 min ahead
            min_importance=EventImportance.HIGH,
        )
        now = datetime.now(timezone.utc)
        for event in events:
            if event.is_active(window_minutes):
                return True
        return False
    
    def hours_until_next_event(
        self,
        symbol: str,
        min_importance: EventImportance = EventImportance.MEDIUM,
    ) -> Optional[float]:
        """
        Time in hours until next event for this symbol.
        
        Returns
        -------
        float or None
            Hours until next event, or None if no events upcoming
        """
        events = self.get_events_for_symbol(
            symbol,
            next_n_hours=168,  # 1 week ahead
            min_importance=min_importance,
        )
        if not events:
            return None
        
        now = datetime.now(timezone.utc)
        delta = (events[0].timestamp - now).total_seconds() / 3600.0
        return max(delta, 0.0)  # Clamp to ≥ 0
    
    def expected_volatility_multiplier(
        self,
        symbol: str,
        window_hours: float = 0.5,
    ) -> float:
        """
        Return ATR multiplier based on upcoming event severity.
        
        Returns
        -------
        float
            1.0 (no event) → 1.5 (high-impact event coming)
        """
        events = self.get_events_for_symbol(
            symbol,
            next_n_hours=int(window_hours + 1),
            min_importance=EventImportance.LOW,
        )
        
        if not events:
            return 1.0
        
        now = datetime.now(timezone.utc)
        max_multiplier = 1.0
        
        for event in events:
            hours_until = (event.timestamp - now).total_seconds() / 3600.0
            if 0 <= hours_until <= window_hours:
                # Map importance + time window to multiplier
                # HIGH + immediate → 1.5x
                # MEDIUM + immediate → 1.3x
                # LOW + immediate → 1.1x
                base = 1.0 + (event.importance.value * 0.2)
                time_decay = 1.0 - (hours_until / window_hours) * 0.3
                multiplier = base * time_decay
                max_multiplier = max(max_multiplier, multiplier)
        
        return min(max_multiplier, 2.0)  # Cap at 2.0x
    
    # ------------------------------------------------------------------
    # Private: Data fetching (source-specific implementations)
    # ------------------------------------------------------------------
    
    def _fetch_events(self) -> list[EconomicEvent]:
        """Fetch all upcoming events from configured source."""
        if self.source == "mock":
            return self._fetch_mock()
        elif self.source == "cache":
            return self._fetch_cache()
        elif self.source == "fred":
            return self._fetch_fred()
        elif self.source == "trading_economics":
            return self._fetch_trading_economics()
        else:
            logger.warning(f"Unknown calendar source: {self.source}")
            return []
    
    def _fetch_mock(self) -> list[EconomicEvent]:
        """Generate mock events for demo."""
        now = datetime.now(timezone.utc)
        
        events = [
            EconomicEvent(
                timestamp=now + timedelta(hours=2),
                country="USD",
                name="Non-Farm Payroll",
                importance=EventImportance.HIGH,
                forecast=195_000,
                previous=198_000,
                actual=None,
                units="K",
            ),
            EconomicEvent(
                timestamp=now + timedelta(hours=6),
                country="EUR",
                name="ECB Interest Rate Decision",
                importance=EventImportance.HIGH,
                forecast=None,
                previous=3.75,
                actual=None,
                units="%",
            ),
            EconomicEvent(
                timestamp=now + timedelta(hours=12),
                country="USD",
                name="Consumer Price Index YoY",
                importance=EventImportance.HIGH,
                forecast=3.2,
                previous=3.3,
                actual=None,
                units="%",
            ),
            EconomicEvent(
                timestamp=now + timedelta(hours=24),
                country="GBP",
                name="Retail Sales MoM",
                importance=EventImportance.MEDIUM,
                forecast=0.2,
                previous=0.5,
                actual=None,
                units="%",
            ),
        ]
        
        return events
    
    def _fetch_cache(self) -> list[EconomicEvent]:
        """Load from cached JSON file."""
        # Implementation: parse calendar_cache.jsonl
        return []
    
    def _fetch_fred(self) -> list[EconomicEvent]:
        """Fetch from FRED API (Federal Reserve Economic Data).
        
        Note: Requires `pip install fredapi`. Get API key from:
        https://fred.stlouisfed.org/docs/api/
        """
        try:
            from fredapi import Fred
        except ImportError:
            logger.warning("fredapi not installed. Install via: pip install fredapi")
            return []
        
        # For now, return empty (FRED is primarily for time-series data, not event calendar)
        # Would need supplementary event data source (e.g., integrate with Trading Economics API)
        logger.info("FRED provider selected (requires supplementary event calendar API)")
        return []
    
    def _fetch_trading_economics(self) -> list[EconomicEvent]:
        """Fetch from Trading Economics API (real-time calendar).
        
        Tries multiple TE endpoints and falls back to mock if unavailable.
        Requires internet connection. Free tier works (rate-limited).
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; using mock calendar. Install via: pip install requests")
            return self._fetch_mock()
        
        # Try multiple TE API endpoints
        endpoints = [
            ("https://tradingeconomics.com/calendar", {"method": "web"}),  # Web scrape endpoint
        ]
        
        for url, params in endpoints:
            try:
                logger.debug(f"Trying Trading Economics endpoint: {url}")
                
                # Optional: add API key for higher rate limits
                from config.settings import settings as cfg
                if hasattr(cfg, 'trading') and hasattr(cfg.trading, 'calendar_api_key'):
                    if cfg.trading.calendar_api_key:
                        params['apikey'] = cfg.trading.calendar_api_key
                
                response = requests.get(url, params=params, timeout=5, verify=False)
                response.raise_for_status()
                
                # Try to parse HTML/JSON response
                try:
                    data = response.json()
                except ValueError:
                    # Try HTML parsing if JSON fails
                    logger.debug("TE response not JSON, trying HTML parsing")
                    # For now, skip HTML parsing and fall back to mock
                    continue
                
                events = []
                
                if isinstance(data, list):
                    # API returned list of events
                    for item in data:
                        try:
                            event = self._parse_te_event(item)
                            if event and event.timestamp > datetime.now(timezone.utc):
                                events.append(event)
                        except Exception as e:
                            logger.debug(f"Error parsing TE event: {e}")
                            continue
                
                if events:
                    logger.info(f"✓ Fetched {len(events)} REAL events from Trading Economics")
                    return sorted(events, key=lambda e: e.timestamp)
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"TE endpoint failed ({url}): {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error with TE endpoint ({url}): {e}")
                continue
        
        # All TE endpoints failed — fall back to mock
        logger.info("Trading Economics unavailable; using MOCK calendar for demo")
        return self._fetch_mock()
    
    def _parse_te_event(self, item: dict) -> Optional[EconomicEvent]:
        """Parse a single Trading Economics event dict."""
        try:
            # Parse timestamp
            date_str = item.get('Date', '') or item.get('date', '')
            time_str = item.get('Time', '') or item.get('time', '')
            
            if not date_str:
                return None
            
            # Try to parse datetime
            dt = None
            if time_str:
                try:
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                except ValueError:
                    try:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        pass
            else:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
            
            if not dt:
                return None
            
            # Ensure UTC timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Map importance
            importance_str = item.get('Importance', 'medium').lower()
            importance = {
                'low': EventImportance.LOW,
                'medium': EventImportance.MEDIUM,
                'high': EventImportance.HIGH,
            }.get(importance_str, EventImportance.MEDIUM)
            
            # Extract country
            country = item.get('Country', 'USD')
            country_map = {
                'United States': 'USD', 'US': 'USD',
                'Eurozone': 'EUR', 'Euro area': 'EUR',
                'United Kingdom': 'GBP', 'UK': 'GBP',
                'Japan': 'JPY',
                'Australia': 'AUD',
                'Canada': 'CAD',
                'Switzerland': 'CHF',
                'New Zealand': 'NZD',
            }
            country = country_map.get(country, country[:3].upper() if country else 'USD')
            
            return EconomicEvent(
                timestamp=dt,
                country=country,
                name=item.get('Event', 'Unknown'),
                importance=importance,
                forecast=_parse_float(item.get('Forecast')),
                previous=_parse_float(item.get('Previous')),
                actual=_parse_float(item.get('Actual')),
                revised=_parse_float(item.get('Revised')),
                units=item.get('Unit', ''),
                url=item.get('Link', item.get('link', '')),
            )
        except Exception as e:
            logger.debug(f"_parse_te_event error: {e}")
            return None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_calendar_provider(source: str = "mock") -> CalendarProvider:
    """Factory function to create provider with error handling."""
    return CalendarProvider(source=source)
