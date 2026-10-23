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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_float(value: any) -> Optional[float]:
    """Safely parse a value to float, handling None and non-numeric strings.

    Trading Economics returns numeric fields as strings with unit suffixes
    (e.g. "216K", "3.2%", "1.05M") — strip any trailing non-numeric
    characters rather than failing outright. Doesn't rescale by the unit
    (K/M/B), which is fine as long as actual/forecast/previous for the same
    event share the same unit (they do), since only their ratio is used.
    """
    if value is None or value == '':
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    import re
    match = re.match(r"^\s*(-?[\d.,]+)", str(value))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
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
        return self.get_upcoming_events(
            next_n_hours, self._symbol_countries(symbol), min_importance,
        )

    def _symbol_countries(self, symbol: str) -> list[str]:
        """Currencies of a trading pair: EURUSD → [EUR, USD], GBPJPY → [GBP, JPY]."""
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
        return countries

    def event_is_active(
        self,
        symbol: str,
        window_minutes: int = 30,
    ) -> bool:
        """True if a HIGH-importance event for this symbol's currencies is
        within ±window_minutes of now — before AND after the release
        (the upcoming-events helpers only see the future)."""
        countries = self._symbol_countries(symbol)
        return any(
            e.country in countries
            and e.importance.value >= EventImportance.HIGH.value
            and e.is_active(window_minutes)
            for e in self._fetch_events()
        )
    
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
        """Fetch upcoming US release dates from FRED (supplementary source).

        FRED itself is a time-series API, not an event calendar — but its
        release/dates REST endpoint does expose the *scheduled* future dates
        of major US data releases when queried with a future realtime_start,
        which is genuinely forward-looking (unlike the historical series
        data). Covers only a small curated list of high-impact US releases;
        use Trading Economics for broader / global coverage.

        Calls the raw REST API directly rather than going through the
        `fredapi` package — that package only wraps FRED's *series* data
        endpoints (get_series, get_series_info, ...), it has no method for
        the release-dates endpoint at all (confirmed against fredapi 0.5.2).

        Note: FRED release dates are date-only (no time-of-day), unlike
        Trading Economics — events are anchored at 00:00 UTC on the release
        date as an approximation.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; skipping FRED calendar source.")
            return []

        from config.settings import settings as cfg
        api_key = getattr(cfg.trading, "fred_api_key", "")
        if not api_key:
            logger.info("TRADING_FRED_API_KEY not set — skipping FRED calendar source.")
            return []

        now     = datetime.now(timezone.utc)
        horizon = now + timedelta(days=60)
        events: list[EconomicEvent] = []

        # Release IDs verified against GET /fred/releases (name match) — do
        # not guess these, FRED's IDs are not obvious from the release name.
        # release_id 101 ("FOMC Press Release") was tried and dropped: it
        # returns a release_date for EVERY single calendar day with no gaps
        # (confirmed by direct query), so it does not represent discrete
        # FOMC decision dates the way this curated list assumes — including
        # it would flag every day as a HIGH-impact event and, combined with
        # RiskEngine Layer 0, block trading almost permanently. Trading
        # Economics (the primary calendar source) covers rate decisions as
        # proper discrete events instead.
        releases = {
            50: ("Employment Situation (NFP)", EventImportance.HIGH),
            10: ("Consumer Price Index",        EventImportance.HIGH),
            53: ("Gross Domestic Product",      EventImportance.HIGH),
            46: ("Producer Price Index",        EventImportance.MEDIUM),
        }

        for release_id, (name, importance) in releases.items():
            try:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/release/dates",
                    params={
                        "release_id": release_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "realtime_start": now.strftime("%Y-%m-%d"),
                        "realtime_end": horizon.strftime("%Y-%m-%d"),
                        "include_release_dates_with_no_data": "true",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.debug(f"FRED release/dates({release_id}) failed: {e}")
                continue

            release_dates = data.get("release_dates", [])
            # Defensive sanity cap: a genuine monthly/quarterly release
            # yields at most ~2 dates in a 60-day window. More than that
            # means this release_id doesn't behave as a discrete calendar
            # event (as happened with 101 above) — skip it rather than
            # spam misleading "events" that would trip the event blackout.
            if len(release_dates) > 4:
                logger.warning(
                    f"FRED release_id={release_id} ('{name}') returned "
                    f"{len(release_dates)} dates in {(horizon - now).days}d — "
                    f"not a discrete event release, skipping."
                )
                continue

            for item in release_dates:
                try:
                    ts = datetime.strptime(item["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except (KeyError, ValueError):
                    continue
                if now < ts <= horizon:
                    events.append(EconomicEvent(
                        timestamp=ts, country="USD", name=name, importance=importance,
                    ))

        if events:
            logger.info(f"✓ Fetched {len(events)} scheduled release dates from FRED")
        return sorted(events, key=lambda e: e.timestamp)

    def _fetch_trading_economics(self) -> list[EconomicEvent]:
        """Fetch from the real Trading Economics Calendar API (JSON REST,
        not the HTML calendar page the previous implementation hit — that
        endpoint never returned parseable data and always fell back to mock).

        Docs: https://docs.tradingeconomics.com/calendar/
        A registered TRADING_CALENDAR_API_KEY is required for anything
        beyond the very limited 'guest:guest' demo tier. Falls back to mock
        data if the key is missing, the request fails, or the response
        shape doesn't match what's expected.
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; using mock calendar. Install via: pip install requests")
            return self._fetch_mock()

        from config.settings import settings as cfg
        api_key = getattr(cfg.trading, "calendar_api_key", "") or "guest:guest"

        now = datetime.now(timezone.utc)
        d1  = now.strftime("%Y-%m-%d")
        d2  = (now + timedelta(days=8)).strftime("%Y-%m-%d")

        # Try the two documented endpoint shapes — the calendar root defaults
        # to all countries on most accounts, but some keys require the
        # explicit /country/all path with a date range.
        endpoints = [
            ("https://api.tradingeconomics.com/calendar", {"c": api_key}),
            ("https://api.tradingeconomics.com/calendar/country/all", {"c": api_key, "d1": d1, "d2": d2}),
        ]

        for url, params in endpoints:
            try:
                logger.debug(f"Trying Trading Economics endpoint: {url}")
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logger.debug(f"TE endpoint failed ({url}): {e}")
                continue
            except ValueError as e:
                logger.debug(f"TE endpoint ({url}) returned non-JSON: {e}")
                continue

            if not isinstance(data, list):
                logger.debug(f"TE endpoint ({url}) returned unexpected shape: {type(data)}")
                continue

            events = []
            for item in data:
                try:
                    event = self._parse_te_event(item)
                    if event and event.timestamp > now:
                        events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing TE event: {e}")
                    continue

            if events:
                logger.info(f"✓ Fetched {len(events)} REAL events from Trading Economics")
                return sorted(events, key=lambda e: e.timestamp)

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
            
            # Map importance — the real TE API returns an int (0-3: 0/1=Low,
            # 2=Medium, 3=High), while string forms ('low'/'medium'/'high')
            # are kept for compatibility with other possible sources.
            raw_importance = item.get('Importance', EventImportance.MEDIUM.value)
            if isinstance(raw_importance, str):
                importance = {
                    'low': EventImportance.LOW,
                    'medium': EventImportance.MEDIUM,
                    'high': EventImportance.HIGH,
                }.get(raw_importance.lower(), EventImportance.MEDIUM)
            else:
                try:
                    imp_int = int(raw_importance)
                except (TypeError, ValueError):
                    imp_int = EventImportance.MEDIUM.value
                importance = {
                    0: EventImportance.LOW,
                    1: EventImportance.LOW,
                    2: EventImportance.MEDIUM,
                    3: EventImportance.HIGH,
                }.get(imp_int, EventImportance.MEDIUM)
            
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
