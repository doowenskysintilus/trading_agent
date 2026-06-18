# Phase 2A: Quick Start Guide - Real Calendar APIs

This guide walks you through implementing real economic calendar data integration (FRED & Trading Economics).

## Prerequisites

✓ Phase 1 & 3 complete (all tests passing)  
✓ CalendarAlpha auto-registered  
✓ .env configured with Phase 1 settings

## Step 1: Get API Credentials

### FRED (Federal Reserve Economic Data)
**Why:** Free, reliable US economic indicators (NFP, CPI, ISM, unemployment, etc.)

1. Go to: https://fred.stlouisfed.org/docs/api/
2. Click "Register for an API Key"
3. Complete registration
4. Copy API key to `.env`:
```bash
FRED_API_KEY=your_api_key_here
```

### Trading Economics (Optional but Recommended)
**Why:** Global calendar coverage (ECB, BoJ, BoE, RBA, etc.)

1. Go to: https://tradingeconomics.com/api/
2. Free tier available (limited requests)
3. Premium tier recommended for historical data
4. Copy API key to `.env`:
```bash
TRADING_ECONOMICS_API_KEY=your_api_key_here
```

## Step 2: Install Dependencies

```bash
# Install required packages
conda activate tradingAI
pip install fred-api tradingeconomics redis pydantic-settings

# Optional: Start Redis for caching
redis-server
# or if Redis not available, set REDIS_URL="" to skip caching
```

## Step 3: Modify calendar_provider.py

The `CalendarProvider` class needs two new data source classes.

### Location
`research/feature_store/calendar_provider.py`

### Add FRED Source

```python
class FREDCalendarSource(CalendarDataSource):
    """Fetch economic events from FRED API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.fred = Alfred(api_key=api_key)
        
        # Map FRED series to events
        self.event_series = {
            'NFP': 'PAYEMS',              # Nonfarm Payroll
            'Unemployment': 'UNRATE',     # Unemployment Rate
            'CPI': 'CPIAUCSL',            # Consumer Price Index
            'PPI': 'PPIACO',              # Producer Price Index
            'ISM': 'MMNRNJ',              # ISM Manufacturing
            'Housing Starts': 'HOUST',    # Housing Starts
        }
    
    def get_events(self, next_n_hours: int) -> list[EconomicEvent]:
        events = []
        
        for event_name, series_id in self.event_series.items():
            try:
                data = self.fred.get_series(series_id, limit=3)
                
                # Parse forecast vs actual
                latest = data[0]
                forecast = data[1] if len(data) > 1 else None
                
                event = EconomicEvent(
                    timestamp=datetime.now(timezone.utc) + timedelta(hours=12),
                    name=event_name,
                    country='USD',
                    importance=EventImportance.HIGH,
                    forecast=forecast,
                    actual=latest,
                    previous=data[2] if len(data) > 2 else None,
                )
                events.append(event)
                
            except Exception as e:
                logger.warning(f"FRED {event_name} fetch failed: {e}")
        
        return events
```

### Add Trading Economics Source

```python
class TradingEconomicsCalendarSource(CalendarDataSource):
    """Fetch global economic calendars from Trading Economics."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get_events(self, next_n_hours: int) -> list[EconomicEvent]:
        events = []
        
        try:
            # Query TE calendar API
            response = self.session.get(
                f"{self.base_url}/calendar",
                params={
                    'limit': 50,
                    'importance': ['high', 'medium'],
                }
            )
            response.raise_for_status()
            
            for event_data in response.json():
                # Parse TE event structure
                event = EconomicEvent(
                    timestamp=datetime.fromisoformat(event_data['date']),
                    name=event_data['title'],
                    country=event_data['country'],
                    importance=self._parse_importance(event_data.get('importance')),
                    forecast=self._parse_number(event_data.get('forecast')),
                    actual=self._parse_number(event_data.get('last')),
                    previous=self._parse_number(event_data.get('previous')),
                )
                events.append(event)
        
        except Exception as e:
            logger.warning(f"Trading Economics fetch failed: {e}")
        
        return events
    
    @staticmethod
    def _parse_importance(importance_str: str) -> EventImportance:
        mapping = {'high': EventImportance.HIGH, 'medium': EventImportance.MEDIUM}
        return mapping.get(importance_str, EventImportance.LOW)
    
    @staticmethod
    def _parse_number(value) -> float | None:
        if value is None or value == '':
            return None
        try:
            return float(value)
        except:
            return None
```

### Update CalendarProvider Constructor

```python
class CalendarProvider:
    def __init__(self, source: str = "auto"):
        self.source = source
        self._setup_sources()
    
    def _setup_sources(self):
        """Initialize configured data sources."""
        
        fred_key = os.getenv("FRED_API_KEY")
        te_key = os.getenv("TRADING_ECONOMICS_API_KEY")
        
        self._fred_source = None
        self._te_source = None
        self._mock_source = MockCalendarSource()
        self._cache = CalendarCache()
        
        if fred_key and self.source in ("fred", "auto"):
            try:
                self._fred_source = FREDCalendarSource(fred_key)
            except Exception as e:
                logger.warning(f"FRED initialization failed: {e}")
        
        if te_key and self.source in ("trading_economics", "auto"):
            try:
                self._te_source = TradingEconomicsCalendarSource(te_key)
            except Exception as e:
                logger.warning(f"TE initialization failed: {e}")
    
    def get_upcoming_events(self, next_n_hours: int, ...) -> list[EconomicEvent]:
        """Try real sources first, fallback to mock."""
        
        cache_key = f"events_{next_n_hours}_{datetime.now().hour}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        # Try FRED
        if self._fred_source and self.source in ("fred", "auto"):
            try:
                events = self._fred_source.get_events(next_n_hours)
                if events:
                    self._cache.set(cache_key, events, ttl=3600)
                    return events
            except Exception as e:
                logger.warning(f"FRED fetch failed: {e}")
        
        # Try Trading Economics
        if self._te_source and self.source in ("trading_economics", "auto"):
            try:
                events = self._te_source.get_events(next_n_hours)
                if events:
                    self._cache.set(cache_key, events, ttl=3600)
                    return events
            except Exception as e:
                logger.warning(f"TE fetch failed: {e}")
        
        # Fallback to mock
        logger.info("Falling back to mock calendar data")
        events = self._mock_source.get_events(next_n_hours)
        self._cache.set(cache_key, events, ttl=1800)
        return events
```

## Step 4: Create Calendar Cache

**File:** `research/feature_store/calendar_cache.py` (NEW)

```python
from typing import Optional
from datetime import datetime, timedelta
import redis
import logging

logger = logging.getLogger(__name__)


class CalendarCache:
    """In-memory + Redis cache for calendar data."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url
        self.memory_cache = {}
        self.ttl = {}
        
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache connected")
            else:
                self.redis_client = None
                logger.info("Using memory cache only (set REDIS_URL for persistent cache)")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[list]:
        """Get from Redis → memory."""
        
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(f"calendar:{key}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Try memory cache
        if key in self.memory_cache:
            if datetime.now() < self.ttl.get(key, datetime.now()):
                return self.memory_cache[key]
            else:
                del self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: list, ttl: int = 3600):
        """Set in Redis + memory."""
        
        # Store in memory
        self.memory_cache[key] = value
        self.ttl[key] = datetime.now() + timedelta(seconds=ttl)
        
        # Try Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"calendar:{key}",
                    ttl,
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
```

## Step 5: Configure .env

```bash
# Calendar configuration
TRADING_INCLUDE_CALENDAR_FEATURES=true
TRADING_CALENDAR_SOURCE=auto          # Try FRED → TE → mock

# API Keys
FRED_API_KEY=your_fred_api_key
TRADING_ECONOMICS_API_KEY=your_te_api_key

# Redis (optional, leave empty for memory-only cache)
REDIS_URL=redis://localhost:6379

# Risk engine (Phase 1)
RISK_EVENT_BLACKOUT_ENABLED=true
RISK_EVENT_BLACKOUT_HOURS=0.5
RISK_EVENT_VOL_MULTIPLIER=1.5
```

## Step 6: Run Validation Tests

```bash
# Test real calendar data retrieval
conda run -n tradingAI python _test_phase2a_real_calendar.py

# Expected output:
# ✓ FRED API returns 6+ event types
# ✓ Trading Economics returns 10+ global events
# ✓ Fallback chain working (API → cache → mock)
# ✓ Cache TTL respected
```

### Create Test File

**File:** `_test_phase2a_real_calendar.py`

```python
#!/usr/bin/env python3
"""Test Phase 2A: Real Calendar APIs"""

import sys
from datetime import datetime, timezone
from research.feature_store.calendar_provider import CalendarProvider, EventImportance

print("=" * 70)
print("TEST 1: FRED API Integration")
print("=" * 70)

try:
    provider = CalendarProvider(source="fred")
    events = provider.get_upcoming_events(next_n_hours=24, min_importance=EventImportance.HIGH)
    
    print(f"✓ FRED API returned {len(events)} events")
    
    for event in events[:3]:
        print(f"  - {event.name} ({event.country}): {event.forecast}")
    
except Exception as e:
    print(f"✗ FRED test failed: {e}")

print("\n" + "=" * 70)
print("TEST 2: Trading Economics Integration")
print("=" * 70)

try:
    provider = CalendarProvider(source="trading_economics")
    events = provider.get_upcoming_events(next_n_hours=48, min_importance=EventImportance.HIGH)
    
    print(f"✓ Trading Economics returned {len(events)} events")
    
    for event in events[:3]:
        print(f"  - {event.name} ({event.country}): {event.forecast}")
    
except Exception as e:
    print(f"✗ TE test failed: {e}")

print("\n" + "=" * 70)
print("TEST 3: Fallback Chain")
print("=" * 70)

try:
    provider = CalendarProvider(source="auto")
    events = provider.get_upcoming_events(next_n_hours=24)
    
    print(f"✓ Fallback chain returned {len(events)} events")
    print(f"  (Source: FRED or TE or mock)")
    
except Exception as e:
    print(f"✗ Fallback test failed: {e}")

print("\n" + "=" * 70)
print("PHASE 2A TESTS COMPLETE")
print("=" * 70)
```

## Step 7: Verify Integration

```bash
# Start the API
cd api
python main.py

# In another terminal, test CalendarAlpha with real data
curl -X POST http://localhost:8000/trading/start \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["EURUSD", "GBPUSD"],
    "strategy": "calendar_alpha",
    "allocation": 0.2
  }'

# Check strategy is using real calendar data
# Log output should show FRED/TE events being fetched
```

## Troubleshooting

### FRED API Key Not Working
- Verify key at https://fred.stlouisfed.org/docs/api/api_key.html
- Check rate limits (120 requests/minute)
- Try FRED_API_KEY in .env again

### Trading Economics Timeout
- Free tier has 100 requests/month
- Use premium tier for production
- Or fallback to FRED (always enabled)

### Redis Connection Failed
- Leave REDIS_URL empty in .env
- System will use in-memory cache
- Performance: ~50ms lookups instead of 5ms

### Still Getting Mock Data
- Check logs for source initialization
- Verify API keys are set in .env
- Run: `conda run -n tradingAI python -c "from research.feature_store.calendar_provider import CalendarProvider; p = CalendarProvider(); print(p._fred_source)"`

## Next: Phase 2B (RL Augmentation)

Once real calendar data is flowing:

```bash
# Collect new experiences with real calendar
TRADING_INCLUDE_CALENDAR_FEATURES=true \
TRADING_CALENDAR_SOURCE=auto \
python live_trading/live_trader.py --mode collect

# Retrain RL agent
python strategies/rl_agent/rl_trainer.py \
  --epochs 100 \
  --learning-rate 3e-4

# Backtest comparison
python backtesting/backtest_engine.py --model models/RecurrentPPO_calendar
```

---

**Ready to implement Phase 2A? Start with Step 1 (API credentials)!**
