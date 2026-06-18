# Economic Calendar Integration - Quick Start Guide

## Activation (3 steps)

### Step 1: Enable Features in .env
Add or modify these lines in your `.env` file:

```bash
# === Economic Calendar Features (Phase 1) ===

# Enable calendar features in feature engineering
TRADING_INCLUDE_CALENDAR_FEATURES=true

# Choose calendar data source: mock | cache | fred | trading_economics
TRADING_CALENDAR_SOURCE=mock

# Enable event-aware risk management
RISK_EVENT_BLACKOUT_ENABLED=true

# How long to block trades before/after major events (hours)
RISK_EVENT_BLACKOUT_HOURS=0.5

# Scale ATR when events expected (e.g., 1.5 = 50% larger stop-loss)
RISK_EVENT_VOL_MULTIPLIER=1.5
```

### Step 2: Restart API Server
The configuration is loaded at startup:
```bash
# Terminal 1: Kill current server
# Then restart:
conda run -n tradingAI python -m api.main
```

### Step 3: Verify via Dashboard
- Start trading from the dashboard
- In the browser console, check that WebSocket receives feature data:
  ```javascript
  // Feature matrix should now include:
  // event_is_active, event_hours_until, event_vol_expectation
  ```

---

## Configuration Presets

### Conservative (Recommended for Live Trading)
```bash
TRADING_INCLUDE_CALENDAR_FEATURES=true
TRADING_CALENDAR_SOURCE=mock
RISK_EVENT_BLACKOUT_ENABLED=true
RISK_EVENT_BLACKOUT_HOURS=1.0      # 1 hour blackout
RISK_EVENT_VOL_MULTIPLIER=2.0      # 2x ATR during events
```

**Effect:** Strong event protection, fewer false signals due to event noise

### Moderate (Development/Testing)
```bash
TRADING_INCLUDE_CALENDAR_FEATURES=true
TRADING_CALENDAR_SOURCE=mock
RISK_EVENT_BLACKOUT_ENABLED=true
RISK_EVENT_BLACKOUT_HOURS=0.5      # 30 min blackout
RISK_EVENT_VOL_MULTIPLIER=1.5      # 1.5x ATR
```

**Effect:** Balanced risk/opportunity. Let some events pass, scale positions.

### Aggressive (Test Regime Robustness)
```bash
TRADING_INCLUDE_CALENDAR_FEATURES=true
TRADING_CALENDAR_SOURCE=mock
RISK_EVENT_BLACKOUT_ENABLED=false  # Events don't block
RISK_EVENT_VOL_MULTIPLIER=1.0      # Normal sizing (no vol scaling)
```

**Effect:** See raw event performance, identify alpha opportunities

### Features Only (ML Training)
```bash
TRADING_INCLUDE_CALENDAR_FEATURES=true
TRADING_CALENDAR_SOURCE=mock
RISK_EVENT_BLACKOUT_ENABLED=false  # No risk layer changes
RISK_EVENT_VOL_MULTIPLIER=1.0      # Normal ATR
```

**Effect:** Augmented features for RL/ML training, normal risk engine. Train on event-rich data.

---

## Verify It's Working

### 1. Check Settings Loaded
```python
python -c "from config.settings import settings; print(f'Calendar features: {settings.trading.include_calendar_features}'); print(f'Blackout hours: {settings.risk_engine.event_blackout_hours}')"
```

### 2. Run Integration Tests
```bash
conda run -n tradingAI python _test_phase1_calendar.py
```

Expected output: `✓ ALL TESTS PASS`

### 3. Check Log Output During Trading
Watch the console for:
```
[2026-06-18 14:30:00] INFO - EVENT BLACKOUT: 1 trade(s) blocked due to nearby economic events
[2026-06-18 14:31:00] INFO - Layer 4 vol check: atr_mult=2.30 (event-aware)
```

---

## Data Sources (Advanced)

### Mock Source (Default)
- **Status:** ✓ Ready
- **Setup:** None required
- **Use case:** Demo, development, testing
- **Events:** 4 synthetic events in next 48h (USD, EUR, GBP)

```python
from research.feature_store.calendar_provider import CalendarProvider
provider = CalendarProvider(source="mock")
events = provider.get_upcoming_events(next_n_hours=24)
for e in events:
    print(f"{e.timestamp} {e.country} {e.name}")
```

### Local Cache (Future)
- **Status:** Stub (TODO)
- **Setup:** Populate `data/calendar_cache.jsonl` manually
- **Format:** One JSON event per line

### FRED API (Future)
- **Status:** Stub (requires API key)
- **Setup:**
  ```bash
  pip install fredapi
  export FRED_API_KEY=your_key_here
  ```
- **Coverage:** US economic indicators

### Trading Economics API (Future)
- **Status:** Stub (requires API key)
- **Setup:**
  ```bash
  pip install tradingeconomics
  export TRADING_ECONOMICS_KEY=your_key_here
  ```
- **Coverage:** 200+ countries, real-time, high-impact events

---

## Monitoring Event Performance

### Check Event PnL in API
```bash
# During live trading, query event stats:
curl -H "X-API-Key: your_key" \
  http://localhost:8000/portfolio/metrics?period=event

# Returns:
# {
#   "pnl_event_period": 145.23,
#   "pnl_quiet_period": 89.50,
#   "accuracy_event": 0.62,
#   "accuracy_quiet": 0.68
# }
```

### Manual Inspection
```python
import pandas as pd
from live_trading.experience_collector import ExperienceCollector

exp = ExperienceCollector()
df = exp.load_experiences("data/storage/datasets")

# See trades around events
df['happened_during_event'] = df['confidence'] > 0.5  # Proxy
print(df[['symbol', 'pnl', 'happened_during_event']].groupby('happened_during_event').agg({'pnl': ['mean', 'count']}))
```

---

## Troubleshooting

### Calendar features not appearing in data
1. Check `.env` setting: `TRADING_INCLUDE_CALENDAR_FEATURES=true`
2. Restart API server (settings loaded at startup)
3. Run test: `python _test_phase1_calendar.py`

### Event blackout blocking legitimate trades
- Increase `RISK_EVENT_BLACKOUT_HOURS` (default 0.5) if too aggressive
- Or disable with `RISK_EVENT_BLACKOUT_ENABLED=false` for testing

### Position sizes suddenly much smaller
- This is **expected** if an event is upcoming (Layer 4 scaling)
- Check logs: `atr_mult=X.XX (event-aware)`
- Adjust `RISK_EVENT_VOL_MULTIPLIER` if too conservative

### API won't start
- Check imports: `from research.feature_store.calendar_provider import CalendarProvider`
- Verify conda env: `conda activate tradingAI && python -c "import pandas; import numpy"`

---

## Next: Phase 2 (Calendar Alpha Strategy)

After Phase 1 stabilizes, Phase 2 will add:

1. **CalendarAlpha strategy** — dedicated alpha model that:
   - Buys when consensus is beaten (actual > forecast)
   - Sells on misses (actual < forecast)
   - Auto-registers at startup

2. **RL augmentation** — expand RL environment observations with:
   - `event_is_active`, `event_hours_until`, `event_vol_expectation`
   - Retrain agent on event-rich data
   - Learn event-specific behavior

3. **Dashboard upgrades** — visualize:
   - Countdown to next 5 events
   - Separate PnL curves (event vs. quiet periods)
   - Event impact heatmaps

---

**Last Updated:** 2026-06-18  
**Status:** Phase 1 Complete ✅  
**Next:** Await Phase 2 implementation
