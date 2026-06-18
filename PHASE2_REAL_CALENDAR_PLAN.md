# Phase 2: Real Calendar Data Integration

## Overview

Phase 2 replaces the mock calendar provider with real economic data sources and adds backtesting/historical analysis capabilities.

**Timeline:** Phase 2A (API integration) → Phase 2B (RL augmentation) → Phase 2C (Dashboard)

---

## Phase 2A: Real Calendar APIs

### Objective

Integrate real economic event data from two sources:

1. **FRED (Federal Reserve Economic Data)**
   - US economic indicators (NFP, CPI, ISM, etc.)
   - Historical actual/forecast data
   - Free, reliable, requires API key

2. **Trading Economics**
   - Global calendars (ECB, BoJ, BoE, etc.)
   - Multiple countries coverage
   - Premium tier for historical forecast accuracy

### Implementation Plan

#### Step 1: FRED Data Source
**File:** `research/feature_store/calendar_provider.py`

```python
class FREDCalendarSource(CalendarDataSource):
    """Fetch economic events from FRED API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
    
    def get_events(self, next_n_hours: int) -> list[EconomicEvent]:
        """
        Query FRED for upcoming releases:
        - Employment (NFP, Unemployment Rate)
        - Inflation (CPI, PPI)
        - Manufacturing (ISM, Factory Orders)
        - Housing (Starts, Building Permits)
        - Sentiment (Consumer Confidence, PMI)
        """
        pass
```

**Setup:**
```bash
# .env
FRED_API_KEY=your_key_here
TRADING_CALENDAR_SOURCE=fred
```

#### Step 2: Trading Economics Data Source
**File:** `research/feature_store/calendar_provider.py`

```python
class TradingEconomicsCalendarSource(CalendarDataSource):
    """Fetch global economic calendars from Trading Economics."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
    
    def get_events(self, next_n_hours: int) -> list[EconomicEvent]:
        """
        Query for global events:
        - Countries: US, EUR, UK, JPY, CAD, AUD, CHF, etc.
        - All HIGH/MEDIUM importance releases
        """
        pass
```

#### Step 3: Caching Layer
**File:** `research/feature_store/calendar_cache.py` (NEW)

```python
class CalendarCache:
    """In-memory cache with Redis fallback."""
    
    def __init__(self, redis_url: str = None):
        self.redis = redis_url
        self.memory_cache = {}
        self.ttl = 3600  # 1 hour
    
    def get(self, key: str) -> list[EconomicEvent] | None:
        """Try Redis first, then memory."""
        pass
    
    def set(self, key: str, value: list[EconomicEvent], ttl: int = 3600):
        """Store in Redis + memory."""
        pass
```

#### Step 4: Fallback Strategy
**Update:** `research/feature_store/calendar_provider.py`

```python
class CalendarProvider:
    """Try real API → Cache → Mock."""
    
    def __init__(self, source: str = "auto"):
        # source: "fred", "trading_economics", "auto", "mock"
        self.source = source
        
    def get_upcoming_events(self, next_n_hours: int) -> list[EconomicEvent]:
        try:
            # Try primary source
            if self.source in ("fred", "auto"):
                return self._fred_source.get_events(next_n_hours)
        except Exception as e:
            logger.warning(f"FRED failed: {e}")
        
        try:
            # Try secondary source
            if self.source in ("trading_economics", "auto"):
                return self._te_source.get_events(next_n_hours)
        except Exception as e:
            logger.warning(f"TE failed: {e}")
        
        # Fallback to cache or mock
        return self._mock_source.get_events(next_n_hours)
```

### Environment Setup

```bash
# .env
TRADING_CALENDAR_SOURCE=auto  # Try real APIs, fallback to mock
FRED_API_KEY=your_fred_key
TRADING_ECONOMICS_API_KEY=your_te_key
REDIS_URL=redis://localhost:6379  # Optional caching
```

### Validation Tests

**Create:** `_test_phase2a_real_calendar.py`

```python
def test_fred_nfp():
    """Verify FRED returns upcoming NFP with forecast."""
    provider = CalendarProvider(source="fred")
    events = provider.get_upcoming_events(next_n_hours=24)
    nfp = [e for e in events if "payroll" in e.name.lower()]
    assert len(nfp) > 0, "No NFP found"
    assert nfp[0].forecast is not None, "NFP forecast missing"

def test_te_ecb():
    """Verify TE returns ECB event."""
    provider = CalendarProvider(source="trading_economics")
    events = provider.get_upcoming_events(next_n_hours=48, countries=["EUR"])
    ecb = [e for e in events if "ECB" in e.name]
    assert len(ecb) > 0, "No ECB found"

def test_fallback_chain():
    """Verify fallback: real API → cache → mock."""
    provider = CalendarProvider(source="auto")
    events = provider.get_upcoming_events(next_n_hours=24)
    assert len(events) >= 3, "Should have fallback events"
```

---

## Phase 2B: RL Environment Augmentation

### Objective

Teach RL agents (PPO, LSTM) to learn event-aware trading strategies.

### Implementation Plan

#### Step 1: Augment Observation Space
**File:** `env/trading_env.py`

```python
class TradingEnv:
    def __init__(self, ..., include_calendar_features=True):
        self.include_calendar_features = include_calendar_features
    
    def _get_observation(self):
        """
        Base observation:
        - OHLCV (5 cols)
        - Technical indicators (20 cols)
        - Risk metrics (3 cols)
        
        New calendar columns (if enabled):
        - event_is_active (1 col): 0/1 binary
        - event_hours_until (1 col): 0-999
        - event_vol_expectation (1 col): 1.0-1.5
        
        → Total: 30 cols (was 28)
        """
        obs = self._build_base_observation()
        
        if self.include_calendar_features and self.calendar_provider:
            event_is_active = self._check_event_active()
            hours_until = self._get_hours_until_next()
            vol_mult = self._get_vol_expectation()
            
            obs = np.concatenate([
                obs,
                [[event_is_active], [hours_until], [vol_mult]]
            ])
        
        return obs
```

#### Step 2: Retrain RL Agents
**Process:**

1. **Collect new experiences** with calendar features enabled
   ```bash
   TRADING_INCLUDE_CALENDAR_FEATURES=true
   python live_trading/live_trader.py --mode collect
   # Generates: data/storage/datasets/experiences_*.jsonl
   ```

2. **Train new RL checkpoint**
   ```bash
   python strategies/rl_agent/rl_trainer.py \
     --dataset data/storage/datasets/experiences_*.jsonl \
     --output models/RecurrentPPO_calendar \
     --epochs 100
   # Output: models/RecurrentPPO_calendar/checkpoint.pt
   ```

3. **Backtest both versions**
   ```bash
   python backtesting/backtest_engine.py \
     --model models/RecurrentPPO_1 --calendar=false \
     --output backtest_no_calendar.json
   
   python backtesting/backtest_engine.py \
     --model models/RecurrentPPO_calendar --calendar=true \
     --output backtest_with_calendar.json
   ```

4. **Compare metrics**
   ```
   No Calendar:
   - Sharpe: 1.2
   - Max DD: -18%
   - Ret/Mo: +1.8%
   
   With Calendar:
   - Sharpe: 1.6 (+33%)
   - Max DD: -12% (-33%)
   - Ret/Mo: +2.1%
   ```

#### Step 3: Feature Importance Analysis
**New file:** `research/rl_analysis/calendar_impact.py`

```python
def analyze_calendar_impact():
    """
    Analyze RL learned weights for calendar features.
    
    Output: Which features most influence RL decisions?
    - event_is_active: Typical weight?
    - event_hours_until: Attenuation curve?
    - event_vol_expectation: Size multiplier?
    """
    pass
```

### Validation Tests

**Create:** `_test_phase2b_rl_calendar.py`

```python
def test_env_observation_shape():
    """Observation should have 3 more cols with calendar."""
    env_no_cal = TradingEnv(include_calendar_features=False)
    env_with_cal = TradingEnv(include_calendar_features=True)
    
    obs_no = env_no_cal.reset()
    obs_with = env_with_cal.reset()
    
    assert obs_with.shape[1] == obs_no.shape[1] + 3

def test_rl_can_learn_event_avoidance():
    """Train RL agent → verify it learns to reduce position size near events."""
    agent = RLAgent(model="RecurrentPPO_calendar")
    # If calendar feature weight > 0, agent is responding to events
    assert agent.policy.get_feature_importance("event_vol_expectation") > 0.1
```

---

## Phase 2C: Dashboard Integration

### Objective

Visualize economic calendar impact in real-time dashboard.

### Components

#### 1. Event Countdown Widget
**Location:** Dashboard sidebar

```jsx
<EventCountdown>
  ├─ Next 5 events (next 24h)
  │  ├─ Time to event (HH:MM)
  │  ├─ Event name
  │  ├─ Importance (color: green/orange/red)
  │  └─ Forecast vs Previous
  └─ Position sizing indicator during event window
```

#### 2. Calendar-Segmented Metrics
**Location:** PnL and accuracy tabs

```
PnL Breakdown:
├─ During Events (red zone)
│  - Avg trades/day: 0.2
│  - Win rate: 42%
│  - Sharpe: 0.8
├─ Pre-Event (yellow zone)
│  - Avg trades/day: 0.5
│  - Win rate: 48%
│  - Sharpe: 1.1
└─ Quiet (green zone)
   - Avg trades/day: 2.1
   - Win rate: 56%
   - Sharpe: 1.8
```

#### 3. Event PnL Heatmap
**New chart type**

```
Y-axis: Event type (NFP, CPI, ECB, etc.)
X-axis: Hours from event (-24 to +24)
Values: Cumulative PnL per cell

→ Visual: Where do we make/lose money relative to events?
```

### Implementation

**Files to modify:**
- `dashboard/src/components/EventCountdown.jsx` (NEW)
- `dashboard/src/components/MetricsBreakdown.jsx` (modified)
- `api/monitoring/metrics_store.py` (add calendar metrics)

---

## Phase 2 Timeline & Dependencies

### Critical Path

```
2A (Real APIs): 1-2 weeks
  ├─ FRED integration
  ├─ TE integration
  └─ Fallback + caching

2B (RL Retrain): 2-3 weeks
  ├─ Collect experiences with calendar
  ├─ Retrain PPO/LSTM models
  └─ Backtest comparison

2C (Dashboard): 1 week
  ├─ Event countdown widget
  ├─ Metrics segmentation
  └─ Deploy

Total: 4-6 weeks
```

### Blockers & Risks

| Risk | Mitigation |
|------|-----------|
| FRED API quota exhaustion | Use cache, batch requests |
| TE API pricing | Use free tier, fallback to FRED |
| RL retraining convergence | Monitor loss curves, adjust LR |
| Historical forecast accuracy | Validate FRED/TE accuracy vs realized |

---

## Phase 2 Success Criteria

- [ ] FRED API returning 10+ event types correctly
- [ ] TE API returning non-US calendars correctly
- [ ] Fallback chain working (API → cache → mock)
- [ ] RL retraining with calendar features converges
- [ ] Backtest comparison shows 20%+ DD reduction
- [ ] Dashboard displaying event countdown + metrics breakdown
- [ ] Zero missing event alerts in live trading

---

## Quick Start (Pseudo-code)

```python
# Phase 2A: Real calendar
from research.feature_store.calendar_provider import CalendarProvider

provider = CalendarProvider(source="auto")  # Try FRED → TE → mock
events = provider.get_upcoming_events(next_n_hours=24)

for event in events:
    print(f"{event.name}: Forecast {event.forecast}, Actual {event.actual}")

# Phase 2B: RL augmentation
from env.trading_env import TradingEnv

env = TradingEnv(include_calendar_features=True)
obs = env.reset()
print(obs.shape)  # Should be (batch, 31) = 28 base + 3 calendar

# Phase 2C: Dashboard
GET /dashboard → EventCountdown widget shows NFP in 2h
```

---

## Files to Create/Modify

| Phase | File | Action | Complexity |
|-------|------|--------|-----------|
| 2A | `research/feature_store/calendar_provider.py` | Modify | High |
| 2A | `research/feature_store/calendar_cache.py` | Create | Medium |
| 2A | `_test_phase2a_real_calendar.py` | Create | Medium |
| 2B | `env/trading_env.py` | Modify | High |
| 2B | `strategies/rl_agent/rl_trainer.py` | Modify | High |
| 2B | `_test_phase2b_rl_calendar.py` | Create | Medium |
| 2C | `dashboard/src/components/EventCountdown.jsx` | Create | Medium |
| 2C | `api/monitoring/metrics_store.py` | Modify | Low |

---

**Next:** Phase 2A - Real Calendar API Integration

Generated: Phase 2 Planning Document  
Status: Ready for development
