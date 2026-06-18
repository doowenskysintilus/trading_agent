#!/usr/bin/env python3
"""
Test script for Phase 1: Economic Calendar Integration

Validates:
1. CalendarProvider initialization and mock data
2. FeatureEngineer with calendar features enabled
3. RiskEngine Layer 0 (blackout) + Layer 4 (event-aware sizing)
"""

import sys
from datetime import datetime, timezone, timedelta

# Test 1: CalendarProvider
print("=" * 70)
print("TEST 1: CalendarProvider")
print("=" * 70)

try:
    from research.feature_store.calendar_provider import (
        CalendarProvider, EventImportance, EconomicEvent
    )
    
    provider = CalendarProvider(source="mock")
    print("✓ CalendarProvider initialized with mock source")
    
    # Get upcoming events
    events = provider.get_upcoming_events(next_n_hours=48, min_importance=EventImportance.MEDIUM)
    print(f"✓ Found {len(events)} mock events in next 48h")
    for e in events[:3]:
        print(f"  - {e.timestamp.strftime('%Y-%m-%d %H:%M')} {e.country} {e.name} (importance={e.importance.name})")
    
    # Test event-specific methods
    is_active = provider.event_is_active("EURUSD", window_minutes=30)
    print(f"✓ Event active for EURUSD: {is_active}")
    
    hours_until = provider.hours_until_next_event("EURUSD")
    print(f"✓ Hours until next USD event: {hours_until}")
    
    vol_mult = provider.expected_volatility_multiplier("EURUSD", window_hours=2.0)
    print(f"✓ Expected vol multiplier (EURUSD): {vol_mult:.2f}x")
    
except Exception as e:
    print(f"✗ CalendarProvider test failed: {e}")
    import traceback
    traceback.print_exc()


# Test 2: FeatureEngineer with calendar features
print("\n" + "=" * 70)
print("TEST 2: FeatureEngineer with Calendar Features")
print("=" * 70)

try:
    import pandas as pd
    import numpy as np
    from research.feature_store.feature_engineer import FeatureEngineer, FeatureConfig
    
    # Create dummy OHLCV data
    n_bars = 100
    dates = pd.date_range('2026-06-01', periods=n_bars, freq='h')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'volume': 1000 * np.random.rand(n_bars),
    }, index=dates)
    
    # Test WITHOUT calendar features
    cfg_no_cal = FeatureConfig(include_calendar_features=False)
    eng_no_cal = FeatureEngineer(cfg_no_cal)
    features_no_cal = eng_no_cal.compute(data, symbol="EURUSD")
    print(f"✓ Features without calendar: {len(features_no_cal.columns)} columns")
    
    # Test WITH calendar features
    cfg_with_cal = FeatureConfig(include_calendar_features=True, calendar_source="mock")
    eng_with_cal = FeatureEngineer(cfg_with_cal)
    features_with_cal = eng_with_cal.compute(data, symbol="EURUSD")
    print(f"✓ Features with calendar: {len(features_with_cal.columns)} columns")
    
    # Check for calendar columns
    expected_cols = ['event_is_active', 'event_hours_until', 'event_vol_expectation']
    for col in expected_cols:
        if col in features_with_cal.columns:
            print(f"  ✓ {col}: {features_with_cal[col].iloc[0]}")
        else:
            print(f"  ✗ {col} NOT FOUND")
    
except Exception as e:
    print(f"✗ FeatureEngineer test failed: {e}")
    import traceback
    traceback.print_exc()


# Test 3: RiskEngine Layer 0 (blackout)
print("\n" + "=" * 70)
print("TEST 3: RiskEngine Layer 0 (Event Blackout)")
print("=" * 70)

try:
    from engines.risk_engine.risk_engine import (
        RiskEngine, RiskConfig, PortfolioState, TradeOrder, OpenPosition
    )
    
    # Create risk engine with calendar
    cfg = RiskConfig(event_blackout_enabled=True, event_blackout_hours=0.5)
    engine = RiskEngine(config=cfg, calendar_provider=provider)
    print("✓ RiskEngine initialized with calendar provider")
    
    # Create a trade for a symbol with an upcoming event
    trade = TradeOrder(
        symbol="EURUSD",
        strategy="test",
        direction=1,
        size=1.0,
        entry_price=1.0850,
        atr=0.0005,
        stop_loss=None,
        take_profit=None,
    )
    
    # Create portfolio state
    ps = PortfolioState(
        equity=100_000,
        balance=100_000,
        peak_equity=100_000,
        daily_start_equity=100_000,
        open_positions=[],
        consecutive_losses=0,
    )
    
    # Check event blackout
    result = engine._check_event_blackout(trade)
    print(f"✓ Event blackout check: passed={result.passed}, reason={result.reason}")
    
except Exception as e:
    print(f"✗ RiskEngine Layer 0 test failed: {e}")
    import traceback
    traceback.print_exc()


# Test 4: RiskEngine Layer 4 (event-aware sizing)
print("\n" + "=" * 70)
print("TEST 4: RiskEngine Layer 4 (Event-Aware Volatility Sizing)")
print("=" * 70)

try:
    # Using same engine from Test 3
    vol_check = engine._check_volatility_size(trade, ps, 1.0)
    print(f"✓ Volatility sizing check: passed={vol_check.passed}")
    if vol_check.suggested_size:
        print(f"  Suggested size: {vol_check.suggested_size:.6f}")
    print(f"  Message: {vol_check.message}")
    
except Exception as e:
    print(f"✗ RiskEngine Layer 4 test failed: {e}")
    import traceback
    traceback.print_exc()


# Test 5: Settings integration
print("\n" + "=" * 70)
print("TEST 5: Settings Integration (config/settings.py)")
print("=" * 70)

try:
    from config.settings import settings
    
    print(f"✓ TradingSettings.include_calendar_features: {settings.trading.include_calendar_features}")
    print(f"✓ TradingSettings.calendar_source: {settings.trading.calendar_source}")
    print(f"✓ RiskEngineSettings.event_blackout_enabled: {settings.risk_engine.event_blackout_enabled}")
    print(f"✓ RiskEngineSettings.event_blackout_hours: {settings.risk_engine.event_blackout_hours}")
    print(f"✓ RiskEngineSettings.event_vol_multiplier: {settings.risk_engine.event_vol_multiplier}")
    
except Exception as e:
    print(f"✗ Settings test failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("PHASE 1 INTEGRATION TESTS COMPLETE")
print("=" * 70)
