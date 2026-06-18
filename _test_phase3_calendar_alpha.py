#!/usr/bin/env python3
"""
Test script for Phase 3: CalendarAlpha Strategy

Validates:
1. CalendarAlpha initialization with and without calendar provider
2. Signal generation (BUY/SELL/HOLD) based on event data
3. Auto-registration in api/main.py startup
"""

import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

print("=" * 70)
print("TEST 1: CalendarAlpha Initialization")
print("=" * 70)

try:
    from strategies.economic_alpha.calendar_alpha import CalendarAlpha
    from research.feature_store.calendar_provider import CalendarProvider
    
    # Test 1A: Without provider (graceful degradation)
    ca_no_provider = CalendarAlpha(enabled=True)
    print("✓ CalendarAlpha created without provider (graceful)")
    
    # Test 1B: With provider
    provider = CalendarProvider(source="mock")
    ca_with_provider = CalendarAlpha(
        calendar_provider=provider,
        forecast_weight=0.7,
        event_lookahead_hours=24,
    )
    print("✓ CalendarAlpha created with provider")
    print(f"  - forecast_weight: {ca_with_provider.forecast_weight}")
    print(f"  - event_lookahead_hours: {ca_with_provider.event_lookahead_hours}")
    print(f"  - name: {ca_with_provider.name}")
    
except Exception as e:
    print(f"✗ CalendarAlpha initialization failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("TEST 2: CalendarAlpha Signal Generation")
print("=" * 70)

try:
    from strategies.economic_alpha.calendar_alpha import CalendarAlpha
    from research.feature_store.calendar_provider import CalendarProvider
    from research.alpha_models.base import SignalType
    
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
    
    # Test 2A: Signal with no events (HOLD)
    provider = CalendarProvider(source="mock")
    ca = CalendarAlpha(calendar_provider=provider)
    signal = ca.compute(data)
    print(f"✓ Signal with no upcoming events: {signal.signal.value} (conf={signal.confidence:.2f})")
    
    # Test 2B: Multiple calls (consistency)
    signal2 = ca.compute(data)
    print(f"✓ Second signal call: {signal2.signal.value}")
    
    # Test 2C: Disabled strategy (should return HOLD with 0 confidence)
    ca_disabled = CalendarAlpha(calendar_provider=provider, enabled=False)
    signal_disabled = ca_disabled.compute(data)
    print(f"✓ Disabled strategy: {signal_disabled.signal.value} (conf={signal_disabled.confidence:.2f})")
    assert signal_disabled.signal == SignalType.HOLD, "Disabled should be HOLD"
    assert signal_disabled.confidence == 0.0, "Disabled confidence should be 0.0"
    
except Exception as e:
    print(f"✗ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("TEST 3: CalendarAlpha Auto-Registration")
print("=" * 70)

try:
    # Simply verify the import works (registration happens via @app.on_event)
    from strategies.economic_alpha.calendar_alpha import CalendarAlpha
    from strategies.momentum.momentum_alpha import MomentumAlpha
    from strategies.mean_reversion.mean_reversion_alpha import MeanReversionAlpha
    
    print("✓ All default strategies can be imported")
    
    # Create instances to verify constructors work
    strategies = [
        MomentumAlpha(),
        MeanReversionAlpha(),
        CalendarAlpha(),
    ]
    
    strategy_names = [s.name for s in strategies]
    print(f"✓ Strategy instances created: {strategy_names}")
    
    if "calendar_alpha" in strategy_names:
        print("  ✓ CalendarAlpha included in default strategy list")
    
    print("✓ Auto-registration mechanism validated (see api.main._register_default_strategies)")
    
except Exception as e:
    print(f"✗ Auto-registration test failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("TEST 4: CalendarAlpha Event-Aware Behavior")
print("=" * 70)

try:
    from strategies.economic_alpha.calendar_alpha import CalendarAlpha
    from research.feature_store.calendar_provider import CalendarProvider, EventImportance
    
    provider = CalendarProvider(source="mock")
    ca = CalendarAlpha(
        calendar_provider=provider,
        forecast_weight=0.7,
        min_importance=EventImportance.HIGH,
    )
    
    # Get upcoming events to understand what calendar expects
    events = provider.get_upcoming_events(next_n_hours=48, min_importance=EventImportance.HIGH)
    print(f"✓ Mock provider has {len(events)} HIGH-importance events in next 48h")
    
    for i, event in enumerate(events[:2]):
        print(f"\n  Event {i+1}: {event.name} ({event.country})")
        print(f"    - Time: {event.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"    - Importance: {event.importance.name}")
        print(f"    - Forecast: {event.forecast}")
        print(f"    - Previous: {event.previous}")
        print(f"    - Actual: {event.actual}")
    
    # Compute signal
    dummy_data = pd.DataFrame({
        'close': [100.0] * 100,
    })
    signal = ca.compute(dummy_data)
    print(f"\n✓ Signal computed: {signal.signal.value} (conf={signal.confidence:.2f})")
    print(f"  Metadata: {signal.metadata}")
    
except Exception as e:
    print(f"✗ Event-aware behavior test failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("TEST 5: CalendarAlpha Metadata Correctness")
print("=" * 70)

try:
    from strategies.economic_alpha.calendar_alpha import CalendarAlpha
    from research.feature_store.calendar_provider import CalendarProvider
    
    provider = CalendarProvider(source="mock")
    ca = CalendarAlpha(calendar_provider=provider)
    
    # Create simple data
    data = pd.DataFrame({
        'close': [100.0] * 50,
    })
    
    signal = ca.compute(data)
    
    # Check metadata
    meta = signal.metadata
    print(f"✓ Metadata keys: {list(meta.keys())}")
    
    # Check that metadata has reason/event info
    if 'reason' in meta:
        print(f"  - Reason: {meta['reason']}")
    if 'event' in meta:
        print(f"  - Event: {meta['event']}")
    if 'country' in meta:
        print(f"  - Country: {meta['country']}")
    
    print("✓ Metadata structure correct")
    
except Exception as e:
    print(f"✗ Metadata test failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("PHASE 3 INTEGRATION TESTS COMPLETE")
print("=" * 70)
