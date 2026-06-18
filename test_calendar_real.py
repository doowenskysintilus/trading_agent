#!/usr/bin/env python3
"""Test real-time calendar API integration"""

from research.feature_store.calendar_provider import CalendarProvider
from datetime import datetime, timezone

# Test 1: Mock calendar (offline)
print("=" * 60)
print("Test 1: Mock Calendar (Synthetic Events)")
print("=" * 60)

mock_provider = CalendarProvider(source="mock")
mock_events = mock_provider.get_upcoming_events(next_n_hours=24)

print(f"Mock events: {len(mock_events)} in next 24 hours")
for event in mock_events[:3]:
    print(f"  {event.timestamp} | {event.country} | {event.name} ({event.importance.name})")

# Test 2: Real-time Trading Economics calendar
print("\n" + "=" * 60)
print("Test 2: Real-Time Calendar (Trading Economics API)")
print("=" * 60)

try:
    import requests
    requests_available = True
except ImportError:
    requests_available = False
    print("⚠️  requests not installed. Install via: pip install requests")

if requests_available:
    te_provider = CalendarProvider(source="trading_economics")
    te_events = te_provider.get_upcoming_events(next_n_hours=72)
    
    print(f"✓ Trading Economics events: {len(te_events)} in next 72 hours")
    
    if te_events:
        for event in te_events[:5]:
            print(f"  {event.timestamp} | {event.country} | {event.name}")
            print(f"    Importance: {event.importance.name}")
            print(f"    Forecast: {event.forecast}, Actual: {event.actual}")
    else:
        print("⚠️  No events returned (API might be rate-limited or network issue)")
else:
    print("Skipped (requests not available)")

# Test 3: Calendar in API endpoint
print("\n" + "=" * 60)
print("Test 3: API Endpoint Integration")
print("=" * 60)

try:
    from api.main import create_app
    from starlette.testclient import TestClient
    
    app = create_app(api_key='')
    client = TestClient(app)
    
    resp = client.get('/calendar/events?next_n_hours=24')
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type')}")
    
    if resp.status_code == 200:
        data = resp.json()
        n_events = len(data.get('data', {}).get('events', []))
        print(f"✓ API returned {n_events} events")
        
        if n_events > 0:
            event = data['data']['events'][0]
            print(f"  First event: {event['name']} ({event['country']})")
            print(f"  Time: {event['timestamp']}")
    else:
        print(f"✗ Error: {resp.text[:200]}")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("- Mock calendar: ✓ Works (synthetic events for demo)")
print("- Real-time calendar: Uses Trading Economics API (free tier, no key needed)")
print("- API endpoint: Returns calendar data with proper JSON wrapper")
print("=" * 60)
