#!/usr/bin/env python3
"""Test calendar API endpoint"""

from api.main import create_app
from starlette.testclient import TestClient

app = create_app(api_key='')
client = TestClient(app)

resp = client.get('/calendar/events')
print(f'Status: {resp.status_code}')
print(f'Content-Type: {resp.headers.get("content-type")}')
print(f'Body:\n{resp.text}')
