"""
WebSocket Connection Manager
=============================
Manages all active dashboard WebSocket connections.
Provides both async broadcast (from async routes) and
thread-safe broadcast (from the sync trading loop via
asyncio.run_coroutine_threadsafe).

Usage inside FastAPI routes
---------------------------
  from api.ws import manager
  await manager.broadcast("trade", payload_dict)

Usage from sync trading thread
-------------------------------
  from api.ws import broadcast_sync
  broadcast_sync("trade", {"symbol": "EURUSD", ...})
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Set

logger = logging.getLogger(__name__)

try:
    from fastapi import WebSocket
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False


class ConnectionManager:
    """Thread-safe WebSocket connection manager."""

    def __init__(self) -> None:
        self._active: Set["WebSocket"] = set()
        self._lock    = asyncio.Lock()
        self._loop:  Optional[asyncio.AbstractEventLoop] = None  # set on first WS connect

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def connect(self, ws: "WebSocket") -> None:
        await ws.accept()
        # Capture the running event loop once so sync threads can use it
        self._loop = asyncio.get_running_loop()
        async with self._lock:
            self._active.add(ws)
        logger.info("WS client connected (%d total)", len(self._active))

    async def disconnect(self, ws: "WebSocket") -> None:
        async with self._lock:
            self._active.discard(ws)
        logger.info("WS client disconnected (%d remaining)", len(self._active))

    # ------------------------------------------------------------------ #
    # Sending                                                              #
    # ------------------------------------------------------------------ #

    async def broadcast(self, msg_type: str, payload: dict) -> None:
        """Broadcast to all connected clients (async context only)."""
        if not self._active:
            return
        text = _encode(msg_type, payload)
        dead: list["WebSocket"] = []
        for ws in list(self._active):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._active.discard(ws)

    async def send_to(self, ws: "WebSocket", msg_type: str, payload: dict) -> None:
        """Send to a single client; removes it if the send fails."""
        try:
            await ws.send_text(_encode(msg_type, payload))
        except Exception:
            await self.disconnect(ws)

    def broadcast_sync(self, msg_type: str, payload: dict) -> None:
        """
        Thread-safe broadcast from a non-async context (e.g. the trading loop).
        No-op if no event loop is available yet (no dashboard connected).
        """
        if self._loop is None or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self.broadcast(msg_type, payload), self._loop
        )

    @property
    def count(self) -> int:
        return len(self._active)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

manager = ConnectionManager()


def broadcast_sync(msg_type: str, payload: dict) -> None:
    """Module-level convenience wrapper for broadcast_sync."""
    manager.broadcast_sync(msg_type, payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(msg_type: str, payload: dict) -> str:
    return json.dumps({
        "type":    msg_type,
        "payload": payload,
        "ts":      datetime.now(timezone.utc).isoformat(),
    }, default=str)
