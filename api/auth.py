"""
API Authentication
==================
API key security with:
  - Bearer token  (Authorization: Bearer <key>)
  - Header        (X-API-Key: <key>)
  - Query param   (fallback for simple dashboards)

Rate limiting per IP:  sliding window, configurable.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Optional

try:
    from fastapi import Depends, HTTPException, Security, status
    from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPAuthorizationCredentials, HTTPBearer
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

# ---------------------------------------------------------------------------
# Rate limiter (in-memory sliding window)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int = 120, window_s: int = 60) -> None:
        self._max   = max_requests
        self._win   = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock  = Lock()

    def check(self, ip: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[ip]
            # Drop expired
            self._hits[ip] = [t for t in hits if now - t < self._win]
            if len(self._hits[ip]) >= self._max:
                return False
            self._hits[ip].append(now)
            return True


_rate_limiter = _RateLimiter()

# ---------------------------------------------------------------------------
# Auth settings (set by create_app)
# ---------------------------------------------------------------------------

_API_KEY:       str  = ""
_RATE_LIMIT_ON: bool = True

def configure(api_key: str, rate_limit: bool = True) -> None:
    global _API_KEY, _RATE_LIMIT_ON
    _API_KEY       = api_key
    _RATE_LIMIT_ON = rate_limit


# ---------------------------------------------------------------------------
# FastAPI security schemes
# ---------------------------------------------------------------------------

if _FASTAPI_OK:
    _bearer_scheme  = HTTPBearer(auto_error=False)
    _header_scheme  = APIKeyHeader(name="X-API-Key",   auto_error=False)
    _query_scheme   = APIKeyQuery(name="api_key",       auto_error=False)

    async def require_api_key(
        bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
        header: Optional[str]                          = Security(_header_scheme),
        query:  Optional[str]                          = Security(_query_scheme),
    ) -> str:
        """
        Dependency: validates the API key from any of three sources.
        Raises HTTP 401 on failure.
        """
        if not _API_KEY:
            # Auth disabled — open access
            return "open"

        provided = None
        if bearer and bearer.credentials:
            provided = bearer.credentials
        elif header:
            provided = header
        elif query:
            provided = query

        if not provided or provided != _API_KEY:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail      = "Invalid or missing API key.",
                headers     = {"WWW-Authenticate": "Bearer"},
            )
        return provided

    async def check_rate_limit(
        request=None,
        _: str = Depends(require_api_key),
    ) -> None:
        """
        Combined auth + rate-limit dependency.
        Use this as the default dependency on all protected endpoints.
        """
        if _RATE_LIMIT_ON and request is not None:
            ip = getattr(getattr(request, "client", None), "host", "unknown")
            if not _rate_limiter.check(ip):
                raise HTTPException(
                    status_code = status.HTTP_429_TOO_MANY_REQUESTS,
                    detail      = "Rate limit exceeded.",
                )

    # Convenience alias
    Auth = Depends(require_api_key)

else:
    Auth = None  # type: ignore
