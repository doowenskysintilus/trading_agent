"""
CommissionModel
===============
Commission per symbol, as a fraction of notional per side, measured on the
account's own closed trades (e.g. FTMO charges ~0.033 %/side on BTCUSD but
~0.0026 %/side on EURUSD — one flat rate was wrong for every symbol).

The measured rates come from an injected provider (MT5 history in live
trading) and are refreshed periodically; symbols without enough history
use the configured default (SCALP_COMMISSION_PCT).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CommissionModel:
    def __init__(
        self,
        default_rate: float,
        provider: Optional[Callable[[], dict[str, dict]]] = None,
        refresh_s: float = 3600.0,
    ) -> None:
        self._default = default_rate
        self._provider = provider
        self._refresh_s = refresh_s
        self._rates: dict[str, float] = {}
        self._details: dict[str, dict] = {}
        self._loaded_at: Optional[float] = None
        self._lock = threading.Lock()

    def rate(self, symbol: str) -> float:
        """Commission per side, fraction of notional."""
        self._maybe_refresh()
        with self._lock:
            return self._rates.get(symbol.upper(), self._default)

    def snapshot(self) -> dict:
        self._maybe_refresh()
        with self._lock:
            return {"default_rate_per_side": self._default, "measured": dict(self._details)}

    def refresh(self) -> None:
        if self._provider is None:
            return
        try:
            measured = self._provider() or {}
        except Exception as exc:          # history unavailable: keep last rates
            logger.warning("Commission rates refresh failed: %s", exc)
            with self._lock:
                self._loaded_at = time.monotonic()
            return
        with self._lock:
            self._details = {s.upper(): d for s, d in measured.items()}
            self._rates = {s: d["rate_per_side"] for s, d in self._details.items()}
            self._loaded_at = time.monotonic()
        if measured:
            logger.info("Commission rates (per side): %s",
                        {s: f"{r:.6f}" for s, r in self._rates.items()})

    def _maybe_refresh(self) -> None:
        if self._provider is None:
            return
        stale = self._loaded_at is None or time.monotonic() - self._loaded_at >= self._refresh_s
        if stale:
            self.refresh()
