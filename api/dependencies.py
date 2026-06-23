"""
Application State & Dependency Injection
=========================================
Holds live references to all subsystems and provides
FastAPI Depends() factories for each router.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — none of these crash startup if missing
# ---------------------------------------------------------------------------

def _try_import(module: str, cls: str):
    try:
        import importlib
        mod = importlib.import_module(module)
        return getattr(mod, cls)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest job tracking
# ---------------------------------------------------------------------------

@dataclass
class BacktestJob:
    job_id:       str
    status:       str                = "queued"   # queued | running | done | failed
    request:      Any                = None
    result:       Optional[Any]      = None
    error:        Optional[str]      = None
    submitted_at: datetime           = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at:   Optional[datetime] = None
    completed_at: Optional[datetime] = None
    _thread:      Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_status_dict(self) -> dict:
        return {
            "job_id":       self.job_id,
            "status":       self.status,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at":   self.started_at.isoformat()   if self.started_at   else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_s":   self.duration_s,
            "error":        self.error,
        }


# ---------------------------------------------------------------------------
# Central app state
# ---------------------------------------------------------------------------

class AppState:
    """
    Singleton that owns references to all subsystems.

    Systems are injected after process startup via `attach_*` methods
    so that the app can serve health checks even before trading starts.
    """

    def __init__(self) -> None:
        self._lock            = threading.Lock()

        # Core subsystems
        self.trader           = None   # LiveTrader
        self.monitor          = None   # TradingMonitor
        self.execution_engine = None   # MT5ExecutionEngine
        self.risk_engine      = None   # RiskEngine
        self.portfolio_engine = None   # PortfolioAllocator
        self.backtest_engine  = None   # BacktestEngine
        self.feature_store    = None   # FeatureStore
        self.data_feed        = None   # DataFeed
        self.retrain_service  = None   # RetrainService (learning models)

        # Strategy registry  { name: AlphaModel }
        self.strategies:      dict[str, Any] = {}

        # Backtest jobs  { job_id: BacktestJob }
        self.backtest_jobs:   dict[str, BacktestJob] = {}

        # Risk config overrides applied via API
        self.risk_overrides:  dict[str, Any] = {}

        # Track API start time
        self.started_at       = datetime.now(timezone.utc)

    # ---- Attach subsystems ------------------------------------------------

    def attach_trader(self, trader) -> None:
        self.trader = trader
        logger.info("AppState: LiveTrader attached")

    def attach_monitor(self, monitor) -> None:
        self.monitor = monitor
        logger.info("AppState: TradingMonitor attached")

    def attach_engines(
        self,
        execution=None,
        risk=None,
        portfolio=None,
        backtest=None,
        feature_store=None,
        data_feed=None,
    ) -> None:
        if execution:    self.execution_engine = execution
        if risk:         self.risk_engine      = risk
        if portfolio:    self.portfolio_engine = portfolio
        if backtest:     self.backtest_engine  = backtest
        if feature_store: self.feature_store   = feature_store
        if data_feed:    self.data_feed        = data_feed
        logger.info("AppState: engines attached")

    def register_strategy(self, name: str, model) -> None:
        self.strategies[name] = model
        logger.info("AppState: strategy %s registered", name)

    # ---- Backtest job lifecycle -------------------------------------------

    def create_job(self, request) -> BacktestJob:
        job = BacktestJob(job_id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self.backtest_jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[BacktestJob]:
        return self.backtest_jobs.get(job_id)

    # ---- Convenience status ----------------------------------------------

    def is_trader_running(self) -> bool:
        return bool(self.trader and getattr(self.trader, "is_running", False))

    def is_emergency_active(self) -> bool:
        if self.trader and hasattr(self.trader, "emergency"):
            return self.trader.emergency.is_active
        return False


# Module-level singleton
_app_state = AppState()


def get_app_state() -> AppState:
    return _app_state


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

try:
    from fastapi import Depends, HTTPException
    _FASTAPI_DEPENDENCIES_OK = True
except ImportError:
    _FASTAPI_DEPENDENCIES_OK = False

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    def Depends(value):  # type: ignore[override]
        return value


def _state() -> AppState:
    return _app_state


def require_trader(state: AppState = Depends(_state)) -> Any:
    if state.trader is None:
        raise HTTPException(503, "LiveTrader not initialised.")
    return state.trader


def require_monitor(state: AppState = Depends(_state)) -> Any:
    if state.monitor is None:
        raise HTTPException(503, "TradingMonitor not initialised.")
    return state.monitor


def require_execution(state: AppState = Depends(_state)) -> Any:
    if state.execution_engine is None:
        raise HTTPException(503, "ExecutionEngine not initialised.")
    return state.execution_engine


def require_risk(state: AppState = Depends(_state)) -> Any:
    if state.risk_engine is None:
        raise HTTPException(503, "RiskEngine not initialised.")
    return state.risk_engine


def require_backtest(state: AppState = Depends(_state)) -> Any:
    if state.backtest_engine is None:
        raise HTTPException(503, "BacktestEngine not initialised.")
    return state.backtest_engine
