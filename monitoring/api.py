"""
Monitoring API
==============
FastAPI REST server exposing real-time trading metrics for dashboards.

Endpoints
---------
  GET /health                         → system health + uptime
  GET /metrics/summary                → full snapshot (equity, risk, strategies)
  GET /metrics/equity                 → equity curve time-series
  GET /metrics/drawdowns              → drawdown statistics + history
  GET /metrics/risk                   → live risk exposure
  GET /metrics/pnl                    → per-strategy PnL
  GET /trades                         → trade history (filterable)
  GET /strategies                     → strategy-level stats
  GET /metrics/risk/history           → risk snapshot history
  POST /control/emergency_stop        → trigger emergency stop (operator only)
  DELETE /control/emergency_stop      → clear emergency stop flag

Run standalone
--------------
  uvicorn monitoring.api:app --host 0.0.0.0 --port 8000 --reload

Or programmatically:
  from monitoring.api import create_app
  app = create_app(monitor=monitor, trader=trader)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI import (optional — trading must not fail without it)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Query, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    logger.warning(
        "FastAPI not installed. "
        "Install with: pip install fastapi uvicorn"
    )

from monitoring.monitor import TradingMonitor

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

if _FASTAPI_OK:
    class EmergencyStopRequest(BaseModel):
        reason: str = "operator_api_call"
        api_key: str = ""          # simple bearer check


# ---------------------------------------------------------------------------
# Global state injected at startup
# ---------------------------------------------------------------------------

_monitor: Optional[TradingMonitor] = None
_trader  = None                     # LiveTrader (optional)
_start_time: float = time.monotonic()
_API_KEY: str = ""                  # set via create_app()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    monitor:  TradingMonitor,
    trader    = None,
    api_key:  str  = "",
    cors_origins: list[str] | None = None,
) -> "FastAPI":
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    monitor  : TradingMonitor  — the central monitoring hub
    trader   : LiveTrader      — optional, enables /control endpoints
    api_key  : str             — if set, required on POST/DELETE endpoints
    cors_origins : list[str]   — allowed CORS origins (default: *)
    """
    if not _FASTAPI_OK:
        raise ImportError("fastapi and uvicorn are required. pip install fastapi uvicorn")

    global _monitor, _trader, _start_time, _API_KEY
    _monitor    = monitor
    _trader     = trader
    _start_time = time.monotonic()
    _API_KEY    = api_key

    app = FastAPI(
        title       = "Quant Fund Monitoring API",
        description = "Real-time metrics and control for the quant-fund-ai trading system.",
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = cors_origins or ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ---- Register routes -----------------------------------------------
    _register_routes(app)

    logger.info("MonitoringAPI created — docs at /docs")
    return app


# ---------------------------------------------------------------------------
# Dependency: require monitor
# ---------------------------------------------------------------------------

def _get_monitor() -> TradingMonitor:
    if _monitor is None:
        raise HTTPException(503, "Monitor not initialised")
    return _monitor


def _check_api_key(api_key: str = Query(default="")) -> None:
    if _API_KEY and api_key != _API_KEY:
        raise HTTPException(403, "Invalid API key")


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def _register_routes(app: "FastAPI") -> None:

    # ------------------------------------------------------------------
    # /health
    # ------------------------------------------------------------------

    @app.get("/health", tags=["system"])
    def health():
        """System health check."""
        uptime_s = time.monotonic() - _start_time
        return {
            "status":       "ok",
            "uptime_s":     round(uptime_s, 1),
            "uptime_human": _fmt_duration(uptime_s),
            "ts":           datetime.now(timezone.utc).isoformat(),
            "monitor_ok":   _monitor is not None,
            "trader_running": _trader.is_running if _trader else None,
        }

    # ------------------------------------------------------------------
    # /metrics/summary
    # ------------------------------------------------------------------

    @app.get("/metrics/summary", tags=["metrics"])
    def metrics_summary(monitor: TradingMonitor = Depends(_get_monitor)):
        """Full snapshot: equity, risk, strategies, recent trades."""
        return monitor.get_summary()

    # ------------------------------------------------------------------
    # /metrics/equity
    # ------------------------------------------------------------------

    @app.get("/metrics/equity", tags=["metrics"])
    def metrics_equity(
        limit: int = Query(500, ge=1, le=5000, description="Max data points"),
        from_db: bool = Query(False, description="Query MySQL instead of in-memory cache"),
        start: Optional[str] = Query(None, description="ISO8601 start datetime"),
        end:   Optional[str] = Query(None, description="ISO8601 end datetime"),
        monitor: TradingMonitor = Depends(_get_monitor),
    ):
        """Portfolio equity curve time-series."""
        if from_db:
            start_dt = _parse_dt(start)
            end_dt   = _parse_dt(end)
            return monitor.query_equity_curve(start=start_dt, end=end_dt, limit=limit)
        return monitor.get_equity_curve(limit=limit)

    # ------------------------------------------------------------------
    # /metrics/drawdowns
    # ------------------------------------------------------------------

    @app.get("/metrics/drawdowns", tags=["metrics"])
    def metrics_drawdowns(
        limit: int = Query(50, ge=1, le=500),
        monitor: TradingMonitor = Depends(_get_monitor),
    ):
        """Drawdown statistics and historical drawdown periods."""
        return {
            "current":        monitor.get_drawdown_stats(),
            "history":        monitor.query_drawdowns(limit=limit),
        }

    # ------------------------------------------------------------------
    # /metrics/risk
    # ------------------------------------------------------------------

    @app.get("/metrics/risk", tags=["metrics"])
    def metrics_risk(monitor: TradingMonitor = Depends(_get_monitor)):
        """Current live risk exposure snapshot."""
        snap = monitor.get_latest_risk()
        if snap is None:
            raise HTTPException(404, "No risk data available yet")
        return snap

    # ------------------------------------------------------------------
    # /metrics/risk/history
    # ------------------------------------------------------------------

    @app.get("/metrics/risk/history", tags=["metrics"])
    def metrics_risk_history(
        limit: int = Query(100, ge=1, le=1000),
        start: Optional[str] = Query(None),
        monitor: TradingMonitor = Depends(_get_monitor),
    ):
        """Historical risk snapshots."""
        start_dt = _parse_dt(start)
        return monitor.query_risk_snapshots(limit=limit, start=start_dt)

    # ------------------------------------------------------------------
    # /metrics/pnl
    # ------------------------------------------------------------------

    @app.get("/metrics/pnl", tags=["metrics"])
    def metrics_pnl(
        strategy: Optional[str] = Query(None, description="Filter by strategy name"),
        start:    Optional[str] = Query(None),
        end:      Optional[str] = Query(None),
        limit:    int = Query(500, ge=1, le=5000),
        monitor:  TradingMonitor = Depends(_get_monitor),
    ):
        """Per-strategy PnL time-series."""
        start_dt = _parse_dt(start)
        end_dt   = _parse_dt(end)
        return monitor.query_strategy_pnl(
            strategy=strategy, start=start_dt, end=end_dt, limit=limit
        )

    # ------------------------------------------------------------------
    # /strategies
    # ------------------------------------------------------------------

    @app.get("/strategies", tags=["strategies"])
    def strategies_summary(monitor: TradingMonitor = Depends(_get_monitor)):
        """Per-strategy statistics: trades, win rate, Sharpe, cumulative PnL."""
        return monitor.get_strategy_summary()

    # ------------------------------------------------------------------
    # /trades
    # ------------------------------------------------------------------

    @app.get("/trades", tags=["trades"])
    def trades(
        symbol:   Optional[str] = Query(None, description="Filter by symbol"),
        strategy: Optional[str] = Query(None, description="Filter by strategy"),
        start:    Optional[str] = Query(None, description="ISO8601 start"),
        end:      Optional[str] = Query(None, description="ISO8601 end"),
        limit:    int  = Query(200, ge=1, le=2000),
        live:     bool = Query(True, description="Use in-memory buffer (fast)"),
        monitor:  TradingMonitor = Depends(_get_monitor),
    ):
        """
        Trade history.
        Use `live=true` for the latest in-memory trades (fast).
        Use `live=false` to query MySQL / JSON logs.
        """
        if live and not (symbol or strategy or start or end):
            return monitor.get_recent_trades(limit=limit)
        start_dt = _parse_dt(start)
        end_dt   = _parse_dt(end)
        return monitor.query_trades(
            symbol=symbol, strategy=strategy,
            start=start_dt, end=end_dt, limit=limit,
        )

    # ------------------------------------------------------------------
    # /trades/stats
    # ------------------------------------------------------------------

    @app.get("/trades/stats", tags=["trades"])
    def trades_stats(
        strategy: Optional[str] = Query(None),
        limit:    int = Query(500),
        monitor:  TradingMonitor = Depends(_get_monitor),
    ):
        """
        Aggregate trade statistics: total trades, win rate,
        profit factor, avg win/loss.
        """
        trades_data = monitor.query_trades(strategy=strategy, limit=limit)
        return _compute_trade_stats(trades_data)

    # ------------------------------------------------------------------
    # /control/emergency_stop  (POST = trigger, DELETE = clear)
    # ------------------------------------------------------------------

    @app.post("/control/emergency_stop", tags=["control"])
    def trigger_emergency_stop(
        body: "EmergencyStopRequest",
        _: None = Depends(_check_api_key),
    ):
        """Trigger the emergency stop (halts all trading immediately)."""
        if _trader is None:
            raise HTTPException(503, "Trader not attached to this API instance")
        _trader.emergency_stop(reason=body.reason)
        return {"status": "triggered", "reason": body.reason}

    @app.delete("/control/emergency_stop", tags=["control"])
    def clear_emergency_stop(
        api_key: str = Query(default=""),
        _: None = Depends(_check_api_key),
    ):
        """Clear the emergency stop flag (allows trader to restart)."""
        if _trader is None:
            raise HTTPException(503, "Trader not attached")
        _trader.emergency.clear()
        return {"status": "cleared"}

    @app.get("/control/status", tags=["control"])
    def trader_status():
        """Trader runtime status."""
        if _trader is None:
            return {"trader": "not_attached"}
        return _trader.get_status()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise HTTPException(400, f"Invalid datetime format: {s!r}")


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _compute_trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0}

    pnls    = [t.get("pnl", 0.0) for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p < 0]

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))

    return {
        "n_trades":     len(trades),
        "n_wins":       len(wins),
        "n_losses":     len(losses),
        "win_rate":     round(len(wins) / len(trades) * 100, 2),
        "total_pnl":    round(sum(pnls), 4),
        "avg_pnl":      round(sum(pnls) / len(pnls), 4),
        "avg_win":      round(sum(wins)   / max(len(wins),   1), 4),
        "avg_loss":     round(sum(losses) / max(len(losses), 1), 4),
        "profit_factor": round(gross_profit / max(gross_loss, 1e-10), 4),
        "best_trade":   round(max(pnls), 4) if pnls else 0,
        "worst_trade":  round(min(pnls), 4) if pnls else 0,
    }


# ---------------------------------------------------------------------------
# Standalone app (for uvicorn)
# ---------------------------------------------------------------------------

# Expose a default app instance so `uvicorn monitoring.api:app` works
# after calling setup() from main entry point.
app: Optional["FastAPI"] = None


def setup(
    monitor:      TradingMonitor,
    trader        = None,
    api_key:      str  = "",
    cors_origins: list[str] | None = None,
) -> "FastAPI":
    """
    Initialise and return the ASGI app.
    Call this before starting uvicorn.

    Example
    -------
    >>> from monitoring.api import setup
    >>> from monitoring.monitor import TradingMonitor
    >>> monitor = TradingMonitor()
    >>> app = setup(monitor=monitor, trader=trader, api_key="secret")
    >>> # uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    global app
    app = create_app(
        monitor      = monitor,
        trader       = trader,
        api_key      = api_key,
        cors_origins = cors_origins,
    )
    return app
