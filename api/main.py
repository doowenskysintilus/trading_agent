"""
Quant Fund — Trading System API
================================
FastAPI backend wiring all subsystems into a secure REST interface.

Endpoint groups
---------------
  /trading          — start, stop, emergency stop, status
  /portfolio        — equity, positions, allocations, equity curve
  /strategies       — per-strategy stats, enable/disable
  /backtest         — submit job, poll status, fetch results
  /trades           — history, open positions, manual override, close position
  /risk             — live risk status, history, config update

Authentication
--------------
  All endpoints require:  X-API-Key: <key>   OR   Authorization: Bearer <key>
  (configurable; disable by leaving api_key blank)

Run
---
  python -m api.main
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import (
        BackgroundTasks, Depends, FastAPI, HTTPException,
        Path, Query, Request, Security, status,
        WebSocket, WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")

from api.auth         import Auth, configure as _configure_auth
from api.schemas      import (
    APIResponse, ok, err,
    TradingStartRequest, TradingStopRequest, EmergencyStopRequest,
    PortfolioStatusResponse, EquityPoint,
    StrategyPerformanceResponse, StrategyToggleRequest,
    BacktestRequest, BacktestJobResponse, BacktestResultResponse,
    BacktestStrategyMetrics, BacktestJobStatus,
    ManualTradeRequest, ClosePositionRequest, TradeExecutionResponse,
    RiskStatusResponse, RiskConfigUpdate, DrawdownPeriod,
)
from api.dependencies import (
    AppState, BacktestJob,
    get_app_state, require_trader, require_monitor,
    require_execution, require_backtest,
)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    app_state:    Optional["AppState"] = None,
    api_key:      str  = "",
    cors_origins: list[str] | None = None,
    title:        str  = "Quant Fund API",
    version:      str  = "1.0.0",
) -> "FastAPI":
    """
    Create and return the configured FastAPI application.

    Parameters
    ----------
    app_state    : inject a pre-built AppState; uses module singleton if None
    api_key      : required on every request (blank = open access)
    cors_origins : list of allowed CORS origins (default: *)
    """
    if not _FASTAPI_OK:
        raise ImportError("pip install fastapi uvicorn")

    _configure_auth(api_key=api_key)

    # Wire WebSocket broadcaster into TradingMonitor so every closed trade
    # and cycle tick is pushed instantly to all connected dashboard clients.
    try:
        from monitoring import monitor as _mon_module
        from api.ws import broadcast_sync as _bcast
        _mon_module.set_ws_broadcaster(_bcast)
        logger.info("WebSocket broadcaster wired into TradingMonitor")
    except Exception as _ws_wire_exc:
        logger.warning("Could not wire WS broadcaster: %s", _ws_wire_exc)

    if app_state is not None:
        # Override the module-level singleton
        import api.dependencies as _dep
        _dep._app_state = app_state

    _state = get_app_state()

    app = FastAPI(
        title       = title,
        version     = version,
        description = __doc__,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = cors_origins or ["*"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*", "X-API-Key"],
    )

    # ----- Global exception handlers ------------------------------------

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code = 500,
            content     = err(f"Internal server error: {type(exc).__name__}"),
        )

    # ====================================================================
    # /health  &  /docs  (no auth)
    # ====================================================================

    @app.get("/health", tags=["system"], include_in_schema=True)
    def health():
        """Liveness probe — no auth required."""
        up = time.monotonic()
        return {
            "status":         "ok",
            "ts":             datetime.now(timezone.utc).isoformat(),
            "trader_running": _state.is_trader_running(),
            "emergency":      _state.is_emergency_active(),
        }

    # ====================================================================
    # /trading
    # ====================================================================

    @app.post("/trading/start", tags=["trading"], dependencies=[Auth])
    def trading_start(body: TradingStartRequest):
        """
        Start the live trading loop.
        Creates a minimal LiveTrader from the provided config and
        wires registered strategies/hooks.
        """
        if _state.is_trader_running():
            return ok({"message": "Trader is already running."})

        try:
            from live_trading.live_trader import LiveTrader, LiveTraderConfig
            from engines.execution_engine.execution_engine import ExecutionConfig

            cfg = LiveTraderConfig(
                symbols            = body.symbols,
                timeframe          = body.timeframe,
                cycle_interval_seconds = body.cycle_interval_s,
                allocation_method  = body.allocation_method,
            )
            trader = LiveTrader(config=cfg)

            # Re-register any strategies already known to AppState
            for name, model in _state.strategies.items():
                trader.register_strategy(model)

            # Attach monitor hook if available
            if _state.monitor:
                trader.register_hook(_state.monitor)

            _state.attach_trader(trader)
            trader.start_background()

            return ok({
                "message":   "LiveTrader started.",
                "symbols":   body.symbols,
                "timeframe": body.timeframe,
            })
        except Exception as exc:
            logger.exception("Failed to start trader")
            raise HTTPException(500, f"Start failed: {exc}") from exc

    @app.post("/trading/stop", tags=["trading"], dependencies=[Auth])
    def trading_stop(body: TradingStopRequest):
        """Gracefully stop the trading loop."""
        if not _state.is_trader_running():
            return ok({"message": "Trader is not running."})
        _state.trader.stop(reason=body.reason)
        return ok({"message": "Stop signal sent.", "reason": body.reason})

    @app.post("/trading/emergency_stop", tags=["trading"], dependencies=[Auth])
    def trading_emergency_stop(body: EmergencyStopRequest):
        """Immediately halt all trading and cancel pending orders."""
        if _state.trader is None:
            raise HTTPException(503, "Trader not initialised.")
        _state.trader.emergency_stop(reason=body.reason)
        return ok({"message": "Emergency stop triggered.", "reason": body.reason})

    @app.delete("/trading/emergency_stop", tags=["trading"], dependencies=[Auth])
    def trading_clear_emergency():
        """Clear the emergency stop flag so trading can resume."""
        if _state.trader is None:
            raise HTTPException(503, "Trader not initialised.")
        _state.trader.emergency.clear()
        return ok({"message": "Emergency stop cleared."})

    @app.get("/trading/status", tags=["trading"], dependencies=[Auth])
    def trading_status():
        """Full trader runtime status."""
        if _state.trader is None:
            return ok({
                "running":          False,
                "emergency_active": False,
                "message":          "Trader not initialised.",
            })
        s = _state.trader.get_status()
        allocs = {}
        try:
            allocs = _state.trader.get_allocation_summary()
        except Exception:
            pass
        return ok({**s, "allocations": allocs})

    # ====================================================================
    # /portfolio
    # ====================================================================

    @app.get("/portfolio/status", tags=["portfolio"], dependencies=[Auth])
    def portfolio_status():
        """
        Live portfolio snapshot: equity, balance, PnL, drawdown,
        open positions, per-strategy allocations.
        """
        if _state.monitor:
            summary = _state.monitor.get_summary()
            return ok(summary)

        # Fallback: extract from trader directly
        if _state.trader is None:
            raise HTTPException(503, "Neither monitor nor trader is initialised.")

        st = _state.trader.get_status()
        return ok(st)

    @app.get("/portfolio/equity_curve", tags=["portfolio"], dependencies=[Auth])
    def portfolio_equity_curve(
        limit:   int           = Query(500,  ge=1, le=5000),
        from_db: bool          = Query(False),
        start:   Optional[str] = Query(None),
        end:     Optional[str] = Query(None),
    ):
        """Equity curve time-series (in-memory or DB)."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        if from_db:
            start_dt = _parse_dt(start)
            end_dt   = _parse_dt(end)
            data = _state.monitor.query_equity_curve(
                start=start_dt, end=end_dt, limit=limit
            )
        else:
            data = _state.monitor.get_equity_curve(limit=limit)
        return ok(data)

    @app.get("/portfolio/allocations", tags=["portfolio"], dependencies=[Auth])
    def portfolio_allocations():
        """Current capital allocation per strategy."""
        if _state.trader is None:
            raise HTTPException(503, "Trader not initialised.")
        try:
            return ok(_state.trader.get_allocation_summary())
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.get("/portfolio/drawdowns", tags=["portfolio"], dependencies=[Auth])
    def portfolio_drawdowns(limit: int = Query(50, ge=1, le=500)):
        """Drawdown statistics and historical drawdown periods."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        return ok({
            "current": _state.monitor.get_drawdown_stats(),
            "history": _state.monitor.query_drawdowns(limit=limit),
        })

    # ====================================================================
    # /strategies
    # ====================================================================

    @app.get("/strategies", tags=["strategies"], dependencies=[Auth])
    def strategies_list():
        """All registered strategies with their performance stats."""
        if _state.monitor:
            return ok(_state.monitor.get_strategy_summary())
        return ok([
            {"strategy": name, "registered": True}
            for name in _state.strategies
        ])

    @app.get("/strategies/{strategy_name}/performance",
             tags=["strategies"], dependencies=[Auth])
    def strategy_performance(strategy_name: str = Path(..., min_length=1)):
        """Detailed performance metrics for a single strategy."""
        if _state.monitor:
            stats = _state.monitor.get_strategy_summary()
            for s in stats:
                name = s.get("strategy") or s.get("name", "")
                if name.lower() == strategy_name.lower():
                    return ok(s)
            raise HTTPException(404, f"Strategy '{strategy_name}' not found in monitor.")

        if strategy_name not in _state.strategies:
            raise HTTPException(404, f"Strategy '{strategy_name}' not registered.")
        return ok({"strategy": strategy_name, "note": "Monitor not attached; no stats available."})

    @app.get("/strategies/{strategy_name}/trades",
             tags=["strategies"], dependencies=[Auth])
    def strategy_trades(
        strategy_name: str       = Path(...),
        limit:         int       = Query(200, ge=1, le=2000),
        start:         Optional[str] = Query(None),
        end:           Optional[str] = Query(None),
    ):
        """Trade history for a specific strategy."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        data = _state.monitor.query_trades(
            strategy = strategy_name,
            start    = _parse_dt(start),
            end      = _parse_dt(end),
            limit    = limit,
        )
        return ok(data)

    @app.post("/strategies/{strategy_name}/disable",
              tags=["strategies"], dependencies=[Auth])
    def strategy_disable(strategy_name: str = Path(...)):
        """Disable a strategy (sets weight to 0 in aggregator)."""
        return _set_strategy_weight(strategy_name, weight=0.0)

    @app.post("/strategies/{strategy_name}/enable",
              tags=["strategies"], dependencies=[Auth])
    def strategy_enable(
        strategy_name: str = Path(...),
        weight: float = Query(1.0, gt=0, le=10.0),
    ):
        """Re-enable a disabled strategy with the specified weight."""
        return _set_strategy_weight(strategy_name, weight=weight)

    def _set_strategy_weight(name: str, weight: float) -> dict:
        if _state.trader is None:
            raise HTTPException(503, "Trader not initialised.")
        try:
            _state.trader._aggregator.set_weight(name, weight)
        except AttributeError:
            raise HTTPException(503, "Aggregator not available on trader.")
        return ok({"strategy": name, "weight": weight})

    # ====================================================================
    # /backtest
    # ====================================================================

    @app.post("/backtest/run", tags=["backtest"],
              dependencies=[Auth], status_code=202)
    def backtest_run(body: BacktestRequest, bg: BackgroundTasks):
        """
        Submit a backtest job.
        Returns immediately with a `job_id`; poll `/backtest/{job_id}/status`.
        """
        if _state.backtest_engine is None:
            # Lazy-init a BacktestEngine with defaults
            try:
                from backtesting.backtest_engine import BacktestEngine, BacktestConfig
                cfg = BacktestConfig(
                    initial_balance = body.initial_balance,
                    commission      = body.commission,
                    spread          = body.spread,
                )
                _state.backtest_engine = BacktestEngine(config=cfg)
            except Exception as exc:
                raise HTTPException(503, f"BacktestEngine unavailable: {exc}") from exc

        job = _state.create_job(body)
        bg.add_task(_run_backtest_job, job, _state)
        return ok(job.to_status_dict())

    @app.get("/backtest/{job_id}/status", tags=["backtest"], dependencies=[Auth])
    def backtest_status(job_id: str = Path(..., min_length=6, max_length=32)):
        """Poll the status of a submitted backtest job."""
        job = _state.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found.")
        return ok(job.to_status_dict())

    @app.get("/backtest/{job_id}/results", tags=["backtest"], dependencies=[Auth])
    def backtest_results(job_id: str = Path(...)):
        """Retrieve the results of a completed backtest job."""
        job = _state.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found.")
        if job.status == "running" or job.status == "queued":
            raise HTTPException(202, "Job not yet complete. Poll /status first.")
        if job.status == "failed":
            raise HTTPException(500, f"Job failed: {job.error}")
        return ok(_format_backtest_result(job))

    @app.get("/backtest/jobs", tags=["backtest"], dependencies=[Auth])
    def backtest_jobs_list(limit: int = Query(20, ge=1, le=200)):
        """List recent backtest jobs."""
        jobs = sorted(
            _state.backtest_jobs.values(),
            key=lambda j: j.submitted_at,
            reverse=True,
        )[:limit]
        return ok([j.to_status_dict() for j in jobs])

    # ====================================================================
    # /trades
    # ====================================================================

    @app.get("/trades/history", tags=["trades"], dependencies=[Auth])
    def trades_history(
        symbol:   Optional[str] = Query(None),
        strategy: Optional[str] = Query(None),
        start:    Optional[str] = Query(None),
        end:      Optional[str] = Query(None),
        limit:    int = Query(200, ge=1, le=2000),
        live:     bool = Query(True, description="Use in-memory cache"),
    ):
        """Trade history with optional filters."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        if live and not (symbol or strategy or start or end):
            return ok(_state.monitor.get_recent_trades(limit=limit))
        return ok(_state.monitor.query_trades(
            symbol=symbol, strategy=strategy,
            start=_parse_dt(start), end=_parse_dt(end),
            limit=limit,
        ))

    @app.get("/trades/open", tags=["trades"], dependencies=[Auth])
    def trades_open():
        """Fetch open positions directly from MT5."""
        if _state.execution_engine is None:
            raise HTTPException(503, "ExecutionEngine not initialised.")
        try:
            positions = _state.execution_engine.get_open_positions()
            return ok(positions)
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @app.post("/trades/override", tags=["trades"], dependencies=[Auth])
    def trade_manual_override(body: ManualTradeRequest):
        """
        Execute a manual trade bypassing signal generation.
        Enforced by execution engine's own risk layer.
        """
        if _state.execution_engine is None:
            raise HTTPException(503, "ExecutionEngine not initialised.")

        try:
            from engines.signal_engine.signal_engine import Signal

            # Map to internal Signal for ExecutionEngine
            signal = {
                "symbol":    body.symbol,
                "direction": body.direction.value,
                "size":      body.size,
                "sl_price":  body.sl_price,
                "tp_price":  body.tp_price,
                "comment":   body.comment,
            }

            result = _state.execution_engine.execute(signal)

            return ok({
                "status":          result.status if hasattr(result, "status") else str(result),
                "ticket":          getattr(result, "ticket",         None),
                "fill_price":      getattr(result, "fill_price",     None),
                "slippage_points": getattr(result, "slippage_points", None),
                "retries":         getattr(result, "retries",         0),
                "comment":         body.comment,
            })
        except Exception as exc:
            logger.exception("Manual trade override failed")
            raise HTTPException(500, f"Trade execution failed: {exc}") from exc

    @app.post("/trades/close", tags=["trades"], dependencies=[Auth])
    def trade_close(body: ClosePositionRequest):
        """
        Close an open position.
        Specify `ticket` for a specific order, or omit to close all
        positions for the symbol.
        """
        if _state.execution_engine is None:
            raise HTTPException(503, "ExecutionEngine not initialised.")
        try:
            if body.ticket:
                result = _state.execution_engine.close_position(
                    ticket  = body.ticket,
                    comment = body.comment,
                )
            else:
                result = _state.execution_engine.close_all(
                    symbol  = body.symbol,
                    comment = body.comment,
                )
            return ok(result if isinstance(result, dict) else {"status": str(result)})
        except Exception as exc:
            logger.exception("Close position failed")
            raise HTTPException(500, f"Close failed: {exc}") from exc

    @app.get("/trades/stats", tags=["trades"], dependencies=[Auth])
    def trades_stats(
        strategy: Optional[str] = Query(None),
        limit:    int = Query(500, ge=1, le=5000),
    ):
        """Aggregate trade statistics: win rate, profit factor, avg win/loss."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        trades = _state.monitor.query_trades(strategy=strategy, limit=limit)
        return ok(_compute_trade_stats(trades))

    # ====================================================================
    # /risk
    # ====================================================================

    @app.get("/risk/status", tags=["risk"], dependencies=[Auth])
    def risk_status():
        """Current live risk exposure snapshot."""
        if _state.monitor:
            snap = _state.monitor.get_latest_risk()
            if snap:
                return ok(snap)

        if _state.risk_engine and _state.trader:
            try:
                # Request a fresh evaluation if possible
                res = _state.risk_engine.get_current_exposure()
                return ok(res)
            except Exception:
                pass

        raise HTTPException(404, "No risk data available yet.")

    @app.get("/risk/history", tags=["risk"], dependencies=[Auth])
    def risk_history(
        limit: int           = Query(100, ge=1, le=1000),
        start: Optional[str] = Query(None),
    ):
        """Historical risk snapshots."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        return ok(_state.monitor.query_risk_snapshots(
            limit=limit, start=_parse_dt(start)
        ))

    @app.get("/risk/drawdowns", tags=["risk"], dependencies=[Auth])
    def risk_drawdowns(limit: int = Query(50, ge=1, le=500)):
        """Drawdown history."""
        if _state.monitor is None:
            raise HTTPException(503, "Monitor not initialised.")
        return ok({
            "stats":   _state.monitor.get_drawdown_stats(),
            "history": _state.monitor.query_drawdowns(limit=limit),
        })

    @app.patch("/risk/config", tags=["risk"], dependencies=[Auth])
    def risk_config_update(body: RiskConfigUpdate):
        """
        Update risk limits at runtime.
        Only provided fields are changed; others are unchanged.
        """
        updated = {}
        overrides = body.model_dump(exclude_none=True)

        if not overrides:
            raise HTTPException(400, "No fields provided to update.")

        _state.risk_overrides.update(overrides)

        # Apply to RiskEngine if attached
        if _state.risk_engine:
            for key, val in overrides.items():
                if hasattr(_state.risk_engine, key):
                    setattr(_state.risk_engine, key, val)
                    updated[key] = val
                elif hasattr(getattr(_state.risk_engine, "config", None), key):
                    setattr(_state.risk_engine.config, key, val)
                    updated[key] = val
                else:
                    updated[key] = val   # cached; applied on next engine init

        return ok({"updated": updated, "all_overrides": _state.risk_overrides})

    @app.get("/risk/config", tags=["risk"], dependencies=[Auth])
    def risk_config_get():
        """Return current risk configuration overrides applied via API."""
        config = {}
        if _state.risk_engine and hasattr(_state.risk_engine, "config"):
            config = vars(_state.risk_engine.config)
        return ok({
            "engine_config":   config,
            "api_overrides":   _state.risk_overrides,
        })

    # ====================================================================
    # /ws  — Real-time WebSocket feed for the dashboard
    # ====================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        Persistent WebSocket connection for real-time dashboard updates.

        Auth  : pass `?api_key=<key>` in the URL (or leave blank if auth disabled).
        Frame : JSON  { type: string, payload: dict, ts: ISO8601 }

        Message types sent by server
        ----------------------------
          snapshot        — full state on connect
          tick            — equity points + portfolio + risk (every 2 s)
          positions       — open positions table (every 10 s)
          strategies      — per-strategy stats (every 10 s)
          heartbeat       — keep-alive ping
        """
        from api.ws import manager as _ws_manager

        # ---- API-key gate ----------------------------------------------
        if api_key:
            params = dict(websocket.query_params)
            token  = (
                params.get("api_key")
                or websocket.headers.get("authorization", "").removeprefix("Bearer ").strip()
            )
            if token != api_key:
                await websocket.close(code=4001)
                return

        await _ws_manager.connect(websocket)
        tick = 0
        try:
            # Full snapshot on connect
            snap = _ws_build_snapshot(_state)
            await _ws_manager.send_to(websocket, "snapshot", snap)

            while True:
                await asyncio.sleep(2.0)
                tick += 1

                # Every tick: equity points + portfolio + risk
                await _ws_manager.send_to(
                    websocket, "tick", _ws_build_tick(_state)
                )

                # Every 10 s: positions + strategies
                if tick % 5 == 0:
                    pos  = _ws_get_positions(_state)
                    strats = _ws_get_strategies(_state)
                    await _ws_manager.send_to(websocket, "positions",  {"positions":  pos})
                    await _ws_manager.send_to(websocket, "strategies", {"strategies": strats})

                # Keep-alive
                await _ws_manager.send_to(websocket, "heartbeat", {"tick": tick})

        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await _ws_manager.disconnect(websocket)

    return app


# ---------------------------------------------------------------------------
# WebSocket payload builders (module-level so they're accessible)
# ---------------------------------------------------------------------------

def _ws_build_snapshot(state: "AppState") -> dict:
    snap: dict = {}
    if state.monitor:
        try:
            snap = state.monitor.get_summary()
            snap["equity_curve"]  = state.monitor.get_equity_curve(limit=300)
            snap["recent_trades"] = state.monitor.get_recent_trades(limit=50)
        except Exception:
            pass
    snap["positions"]  = _ws_get_positions(state)
    snap["strategies"] = _ws_get_strategies(state)
    return snap


def _ws_build_tick(state: "AppState") -> dict:
    payload: dict = {}
    if state.monitor:
        try:
            payload["equity_curve"] = state.monitor.get_equity_curve(limit=10)
            summary = state.monitor.get_summary()
            payload["portfolio"]   = summary.get("portfolio", {})
            payload["risk"]        = state.monitor.get_latest_risk() or {}
            payload["trades"]      = state.monitor.get_recent_trades(limit=5)
        except Exception:
            pass
    return payload


def _ws_get_positions(state: "AppState") -> list:
    if state.execution_engine:
        try:
            return state.execution_engine.get_open_positions()
        except Exception:
            pass
    return []


def _ws_get_strategies(state: "AppState") -> list:
    if state.monitor:
        try:
            return state.monitor.get_strategy_summary()
        except Exception:
            pass
    return []


# Re-remove the old `return app` that was above — it's now in the WS block above.


# ---------------------------------------------------------------------------
# Background task: run backtest in thread
# ---------------------------------------------------------------------------

def _run_backtest_job(job: BacktestJob, state: AppState) -> None:
    """Executed by FastAPI BackgroundTasks in a thread pool thread."""
    job.status     = BacktestJobStatus.RUNNING.value
    job.started_at = datetime.now(timezone.utc)

    try:
        from backtesting.backtest_engine import BacktestConfig

        req: BacktestRequest = job.request

        # Build strategy list from registry
        strategy_models = []
        for name in req.strategies:
            model = state.strategies.get(name)
            if model is None:
                raise ValueError(f"Strategy '{name}' not registered in AppState.")
            strategy_models.append(model)

        if not strategy_models:
            raise ValueError("No valid strategies found for backtest.")

        # Load data from FeatureStore
        if state.feature_store is None:
            raise RuntimeError("FeatureStore not attached.")

        data = state.feature_store.load(symbol=req.symbol, timeframe=req.timeframe)
        if data is None or data.empty:
            raise RuntimeError(f"No data for {req.symbol}/{req.timeframe}")

        # Reconfigure engine
        state.backtest_engine.config = BacktestConfig(
            initial_balance = req.initial_balance,
            commission      = req.commission,
            spread          = req.spread,
        )

        result = state.backtest_engine.run(
            strategies = strategy_models,
            data       = data,
            seed       = req.seed,
        )

        job.result       = result
        job.status       = BacktestJobStatus.DONE.value
        job.completed_at = datetime.now(timezone.utc)

    except Exception as exc:
        logger.exception("Backtest job %s failed", job.job_id)
        job.error        = str(exc)
        job.status       = BacktestJobStatus.FAILED.value
        job.completed_at = datetime.now(timezone.utc)


def _format_backtest_result(job: BacktestJob) -> dict:
    result = job.result
    if result is None:
        return {"job_id": job.job_id, "status": "done", "note": "No result data."}

    strategy_metrics = []
    if hasattr(result, "strategy_results"):
        for name, res in result.strategy_results.items():
            m = getattr(res, "metrics", None)
            if m:
                strategy_metrics.append({
                    "strategy":         name,
                    "total_return_pct": getattr(m, "total_return_pct", 0.0),
                    "cagr":             getattr(m, "cagr",             0.0),
                    "sharpe":           getattr(m, "sharpe",           0.0),
                    "sortino":          getattr(m, "sortino",          0.0),
                    "calmar":           getattr(m, "calmar",           0.0),
                    "max_drawdown_pct": getattr(m, "max_drawdown_pct", 0.0),
                    "n_trades":         getattr(m, "n_trades",         0),
                    "win_rate":         getattr(m, "win_rate",         0.0),
                    "profit_factor":    getattr(m, "profit_factor",    0.0),
                    "var_95":           getattr(m, "var_95",           0.0),
                })

    portfolio_m = None
    pm = getattr(result, "portfolio_metrics", None)
    if pm:
        portfolio_m = {
            "strategy":         "portfolio",
            "total_return_pct": getattr(pm, "total_return_pct", 0.0),
            "cagr":             getattr(pm, "cagr",             0.0),
            "sharpe":           getattr(pm, "sharpe",           0.0),
            "sortino":          getattr(pm, "sortino",          0.0),
            "calmar":           getattr(pm, "calmar",           0.0),
            "max_drawdown_pct": getattr(pm, "max_drawdown_pct", 0.0),
            "n_trades":         getattr(pm, "n_trades",         0),
            "win_rate":         getattr(pm, "win_rate",         0.0),
            "profit_factor":    getattr(pm, "profit_factor",    0.0),
            "var_95":           getattr(pm, "var_95",           0.0),
        }

    return {
        "job_id":            job.job_id,
        "status":            job.status,
        "strategy_metrics":  strategy_metrics,
        "portfolio_metrics": portfolio_m,
        "n_trades_total":    sum(s.get("n_trades", 0) for s in strategy_metrics),
        "duration_s":        job.duration_s,
    }


# ---------------------------------------------------------------------------
# Trade statistics helper
# ---------------------------------------------------------------------------

def _compute_trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0}
    pnls   = [t.get("pnl", 0.0) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    return {
        "n_trades":      len(trades),
        "n_wins":        len(wins),
        "n_losses":      len(losses),
        "win_rate":      round(len(wins) / len(trades) * 100, 2),
        "total_pnl":     round(sum(pnls), 4),
        "avg_pnl":       round(sum(pnls) / len(pnls), 4),
        "avg_win":       round(gross_profit / max(len(wins),   1), 4),
        "avg_loss":      round(-gross_loss  / max(len(losses), 1), 4),
        "profit_factor": round(gross_profit / max(gross_loss, 1e-10), 4),
        "best_trade":    round(max(pnls), 4),
        "worst_trade":   round(min(pnls), 4),
    }


# ---------------------------------------------------------------------------
# Datetime parser helper
# ---------------------------------------------------------------------------

def _parse_dt(s: Optional[str]):
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(400, f"Invalid datetime: {s!r}")


# ---------------------------------------------------------------------------
# Module-level app (for `uvicorn api.main:app`)
# ---------------------------------------------------------------------------

app = None  # populated by calling create_app() in your entry point


def _build_default_app() -> "FastAPI":
    """
    Called when the module is loaded by uvicorn directly.
    Override by setting environment variables or calling create_app() manually.
    """
    import os
    key = os.environ.get("QUANT_API_KEY", "")
    return create_app(api_key=key)


if _FASTAPI_OK:
    app = _build_default_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Quant Fund Trading API")
    parser.add_argument("--host",    default="0.0.0.0",    help="Bind host")
    parser.add_argument("--port",    default=8000, type=int, help="Bind port")
    parser.add_argument("--api-key", default=os.environ.get("QUANT_API_KEY", ""),
                        help="API key (or set QUANT_API_KEY env var)")
    parser.add_argument("--reload",  action="store_true",  help="Auto-reload on change")
    args = parser.parse_args()

    if not _FASTAPI_OK:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        raise SystemExit(1)

    app = create_app(api_key=args.api_key)

    print(f"\n  Quant Fund API  →  http://{args.host}:{args.port}/docs\n")
    uvicorn.run(
        app,
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        workers = 1,       # single worker = shared AppState
    )
