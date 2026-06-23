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
import json
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
    TradeDirection,
    RiskStatusResponse, RiskConfigUpdate, DrawdownPeriod,
    RetrainRequest,
    EconomicEventResponse, CalendarEventsResponse,
)
from api.dependencies import (
    AppState, BacktestJob,
    get_app_state, require_trader, require_monitor,
    require_execution, require_backtest,
)
from research.feature_store.calendar_provider import CalendarProvider, EventImportance


# ---------------------------------------------------------------------------
# RL feature provider
# ---------------------------------------------------------------------------

def _fetch_full_mt5_history(symbol: str, timeframe: str) -> "np.ndarray | None":
    """Fetch the complete available MT5 history for a symbol/timeframe.

    Uses copy_rates_range() from the earliest tradeable date (year 2000) to
    now so we get every bar the broker has on record — regardless of how many
    bars the terminal has loaded in RAM.

    For EURUSD H1 this typically yields 80 000 – 200 000 bars (2001 → today).
    For BTCUSD H1 it yields whatever the broker has stored (often from 2014+).

    Returns a numpy structured array (same format as copy_rates_from_pos) or
    None on failure.
    """
    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timezone
        from live_trading.live_trader import _get_mt5_tf

        tf  = _get_mt5_tf(timeframe)
        t0  = datetime(2000, 1, 1, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        rates = mt5.copy_rates_range(symbol, tf, t0, now)
        got   = len(rates) if rates is not None else 0
        logger.info(
            "Full history fetch: %s %s — %d bars from %s to %s",
            symbol, timeframe, got, t0.date(), now.date(),
        )
        return rates if got > 0 else None
    except Exception as exc:
        logger.warning("Full history fetch failed for %s %s: %s", symbol, timeframe, exc)
        return None


def _make_rl_feature_provider(
    symbol: str,
    timeframe: str,
    n_bars: int,
    feature_config=None,
):
    """Build a callable that returns a complete historical feature DataFrame.

    Strategy (in order):
    1. copy_rates_range(symbol, tf, 2000-01-01, now)  — full broker history
    2. If that fails, fall back to copy_rates_from_pos with a large n_bars
       (or auto-detected maximum when n_bars=0).

    The result is every single bar the broker has available for the symbol
    and timeframe — potentially 20+ years of H1 data for major FX pairs.
    """
    def _provider():
        import pandas as pd
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.warning("RL feature provider: MetaTrader5 unavailable.")
            return None

        from live_trading.live_trader import _get_mt5_tf
        from research.feature_store.feature_engineer import FeatureConfig, FeatureEngineer

        # --- Step 1: fetch ALL available history via date range ---------------
        rates = _fetch_full_mt5_history(symbol, timeframe)

        # --- Step 2: fallback — large positional fetch if range failed --------
        if rates is None or len(rates) == 0:
            tf_mt5    = _get_mt5_tf(timeframe)
            term      = mt5.terminal_info()
            maxbars   = int(getattr(term, "maxbars", 0) or 0) or 300_000
            fetch_n   = n_bars if n_bars > 0 else maxbars
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, fetch_n)

        if rates is None or len(rates) == 0:
            logger.warning("RL feature provider: no bars available for %s %s.", symbol, timeframe)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
        df.columns = [c.lower() for c in df.columns]

        engineer = FeatureEngineer(feature_config or FeatureConfig())
        features = engineer.compute(df).ffill().fillna(0.0)

        first = df.index[0].strftime("%Y-%m-%d") if len(df) else "?"
        last  = df.index[-1].strftime("%Y-%m-%d") if len(df) else "?"
        logger.info(
            "RL feature provider: %s %s | %d OHLCV bars (%s → %s) → %d features × %d cols",
            symbol, timeframe,
            len(df), first, last,
            features.shape[0], features.shape[1],
        )
        return features

    return _provider


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

    # CORS: allow_credentials=True is incompatible with allow_origins=["*"]
    # (browsers reject it). If no explicit origins are configured we fall back
    # to a non-credentialed wildcard so the dashboard still works.
    _origins = cors_origins or []
    _credentials = bool(_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = _origins if _origins else ["*"],
        allow_credentials = _credentials,
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
            import os
            from live_trading.live_trader import LiveTrader, LiveTraderConfig
            from engines.execution_engine.execution_engine import ExecutionConfig
            from engines.portfolio_engine.portfolio_engine import AllocationMethod
            from config.settings import settings

            # Resolve allocation method from its enum NAME (e.g. "RISK_PARITY"),
            # falling back to RISK_PARITY if an unknown value is supplied.
            try:
                alloc_method = AllocationMethod[body.allocation_method]
            except KeyError:
                alloc_method = AllocationMethod.RISK_PARITY

            cfg = LiveTraderConfig(
                symbols                = body.symbols,
                timeframe              = body.timeframe,
                warmup_bars            = body.warmup_bars,
                cycle_interval_seconds = body.cycle_interval_s,
                allocation_method      = alloc_method,
                initial_balance        = body.initial_balance,
                htf_enabled            = body.htf_enabled,
                htf_timeframe          = body.htf_timeframe,
                verbose_signals        = body.verbose_signals,
                ml_filter_enabled      = body.ml_filter_enabled,
                ml_min_win_proba       = body.ml_min_win_proba,
                sl_atr_multiplier      = body.sl_atr_multiplier,
                tp_atr_multiplier      = body.tp_atr_multiplier,
                # MT5 credentials sourced from .env (optional — an
                # already-logged-in terminal works without them).
                mt5_login    = settings.mt5.login,
                mt5_password = settings.mt5.password,
                mt5_server   = settings.mt5.server,
                mt5_path     = settings.mt5.path,
                magic_number = settings.mt5.magic_number,
            )
            trader = LiveTrader(config=cfg)

            # Re-register any strategies already known to AppState
            for name, model in _state.strategies.items():
                trader.register_strategy(model)

            # Attach monitor hook if available
            if _state.monitor:
                trader.register_hook(_state.monitor)

            _state.attach_trader(trader)
            # Expose the trader's execution engine so /trades/* endpoints work.
            _state.attach_engines(
                execution = trader._exec_engine,
                risk      = trader._risk_engine,
                portfolio = trader._allocator,
            )
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

    @app.get("/trading/config", tags=["trading"], dependencies=[Auth])
    def trading_config():
        """Default trading parameters from .env (pairs, timeframe, sizing).

        Used by the dashboard to pre-fill the "Start Trading" panel.
        """
        from config.settings import settings as _cfg
        return ok({
            "symbols":           list(_cfg.trading.symbols),
            "timeframe":         _cfg.trading.timeframe,
            "cycle_interval_s":  _cfg.trading.cycle_seconds,
            "warmup_bars":       _cfg.trading.warmup_bars,
            "initial_balance":   _cfg.trading.initial_balance,
            "allocation_method": _cfg.trading.allocation_method,
            "htf_enabled":       _cfg.trading.htf_enabled,
            "htf_timeframe":     _cfg.trading.htf_timeframe,
            "ml_filter_enabled": _cfg.trading.ml_filter_enabled,
            "ml_min_win_proba":  _cfg.trading.ml_min_win_proba,
            "sl_atr_multiplier": _cfg.trading.sl_atr_multiplier,
            "tp_atr_multiplier": _cfg.trading.tp_atr_multiplier,
        })

    # ====================================================================
    # /rl  — learning models (manual retraining from collected results)
    # ====================================================================

    @app.post("/rl/retrain", tags=["learning"], dependencies=[Auth])
    def rl_retrain(body: Optional[RetrainRequest] = None):
        """Retrain the learning models from the system's own trade results.

        * ML win/loss classifier — trained on collected trade experiences.
        * RL agent (RecurrentPPO) — optional, retrained on market history.

        Runs in the background; poll GET /rl/status for progress/results.
        """
        from research.training.retrain_service import RetrainService

        svc = getattr(_state, "retrain_service", None)
        if svc is None:
            svc = RetrainService()
            _state.retrain_service = svc

        req = body or RetrainRequest()

        # When the trader is running, retrain RL on its configured symbol/TF.
        rl_symbol = rl_tf = None
        feature_config = None
        if _state.trader is not None:
            cfg = getattr(_state.trader, "cfg", None)
            if cfg is not None:
                syms = getattr(cfg, "symbols", None)
                rl_symbol = syms[0] if syms else None
                # RL training uses its own timeframe (H1 by default), separate
                # from the live-trading timeframe (e.g. M1). This gives RL
                # access to years of H1 history while live decisions run on M1.
                rl_tf = (
                    _cfg.trading.rl_timeframe
                    or getattr(cfg, "timeframe", None)
                    or "H1"
                )
                feature_config = getattr(cfg, "feature_config", None)

        # RL needs a long, continuous price-series history (not the sparse
        # per-trade experiences). Pull maximum available bars from MT5 and
        # compute the SAME features the live trader uses so the trained agent
        # stays consistent with live inference.
        rl_feature_provider = _make_rl_feature_provider(
            symbol         = rl_symbol or "EURUSD",
            timeframe      = rl_tf,
            n_bars         = req.rl_history_bars,   # 0 = auto all available
            feature_config = feature_config,
        )

        try:
            status_obj = svc.start(
                train_ml            = req.train_ml,
                train_rl            = req.train_rl,
                rl_timesteps        = req.rl_timesteps,
                rl_continuous       = req.rl_continuous,
                rl_interval_s       = req.rl_interval_s,
                rl_symbol           = rl_symbol,
                rl_timeframe        = rl_tf,
                rl_feature_provider = rl_feature_provider,
            )
            return ok(status_obj)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/rl/stop", tags=["learning"], dependencies=[Auth])
    def rl_stop():
        """Request graceful stop for an active retraining loop."""
        svc = getattr(_state, "retrain_service", None)
        if svc is None:
            return ok({"state": "idle", "message": "No retraining run yet."})
        return ok(svc.stop())

    @app.get("/rl/status", tags=["learning"], dependencies=[Auth])
    def rl_status():
        """Current state and last results of the retraining service."""
        svc = getattr(_state, "retrain_service", None)
        if svc is None:
            return ok({"state": "idle", "message": "No retraining run yet."})
        return ok(svc.status())

    @app.get("/rl/progress", tags=["learning"], dependencies=[Auth])
    def rl_progress(limit: int = 500):
        """Reward-vs-time history of the RL agent for the live dashboard chart.

        Reads the append-only progress log written during training (one point
        per episode). Returns the most recent ``limit`` points so the curve can
        confirm the reward trends upward over time.
        """
        from pathlib import Path as _Path
        from strategies.rl_agent.rl_trainer import RLTrainerConfig

        limit = max(1, min(int(limit), 5000))
        progress_file = _Path(RLTrainerConfig().progress_path)
        if not progress_file.exists():
            return ok({"points": [], "message": "No RL training history yet."})

        points: list[dict] = []
        try:
            with progress_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as exc:
            raise HTTPException(500, f"Could not read RL progress: {exc}") from exc

        return ok({"points": points, "n": len(points)})

    @app.get("/rl/episode", tags=["learning"], dependencies=[Auth])
    def rl_episode(bars: int = 600, deterministic: bool = True):
        """Replay one episode of the trained RL agent for environment
        visualisation.

        Pulls recent market history, computes the same features used in
        training, then steps the trained agent through a fresh RLTradingEnv,
        recording price, the action taken (HOLD/BUY/SELL), the position, and
        equity at every bar. The dashboard plots this as price + entry/exit
        markers + an equity curve.
        """
        from pathlib import Path as _Path
        from strategies.rl_agent.rl_trainer import (
            RLTrainerConfig, RLTradingEnv, replay_episode,
        )

        bars = max(64, min(int(bars), 20_000))

        rl_cfg = RLTrainerConfig()
        model_file = _Path(rl_cfg.model_dir) / f"{rl_cfg.model_name}.zip"
        if not model_file.exists():
            return ok({
                "trajectory": [],
                "message": "No trained RL model yet. Train it via Retrain · incl. RL.",
            })

        # Resolve symbol/timeframe/feature_config from the running trader.
        symbol = rl_cfg.symbol
        timeframe = rl_cfg.timeframe
        feature_config = None
        if _state.trader is not None:
            cfg = getattr(_state.trader, "cfg", None)
            if cfg is not None:
                syms = getattr(cfg, "symbols", None)
                symbol = (syms[0] if syms else None) or symbol
                timeframe = getattr(cfg, "timeframe", None) or timeframe
                feature_config = getattr(cfg, "feature_config", None)

        # Fetch OHLCV + compute features, keeping the real close aligned.
        try:
            import MetaTrader5 as mt5
        except ImportError:
            return ok({"trajectory": [], "message": "MetaTrader5 unavailable."})

        from live_trading.live_trader import _get_mt5_tf
        from research.feature_store.feature_engineer import (
            FeatureConfig, FeatureEngineer,
        )

        rates = mt5.copy_rates_from_pos(symbol, _get_mt5_tf(timeframe), 0, bars)
        if rates is None or len(rates) == 0:
            return ok({"trajectory": [], "message": f"MT5 returned no bars for {symbol} {timeframe}."})

        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
        df.columns = [c.lower() for c in df.columns]

        engineer = FeatureEngineer(feature_config or FeatureConfig())
        feats = engineer.compute(df).ffill().fillna(0.0)
        if len(feats) <= rl_cfg.window_size + 2:
            return ok({"trajectory": [], "message": "Not enough bars to replay an episode."})

        # Align the real close to the (NaN-dropped) feature rows.
        close = df.loc[feats.index, "close"].to_numpy(dtype=float)
        features = feats.to_numpy(dtype="float32")

        algo = "RecurrentPPO" if rl_cfg.use_recurrent_ppo else "PPO"
        try:
            trajectory = replay_episode(
                features        = features,
                model_path      = str(model_file),
                algo            = algo,
                window_size     = rl_cfg.window_size,
                reward_cfg      = rl_cfg.reward_config,
                initial_balance = rl_cfg.initial_balance,
                spread          = rl_cfg.spread,
                sl_pct          = rl_cfg.sl_pct,
                tp_pct          = rl_cfg.tp_pct,
                position_size_pct = rl_cfg.position_size_pct,
                close_prices    = close,
                deterministic   = deterministic,
            )
        except Exception as exc:
            logger.exception("RL episode replay failed")
            raise HTTPException(500, f"Episode replay failed: {exc}") from exc

        n_buy  = sum(1 for p in trajectory if p["action"] == 1)
        n_sell = sum(1 for p in trajectory if p["action"] == 2)
        final_equity = trajectory[-1]["equity"] if trajectory else rl_cfg.initial_balance
        return ok({
            "trajectory":      trajectory,
            "n":               len(trajectory),
            "symbol":          symbol,
            "timeframe":       timeframe,
            "initial_balance": rl_cfg.initial_balance,
            "final_equity":    final_equity,
            "n_buy":           n_buy,
            "n_sell":          n_sell,
        })

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

    @app.get("/calendar/events", tags=["calendar"], dependencies=[Auth])
    def calendar_events(
        next_n_hours: int = Query(72, ge=1, le=168),
        source: Optional[str] = Query(None, description="Override calendar source"),
        importance: str = Query("MEDIUM", pattern="^(LOW|MEDIUM|HIGH)$"),
        countries: Optional[str] = Query(None, description="Comma-separated country codes"),
    ):
        """Upcoming economic calendar events."""
        from config.settings import settings as _cfg

        provider = CalendarProvider(source or _cfg.trading.calendar_source)
        min_importance = EventImportance[importance.upper()]
        country_list = [c.strip().upper() for c in (countries or "").split(",") if c.strip()]

        events = provider.get_upcoming_events(
            next_n_hours=next_n_hours,
            countries=country_list or None,
            min_importance=min_importance,
        )

        payload = [
            {
                "timestamp": e.timestamp.isoformat(),
                "country": e.country,
                "name": e.name,
                "importance": e.importance.name,
                "forecast": e.forecast,
                "previous": e.previous,
                "actual": e.actual,
                "revised": e.revised,
                "units": e.units,
                "url": e.url,
            }
            for e in events
        ]
        return ok({"events": payload})

    @app.get("/portfolio/account", tags=["portfolio"], dependencies=[Auth])
    def portfolio_account():
        """Real MT5 account info: balance, equity, margin, currency, leverage.

        Returns 503 when no execution engine is attached yet (trading not
        started) or when MT5 is not connected.
        """
        if _state.execution_engine is None:
            raise HTTPException(503, "Execution engine not initialised (start trading first).")
        info = _state.execution_engine.get_account_info()
        if not info:
            raise HTTPException(503, "MT5 not connected — no account info available.")
        return ok(info)

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
            if _state.trader is not None and hasattr(_state.trader, "get_live_positions"):
                positions = _state.trader.get_live_positions()
            else:
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
            from engines.signal_engine.signal_aggregator import AggregatedDecision
            from engines.risk_engine.risk_engine import TradeOrder
            from research.alpha_models.base import SignalType

            is_buy    = body.direction == TradeDirection.BUY
            signal    = SignalType.BUY if is_buy else SignalType.SELL
            direction = 1 if is_buy else -1

            # Build a synthetic high-confidence decision for the manual order.
            decision = AggregatedDecision(
                signal     = signal,
                confidence = 1.0,
                buy_score  = 1.0 if is_buy else 0.0,
                sell_score = 0.0 if is_buy else 1.0,
                net_score  = 1.0 if is_buy else -1.0,
                metadata   = {"source": "manual_override", "comment": body.comment},
            )

            trade = TradeOrder(
                symbol      = body.symbol,
                strategy    = "manual",
                direction   = direction,
                size        = body.size,
                entry_price = body.sl_price or 0.0,
                atr         = 0.0,
                stop_loss   = body.sl_price,
                take_profit = body.tp_price,
            )

            result = _state.execution_engine.execute(decision, trade)

            return ok({
                "status":          result.status.value if hasattr(result.status, "value") else str(result.status),
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
            result = _state.execution_engine.close_position(
                symbol = body.symbol,
                ticket = body.ticket,
            )
            return ok({
                "status":     result.status.value if hasattr(result.status, "value") else str(result.status),
                "ticket":     getattr(result, "ticket",     None),
                "fill_price": getattr(result, "fill_price", None),
                "message":    getattr(result, "message",    ""),
            })
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
                    tick            — full incremental state (every 2 s)
                    positions       — open positions table (compat stream)
                    strategies      — per-strategy stats (compat stream)
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

                # Compatibility stream for legacy consumers
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

    # ====================================================================
    # Startup — open a read-only MT5 connection so the dashboard can show
    # the real account balance even before live trading is started, register
    # the default alpha strategies, and attach a monitor for the dashboard.
    # Best-effort: each step is skipped silently on failure.
    # ====================================================================
    @app.on_event("startup")
    def _connect_mt5_account():
        if _state.execution_engine is not None:
            return  # a trader already attached its engine
        try:
            from engines.execution_engine.execution_engine import (
                ExecutionConfig, MT5ExecutionEngine,
            )
            from config.settings import settings

            exec_cfg = ExecutionConfig(
                mt5_path     = settings.mt5.path or None,
                login        = settings.mt5.login or None,
                password     = settings.mt5.password or None,
                server       = settings.mt5.server or None,
                magic_number = settings.mt5.magic_number,
            )
            engine = MT5ExecutionEngine(config=exec_cfg)
            if engine.connect():
                _state.attach_engines(execution=engine)
                logger.info("Read-only MT5 connection established for account info.")
            else:
                logger.info("MT5 account connection skipped (not logged in / unavailable).")
        except Exception as exc:
            logger.info("MT5 account connection skipped: %s", exc)

    @app.on_event("startup")
    def _register_default_strategies():
        """Register the built-in alpha strategies so /trading/start works
        out of the box (the dashboard START button needs at least one)."""
        if _state.strategies:
            return  # already populated
        try:
            from strategies.momentum.momentum_alpha import MomentumAlpha
            from strategies.mean_reversion.mean_reversion_alpha import MeanReversionAlpha
            from strategies.economic_alpha.calendar_alpha import CalendarAlpha

            for model in (MomentumAlpha(), MeanReversionAlpha(), CalendarAlpha()):
                _state.register_strategy(model.name, model)
            logger.info(
                "Default strategies registered: %s",
                list(_state.strategies.keys()),
            )
        except Exception as exc:
            logger.warning("Could not register default strategies: %s", exc)

        # Register the trained RL agent as an additional strategy, but only if
        # a saved model exists (it is produced by /rl/retrain). Skipped
        # silently otherwise so the system still runs on the rule-based
        # strategies until the RL agent has been trained at least once.
        try:
            from pathlib import Path as _Path
            from strategies.rl_agent.rl_trainer import RLTrainerConfig

            _rl_cfg   = RLTrainerConfig()
            _rl_model = _Path(_rl_cfg.model_dir) / f"{_rl_cfg.model_name}.zip"
            if _rl_model.exists() and "rl_agent" not in _state.strategies:
                from strategies.rl_agent.rl_alpha import RLAlpha

                _algo = "RecurrentPPO" if _rl_cfg.use_recurrent_ppo else "PPO"
                rl_alpha = RLAlpha(
                    model_path  = str(_rl_model),
                    algo        = _algo,
                    window_size = _rl_cfg.window_size,
                )
                _state.register_strategy(rl_alpha.name, rl_alpha)
                logger.info("RL agent registered from %s (%s).", _rl_model, _algo)
            elif not _rl_model.exists():
                logger.info(
                    "No trained RL model yet (%s) — RL agent not registered. "
                    "Train it via /rl/retrain.", _rl_model,
                )
        except Exception as exc:
            logger.warning("Could not register RL agent: %s", exc)

    @app.on_event("startup")
    def _attach_default_monitor():
        """Attach a TradingMonitor so the dashboard receives portfolio,
        equity-curve and trade data."""
        if _state.monitor is not None:
            return
        try:
            from monitoring.monitor import TradingMonitor, MonitorConfig
            from monitoring.metrics_store import MySQLConfig
            from config.settings import settings

            mon_cfg = MonitorConfig()
            try:
                if settings.mysql.enabled:
                    mon_cfg.mysql_config = MySQLConfig.from_env()
            except Exception:
                pass

            _state.attach_monitor(TradingMonitor(config=mon_cfg))
            logger.info("TradingMonitor attached.")
        except Exception as exc:
            logger.warning("Could not attach monitor: %s", exc)

    @app.on_event("startup")
    def _start_rl_always_on():
        """Keep RL retraining loop active while the API process is running."""
        try:
            from config.settings import settings as _cfg
            if not _cfg.trading.rl_always_on:
                return

            from research.training.retrain_service import RetrainService

            svc = getattr(_state, "retrain_service", None)
            if svc is None:
                svc = RetrainService()
                _state.retrain_service = svc

            # Reuse running trader context when available; otherwise defaults.
            symbol = (_cfg.trading.symbols[0] if _cfg.trading.symbols else "EURUSD")
            timeframe = _cfg.trading.timeframe
            feature_config = None
            if _state.trader is not None:
                cfg = getattr(_state.trader, "cfg", None)
                if cfg is not None:
                    syms = getattr(cfg, "symbols", None)
                    symbol = (syms[0] if syms else None) or symbol
                    timeframe = getattr(cfg, "timeframe", None) or timeframe
                    feature_config = getattr(cfg, "feature_config", None)

            # RL always trains on its own dedicated timeframe (H1 by default),
            # independent of the M1/M5 live-trading timeframe.
            rl_tf = _cfg.trading.rl_timeframe or "H1"
            provider = _make_rl_feature_provider(
                symbol=symbol,
                timeframe=rl_tf,
                n_bars=_cfg.trading.rl_history_bars,   # 0 = auto all available
                feature_config=feature_config,
            )

            if not svc.is_running():
                svc.start(
                    train_ml=False,
                    train_rl=True,
                    rl_timesteps=max(1_000, int(_cfg.trading.rl_timesteps)),
                    rl_continuous=True,
                    rl_interval_s=max(60, int(_cfg.trading.rl_interval_s)),
                    rl_symbol=symbol,
                    rl_timeframe=timeframe,
                    rl_feature_provider=provider,
                )
                logger.info(
                    "RL always-on loop started (symbol=%s tf=%s interval=%ss timesteps=%s).",
                    symbol,
                    timeframe,
                    int(_cfg.trading.rl_interval_s),
                    int(_cfg.trading.rl_timesteps),
                )
        except Exception as exc:
            logger.warning("Could not start RL always-on loop: %s", exc)

    return app


# ---------------------------------------------------------------------------
# WebSocket payload builders (module-level so they're accessible)
# ---------------------------------------------------------------------------

def _ws_build_snapshot(state: "AppState") -> dict:
    snap: dict = {
        "portfolio": {},
        "equity_curve": [],
        "recent_trades": [],
    }
    if state.monitor:
        try:
            snap = state.monitor.get_summary()
            snap["equity_curve"]  = state.monitor.get_equity_curve(limit=300)
            snap["recent_trades"] = state.monitor.get_recent_trades(limit=50)
            snap["portfolio"] = _ws_get_portfolio(state)
        except Exception:
            pass
    snap["positions"]  = _ws_get_positions(state)
    snap["strategies"] = _ws_get_strategies(state)
    snap["account"]    = _ws_get_account(state)
    snap["status"]     = _ws_get_status(state)
    return snap


def _ws_build_tick(state: "AppState") -> dict:
    payload: dict = {}
    if state.monitor:
        try:
            payload["equity_curve"] = state.monitor.get_equity_curve(limit=10)
            payload["portfolio"]   = _ws_get_portfolio(state)
            payload["risk"]        = state.monitor.get_latest_risk() or {}
            payload["trades"]      = state.monitor.get_recent_trades(limit=5)
        except Exception:
            pass
    payload["positions"]  = _ws_get_positions(state)
    payload["strategies"] = _ws_get_strategies(state)
    payload["account"] = _ws_get_account(state)
    payload["status"]  = _ws_get_status(state)
    return payload


def _ws_get_portfolio(state: "AppState") -> dict:
    """Normalised live portfolio snapshot for the dashboard."""
    if state.monitor:
        try:
            summary = state.monitor.get_summary()
            risk = summary.get("risk") or {}
            dd = summary.get("drawdown") or {}
            return {
                "equity":       float(summary.get("equity", 0.0)),
                "balance":      float(summary.get("balance", 0.0)),
                "daily_pnl":    float(risk.get("daily_pnl", 0.0)),
                "daily_pnl_pct": float(risk.get("daily_pnl_pct", 0.0)),
                "drawdown_pct": float(dd.get("current_drawdown_pct", 0.0)),
                "high_water":   float(dd.get("high_water_mark", 0.0)),
                "n_positions":  int(risk.get("n_positions", 0)),
                "risk":         risk,
            }
        except Exception:
            pass
    return {}


def _ws_get_positions(state: "AppState") -> list:
    if state.trader is not None and hasattr(state.trader, "get_live_positions"):
        try:
            return state.trader.get_live_positions()
        except Exception:
            pass
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


def _ws_get_account(state: "AppState") -> dict:
    """Real MT5 account snapshot (balance, equity, margin, currency).

    Returns an empty dict when MT5 is not connected (e.g. no live trader
    running yet), so the dashboard simply falls back to internal figures.
    """
    if state.execution_engine:
        try:
            info = state.execution_engine.get_account_info()
            if info:
                return info
        except Exception:
            pass
    return {}


def _ws_get_status(state: "AppState") -> dict:
    """Lightweight trading-loop status for the dashboard controls."""
    try:
        return {
            "running":          state.is_trader_running(),
            "emergency_active": state.is_emergency_active(),
        }
    except Exception:
        return {"running": False, "emergency_active": False}


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
    Reads configuration from `.env` (see config.settings). Override by
    calling create_app() manually with explicit arguments.
    """
    from config.settings import settings
    return create_app(
        api_key      = settings.api.api_key,
        cors_origins = settings.cors.origins,
        title        = settings.api.title,
        version      = settings.api.version,
    )


if _FASTAPI_OK:
    app = _build_default_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from config.settings import settings

    parser = argparse.ArgumentParser(description="Quant Fund Trading API")
    parser.add_argument("--host",    default=settings.api.host,    help="Bind host")
    parser.add_argument("--port",    default=settings.api.port, type=int, help="Bind port")
    parser.add_argument("--api-key", default=settings.api.api_key,
                        help="API key (or set QUANT_API_KEY in .env)")
    parser.add_argument("--reload",  action="store_true", default=settings.api.reload,
                        help="Auto-reload on change")
    args = parser.parse_args()

    if not _FASTAPI_OK:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        raise SystemExit(1)

    app = create_app(
        api_key      = args.api_key,
        cors_origins = settings.cors.origins,
        title        = settings.api.title,
        version      = settings.api.version,
    )

    print(f"\n  Quant Fund API  →  http://{args.host}:{args.port}/docs\n")
    uvicorn.run(
        app,
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        workers = 1,       # single worker = shared AppState
    )
