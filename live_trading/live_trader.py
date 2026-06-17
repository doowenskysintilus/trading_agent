"""
LiveTrader
==========
Main trading orchestrator for the quant-fund-ai system.

Full cycle (executed on every scheduled tick)
----------------------------------------------
  1. fetch_data        → raw OHLCV bars from MT5 (per symbol)
  2. compute_features  → FeatureEngineer → feature matrix
  3. run_alpha_models  → list[AlphaSignal]  (all registered strategies)
  4. aggregate_signals → AggregatedDecision (SignalAggregator)
  5. apply_risk        → RiskDecision       (RiskEngine)
  6. allocate_capital  → AllocationResult   (PortfolioAllocator)
  7. execute_trades    → list[ExecutionResult] (MT5ExecutionEngine)
  8. log & dispatch    → structured log + monitoring hooks

Safety mechanisms
-----------------
  Emergency Stop   — threading.Event + on-disk flag file
                     Triggered programmatically or by SIGTERM/SIGINT
  Auto-recovery    — exponential backoff (base^attempt, capped at 5 min)
                     Consecutive-success counter resets retry budget
  Watchdog         — per-cycle timeout kills a stalled cycle

Monitoring hooks
----------------
  Register any callable(TradingCycleResult) via register_hook().
  Built-in hooks: JsonFileLogger (written to log_dir).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from engines.execution_engine.execution_engine import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    MT5ExecutionEngine,
)
from engines.portfolio_engine.portfolio_engine import (
    AllocationMethod,
    AllocationResult,
    PortfolioAllocator,
)
from engines.risk_engine.risk_engine import (
    OpenPosition,
    PortfolioState,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    TradeOrder,
)
from engines.signal_engine.signal_aggregator import (
    AggregatedDecision,
    SignalAggregator,
)
from research.alpha_models.base import AlphaModel, AlphaSignal, SignalType
from research.feature_store.feature_engineer import FeatureConfig, FeatureEngineer
from live_trading.experience_collector import ExperienceCollector, extract_observation
from strategies.ml_agent.win_classifier import WinClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MT5 timeframe map (loaded lazily)
# ---------------------------------------------------------------------------

_MT5_TF_MAP: dict[str, int] = {}

def _get_mt5_tf(timeframe: str) -> int:
    global _MT5_TF_MAP
    if not _MT5_TF_MAP:
        try:
            import MetaTrader5 as mt5
            _MT5_TF_MAP = {
                "M1":  mt5.TIMEFRAME_M1,
                "M5":  mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1":  mt5.TIMEFRAME_H1,
                "H4":  mt5.TIMEFRAME_H4,
                "D1":  mt5.TIMEFRAME_D1,
                "W1":  mt5.TIMEFRAME_W1,
            }
        except ImportError:
            _MT5_TF_MAP = {}
    return _MT5_TF_MAP.get(timeframe.upper(), 16385)  # 16385 = H1 fallback


# Ordered list of supported timeframes (shortest → longest).
_TF_ORDER: list[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]


def _derive_htf(timeframe: str, steps: int = 2) -> str:
    """Pick a higher timeframe `steps` positions above the entry timeframe.

    Example: H1 + 2 → D1, M15 + 2 → H1. Caps at the longest timeframe (W1).
    """
    tf = timeframe.upper()
    if tf not in _TF_ORDER:
        return "D1"
    idx = min(_TF_ORDER.index(tf) + steps, len(_TF_ORDER) - 1)
    return _TF_ORDER[idx]


def _fmt_meta(meta: dict) -> str:
    """Compact one-line render of a signal's diagnostic metadata."""
    if not meta:
        return ""
    parts = []
    for k, v in meta.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Cycle status
# ---------------------------------------------------------------------------

class CycleStatus(Enum):
    OK            = auto()
    NO_SIGNAL     = auto()
    ML_FILTERED   = auto()
    RISK_BLOCKED  = auto()
    EXECUTION_ERR = auto()
    DATA_ERROR    = auto()
    EMERGENCY     = auto()
    ERROR         = auto()


# ---------------------------------------------------------------------------
# Cycle result
# ---------------------------------------------------------------------------

@dataclass
class TradingCycleResult:
    """Complete output of a single trading cycle for one symbol."""

    cycle_id:        int
    symbol:          str
    timestamp:       datetime
    status:          CycleStatus

    # Per-stage outputs (None if stage was not reached)
    n_bars:          int                           = 0
    n_features:      int                           = 0
    signals:         Optional[list[AlphaSignal]]   = None
    decision:        Optional[AggregatedDecision]  = None
    risk_decision:   Optional[RiskDecision]        = None
    allocation:      Optional[AllocationResult]    = None
    executions:      Optional[list[ExecutionResult]] = None
    portfolio_state: Optional[PortfolioState]      = None

    duration_ms:     float                         = 0.0
    error:           Optional[str]                 = None

    def to_dict(self) -> dict:
        return {
            "cycle_id":       self.cycle_id,
            "symbol":         self.symbol,
            "timestamp":      self.timestamp.isoformat(),
            "status":         self.status.name,
            "n_bars":         self.n_bars,
            "n_features":     self.n_features,
            "decision":       self.decision.signal.name if self.decision else None,
            "confidence":     round(self.decision.confidence, 4) if self.decision else None,
            "risk_decision":  self.risk_decision.verdict.name if self.risk_decision else None,
            "n_executions":   len(self.executions) if self.executions else 0,
            "duration_ms":    round(self.duration_ms, 2),
            "error":          self.error,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LiveTraderConfig:
    """Full configuration for the live trading orchestrator."""

    # ---- Trading universe -----------------------------------------------
    symbols:   list[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe: str       = "H1"
    warmup_bars: int     = 200     # 0 => auto max MT5 bars for timeframe

    # ---- Multi-timeframe trend filter -----------------------------------
    # Confirms each entry signal against the trend of a higher timeframe
    # (e.g. trade H1 signals only in the direction of the D1 trend).
    htf_enabled:   bool = True
    htf_timeframe: str  = ""    # "" → auto-derived from `timeframe` (+2 steps)
    htf_bars:      int  = 200   # 0 => auto max MT5 bars for HTF
    htf_fast_ema:  int  = 20    # fast EMA period for the HTF trend
    htf_slow_ema:  int  = 50    # slow EMA period for the HTF trend

    # ---- Diagnostics ----------------------------------------------------
    # When True, each cycle logs WHY a signal is HOLD/actionable (indicator
    # values: EMA gap, RSI, z-score, and the higher-timeframe trend).
    verbose_signals: bool = True

    # ---- Scheduling ------------------------------------------------------
    cycle_interval_seconds: int = 3600    # 3600 = hourly for H1
    cycle_timeout_seconds:  int = 120     # max time per cycle before kill

    # ---- MT5 connection --------------------------------------------------
    mt5_login:    int  = 0
    mt5_password: str  = ""
    mt5_server:   str  = ""
    mt5_path:     str  = ""
    magic_number: int  = 20260524

    # ---- Feature engineering --------------------------------------------
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # ---- Signal aggregation ---------------------------------------------
    min_confidence: float = 0.45    # signals below this are ignored

    # ---- Risk engine ----------------------------------------------------
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # ---- Portfolio allocation -------------------------------------------
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY
    initial_balance:   float            = 100_000.0

    # ---- Auto-recovery --------------------------------------------------
    max_retries:           int   = 5
    retry_backoff_base:    float = 2.0   # seconds; delay = base ^ attempt
    max_retry_delay:       float = 300.0 # cap at 5 minutes
    success_reset_count:   int   = 10    # consecutive successes to reset retries

    # ---- Emergency stop -------------------------------------------------
    stop_flag_path: str = "data/storage/EMERGENCY_STOP"

    # ---- I/O -----------------------------------------------------------
    log_dir: str = "data/storage/logs/live"

    # ---- Training-data collection --------------------------------------
    # When True, every executed trade's entry observation + realized outcome
    # is written to dataset_dir (one record per trade — no per-cycle bloat),
    # for offline RL / ML retraining.
    collect_experiences: bool = True
    dataset_dir: str = "data/storage/datasets"

    # ---- ML win/loss confidence filter ---------------------------------
    # When enabled, the trained WinClassifier predicts P(win) from the entry
    # observation; trades below ml_min_win_proba are skipped. If no model is
    # trained yet (or sklearn missing), the filter passes everything through.
    ml_filter_enabled: bool = True
    ml_min_win_proba:  float = 0.50
    ml_model_path:     str   = "data/storage/models/win_classifier.joblib"


# ---------------------------------------------------------------------------
# Emergency stop manager
# ---------------------------------------------------------------------------

class EmergencyStopManager:
    """
    Thread-safe emergency stop with two independent triggers:
      1. In-memory threading.Event (instant, per-process)
      2. On-disk flag file (persists across restarts)

    Either trigger independently halts the trading loop.
    """

    def __init__(self, flag_path: str) -> None:
        self._flag_path = Path(flag_path)
        self._event     = threading.Event()

    @property
    def is_active(self) -> bool:
        return self._event.is_set() or self._flag_path.exists()

    def trigger(self, reason: str = "manual") -> None:
        logger.critical("EMERGENCY STOP triggered — reason: %s", reason)
        self._event.set()
        self._flag_path.parent.mkdir(parents=True, exist_ok=True)
        self._flag_path.write_text(
            f"{datetime.now(timezone.utc).isoformat()} | {reason}"
        )

    def clear(self) -> None:
        """Reset the emergency stop (requires explicit operator action)."""
        self._event.clear()
        if self._flag_path.exists():
            self._flag_path.unlink()
        logger.warning("Emergency stop cleared.")

    def check_or_raise(self) -> None:
        if self.is_active:
            raise RuntimeError("Emergency stop is active — trading halted.")


# ---------------------------------------------------------------------------
# Built-in JSON monitoring hook
# ---------------------------------------------------------------------------

class JsonFileLogger:
    """Appends each TradingCycleResult as a JSON line to a daily file."""

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, result: TradingCycleResult) -> None:
        date_str  = result.timestamp.strftime("%Y%m%d")
        log_file  = self._log_dir / f"cycles_{date_str}.jsonl"
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except OSError as exc:
            logger.error("JsonFileLogger write failed: %s", exc)


# ---------------------------------------------------------------------------
# Monitoring hook type
# ---------------------------------------------------------------------------

MonitoringHook = Callable[[TradingCycleResult], None]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class LiveTrader:
    """
    Live trading orchestrator.

    Parameters
    ----------
    config : LiveTraderConfig

    Quick start
    -----------
    >>> from live_trading.live_trader import LiveTrader, LiveTraderConfig
    >>> from strategies.momentum.momentum_alpha import MomentumAlpha
    >>> from strategies.mean_reversion.mean_reversion_alpha import MeanReversionAlpha
    >>>
    >>> cfg    = LiveTraderConfig(symbols=["EURUSD", "GBPUSD"])
    >>> trader = LiveTrader(cfg)
    >>> trader.register_strategy(MomentumAlpha(),    weight=1.0)
    >>> trader.register_strategy(MeanReversionAlpha(), weight=0.8)
    >>> trader.start()   # blocking
    """

    def __init__(self, config: LiveTraderConfig | None = None) -> None:
        self.cfg = config or LiveTraderConfig()

        # Resolve the higher timeframe once (auto-derive if left blank).
        if self.cfg.htf_enabled and not self.cfg.htf_timeframe:
            self.cfg.htf_timeframe = _derive_htf(self.cfg.timeframe)

        # ---- Sub-engines ------------------------------------------------
        self._feature_eng  = FeatureEngineer(self.cfg.feature_config)
        self._aggregator   = SignalAggregator()
        self._risk_engine  = RiskEngine(self.cfg.risk_config)
        self._allocator    = PortfolioAllocator(
            total_capital   = self.cfg.initial_balance,
        )
        self._exec_engine  = MT5ExecutionEngine(
            ExecutionConfig(
                login        = self.cfg.mt5_login,
                password     = self.cfg.mt5_password,
                server       = self.cfg.mt5_server,
                mt5_path     = self.cfg.mt5_path,
                magic_number = self.cfg.magic_number,
            )
        )

        # ---- State -------------------------------------------------------
        self._portfolio_state = PortfolioState(
            equity             = self.cfg.initial_balance,
            balance            = self.cfg.initial_balance,
            peak_equity        = self.cfg.initial_balance,
            daily_start_equity = self.cfg.initial_balance,
            open_positions     = [],
        )
        # Daily-loss / drawdown references are re-anchored to the REAL account
        # equity on the first live read (see _sync_account_state). Until then
        # they use the configured initial_balance.
        self._account_anchored = False
        self._anchor_day: Optional[str] = None
        self._cycle_id         = 0
        self._consecutive_ok   = 0
        self._retry_count      = 0
        self._running          = False
        self._stop_event       = threading.Event()
        self._background_thread: Optional[threading.Thread] = None
        self._auto_bars_cache: dict[tuple[str, str], int] = {}

        # ---- Emergency stop ---------------------------------------------
        self._emergency = EmergencyStopManager(self.cfg.stop_flag_path)

        # ---- Strategies -------------------------------------------------
        self._strategies: list[AlphaModel] = []

        # ---- Monitoring hooks -------------------------------------------
        self._hooks: list[MonitoringHook] = [
            JsonFileLogger(self.cfg.log_dir)
        ]
        # Direct reference to a TradingMonitor (if registered) so closed
        # trades can be recorded with their realized PnL. Detected in
        # register_hook by the presence of a record_trade() method.
        self._monitor = None

        # Open trades pending PnL reconciliation: ticket → entry metadata.
        # When a position disappears from MT5 (closed by SL/TP), its realized
        # PnL is fetched and fed to the learning components.
        self._open_trades: dict[int, dict] = {}

        # Latest feature observation per symbol (captured each cycle in
        # stage 2), used to label each placed order with its entry state.
        self._feature_snapshot: dict[str, dict] = {}

        # Training-data collector — one record per closed trade.
        self._experience = (
            ExperienceCollector(self.cfg.dataset_dir)
            if self.cfg.collect_experiences else None
        )

        # ML win/loss confidence filter. Loads the latest trained model if it
        # exists; reloaded automatically when the model file changes on disk
        # (e.g. after a /rl/retrain run) so live trading uses fresh weights.
        self._win_clf: Optional[WinClassifier] = None
        self._win_clf_mtime: float = 0.0
        if self.cfg.ml_filter_enabled:
            self._win_clf = WinClassifier(self.cfg.ml_model_path)
            self._maybe_reload_classifier()

        # ---- OHLCV cache (symbol → DataFrame) ---------------------------
        self._data_cache: dict[str, pd.DataFrame] = {}

        self._setup_signal_handlers()
        logger.info("LiveTrader initialised — symbols=%s tf=%s",
                    self.cfg.symbols, self.cfg.timeframe)

    # ------------------------------------------------------------------
    # Strategy and hook registration
    # ------------------------------------------------------------------

    def register_strategy(
        self,
        model: AlphaModel,
        weight: float = 1.0,
        initial_capital: float | None = None,
    ) -> "LiveTrader":
        """Register an AlphaModel and its signal weight."""
        self._strategies.append(model)
        self._aggregator.set_weight(model.name, weight)
        self._allocator.register(
            name           = model.name,
            initial_weight = weight,
        )
        logger.info("Strategy registered: %s (weight=%.2f)", model.name, weight)
        return self

    def register_hook(self, hook: MonitoringHook) -> "LiveTrader":
        """Register a monitoring callback. Called after every cycle."""
        self._hooks.append(hook)
        # If the hook can record closed trades (TradingMonitor), keep a direct
        # reference so realized PnL can be persisted on trade close.
        if self._monitor is None and callable(getattr(hook, "record_trade", None)):
            self._monitor = hook
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the trading loop (blocking). Press Ctrl+C to stop."""
        if self._emergency.is_active:
            logger.critical(
                "Emergency stop is active. Run trader.emergency_stop.clear() "
                "before starting."
            )
            return
        if not self._strategies:
            raise RuntimeError("No strategies registered. Call register_strategy() first.")

        self._running = True
        logger.info("LiveTrader starting — %d strategies", len(self._strategies))

        try:
            with self._exec_engine:
                self._main_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received — shutting down.")
        finally:
            self._running = False
            logger.info("LiveTrader stopped.")

    def start_background(self) -> threading.Thread:
        """Start the trading loop in a daemon thread."""
        t = threading.Thread(target=self.start, daemon=True, name="LiveTrader")
        t.start()
        self._background_thread = t
        return t

    def stop(self, reason: str = "operator") -> None:
        """Graceful stop — finishes the current cycle."""
        logger.info("Stop requested: %s", reason)
        self._stop_event.set()
        self._running = False

    def emergency_stop(self, reason: str = "manual") -> None:
        """Immediate halt + persist flag to disk."""
        self._emergency.trigger(reason)
        self.stop(reason=f"emergency: {reason}")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def emergency(self) -> EmergencyStopManager:
        """Access the emergency stop manager directly."""
        return self._emergency

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        """
        Outer scheduling loop.
        Runs each symbol through a full cycle, then sleeps until the next
        interval boundary.
        """
        while self._running and not self._stop_event.is_set():

            # Emergency check (event + file)
            if self._emergency.is_active:
                logger.critical("Emergency stop active — halting main loop.")
                break

            cycle_start = time.monotonic()
            self._cycle_id += 1
            logger.info("─── Cycle #%d ───────────────────────────────", self._cycle_id)

            # Run each symbol
            for symbol in self.cfg.symbols:
                if self._stop_event.is_set() or self._emergency.is_active:
                    break
                self._run_with_recovery(symbol)

            # ---- Record PnL of trades closed by SL/TP (learning loop) ---
            self._reconcile_closed_trades()

            # ---- Rebalance allocations after all symbols ----------------
            self._rebalance_portfolio()

            # ---- Update trailing stops ----------------------------------
            self._update_trailing_stops()

            # ---- Sleep until next interval boundary ---------------------
            elapsed  = time.monotonic() - cycle_start
            sleep_s  = max(0.0, self.cfg.cycle_interval_seconds - elapsed)
            logger.info(
                "Cycle #%d complete in %.1fs — sleeping %.1fs",
                self._cycle_id, elapsed, sleep_s,
            )
            self._interruptible_sleep(sleep_s)

    # ------------------------------------------------------------------
    # Auto-recovery wrapper
    # ------------------------------------------------------------------

    def _run_with_recovery(self, symbol: str) -> None:
        """
        Run one symbol cycle, retrying on non-fatal errors with
        exponential backoff.
        """
        for attempt in range(self.cfg.max_retries + 1):
            try:
                result = self._run_cycle(symbol)
                self._dispatch_hooks(result)

                if result.status not in (CycleStatus.ERROR, CycleStatus.DATA_ERROR):
                    self._consecutive_ok += 1
                    if self._consecutive_ok >= self.cfg.success_reset_count:
                        self._retry_count  = 0
                        self._consecutive_ok = 0
                return

            except RuntimeError as exc:
                if "Emergency stop" in str(exc):
                    raise
                self._handle_recovery(symbol, exc, attempt)
            except Exception as exc:          # pylint: disable=broad-except
                self._handle_recovery(symbol, exc, attempt)

        # Exhausted retries
        logger.error(
            "[%s] Exhausted %d retries — skipping this cycle.",
            symbol, self.cfg.max_retries,
        )

    def _handle_recovery(self, symbol: str, exc: Exception, attempt: int) -> None:
        """Log, apply backoff, decide whether to escalate to emergency stop."""
        self._consecutive_ok = 0
        self._retry_count   += 1
        delay = min(
            self.cfg.retry_backoff_base ** (attempt + 1),
            self.cfg.max_retry_delay,
        )
        logger.error(
            "[%s] Cycle error (attempt %d/%d) — %s: %s — retrying in %.0fs",
            symbol, attempt + 1, self.cfg.max_retries,
            type(exc).__name__, exc, delay,
        )
        logger.debug(traceback.format_exc())

        # Escalate to emergency stop if retries exhausted consistently
        if self._retry_count >= self.cfg.max_retries * 3:
            self.emergency_stop(
                f"Auto-escalation after {self._retry_count} consecutive errors"
            )

        self._interruptible_sleep(delay)

    # ------------------------------------------------------------------
    # Full cycle for one symbol
    # ------------------------------------------------------------------

    def _run_cycle(self, symbol: str) -> TradingCycleResult:
        """Execute all 8 stages for one symbol."""
        ts    = datetime.now(timezone.utc)
        t0    = time.monotonic()
        result = TradingCycleResult(
            cycle_id  = self._cycle_id,
            symbol    = symbol,
            timestamp = ts,
            status    = CycleStatus.ERROR,
        )

        try:
            self._emergency.check_or_raise()

            # Refresh equity/balance from the live account so EVERY cycle
            # (including NO_SIGNAL / RISK_BLOCKED) reports the real portfolio
            # state. Without this, early-return cycles leave portfolio_state
            # = None → monitor sees equity 0 → false 100% drawdown alerts.
            self._sync_account_state()
            result.portfolio_state = self._portfolio_state

            # 1. Fetch data -----------------------------------------------
            data = self._fetch_data(symbol)
            if data is None or len(data) < self.cfg.warmup_bars // 2:
                result.status = CycleStatus.DATA_ERROR
                result.error  = "Insufficient bars fetched"
                return result
            result.n_bars = len(data)

            # 2. Compute features -----------------------------------------
            features = self._compute_features(data)
            result.n_features = features.shape[1]
            # Snapshot the entry observation so any order placed this cycle
            # can be labelled with the exact state that produced it.
            if self._experience is not None:
                self._feature_snapshot[symbol] = extract_observation(features)

            # 3. Run alpha models -----------------------------------------
            if self.cfg.verbose_signals:
                logger.info("[%s] analysing %d bars (%s)…",
                            symbol, len(data), self.cfg.timeframe)
            signals = self._run_alpha_models(features)
            result.signals = signals
            if not signals:
                result.status = CycleStatus.NO_SIGNAL
                return result

            # 4. Aggregate signals ----------------------------------------
            decision = self._aggregate_signals(signals)
            result.decision = decision
            if self.cfg.verbose_signals:
                logger.info(
                    "    AGG → %-4s conf=%.3f (min=%.2f)",
                    decision.signal.name, decision.confidence,
                    self.cfg.min_confidence,
                )
            if decision.signal == SignalType.HOLD:
                if self.cfg.verbose_signals:
                    logger.info("    [%s] HOLD — strategies disagree / no setup", symbol)
                result.status = CycleStatus.NO_SIGNAL
                return result
            if decision.confidence < self.cfg.min_confidence:
                if self.cfg.verbose_signals:
                    logger.info(
                        "    [%s] %s skipped — confidence %.3f < min %.2f",
                        symbol, decision.signal.name, decision.confidence,
                        self.cfg.min_confidence,
                    )
                result.status = CycleStatus.NO_SIGNAL
                return result

            # 4b. Multi-timeframe trend filter ----------------------------
            # Only trade in the direction of the higher-timeframe trend.
            if self.cfg.htf_enabled:
                htf_trend = self._htf_trend(symbol)
                if self.cfg.verbose_signals:
                    logger.info(
                        "    HTF(%s) trend = %s",
                        self.cfg.htf_timeframe, htf_trend.name,
                    )
                if htf_trend != SignalType.HOLD and decision.signal != htf_trend:
                    logger.info(
                        "    [%s] HTF filter: %s entry vs %s trend \u2192 skip",
                        symbol, decision.signal.name, htf_trend.name,
                    )
                    result.status = CycleStatus.NO_SIGNAL
                    return result

            # 4c. ML win/loss confidence filter ---------------------------
            # The classifier learns from the system's own closed-trade results
            # and predicts P(win) for this entry. Skip trades it judges weak.
            if self._win_clf is not None:
                self._maybe_reload_classifier()
                obs = self._feature_snapshot.get(symbol)
                if obs is None:
                    obs = extract_observation(features)
                win_p = self._win_clf.predict_win_proba(obs)
                if win_p is not None:
                    if self.cfg.verbose_signals:
                        logger.info(
                            "    ML P(win) = %.3f (min=%.2f)",
                            win_p, self.cfg.ml_min_win_proba,
                        )
                    if win_p < self.cfg.ml_min_win_proba:
                        logger.info(
                            "    [%s] ML filter: P(win) %.3f < min %.2f \u2192 skip",
                            symbol, win_p, self.cfg.ml_min_win_proba,
                        )
                        result.status = CycleStatus.ML_FILTERED
                        return result

            # 5. Apply risk engine ----------------------------------------
            trade_order  = self._build_trade_order(symbol, decision)
            risk_decision = self._apply_risk(trade_order)
            result.risk_decision = risk_decision
            if risk_decision.verdict.name in ("BLOCK",):
                result.status = CycleStatus.RISK_BLOCKED
                logger.info("[%s] Risk BLOCK — %s", symbol, risk_decision.blocking_reasons)
                return result

            # 6. Allocate capital -----------------------------------------
            allocation = self._allocate_capital()
            result.allocation = allocation

            # 7. Execute trades -------------------------------------------
            executions = self._execute_trade(decision, trade_order, risk_decision)
            result.executions = executions

            # 8. Post-cycle updates ---------------------------------------
            self._post_cycle_update(symbol, decision, executions)
            result.portfolio_state = self._portfolio_state

            result.status = CycleStatus.OK

        except RuntimeError:
            raise
        except Exception as exc:
            result.status = CycleStatus.ERROR
            result.error  = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] Unhandled error in cycle", symbol)

        finally:
            result.duration_ms = (time.monotonic() - t0) * 1000
            self._log_cycle(result)

        return result

    # ------------------------------------------------------------------
    # Stage 1 — Fetch data
    # ------------------------------------------------------------------

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch the last `warmup_bars` OHLCV bars from MT5.
        Returns a DataFrame with columns: open, high, low, close, volume.
        """
        try:
            import MetaTrader5 as mt5

            n_bars = self._resolve_fetch_bars(
                symbol=symbol,
                timeframe=self.cfg.timeframe,
                requested_bars=self.cfg.warmup_bars,
                min_bars=200,
            )

            rates = mt5.copy_rates_from_pos(
                symbol,
                _get_mt5_tf(self.cfg.timeframe),
                0,                          # start from the most recent
                n_bars,
            )
            if rates is None or len(rates) == 0:
                logger.warning("[%s] MT5 returned no data", symbol)
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={
                "tick_volume": "volume",
                "spread":      "spread",
            }).set_index("time")
            df.columns = [c.lower() for c in df.columns]
            self._data_cache[symbol] = df
            return df

        except ImportError:
            # MT5 not available — use cached data for testing
            if symbol in self._data_cache:
                logger.debug("[%s] MT5 unavailable, using cached data", symbol)
                return self._data_cache[symbol]
            logger.error("[%s] MT5 not installed and no cached data", symbol)
            return None

    # ------------------------------------------------------------------
    # Stage 1b — Higher-timeframe trend
    # ------------------------------------------------------------------

    def _htf_trend(self, symbol: str) -> SignalType:
        """
        Determine the higher-timeframe trend via EMA crossover.

        Returns BUY (uptrend), SELL (downtrend), or HOLD when the trend is
        neutral / undetermined (which lets the entry signal pass through).
        """
        try:
            import MetaTrader5 as mt5

            htf_bars = self._resolve_fetch_bars(
                symbol=symbol,
                timeframe=self.cfg.htf_timeframe,
                requested_bars=self.cfg.htf_bars,
                min_bars=max(self.cfg.htf_slow_ema + 2, 60),
            )

            rates = mt5.copy_rates_from_pos(
                symbol,
                _get_mt5_tf(self.cfg.htf_timeframe),
                0,
                htf_bars,
            )
            if rates is None or len(rates) < self.cfg.htf_slow_ema + 2:
                return SignalType.HOLD

            close = pd.Series(rates["close"], dtype=float)
            fast = close.ewm(span=self.cfg.htf_fast_ema, adjust=False).mean().iloc[-1]
            slow = close.ewm(span=self.cfg.htf_slow_ema, adjust=False).mean().iloc[-1]

            if fast > slow:
                return SignalType.BUY
            if fast < slow:
                return SignalType.SELL
            return SignalType.HOLD

        except ImportError:
            return SignalType.HOLD
        except Exception as exc:
            logger.warning("[%s] HTF trend computation failed: %s", symbol, exc)
            return SignalType.HOLD

    def _resolve_fetch_bars(
        self,
        *,
        symbol: str,
        timeframe: str,
        requested_bars: int,
        min_bars: int,
    ) -> int:
        """Resolve MT5 bar count for live fetches.

        requested_bars > 0: use requested value.
        requested_bars <= 0: auto-detect max available bars for symbol/timeframe,
        cache the result, and enforce a minimum floor.
        """
        req = int(requested_bars)
        if req > 0:
            return req

        key = (symbol, timeframe.upper())
        cached = self._auto_bars_cache.get(key)
        if cached is not None:
            return max(int(min_bars), int(cached))

        try:
            import MetaTrader5 as mt5

            tf = _get_mt5_tf(timeframe)
            term = mt5.terminal_info()
            maxbars = int(getattr(term, "maxbars", 0) or 0)
            hard_cap = maxbars if maxbars > 0 else 300_000

            probes = [2_000, 10_000, 50_000, 100_000, hard_cap]
            probes = sorted({p for p in probes if 0 < p <= hard_cap})

            best = 0
            for count in probes:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
                got = len(rates) if rates is not None else 0
                best = max(best, got)
                if got < count:
                    resolved = max(int(min_bars), int(got))
                    self._auto_bars_cache[key] = resolved
                    logger.info(
                        "[%s] Auto bars resolved for %s: %d",
                        symbol,
                        timeframe,
                        resolved,
                    )
                    return resolved

            resolved = max(int(min_bars), int(best))
            self._auto_bars_cache[key] = resolved
            logger.info("[%s] Auto bars resolved for %s: %d", symbol, timeframe, resolved)
            return resolved
        except Exception as exc:
            fallback = max(int(min_bars), 200)
            logger.warning(
                "[%s] Auto bars resolution failed for %s: %s (fallback=%d)",
                symbol,
                timeframe,
                exc,
                fallback,
            )
            return fallback

    # ------------------------------------------------------------------
    # ML confidence filter — hot-reload of the trained model
    # ------------------------------------------------------------------

    def _maybe_reload_classifier(self) -> None:
        """(Re)load the win/loss model when its file appears or changes.

        Lets the live trader pick up a freshly retrained model (after a
        /rl/retrain run) without a restart. Cheap stat() check per cycle.
        """
        if self._win_clf is None:
            return
        try:
            mtime = self._win_clf.model_path.stat().st_mtime
        except OSError:
            return  # no model file yet — filter passes everything through
        if mtime <= self._win_clf_mtime:
            return
        if self._win_clf.load():
            self._win_clf_mtime = mtime
            logger.info("WinClassifier model loaded (mtime=%.0f).", mtime)

    # ------------------------------------------------------------------
    # Stage 2 — Compute features
    # ------------------------------------------------------------------

    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run FeatureEngineer on raw OHLCV data."""
        features = self._feature_eng.compute(data)
        features = features.ffill().fillna(0.0)
        return features

    # ------------------------------------------------------------------
    # Stage 3 — Run alpha models
    # ------------------------------------------------------------------

    def _run_alpha_models(self, features: pd.DataFrame) -> list[AlphaSignal]:
        """Run all registered strategies; filter disabled or low-confidence."""
        signals: list[AlphaSignal] = []
        for model in self._strategies:
            if not model.enabled:
                continue
            try:
                sig = model.compute(features)
                signals.append(sig)
                if self.cfg.verbose_signals:
                    logger.info(
                        "    %-14s → %-4s conf=%.3f %s",
                        model.name, sig.signal.name, sig.confidence,
                        _fmt_meta(sig.metadata),
                    )
                else:
                    logger.debug(
                        "  [%s] → %s (conf=%.3f)",
                        model.name, sig.signal.name, sig.confidence,
                    )
            except Exception as exc:
                logger.warning("Strategy %s raised: %s", model.name, exc)
        return signals

    # ------------------------------------------------------------------
    # Stage 4 — Aggregate signals
    # ------------------------------------------------------------------

    def _aggregate_signals(self, signals: list[AlphaSignal]) -> AggregatedDecision:
        return self._aggregator.aggregate(signals)

    # ------------------------------------------------------------------
    # Stage 5 — Apply risk
    # ------------------------------------------------------------------

    def _build_trade_order(
        self,
        symbol: str,
        decision: AggregatedDecision,
    ) -> TradeOrder:
        equity      = self._portfolio_state.equity
        atr_proxy   = self._get_atr_proxy(symbol)
        entry_price = self._get_last_price(symbol)
        direction   = 1 if decision.signal == SignalType.BUY else -1

        # Initial volatility-based size: risk a fixed fraction of equity per
        # ATR-based stop distance. The RiskEngine re-caps this in Layer 4.
        risk_cfg    = self._risk_engine.config
        risk_amount = equity * risk_cfg.risk_per_trade_pct
        sl_distance = max(atr_proxy * risk_cfg.atr_multiplier, 1e-9)
        size        = round(risk_amount / sl_distance, 2)

        return TradeOrder(
            symbol      = symbol,
            strategy    = "aggregated",
            direction   = direction,
            size        = size,
            entry_price = entry_price,
            atr         = atr_proxy,
        )

    def _apply_risk(self, order: TradeOrder) -> RiskDecision:
        return self._risk_engine.evaluate_portfolio_risk(
            trades          = [order],
            portfolio_state = self._portfolio_state,
        )

    def _get_last_price(self, symbol: str) -> float:
        """Latest close price from cached OHLCV data (fallback 0.0)."""
        data = self._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return 0.0
        return float(data["close"].values[-1])

    def _get_atr_proxy(self, symbol: str) -> float:
        """Estimate ATR from cached data (last 14 bars)."""
        data = self._data_cache.get(symbol)
        if data is None or len(data) < 14:
            return 0.001   # safe fallback
        highs  = data["high"].values[-14:]
        lows   = data["low"].values[-14:]
        closes = data["close"].values[-14:]
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        )
        return float(np.mean(tr))

    # ------------------------------------------------------------------
    # Stage 6 — Allocate capital
    # ------------------------------------------------------------------

    def _allocate_capital(self) -> AllocationResult:
        return self._allocator.allocate(method=self.cfg.allocation_method)

    # ------------------------------------------------------------------
    # Stage 7 — Execute trades
    # ------------------------------------------------------------------

    def _execute_trade(
        self,
        decision: AggregatedDecision,
        order: TradeOrder,
        risk_decision: RiskDecision,
    ) -> list[ExecutionResult]:
        results = []
        try:
            # The RiskEngine already applied any size reduction (REDUCE verdict)
            # to the orders in approved_trades. Execute those directly.
            approved = risk_decision.approved_trades or [order]

            for approved_order in approved:
                result = self._exec_engine.execute(decision, approved_order)
                results.append(result)

                if result.status == ExecutionStatus.FILLED:
                    logger.info(
                        "\u2713 FILLED %s | ticket=%s | price=%.5f | size=%.2f",
                        approved_order.symbol,
                        result.ticket,
                        result.fill_price,
                        approved_order.size,
                    )
                    # Track this trade so its realized PnL can be recorded
                    # when the position closes (SL/TP). Attribute it to the
                    # strategies that voted in the trade's direction.
                    contributors = [
                        v["strategy"] for v in (decision.votes or [])
                        if v.get("counted") and v.get("signal") == decision.signal.value
                    ] or [m.name for m in self._strategies]
                    self._open_trades[result.ticket] = {
                        "symbol":      approved_order.symbol,
                        "strategies":  contributors,
                        "direction":   "BUY" if approved_order.direction == +1 else "SELL",
                        "size":        approved_order.size,
                        "entry_price": result.fill_price,
                        "entry_ts":    datetime.now(timezone.utc),
                        "entry_cycle": self._cycle_id,
                        "confidence":  decision.confidence,
                        # Entry observation for training (only kept if the
                        # experience collector is enabled).
                        "features":    self._feature_snapshot.get(approved_order.symbol)
                                       if self._experience is not None else None,
                    }
                else:
                    logger.warning(
                        "\u2717 REJECTED %s | status=%s | retries=%d | reason: %s",
                        approved_order.symbol, result.status, result.retries,
                        result.message or "(no detail)",
                    )
        except Exception as exc:
            logger.error("Execution error for %s: %s", order.symbol, exc)

        return results

    # ------------------------------------------------------------------
    # Post-cycle updates
    # ------------------------------------------------------------------

    def _sync_account_state(self) -> None:
        """Refresh equity/balance from the live MT5 account (best-effort).

        Called at the start of every cycle so portfolio_state always carries
        a real, non-zero equity even on cycles that take no trade.

        On the first real read (and at each new day) the daily-loss and
        drawdown reference points are re-anchored to the *actual* account
        equity. Without this, daily_start_equity stays at the configured
        initial_balance (e.g. 100k) while real equity is ~10k → the risk
        engine sees a fake 90% daily loss and trips the kill switch.
        """
        try:
            account = self._exec_engine.get_account_info()
            if not account:
                return
            eq  = account.get("equity")
            bal = account.get("balance")
            if eq is not None and eq > 0:
                self._portfolio_state.equity = eq
            if bal is not None and bal > 0:
                self._portfolio_state.balance = bal

            # Track the running high-water mark for drawdown calculations.
            if eq is not None and eq > self._portfolio_state.peak_equity:
                self._portfolio_state.peak_equity = eq

            # Re-anchor references to the real equity on first read / new day.
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            if eq is not None and eq > 0 and (
                not self._account_anchored or self._anchor_day != today
            ):
                self._portfolio_state.daily_start_equity = eq
                if not self._account_anchored:
                    self._portfolio_state.peak_equity = eq
                self._account_anchored = True
                self._anchor_day = today
                logger.info(
                    "Risk references anchored to live equity: %.2f", eq
                )
        except Exception as exc:
            logger.debug("Account equity sync failed: %s", exc)

    def _post_cycle_update(
        self,
        symbol: str,
        decision: AggregatedDecision,
        executions: list[ExecutionResult],
    ) -> None:
        """Refresh portfolio state from MT5 account info."""
        try:
            account = self._exec_engine.get_account_info()
            if account:
                self._portfolio_state.equity  = account.get("equity", self._portfolio_state.equity)
                self._portfolio_state.balance = account.get("balance", self._portfolio_state.balance)

            positions = self._exec_engine.get_open_positions()
            if positions is not None:
                self._portfolio_state.open_positions = [
                    OpenPosition(
                        symbol        = p.symbol,
                        strategy      = "live",
                        direction     = p.direction,
                        size          = p.volume,
                        entry_price   = p.open_price,
                        current_price = p.current_price,
                    )
                    for p in positions
                ]
        except Exception as exc:
            logger.warning("Portfolio state update failed: %s", exc)

    def _rebalance_portfolio(self) -> None:
        """Trigger allocation rebalance after every full symbol sweep."""
        try:
            result = self._allocator.rebalance(method=self.cfg.allocation_method)
            if result.rebalanced:
                logger.info("Portfolio rebalanced — allocations: %s",
                            {k: round(v, 0) for k, v in result.allocations.items()})
        except Exception as exc:
            logger.warning("Rebalance failed: %s", exc)

    # ------------------------------------------------------------------
    # Closed-trade reconciliation — the learning feedback loop
    # ------------------------------------------------------------------

    def _reconcile_closed_trades(self) -> None:
        """Detect positions closed by MT5 (SL/TP), record their realized PnL,
        and feed it to the learning components so the system improves:

          * SignalAggregator.record_trade  -> reweights strategies by win-rate
          * PortfolioAllocator.record_pnl  -> reallocates capital by Sharpe
          * TradingMonitor.record_trade    -> persists the closed trade

        Runs once per cycle sweep. Compares tracked entry tickets against the
        positions still open in MT5; any missing ticket has been closed.
        """
        if not self._open_trades:
            return

        try:
            open_now = {p.ticket for p in self._exec_engine.get_open_positions()}
        except Exception as exc:
            logger.debug("Reconciliation skipped — cannot list positions: %s", exc)
            return

        closed_tickets = [t for t in list(self._open_trades) if t not in open_now]
        for ticket in closed_tickets:
            info = self._open_trades.pop(ticket)

            res = None
            try:
                res = self._exec_engine.get_closed_pnl(ticket)
            except Exception as exc:
                logger.debug("get_closed_pnl(%s) failed: %s", ticket, exc)
            if not res:
                # PnL not resolvable yet — re-track so we retry next sweep.
                self._open_trades[ticket] = info
                continue

            pnl        = float(res.get("pnl", 0.0))
            exit_price = float(res.get("exit_price", 0.0))
            ts         = datetime.now(timezone.utc)
            duration_s = int((ts - info["entry_ts"]).total_seconds())

            # 1) Feed the learning components, per contributing strategy.
            for strat in info["strategies"]:
                try:
                    self._aggregator.record_trade(strategy=strat, pnl=pnl)
                except Exception as exc:
                    logger.debug("aggregator.record_trade(%s) failed: %s", strat, exc)
                try:
                    self._allocator.record_pnl(strategy=strat, pnl=pnl)
                except KeyError:
                    pass  # strategy not registered in the allocator
                except Exception as exc:
                    logger.debug("allocator.record_pnl(%s) failed: %s", strat, exc)

            # 2) Persist the closed trade (JSON/MySQL + WebSocket push).
            primary = info["strategies"][0] if info["strategies"] else "live"
            if self._monitor is not None:
                try:
                    self._monitor.record_trade(
                        ts          = ts,
                        symbol      = info["symbol"],
                        strategy    = primary,
                        direction   = info["direction"],
                        size        = info["size"],
                        entry_price = info["entry_price"],
                        exit_price  = exit_price,
                        pnl         = pnl,
                        duration_s  = duration_s,
                        exit_reason = "closed",
                        ticket      = str(ticket),
                    )
                except Exception as exc:
                    logger.warning("monitor.record_trade failed: %s", exc)

            outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
            logger.info(
                "Trade closed [%s] ticket=%s pnl=%.2f %s (%s) -> learning updated",
                info["symbol"], ticket, pnl, outcome,
                ", ".join(info["strategies"]),
            )

            # 3) Persist a compact training experience (one record per trade:
            #    entry observation + action + realized outcome — no bloat).
            if self._experience is not None:
                self._experience.record({
                    "ticket":      str(ticket),
                    "symbol":      info["symbol"],
                    "timeframe":   self.cfg.timeframe,
                    "direction":   info["direction"],
                    "size":        info["size"],
                    "entry_ts":    info["entry_ts"],
                    "entry_price": info["entry_price"],
                    "exit_ts":     ts,
                    "exit_price":  exit_price,
                    "pnl":         pnl,
                    "duration_s":  duration_s,
                    "outcome":     outcome,
                    "confidence":  info.get("confidence"),
                    "strategies":  info["strategies"],
                    "exit_reason": "closed",
                    "features":    info.get("features") or {},
                })

    def _update_trailing_stops(self) -> None:
        """Push trailing stop modifications to MT5."""
        try:
            self._exec_engine.apply_trailing_stops()
        except Exception as exc:
            logger.warning("Trailing stop update failed: %s", exc)

    # ------------------------------------------------------------------
    # Stage 8 — Logging and monitoring hooks
    # ------------------------------------------------------------------

    def _log_cycle(self, result: TradingCycleResult) -> None:
        level = logging.INFO if result.status == CycleStatus.OK else logging.WARNING
        logger.log(
            level,
            "[%s] Cycle #%d %s | conf=%.3f | risk=%s | t=%.0fms",
            result.symbol,
            result.cycle_id,
            result.status.name,
            result.decision.confidence if result.decision else 0.0,
            result.risk_decision.verdict.name if result.risk_decision else "—",
            result.duration_ms,
        )
        if result.error:
            logger.error("  └─ Error: %s", result.error)

    def _dispatch_hooks(self, result: TradingCycleResult) -> None:
        for hook in self._hooks:
            try:
                hook(result)
            except Exception as exc:
                logger.error("Monitoring hook %s failed: %s", hook, exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in small increments so the stop event is checked often."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop_event.is_set() or self._emergency.is_active:
                return
            time.sleep(min(1.0, deadline - time.monotonic()))

    def _setup_signal_handlers(self) -> None:
        """Handle SIGTERM / SIGINT for graceful shutdown."""
        def _handler(signum, _frame):
            sig_name = signal.Signals(signum).name
            logger.warning("Signal %s received — initiating graceful stop.", sig_name)
            self.stop(reason=f"os_signal:{sig_name}")

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT,  _handler)
        except (OSError, ValueError):
            # Signal handlers can only be set in the main thread
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of the current trader state."""
        return {
            "running":          self._running,
            "cycle_id":         self._cycle_id,
            "emergency_active": self._emergency.is_active,
            "retry_count":      self._retry_count,
            "consecutive_ok":   self._consecutive_ok,
            "strategies":       [m.name for m in self._strategies],
            "symbols":          self.cfg.symbols,
            "equity":           self._portfolio_state.equity,
            "balance":          self._portfolio_state.balance,
            "open_positions":   len(self._portfolio_state.open_positions),
        }

    def get_allocation_summary(self) -> dict:
        """Current capital allocation per strategy."""
        return self._allocator.get_summary()
