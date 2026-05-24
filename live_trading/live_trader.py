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


# ---------------------------------------------------------------------------
# Cycle status
# ---------------------------------------------------------------------------

class CycleStatus(Enum):
    OK            = auto()
    NO_SIGNAL     = auto()
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
    warmup_bars: int     = 200     # bars fetched for feature computation

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

        # ---- Sub-engines ------------------------------------------------
        self._feature_eng  = FeatureEngineer(self.cfg.feature_config)
        self._aggregator   = SignalAggregator()
        self._risk_engine  = RiskEngine(self.cfg.risk_config)
        self._allocator    = PortfolioAllocator(
            total_capital   = self.cfg.initial_balance,
            default_method  = self.cfg.allocation_method,
        )
        self._exec_engine  = MT5ExecutionEngine(
            ExecutionConfig(
                login        = self.cfg.mt5_login,
                password     = self.cfg.mt5_password,
                server       = self.cfg.mt5_server,
                path         = self.cfg.mt5_path,
                magic_number = self.cfg.magic_number,
            )
        )

        # ---- State -------------------------------------------------------
        self._portfolio_state = PortfolioState(
            equity        = self.cfg.initial_balance,
            balance       = self.cfg.initial_balance,
            open_positions= [],
            daily_pnl     = 0.0,
            daily_trades  = 0,
        )
        self._cycle_id         = 0
        self._consecutive_ok   = 0
        self._retry_count      = 0
        self._running          = False
        self._stop_event       = threading.Event()
        self._background_thread: Optional[threading.Thread] = None

        # ---- Emergency stop ---------------------------------------------
        self._emergency = EmergencyStopManager(self.cfg.stop_flag_path)

        # ---- Strategies -------------------------------------------------
        self._strategies: list[AlphaModel] = []

        # ---- Monitoring hooks -------------------------------------------
        self._hooks: list[MonitoringHook] = [
            JsonFileLogger(self.cfg.log_dir)
        ]

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
            strategy_name   = model.name,
            initial_capital = initial_capital or (
                self.cfg.initial_balance / max(len(self._strategies), 1)
            ),
        )
        logger.info("Strategy registered: %s (weight=%.2f)", model.name, weight)
        return self

    def register_hook(self, hook: MonitoringHook) -> "LiveTrader":
        """Register a monitoring callback. Called after every cycle."""
        self._hooks.append(hook)
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

            # 3. Run alpha models -----------------------------------------
            signals = self._run_alpha_models(features)
            result.signals = signals
            if not signals:
                result.status = CycleStatus.NO_SIGNAL
                return result

            # 4. Aggregate signals ----------------------------------------
            decision = self._aggregate_signals(signals)
            result.decision = decision
            if (
                decision.signal == SignalType.HOLD
                or decision.confidence < self.cfg.min_confidence
            ):
                result.status = CycleStatus.NO_SIGNAL
                return result

            # 5. Apply risk engine ----------------------------------------
            trade_order  = self._build_trade_order(symbol, decision)
            risk_decision = self._apply_risk(trade_order)
            result.risk_decision = risk_decision
            if risk_decision.verdict.name in ("BLOCK",):
                result.status = CycleStatus.RISK_BLOCKED
                logger.info("[%s] Risk BLOCK — %s", symbol, risk_decision.reason)
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

            rates = mt5.copy_rates_from_pos(
                symbol,
                _get_mt5_tf(self.cfg.timeframe),
                0,                          # start from the most recent
                self.cfg.warmup_bars,
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
        equity     = self._portfolio_state.equity
        atr_proxy  = self._get_atr_proxy(symbol)
        size       = self._risk_engine.compute_atr_size(equity, atr_proxy)

        return TradeOrder(
            symbol    = symbol,
            direction = decision.signal.name,
            size      = size,
            strategy  = "aggregated",
            confidence= decision.confidence,
        )

    def _apply_risk(self, order: TradeOrder) -> RiskDecision:
        return self._risk_engine.evaluate_portfolio_risk(
            proposed_trades  = [order],
            portfolio_state  = self._portfolio_state,
        )

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
            # Apply size reduction from risk engine (REDUCE verdict)
            if risk_decision.verdict.name == "REDUCE":
                order.size *= risk_decision.size_multiplier

            result = self._exec_engine.execute(decision, order)
            results.append(result)

            if result.status == "FILLED":
                logger.info(
                    "✓ FILLED %s | ticket=%s | price=%.5f | size=%.2f",
                    order.symbol,
                    result.ticket,
                    result.fill_price,
                    order.size,
                )
                # Record trade outcome for performance weighting
                for model in self._strategies:
                    self._aggregator.record_trade(
                        strategy_name = model.name,
                        pnl           = 0.0,    # filled at entry — PnL unknown yet
                    )
            else:
                logger.warning(
                    "✗ REJECTED %s | status=%s | retries=%d",
                    order.symbol, result.status, result.retries,
                )
        except Exception as exc:
            logger.error("Execution error for %s: %s", order.symbol, exc)

        return results

    # ------------------------------------------------------------------
    # Post-cycle updates
    # ------------------------------------------------------------------

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
                        symbol    = p.symbol,
                        direction = p.direction,
                        size      = p.volume,
                        entry_price = p.price_open,
                        strategy  = "live",
                        unrealized_pnl = p.profit,
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
