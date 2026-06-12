"""
TradingMonitor
==============
Central monitoring hub for the quant-fund-ai live trading system.

Integrates with LiveTrader via a MonitoringHook callback:

    trader.register_hook(monitor)

On every cycle result it:
  1. Updates in-memory ring buffers (fast API reads)
  2. Computes drawdown tracking (open/close drawdown periods)
  3. Computes rolling per-strategy stats
  4. Writes equity snapshot    → MetricsStore
  5. Writes strategy PnL rows  → MetricsStore
  6. Writes risk snapshot      → MetricsStore
  7. Writes completed trades   → MetricsStore

Separate method  record_trade(TradeRecord)  is called by LiveTrader
or BacktestEngine when a trade closes.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from monitoring.metrics_store import MetricsStore, MySQLConfig
from live_trading.live_trader import CycleStatus, TradingCycleResult

# ---------------------------------------------------------------------------
# WebSocket broadcaster — wired by api.main.create_app()
# ---------------------------------------------------------------------------

_ws_broadcaster: Optional[Callable[[str, dict], None]] = None


def set_ws_broadcaster(fn: Callable[[str, dict], None]) -> None:
    """
    Register the real-time WebSocket push function.

    Called once from api.main.create_app() to inject broadcast_sync:

        from monitoring import monitor as _mon
        from api.ws import broadcast_sync
        _mon.set_ws_broadcaster(broadcast_sync)

    After registration every closed trade, cycle tick, and risk alert
    is pushed instantly to all connected dashboard clients.
    """
    global _ws_broadcaster
    _ws_broadcaster = fn
    logger.info("WS broadcaster registered: %s", fn.__name__)


def _broadcast(msg_type: str, payload: dict) -> None:
    """Fire-and-forget push to connected WebSocket clients. Never raises."""
    if _ws_broadcaster is not None:
        try:
            _ws_broadcaster(msg_type, payload)
        except Exception as exc:
            logger.debug("WS broadcast error [%s]: %s", msg_type, exc)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory snapshot types
# ---------------------------------------------------------------------------

@dataclass
class EquityPoint:
    ts:           datetime
    equity:       float
    balance:      float
    open_pnl:     float
    drawdown_pct: float
    n_positions:  int


@dataclass
class StrategyStats:
    """Rolling statistics for one strategy."""

    name:           str
    n_trades:       int           = 0
    n_wins:         int           = 0
    cumulative_pnl: float         = 0.0
    pnl_history:    deque         = field(default_factory=lambda: deque(maxlen=100))
    last_updated:   Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        return self.n_wins / max(self.n_trades, 1)

    @property
    def sharpe(self) -> float:
        arr = np.array(self.pnl_history)
        if len(arr) < 2:
            return 0.0
        return float(np.mean(arr) / (np.std(arr) + 1e-8)) * np.sqrt(min(len(arr), 252))

    @property
    def avg_pnl(self) -> float:
        return float(np.mean(self.pnl_history)) if self.pnl_history else 0.0


@dataclass
class DrawdownState:
    """Tracks the current open drawdown episode."""

    in_drawdown:   bool    = False
    peak_equity:   float   = 0.0
    trough_equity: float   = 0.0
    start_ts:      Optional[datetime] = None
    max_depth_pct: float   = 0.0


@dataclass
class RiskSnapshot:
    ts:             datetime
    total_exposure: float
    daily_pnl:      float
    daily_pnl_pct:  float
    leverage:       float
    n_positions:    int
    var_95:         float
    verdict:        str


# ---------------------------------------------------------------------------
# Monitor configuration
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    # Memory ring buffer sizes
    equity_buffer_size:   int = 2880    # 2 days at 1-min resolution
    risk_buffer_size:     int = 500
    trade_buffer_size:    int = 1000

    # Drawdown threshold to open a new drawdown period record
    drawdown_threshold:   float = 0.005  # 0.5%

    # MySQL
    mysql_config: Optional[MySQLConfig] = None
    log_dir:      str = "data/storage/logs/metrics"

    # Alert hooks
    alert_on_drawdown_pct: float = 0.10  # alert when DD > 10%
    alert_on_daily_loss:   float = 0.05  # alert when daily loss > 5%


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

class TradingMonitor:
    """
    Central monitoring hub.

    Thread-safe: all public methods acquire a reentrant lock.

    Parameters
    ----------
    config : MonitorConfig
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.cfg  = config or MonitorConfig()
        self._lock = threading.RLock()

        # In-memory buffers
        self._equity_curve: deque[EquityPoint]  = deque(maxlen=self.cfg.equity_buffer_size)
        self._risk_buffer:  deque[RiskSnapshot] = deque(maxlen=self.cfg.risk_buffer_size)
        self._trade_buffer: deque[dict]         = deque(maxlen=self.cfg.trade_buffer_size)

        # Per-strategy stats
        self._strategy_stats: dict[str, StrategyStats] = {}

        # Drawdown tracking
        self._dd_state     = DrawdownState()
        self._high_water   = 0.0
        self._initial_equity = 0.0
        self._daily_start_equity = 0.0
        self._daily_date: Optional[str] = None

        # Alert callbacks
        self._alert_hooks: list[Callable[[str, dict], None]] = []

        # Persistence
        self._store = MetricsStore(
            mysql_config = self.cfg.mysql_config,
            log_dir      = self.cfg.log_dir,
        )
        self._store.connect()

        logger.info("TradingMonitor initialised.")

    # ------------------------------------------------------------------
    # LiveTrader hook entry point
    # ------------------------------------------------------------------

    def __call__(self, result: TradingCycleResult) -> None:
        """
        Called by LiveTrader after every cycle.
        Implements the MonitoringHook protocol.
        """
        with self._lock:
            try:
                self._process_cycle(result)
            except Exception as exc:
                logger.error("Monitor processing error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Trade recording (called when a trade closes)
    # ------------------------------------------------------------------

    def record_trade(
        self,
        ts:          datetime,
        symbol:      str,
        strategy:    str,
        direction:   str,
        size:        float,
        entry_price: float,
        exit_price:  float,
        pnl:         float,
        duration_s:  int   = 0,
        exit_reason: str   = "",
        ticket:      str   = "",
    ) -> None:
        """Record a closed trade. Call from LiveTrader / BacktestEngine."""
        with self._lock:
            # Update strategy stats
            stats = self._get_stats(strategy)
            stats.n_trades       += 1
            stats.cumulative_pnl += pnl
            stats.pnl_history.append(pnl)
            stats.last_updated    = ts
            if pnl > 0:
                stats.n_wins += 1

            trade_row = {
                "ts":          ts.isoformat(),
                "symbol":      symbol,
                "strategy":    strategy,
                "direction":   direction,
                "size":        round(size, 4),
                "entry_price": round(entry_price, 5),
                "exit_price":  round(exit_price, 5),
                "pnl":         round(pnl, 4),
                "duration_s":  duration_s,
                "exit_reason": exit_reason,
                "ticket":      ticket,
            }
            self._trade_buffer.append(trade_row)

            # Real-time WebSocket push
            _broadcast("trade", trade_row)

            # Persist
            self._store.write_trade(
                ts=ts, symbol=symbol, strategy=strategy,
                direction=direction, size=size,
                entry_price=entry_price, exit_price=exit_price,
                pnl=pnl, duration_s=duration_s,
                exit_reason=exit_reason, ticket=ticket,
            )

            logger.info(
                "Trade recorded [%s/%s] dir=%s pnl=%.2f",
                symbol, strategy, direction, pnl,
            )

    # ------------------------------------------------------------------
    # Core cycle processing
    # ------------------------------------------------------------------

    def _process_cycle(self, result: TradingCycleResult) -> None:
        ts      = result.timestamp
        ps      = result.portfolio_state
        equity  = ps.equity  if ps else 0.0
        balance = ps.balance if ps else 0.0

        # Defensive guard: a cycle without a portfolio snapshot reports
        # equity 0, which would register a false 100% drawdown against the
        # established high-water mark. Carry forward the last known equity.
        if equity <= 0:
            equity = self._high_water if self._high_water > 0 else 0.0
        if balance <= 0:
            balance = equity

        # Initialise reference values on first cycle
        if self._initial_equity == 0.0 and equity > 0:
            self._initial_equity      = equity
            self._high_water          = equity
            self._daily_start_equity  = equity
            self._daily_date          = ts.strftime("%Y%m%d")

        # Daily reset
        today = ts.strftime("%Y%m%d")
        if today != self._daily_date:
            self._daily_start_equity = equity
            self._daily_date         = today

        # Computed metrics
        n_positions = len(ps.open_positions) if ps else 0
        open_pnl    = sum(p.unrealized_pnl for p in ps.open_positions) if ps else 0.0
        daily_pnl   = equity - self._daily_start_equity
        daily_pnl_pct = daily_pnl / (self._daily_start_equity + 1e-10)

        # High-water update
        if equity > self._high_water:
            self._high_water = equity
        drawdown_pct = max(0.0, (self._high_water - equity) / (self._high_water + 1e-10))

        # --- Equity snapshot -------------------------------------------
        ep = EquityPoint(
            ts           = ts,
            equity       = equity,
            balance      = balance,
            open_pnl     = open_pnl,
            drawdown_pct = drawdown_pct,
            n_positions  = n_positions,
        )
        self._equity_curve.append(ep)
        self._store.write_equity_snapshot(
            ts=ts, equity=equity, balance=balance,
            open_pnl=open_pnl, drawdown_pct=drawdown_pct,
            high_water=self._high_water, n_positions=n_positions,
        )

        # --- Drawdown tracking -----------------------------------------
        self._update_drawdown(ts, equity, drawdown_pct)

        # --- Strategy PnL rows -----------------------------------------
        if result.signals:
            for sig in result.signals:
                stats = self._get_stats(sig.strategy_name)
                self._store.write_strategy_pnl(
                    ts             = ts,
                    strategy_name  = sig.strategy_name,
                    cycle_pnl      = 0.0,
                    cumulative_pnl = stats.cumulative_pnl,
                    n_trades       = stats.n_trades,
                    win_rate       = stats.win_rate,
                    sharpe         = stats.sharpe,
                )

        # --- Risk snapshot ---------------------------------------------
        risk_d  = result.risk_decision
        verdict = risk_d.verdict.name if risk_d else ""
        exposure = sum(
            p.size * p.entry_price for p in (ps.open_positions if ps else [])
        )
        leverage = exposure / (equity + 1e-10) if equity > 0 else 0.0
        var_95   = self._compute_var95()

        rs = RiskSnapshot(
            ts             = ts,
            total_exposure = exposure,
            daily_pnl      = daily_pnl,
            daily_pnl_pct  = daily_pnl_pct,
            leverage       = leverage,
            n_positions    = n_positions,
            var_95         = var_95,
            verdict        = verdict,
        )
        self._risk_buffer.append(rs)
        self._store.write_risk_snapshot(
            ts=ts, total_exposure=exposure,
            daily_pnl=daily_pnl, daily_pnl_pct=daily_pnl_pct,
            leverage=leverage, n_positions=n_positions,
            var_95=var_95, risk_verdict=verdict,
        )

        # --- Alerts ----------------------------------------------------
        self._check_alerts(drawdown_pct, daily_pnl_pct, result)

        # --- Real-time WebSocket tick ----------------------------------
        _broadcast("tick", {
            "equity_curve": self.get_equity_curve(limit=1),
            "portfolio": {
                "equity":        round(equity,  2),
                "balance":       round(balance, 2),
                "daily_pnl":     round(daily_pnl,      2),
                "daily_pnl_pct": round(daily_pnl_pct * 100, 4),
                "drawdown_pct":  round(drawdown_pct  * 100, 4),
                "high_water":    round(self._high_water, 2),
                "n_positions":   n_positions,
            },
            "risk": self.get_latest_risk() or {},
        })

        if result.status == CycleStatus.OK:
            logger.debug(
                "Monitor [%s] equity=%.2f dd=%.2f%% daily_pnl=%.2f",
                result.symbol, equity, drawdown_pct * 100, daily_pnl,
            )

    # ------------------------------------------------------------------
    # Drawdown tracking
    # ------------------------------------------------------------------

    def _update_drawdown(
        self,
        ts:           datetime,
        equity:       float,
        drawdown_pct: float,
    ) -> None:
        dd = self._dd_state

        if not dd.in_drawdown and drawdown_pct >= self.cfg.drawdown_threshold:
            dd.in_drawdown   = True
            dd.peak_equity   = self._high_water
            dd.trough_equity = equity
            dd.start_ts      = ts
            dd.max_depth_pct = drawdown_pct
            logger.warning(
                "Drawdown started — depth=%.2f%% peak=%.2f",
                drawdown_pct * 100, dd.peak_equity,
            )

        elif dd.in_drawdown:
            if equity < dd.trough_equity:
                dd.trough_equity = equity
                dd.max_depth_pct = drawdown_pct

            if drawdown_pct < self.cfg.drawdown_threshold / 2:
                # Drawdown recovered
                duration_s = int((ts - dd.start_ts).total_seconds()) if dd.start_ts else 0
                self._store.write_drawdown_period(
                    start_ts      = dd.start_ts,
                    end_ts        = ts,
                    peak_equity   = dd.peak_equity,
                    trough_equity = dd.trough_equity,
                    depth_pct     = dd.max_depth_pct,
                    duration_s    = duration_s,
                    recovered     = True,
                )
                logger.info(
                    "Drawdown recovered — max=%.2f%% duration=%ds",
                    dd.max_depth_pct * 100, duration_s,
                )
                dd.in_drawdown = False

    # ------------------------------------------------------------------
    # Alert system
    # ------------------------------------------------------------------

    def register_alert(self, callback: Callable[[str, dict], None]) -> "TradingMonitor":
        """
        Register an alert callback.
        Signature: callback(alert_type: str, data: dict)
        """
        self._alert_hooks.append(callback)
        return self

    def _check_alerts(
        self,
        drawdown_pct:  float,
        daily_pnl_pct: float,
        result:        TradingCycleResult,
    ) -> None:
        alerts = []
        if drawdown_pct >= self.cfg.alert_on_drawdown_pct:
            alerts.append(("DRAWDOWN_ALERT", {
                "drawdown_pct": round(drawdown_pct * 100, 2),
                "equity":       result.portfolio_state.equity if result.portfolio_state else 0,
                "ts":           result.timestamp.isoformat(),
            }))
        if daily_pnl_pct <= -self.cfg.alert_on_daily_loss:
            alerts.append(("DAILY_LOSS_ALERT", {
                "daily_pnl_pct": round(daily_pnl_pct * 100, 2),
                "ts":            result.timestamp.isoformat(),
            }))
        if result.status == CycleStatus.EMERGENCY:
            alerts.append(("EMERGENCY_STOP", {
                "symbol": result.symbol,
                "error":  result.error,
                "ts":     result.timestamp.isoformat(),
            }))

        for alert_type, data in alerts:
            logger.warning("ALERT [%s]: %s", alert_type, data)
            _broadcast("alert", {"type": alert_type, "message": f"{alert_type}: {data}", **data})
            for hook in self._alert_hooks:
                try:
                    hook(alert_type, data)
                except Exception as exc:
                    logger.error("Alert hook error: %s", exc)

    # ------------------------------------------------------------------
    # In-memory reads (used by API)
    # ------------------------------------------------------------------

    def get_equity_curve(self, limit: int = 500) -> list[dict]:
        with self._lock:
            points = list(self._equity_curve)[-limit:]
            return [
                {
                    "ts":           p.ts.isoformat(),
                    "equity":       round(p.equity, 2),
                    "balance":      round(p.balance, 2),
                    "open_pnl":     round(p.open_pnl, 2),
                    "drawdown_pct": round(p.drawdown_pct * 100, 4),
                    "n_positions":  p.n_positions,
                }
                for p in points
            ]

    def get_strategy_summary(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "strategy":       s.name,
                    "n_trades":       s.n_trades,
                    "win_rate":       round(s.win_rate * 100, 2),
                    "cumulative_pnl": round(s.cumulative_pnl, 2),
                    "avg_pnl":        round(s.avg_pnl, 4),
                    "sharpe":         round(s.sharpe, 3),
                    "last_updated":   s.last_updated.isoformat() if s.last_updated else None,
                }
                for s in self._strategy_stats.values()
            ]

    def get_latest_risk(self) -> Optional[dict]:
        with self._lock:
            if not self._risk_buffer:
                return None
            rs = self._risk_buffer[-1]
            return {
                "ts":             rs.ts.isoformat(),
                "total_exposure": round(rs.total_exposure, 2),
                "daily_pnl":      round(rs.daily_pnl, 2),
                "daily_pnl_pct":  round(rs.daily_pnl_pct * 100, 4),
                "leverage":       round(rs.leverage, 4),
                "n_positions":    rs.n_positions,
                "var_95":         round(rs.var_95, 2),
                "verdict":        rs.verdict,
            }

    def get_drawdown_stats(self) -> dict:
        with self._lock:
            eq = [p.equity for p in self._equity_curve]
            if not eq:
                return {}
            eq_arr     = np.array(eq)
            hwm        = np.maximum.accumulate(eq_arr)
            dd_series  = (hwm - eq_arr) / (hwm + 1e-10)
            current_dd = float(dd_series[-1]) * 100

            return {
                "current_drawdown_pct":  round(current_dd, 4),
                "max_drawdown_pct":      round(float(np.max(dd_series)) * 100, 4),
                "high_water_mark":       round(float(self._high_water), 2),
                "in_drawdown":           self._dd_state.in_drawdown,
                "drawdown_start":        (
                    self._dd_state.start_ts.isoformat()
                    if self._dd_state.start_ts else None
                ),
            }

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        with self._lock:
            trades = list(self._trade_buffer)
            return trades[-limit:]

    def get_summary(self) -> dict:
        """Full snapshot — used by the /metrics/summary API endpoint."""
        with self._lock:
            latest_eq = self._equity_curve[-1] if self._equity_curve else None
            return {
                "equity":            round(latest_eq.equity,  2) if latest_eq else 0,
                "balance":           round(latest_eq.balance, 2) if latest_eq else 0,
                "drawdown":          self.get_drawdown_stats(),
                "risk":              self.get_latest_risk(),
                "strategies":        self.get_strategy_summary(),
                "recent_trades":     self.get_recent_trades(10),
                "total_trades":      sum(
                    s.n_trades for s in self._strategy_stats.values()
                ),
                "total_pnl":         round(
                    sum(s.cumulative_pnl for s in self._strategy_stats.values()), 2
                ),
            }

    # ------------------------------------------------------------------
    # Persistence queries (delegates to store)
    # ------------------------------------------------------------------

    def query_equity_curve(self, **kwargs) -> list[dict]:
        return self._store.query_equity_curve(**kwargs)

    def query_trades(self, **kwargs) -> list[dict]:
        return self._store.query_trades(**kwargs)

    def query_strategy_pnl(self, **kwargs) -> list[dict]:
        return self._store.query_strategy_pnl(**kwargs)

    def query_risk_snapshots(self, **kwargs) -> list[dict]:
        return self._store.query_risk_snapshots(**kwargs)

    def query_drawdowns(self, **kwargs) -> list[dict]:
        return self._store.query_drawdowns(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_stats(self, strategy: str) -> StrategyStats:
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = StrategyStats(name=strategy)
        return self._strategy_stats[strategy]

    def _compute_var95(self) -> float:
        """95% VaR from recent equity returns."""
        if len(self._equity_curve) < 20:
            return 0.0
        eq = np.array([p.equity for p in self._equity_curve])
        returns = np.diff(eq) / (eq[:-1] + 1e-10)
        return float(np.percentile(returns, 5))  # 5th percentile (negative)
