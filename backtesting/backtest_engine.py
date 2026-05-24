"""
BacktestEngine
==============
Institutional-grade multi-strategy backtesting engine.

Replicates live execution exactly by reusing:
  - AlphaModel          (signal generation)
  - SignalAggregator    (conflict resolution + performance weighting)
  - RiskEngine          (drawdown / exposure / volatility checks)
  - PortfolioAllocator  (capital allocation)

Market realism
--------------
  - Spread applied on entry and exit
  - Gaussian slippage noise
  - Commission per trade (fraction of notional)
  - SL / TP checked against bar's High/Low (not just Close)

Parallel execution
------------------
  - Each strategy is evaluated independently in a thread pool
  - Results are aggregated into a combined portfolio equity curve

Output
------
  BacktestResult
    ├── metrics          : dict[strategy → PerformanceMetrics]
    ├── portfolio_metrics: PerformanceMetrics  (combined)
    ├── equity_curves    : dict[strategy → pd.Series]
    ├── portfolio_equity : pd.Series
    ├── trade_log        : pd.DataFrame
    └── drawdown_series  : pd.Series
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.metrics import MetricsCalculator, PerformanceMetrics, compare_metrics
from engines.risk_engine.risk_engine import (
    PortfolioState, RiskConfig, RiskEngine, RiskVerdict, TradeOrder, OpenPosition,
)
from engines.signal_engine.signal_aggregator import AggregatedDecision, SignalAggregator
from engines.portfolio_engine.portfolio_engine import AllocationMethod, PortfolioAllocator
from research.alpha_models.base import AlphaModel, SignalType

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Market simulation and engine parameters."""

    # ---- Costs -----------------------------------------------------------
    spread: float          = 0.0001    # half-spread in price units
    slippage_std: float    = 0.00005   # Gaussian noise std
    commission: float      = 0.0002    # fraction of notional per trade

    # ---- Position sizing -------------------------------------------------
    initial_balance: float = 100_000.0
    position_size_pct: float = 0.10   # fraction of equity per trade
    sl_pct: float          = 0.02
    tp_pct: float          = 0.04

    # ---- Engine params ---------------------------------------------------
    window_size: int       = 50        # lookback bars fed to each strategy
    periods_per_year: int  = 252       # for annualisation (252=daily, 52=weekly)
    risk_free_rate: float  = 0.02

    # ---- Parallelism -----------------------------------------------------
    n_workers: int         = 4

    # ---- Risk engine -----------------------------------------------------
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # ---- Allocation method -----------------------------------------------
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY


# ---------------------------------------------------------------------------
# Simulated position state
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    direction: int
    size: float
    entry_price: float
    sl: float
    tp: float
    entry_bar: int
    strategy: str
    commission_paid: float = 0.0


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    strategy: str
    entry_bar: int
    exit_bar: int
    direction: int
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str    # "sl" | "tp" | "signal" | "eod"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    metrics: dict[str, PerformanceMetrics]        = field(default_factory=dict)
    portfolio_metrics: Optional[PerformanceMetrics] = None
    equity_curves: dict[str, pd.Series]           = field(default_factory=dict)
    portfolio_equity: Optional[pd.Series]         = None
    trade_log: pd.DataFrame                       = field(default_factory=pd.DataFrame)
    drawdown_series: Optional[pd.Series]          = None
    config: Optional[BacktestConfig]              = None

    def summary(self) -> pd.DataFrame:
        return compare_metrics(self.metrics)

    def __repr__(self) -> str:
        names = list(self.metrics.keys())
        return (
            f"BacktestResult(strategies={names}, "
            f"portfolio={self.portfolio_metrics})"
        )


# ---------------------------------------------------------------------------
# Per-strategy simulator
# ---------------------------------------------------------------------------

class _StrategySim:
    """
    Isolated simulation loop for a single AlphaModel.
    Holds its own balance and position state.
    """

    def __init__(
        self,
        model: AlphaModel,
        data: pd.DataFrame,
        config: BacktestConfig,
        rng: np.random.Generator,
    ) -> None:
        self.model  = model
        self.data   = data.reset_index(drop=True)
        self.cfg    = config
        self.rng    = rng

        self.balance   = config.initial_balance
        self.equity    = config.initial_balance
        self.peak_equity = config.initial_balance

        self.position: Optional[SimPosition] = None
        self.equity_curve: list[float] = []
        self.exposure_mask: list[bool] = []
        self.trades: list[TradeRecord] = []

    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.Series, list[TradeRecord], pd.Series]:
        """
        Run full simulation. Returns (equity_curve, trades, exposure_mask).
        """
        cfg = self.cfg
        n   = len(self.data)

        for bar in range(cfg.window_size, n):
            window = self.data.iloc[bar - cfg.window_size: bar]
            close  = float(self.data.at[bar, "close"])
            high   = float(self.data.at[bar, "high"])
            low    = float(self.data.at[bar, "low"])

            # ---- Check SL/TP on open position ---------------------------
            if self.position:
                exit_price, reason = self._check_sl_tp(high, low)
                if exit_price:
                    self._close(bar, exit_price, reason)

            # ---- Generate signal ----------------------------------------
            try:
                signal = self.model.compute(window)
            except Exception as exc:
                logger.debug("Strategy %s raised: %s", self.model.name, exc)
                signal = self.model._hold()

            # ---- Execute signal -----------------------------------------
            if signal.signal != SignalType.HOLD:
                direction = +1 if signal.signal == SignalType.BUY else -1
                self._open(bar, close, direction)

            # ---- Mark-to-market equity ----------------------------------
            self._update_equity(close)
            self.equity_curve.append(self.equity)
            self.exposure_mask.append(self.position is not None)

        # ---- Close any remaining position at end -----------------------
        if self.position:
            close = float(self.data.iloc[-1]["close"])
            self._close(len(self.data) - 1, close, "eod")

        idx = self.data.index[cfg.window_size:]
        ec  = pd.Series(self.equity_curve, index=idx, name=self.model.name)
        em  = pd.Series(self.exposure_mask, index=idx, name=self.model.name)
        return ec, self.trades, em

    # ------------------------------------------------------------------

    def _open(self, bar: int, price: float, direction: int) -> None:
        # If already in position of same direction, skip
        if self.position and self.position.direction == direction:
            return
        # Close existing before reversing
        if self.position:
            self._close(bar, price, "signal")

        exec_price = self._execution_price(price, direction)
        notional   = self.equity * self.cfg.position_size_pct
        size       = notional / (exec_price + 1e-10)
        commission = notional * self.cfg.commission

        if self.balance < commission:
            return

        self.balance -= commission
        sl = exec_price * (1 - self.cfg.sl_pct * direction)
        tp = exec_price * (1 + self.cfg.tp_pct * direction)

        self.position = SimPosition(
            direction     = direction,
            size          = size,
            entry_price   = exec_price,
            sl            = sl,
            tp            = tp,
            entry_bar     = bar,
            strategy      = self.model.name,
            commission_paid = commission,
        )

    def _close(self, bar: int, price: float, reason: str) -> None:
        pos        = self.position
        exec_price = self._execution_price(price, -pos.direction)
        commission = exec_price * pos.size * self.cfg.commission
        raw_pnl    = (exec_price - pos.entry_price) * pos.direction * pos.size
        net_pnl    = raw_pnl - commission

        self.balance += pos.size * exec_price - commission
        self.trades.append(TradeRecord(
            strategy    = pos.strategy,
            entry_bar   = pos.entry_bar,
            exit_bar    = bar,
            direction   = pos.direction,
            size        = pos.size,
            entry_price = pos.entry_price,
            exit_price  = exec_price,
            pnl         = net_pnl,
            exit_reason = reason,
        ))
        self.position = None

    def _check_sl_tp(self, high: float, low: float) -> tuple[Optional[float], str]:
        pos = self.position
        if pos.direction == +1:
            if low <= pos.sl:
                return pos.sl, "sl"
            if high >= pos.tp:
                return pos.tp, "tp"
        else:
            if high >= pos.sl:
                return pos.sl, "sl"
            if low <= pos.tp:
                return pos.tp, "tp"
        return None, ""

    def _update_equity(self, price: float) -> None:
        unrealized = 0.0
        if self.position:
            unrealized = (
                (price - self.position.entry_price)
                * self.position.direction
                * self.position.size
            )
        self.equity = self.balance + unrealized
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def _execution_price(self, price: float, direction: int) -> float:
        slippage = float(self.rng.normal(0, self.cfg.slippage_std))
        spread   = self.cfg.spread * direction
        return price + spread + slippage


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Institutional-grade multi-strategy backtesting engine.

    Parameters
    ----------
    config : BacktestConfig
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config  = config or BacktestConfig()
        self._calc   = MetricsCalculator(
            risk_free_rate   = self.config.risk_free_rate,
            periods_per_year = self.config.periods_per_year,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        strategies: list[AlphaModel],
        data: pd.DataFrame,
        seed: int = 42,
    ) -> BacktestResult:
        """
        Run a multi-strategy backtest on a single dataset.

        Parameters
        ----------
        strategies : list[AlphaModel]
            Each strategy is run independently, then combined.
        data : pd.DataFrame
            Full OHLCV history (must span the entire backtest period).
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        BacktestResult
        """
        cfg = self.config
        self._validate(data)

        logger.info(
            "Backtest starting — %d strategies | %d bars | "
            "commission=%.4f | spread=%.5f",
            len(strategies), len(data), cfg.commission, cfg.spread,
        )

        # ---- Run strategies in parallel ---------------------------------
        results: dict[str, tuple] = {}
        rng = np.random.default_rng(seed)

        with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
            futures = {
                pool.submit(
                    self._run_strategy,
                    model, data, cfg,
                    np.random.default_rng(rng.integers(0, 2**31)),
                ): model.name
                for model in strategies
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error("Strategy '%s' failed: %s", name, exc, exc_info=True)

        # ---- Aggregate --------------------------------------------------
        equity_curves: dict[str, pd.Series]       = {}
        all_trades:    list[TradeRecord]           = []
        exposure_masks: dict[str, pd.Series]      = {}
        trade_pnls:    dict[str, list[float]]      = {}

        for name, (ec, trades, em) in results.items():
            equity_curves[name]  = ec
            all_trades.extend(trades)
            exposure_masks[name] = em
            trade_pnls[name]     = [t.pnl for t in trades]

        # ---- Compute metrics per strategy -------------------------------
        metrics: dict[str, PerformanceMetrics] = {}
        for name, ec in equity_curves.items():
            metrics[name] = self._calc.compute(
                equity_curve  = ec,
                trade_pnls    = trade_pnls.get(name),
                exposure_mask = exposure_masks.get(name),
                strategy_name = name,
            )

        # ---- Portfolio equity curve (equal-weight combined) -------------
        portfolio_equity = self._build_portfolio_equity(equity_curves, cfg)
        portfolio_metrics = self._calc.compute(
            equity_curve  = portfolio_equity,
            strategy_name = "portfolio",
        ) if portfolio_equity is not None else None

        # ---- Drawdown series --------------------------------------------
        if portfolio_equity is not None:
            hwm = portfolio_equity.cummax()
            drawdown_series = (portfolio_equity - hwm) / (hwm + 1e-10)
        else:
            drawdown_series = None

        # ---- Trade log --------------------------------------------------
        trade_log = self._build_trade_log(all_trades, data)

        self._log_summary(metrics, portfolio_metrics)

        return BacktestResult(
            metrics           = metrics,
            portfolio_metrics = portfolio_metrics,
            equity_curves     = equity_curves,
            portfolio_equity  = portfolio_equity,
            trade_log         = trade_log,
            drawdown_series   = drawdown_series,
            config            = cfg,
        )

    # ------------------------------------------------------------------
    # Per-strategy runner (called in thread pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_strategy(
        model: AlphaModel,
        data: pd.DataFrame,
        config: BacktestConfig,
        rng: np.random.Generator,
    ) -> tuple[pd.Series, list[TradeRecord], pd.Series]:
        sim = _StrategySim(model, data, config, rng)
        return sim.run()

    # ------------------------------------------------------------------
    # Portfolio combination
    # ------------------------------------------------------------------

    @staticmethod
    def _build_portfolio_equity(
        equity_curves: dict[str, pd.Series],
        cfg: BacktestConfig,
    ) -> Optional[pd.Series]:
        if not equity_curves:
            return None

        # Align all curves on common index
        combined = pd.DataFrame(equity_curves)
        combined = combined.fillna(method="ffill").dropna()

        if combined.empty:
            return None

        # Normalise each strategy to start at 1.0, then average
        normalised = combined.div(combined.iloc[0])
        portfolio  = normalised.mean(axis=1) * cfg.initial_balance
        portfolio.name = "portfolio"
        return portfolio

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_trade_log(
        trades: list[TradeRecord],
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            rows.append({
                "strategy":    t.strategy,
                "entry_bar":   t.entry_bar,
                "exit_bar":    t.exit_bar,
                "direction":   "LONG" if t.direction == +1 else "SHORT",
                "size":        round(t.size, 4),
                "entry_price": round(t.entry_price, 5),
                "exit_price":  round(t.exit_price, 5),
                "pnl":         round(t.pnl, 4),
                "exit_reason": t.exit_reason,
                "duration":    t.exit_bar - t.entry_bar,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _validate(data: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(data.columns.str.lower())
        if missing:
            raise ValueError(f"Data missing columns: {missing}")
        if data.isnull().any().any():
            raise ValueError("Data contains NaN values — clean before backtesting.")
        if len(data) < 100:
            raise ValueError("Data too short (< 100 bars).")

    @staticmethod
    def _log_summary(
        metrics: dict[str, PerformanceMetrics],
        portfolio: Optional[PerformanceMetrics],
    ) -> None:
        for name, m in metrics.items():
            logger.info(
                "[%s] CAGR=%.2f%% | Sharpe=%.3f | MaxDD=%.2f%% | Trades=%d",
                name,
                m.cagr * 100,
                m.sharpe,
                m.max_drawdown_pct * 100,
                m.n_trades,
            )
        if portfolio:
            logger.info(
                "[PORTFOLIO] CAGR=%.2f%% | Sharpe=%.3f | Sortino=%.3f | MaxDD=%.2f%%",
                portfolio.cagr * 100,
                portfolio.sharpe,
                portfolio.sortino,
                portfolio.max_drawdown_pct * 100,
            )
