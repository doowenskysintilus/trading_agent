"""
Performance Metrics
===================
Institutional-grade statistics computed from an equity curve and
return series.

Metrics
-------
- CAGR
- Sharpe ratio (annualised)
- Sortino ratio (annualised)
- Max drawdown & recovery time
- Calmar ratio
- Win rate, avg win / avg loss, profit factor
- VaR (95 %) and CVaR / Expected Shortfall
- Skewness, kurtosis
- Exposure time (pct of bars with open position)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for all computed performance statistics."""

    # ---- Return metrics --------------------------------------------------
    total_return_pct: float     = 0.0
    cagr: float                 = 0.0

    # ---- Risk-adjusted ---------------------------------------------------
    sharpe: float               = 0.0
    sortino: float              = 0.0
    calmar: float               = 0.0

    # ---- Drawdown --------------------------------------------------------
    max_drawdown_pct: float     = 0.0
    avg_drawdown_pct: float     = 0.0
    max_drawdown_duration: int  = 0    # bars to recovery

    # ---- Trade stats -----------------------------------------------------
    n_trades: int               = 0
    win_rate: float             = 0.0
    avg_win: float              = 0.0
    avg_loss: float             = 0.0
    profit_factor: float        = 0.0
    avg_trade_return: float     = 0.0
    best_trade: float           = 0.0
    worst_trade: float          = 0.0

    # ---- Tail risk -------------------------------------------------------
    var_95: float               = 0.0     # daily VaR at 95 %
    cvar_95: float              = 0.0     # Expected Shortfall
    skewness: float             = 0.0
    excess_kurtosis: float      = 0.0

    # ---- Activity --------------------------------------------------------
    exposure_pct: float         = 0.0     # fraction of bars with open position
    volatility_annual: float    = 0.0

    # ---- Meta ------------------------------------------------------------
    strategy_name: str          = ""
    start_date: Optional[str]   = None
    end_date: Optional[str]     = None
    n_bars: int                 = 0

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics("
            f"CAGR={self.cagr:.2%}, "
            f"Sharpe={self.sharpe:.3f}, "
            f"Sortino={self.sortino:.3f}, "
            f"MaxDD={self.max_drawdown_pct:.2%}, "
            f"WinRate={self.win_rate:.2%})"
        )


class MetricsCalculator:
    """
    Computes PerformanceMetrics from an equity curve and optional trade log.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (default 0.02 = 2 %).
    periods_per_year : int
        Number of bars per year (252 for daily, 8760 for hourly, etc.).
    """

    def __init__(
        self,
        risk_free_rate: float    = 0.02,
        periods_per_year: int    = 252,
    ) -> None:
        self.risk_free_rate    = risk_free_rate
        self.periods_per_year  = periods_per_year

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        equity_curve: pd.Series,
        trade_pnls: Optional[list[float]] = None,
        exposure_mask: Optional[pd.Series] = None,
        strategy_name: str = "",
    ) -> PerformanceMetrics:
        """
        Compute full performance metrics.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio value at each bar (indexed by datetime or int).
        trade_pnls : list[float] | None
            Net PnL of each closed trade. If None, derived from equity curve.
        exposure_mask : pd.Series[bool] | None
            True when a position is open at that bar.
        strategy_name : str

        Returns
        -------
        PerformanceMetrics
        """
        if len(equity_curve) < 2:
            return PerformanceMetrics(strategy_name=strategy_name)

        eq      = equity_curve.dropna().astype(float)
        returns = eq.pct_change().dropna()
        ppy     = self.periods_per_year
        rfr_bar = (1 + self.risk_free_rate) ** (1 / ppy) - 1

        m = PerformanceMetrics(strategy_name=strategy_name)
        m.n_bars = len(eq)

        # ---- Dates -------------------------------------------------------
        if hasattr(eq.index, "strftime"):
            m.start_date = str(eq.index[0])
            m.end_date   = str(eq.index[-1])

        # ---- Return metrics ----------------------------------------------
        m.total_return_pct = float((eq.iloc[-1] / eq.iloc[0]) - 1)
        n_years = len(eq) / ppy
        m.cagr  = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(n_years, 1e-6)) - 1)

        # ---- Volatility --------------------------------------------------
        m.volatility_annual = float(returns.std(ddof=1) * math.sqrt(ppy))

        # ---- Sharpe ------------------------------------------------------
        excess    = returns - rfr_bar
        m.sharpe  = float(
            excess.mean() / (excess.std(ddof=1) + 1e-10) * math.sqrt(ppy)
        )

        # ---- Sortino (downside deviation only) ---------------------------
        downside   = returns[returns < rfr_bar] - rfr_bar
        down_std   = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 1e-10
        m.sortino  = float(
            (returns.mean() - rfr_bar) / (down_std + 1e-10) * math.sqrt(ppy)
        )

        # ---- Drawdown ----------------------------------------------------
        m.max_drawdown_pct, m.avg_drawdown_pct, m.max_drawdown_duration = \
            self._drawdown_stats(eq)

        # ---- Calmar ------------------------------------------------------
        m.calmar = float(m.cagr / (abs(m.max_drawdown_pct) + 1e-10))

        # ---- Trade stats -------------------------------------------------
        pnls = trade_pnls or self._estimate_trade_pnls(eq)
        if pnls:
            m.n_trades         = len(pnls)
            wins               = [p for p in pnls if p > 0]
            losses             = [p for p in pnls if p <= 0]
            m.win_rate         = len(wins) / len(pnls)
            m.avg_win          = float(np.mean(wins))   if wins   else 0.0
            m.avg_loss         = float(np.mean(losses)) if losses else 0.0
            m.profit_factor    = (
                sum(wins) / (abs(sum(losses)) + 1e-10) if losses else float("inf")
            )
            m.avg_trade_return = float(np.mean(pnls))
            m.best_trade       = float(max(pnls))
            m.worst_trade      = float(min(pnls))

        # ---- Tail risk ---------------------------------------------------
        m.var_95  = float(np.percentile(returns, 5))
        tail      = returns[returns <= m.var_95]
        m.cvar_95 = float(tail.mean()) if len(tail) > 0 else m.var_95

        ret_arr          = returns.values
        m.skewness       = float(self._skewness(ret_arr))
        m.excess_kurtosis = float(self._kurtosis(ret_arr) - 3)

        # ---- Exposure ----------------------------------------------------
        if exposure_mask is not None and len(exposure_mask) > 0:
            m.exposure_pct = float(exposure_mask.mean())

        return m

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    @staticmethod
    def _drawdown_stats(eq: pd.Series) -> tuple[float, float, int]:
        """Returns (max_dd, avg_dd, max_duration_bars)."""
        hwm        = eq.cummax()
        dd_series  = (eq - hwm) / (hwm + 1e-10)

        max_dd = float(dd_series.min())
        avg_dd = float(dd_series[dd_series < 0].mean()) if (dd_series < 0).any() else 0.0

        # Max recovery duration
        in_drawdown  = dd_series < 0
        max_duration = 0
        current_dur  = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_dur += 1
                max_duration = max(max_duration, current_dur)
            else:
                current_dur = 0

        return max_dd, avg_dd, max_duration

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_trade_pnls(eq: pd.Series) -> list[float]:
        """
        Rough estimate: treat every sign-change in returns as a trade.
        For accurate stats pass actual trade_pnls.
        """
        returns = eq.pct_change().dropna()
        sign    = np.sign(returns.values)
        trades: list[float] = []
        current_pnl = 0.0
        for i, (ret, s) in enumerate(zip(returns.values, sign)):
            if i == 0:
                current_pnl = ret
                prev_sign   = s
                continue
            if s != prev_sign and prev_sign != 0:
                trades.append(current_pnl)
                current_pnl = ret
            else:
                current_pnl += ret
            prev_sign = s
        if current_pnl != 0:
            trades.append(current_pnl)
        return trades

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        n  = len(x)
        if n < 3:
            return 0.0
        m  = x.mean()
        s  = x.std(ddof=1) + 1e-10
        return float(np.mean(((x - m) / s) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        n  = len(x)
        if n < 4:
            return 3.0
        m  = x.mean()
        s  = x.std(ddof=1) + 1e-10
        return float(np.mean(((x - m) / s) ** 4))


# ---------------------------------------------------------------------------
# Pretty-print a comparison table of multiple strategies
# ---------------------------------------------------------------------------

def compare_metrics(results: dict[str, PerformanceMetrics]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of {name: PerformanceMetrics}.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Strategy":     name,
            "CAGR":         f"{m.cagr:.2%}",
            "Sharpe":       f"{m.sharpe:.3f}",
            "Sortino":      f"{m.sortino:.3f}",
            "Calmar":       f"{m.calmar:.3f}",
            "MaxDD":        f"{m.max_drawdown_pct:.2%}",
            "WinRate":      f"{m.win_rate:.2%}",
            "ProfitFactor": f"{m.profit_factor:.2f}",
            "Trades":       m.n_trades,
            "VaR95":        f"{m.var_95:.4%}",
            "CVaR95":       f"{m.cvar_95:.4%}",
            "Skew":         f"{m.skewness:.3f}",
        })
    return pd.DataFrame(rows).set_index("Strategy")
