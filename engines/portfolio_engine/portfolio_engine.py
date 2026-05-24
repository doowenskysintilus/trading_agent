"""
PortfolioEngine
===============
Dynamic capital allocation engine for a multi-strategy quant fund.

Allocation methods
------------------
EQUAL_WEIGHT        : uniform allocation across active strategies
RISK_PARITY         : weight inversely proportional to return volatility
PERFORMANCE_WEIGHTED: weight proportional to risk-adjusted recent PnL
KELLY               : fractional Kelly sizing per strategy
CUSTOM              : user-supplied fixed weights

Features
--------
- Per-strategy PnL and return tracking (rolling window)
- Automatic weight adjustment on rebalance()
- Soft floor weight (no strategy fully starved)
- Drawdown-triggered de-allocation per strategy
- Full audit trail via AllocationResult

Entry points
------------
    allocator = PortfolioAllocator(total_capital=1_000_000)
    allocator.register("momentum",      initial_weight=1.0)
    allocator.register("mean_reversion", initial_weight=1.0)
    allocator.register("rl_agent",       initial_weight=1.5)

    allocator.record_pnl("momentum", pnl=1200, trade_duration=8)
    result = allocator.rebalance(method=AllocationMethod.RISK_PARITY)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AllocationMethod(str, Enum):
    EQUAL_WEIGHT         = "equal_weight"
    RISK_PARITY          = "risk_parity"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    KELLY                = "kelly"
    CUSTOM               = "custom"


class StrategyStatus(str, Enum):
    ACTIVE   = "active"
    REDUCED  = "reduced"    # drawdown-triggered half allocation
    SUSPENDED = "suspended" # max drawdown breached → 0 allocation


# ---------------------------------------------------------------------------
# Per-strategy state
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Rolling performance statistics for one strategy."""

    name: str
    base_weight: float          # static hint weight set at registration
    status: StrategyStatus = StrategyStatus.ACTIVE

    # Rolling history
    pnl_history: deque     = field(default_factory=lambda: deque(maxlen=50))
    return_history: deque  = field(default_factory=lambda: deque(maxlen=50))

    # Cumulative
    total_pnl: float       = 0.0
    n_trades: int          = 0
    n_wins: int            = 0
    peak_pnl: float        = 0.0
    strategy_drawdown: float = 0.0

    # Current allocation (set by engine)
    allocated_capital: float = 0.0
    current_weight: float    = 0.0

    def record(self, pnl: float, allocated_at: float) -> None:
        self.pnl_history.append(pnl)
        ret = pnl / (allocated_at + 1e-10)
        self.return_history.append(ret)
        self.total_pnl += pnl
        self.n_trades  += 1
        if pnl > 0:
            self.n_wins += 1
        # Strategy-level drawdown tracking
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        self.strategy_drawdown = (
            (self.peak_pnl - self.total_pnl) / (abs(self.peak_pnl) + 1e-10)
            if self.peak_pnl > 0 else 0.0
        )

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades > 0 else 0.5

    @property
    def mean_return(self) -> float:
        if not self.return_history:
            return 0.0
        return float(np.mean(list(self.return_history)))

    @property
    def volatility(self) -> float:
        if len(self.return_history) < 2:
            return 1.0  # neutral fallback
        return float(np.std(list(self.return_history), ddof=1)) + 1e-10

    @property
    def sharpe(self) -> float:
        return self.mean_return / (self.volatility + 1e-10)

    @property
    def avg_win(self) -> float:
        wins = [p for p in self.pnl_history if p > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(p) for p in self.pnl_history if p < 0]
        return float(np.mean(losses)) if losses else 1.0

    def kelly_fraction(self, kelly_fraction: float = 0.25) -> float:
        """
        Fractional Kelly: f* = p - (1-p)/b, where b = avg_win/avg_loss.
        Clamped to [0, 1], then scaled by kelly_fraction.
        """
        p = self.win_rate
        b = self.avg_win / (self.avg_loss + 1e-10)
        f_star = p - (1 - p) / (b + 1e-10)
        return float(np.clip(f_star, 0.0, 1.0)) * kelly_fraction

    def summary(self) -> dict:
        return {
            "status":       self.status.value,
            "weight":       round(self.current_weight, 4),
            "allocated":    round(self.allocated_capital, 2),
            "total_pnl":    round(self.total_pnl, 2),
            "n_trades":     self.n_trades,
            "win_rate":     round(self.win_rate, 3),
            "sharpe":       round(self.sharpe, 3),
            "drawdown":     round(self.strategy_drawdown, 4),
            "volatility":   round(self.volatility, 6),
        }


# ---------------------------------------------------------------------------
# Allocation result
# ---------------------------------------------------------------------------

@dataclass
class AllocationResult:
    """
    Output of every allocate() / rebalance() call.

    Attributes
    ----------
    method : AllocationMethod
    weights : dict[str, float]
        Normalised weights (sum ≈ 1.0 over active strategies).
    allocations : dict[str, float]
        Capital in base currency assigned to each strategy.
    total_capital : float
        Total capital distributed.
    rebalanced : bool
        True when weights changed since last allocation.
    metadata : dict
        Per-strategy breakdown and diagnostics.
    """

    method: AllocationMethod
    weights: dict[str, float]
    allocations: dict[str, float]
    total_capital: float
    rebalanced: bool = False
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        alloc_str = ", ".join(
            f"{k}={v:,.0f}" for k, v in self.allocations.items()
        )
        return (
            f"AllocationResult(method={self.method.value}, "
            f"total={self.total_capital:,.0f}, [{alloc_str}])"
        )


# ---------------------------------------------------------------------------
# Portfolio Engine
# ---------------------------------------------------------------------------

class PortfolioAllocator:
    """
    Allocates capital across multiple strategies with dynamic rebalancing.

    Parameters
    ----------
    total_capital : float
        Total capital available for allocation.
    floor_weight : float
        Minimum weight fraction any active strategy receives (prevents
        starvation). Default 0.05 (5 %).
    strategy_max_drawdown : float
        Per-strategy drawdown that triggers a REDUCED status.
    strategy_kill_drawdown : float
        Per-strategy drawdown that triggers SUSPENDED status (0 allocation).
    kelly_fraction : float
        Fractional Kelly multiplier (e.g. 0.25 = quarter-Kelly).
    performance_window : int
        Trailing N trades used for performance-based allocation.
    """

    def __init__(
        self,
        total_capital: float,
        floor_weight: float          = 0.05,
        strategy_max_drawdown: float = 0.20,
        strategy_kill_drawdown: float = 0.35,
        kelly_fraction: float        = 0.25,
        performance_window: int      = 20,
    ) -> None:
        self.total_capital          = total_capital
        self.floor_weight           = floor_weight
        self.strategy_max_drawdown  = strategy_max_drawdown
        self.strategy_kill_drawdown = strategy_kill_drawdown
        self.kelly_fraction         = kelly_fraction
        self.performance_window     = performance_window

        self._strategies: dict[str, StrategyStats] = {}
        self._custom_weights: dict[str, float]     = {}
        self._last_result: Optional[AllocationResult] = None

    # ------------------------------------------------------------------
    # Registration & management
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        initial_weight: float = 1.0,
        custom_weight: Optional[float] = None,
    ) -> None:
        """Register a strategy for allocation tracking."""
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered.")
        self._strategies[name] = StrategyStats(name=name, base_weight=initial_weight)
        if custom_weight is not None:
            self._custom_weights[name] = custom_weight
        logger.info("Registered strategy '%s' with base weight %.2f", name, initial_weight)

    def deregister(self, name: str) -> None:
        self._strategies.pop(name, None)
        self._custom_weights.pop(name, None)

    def suspend(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].status = StrategyStatus.SUSPENDED

    def reactivate(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].status = StrategyStatus.ACTIVE

    def update_capital(self, new_total: float) -> None:
        """Update total capital (e.g. after deposit / withdrawal)."""
        self.total_capital = new_total

    # ------------------------------------------------------------------
    # PnL recording
    # ------------------------------------------------------------------

    def record_pnl(
        self,
        strategy: str,
        pnl: float,
        trade_duration: int = 1,
    ) -> None:
        """
        Record a closed trade outcome.

        Parameters
        ----------
        strategy : str
            Strategy name (must be registered).
        pnl : float
            Net PnL of the trade.
        trade_duration : int
            Number of bars the trade was open (unused currently, reserved).
        """
        if strategy not in self._strategies:
            raise KeyError(f"Strategy '{strategy}' is not registered.")

        stats = self._strategies[strategy]
        stats.record(pnl, allocated_at=stats.allocated_capital)

        # Auto-update strategy status based on per-strategy drawdown
        if stats.strategy_drawdown >= self.strategy_kill_drawdown:
            if stats.status != StrategyStatus.SUSPENDED:
                stats.status = StrategyStatus.SUSPENDED
                logger.warning(
                    "Strategy '%s' SUSPENDED — drawdown %.1f%%",
                    strategy, stats.strategy_drawdown * 100,
                )
        elif stats.strategy_drawdown >= self.strategy_max_drawdown:
            if stats.status == StrategyStatus.ACTIVE:
                stats.status = StrategyStatus.REDUCED
                logger.warning(
                    "Strategy '%s' REDUCED — drawdown %.1f%%",
                    strategy, stats.strategy_drawdown * 100,
                )
        else:
            # Recovery
            if stats.status == StrategyStatus.REDUCED:
                stats.status = StrategyStatus.ACTIVE

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        method: AllocationMethod = AllocationMethod.RISK_PARITY,
        total_capital: Optional[float] = None,
    ) -> AllocationResult:
        """
        Compute capital allocation across all registered strategies.

        Parameters
        ----------
        method : AllocationMethod
        total_capital : float | None
            Override stored total capital for this call only.

        Returns
        -------
        AllocationResult
        """
        capital = total_capital or self.total_capital

        if not self._strategies:
            raise RuntimeError("No strategies registered.")

        raw_weights = self._compute_weights(method)
        weights     = self._apply_status_adjustments(raw_weights)
        weights     = self._normalise(weights)
        allocations = {k: round(v * capital, 2) for k, v in weights.items()}

        rebalanced = self._detect_rebalance(weights)

        # Persist allocation into each StrategyStats
        for name, alloc in allocations.items():
            self._strategies[name].allocated_capital = alloc
            self._strategies[name].current_weight    = weights[name]

        result = AllocationResult(
            method=method,
            weights={k: round(v, 6) for k, v in weights.items()},
            allocations=allocations,
            total_capital=capital,
            rebalanced=rebalanced,
            metadata={
                "raw_weights":     {k: round(v, 6) for k, v in raw_weights.items()},
                "strategy_status": {
                    n: s.status.value for n, s in self._strategies.items()
                },
            },
        )
        self._last_result = result

        logger.info(
            "Allocation [%s] — total: %.0f — weights: %s",
            method.value,
            capital,
            {k: f"{v:.1%}" for k, v in weights.items()},
        )
        return result

    def rebalance(
        self,
        method: AllocationMethod = AllocationMethod.RISK_PARITY,
        total_capital: Optional[float] = None,
    ) -> AllocationResult:
        """
        Alias for allocate() — semantically indicates a rebalance event.
        Logs a rebalance notice and returns the new AllocationResult.
        """
        logger.info("Rebalancing portfolio using method: %s", method.value)
        return self.allocate(method=method, total_capital=total_capital)

    # ------------------------------------------------------------------
    # Weight computation methods
    # ------------------------------------------------------------------

    def _compute_weights(self, method: AllocationMethod) -> dict[str, float]:
        methods = {
            AllocationMethod.EQUAL_WEIGHT:         self._equal_weight,
            AllocationMethod.RISK_PARITY:          self._risk_parity,
            AllocationMethod.PERFORMANCE_WEIGHTED: self._performance_weighted,
            AllocationMethod.KELLY:                self._kelly,
            AllocationMethod.CUSTOM:               self._custom,
        }
        return methods[method]()

    def _equal_weight(self) -> dict[str, float]:
        n = len(self._strategies)
        return {name: 1.0 / n for name in self._strategies}

    def _risk_parity(self) -> dict[str, float]:
        """Inverse-volatility weighting."""
        inv_vol = {
            name: 1.0 / stats.volatility
            for name, stats in self._strategies.items()
        }
        total = sum(inv_vol.values()) + 1e-10
        return {k: v / total for k, v in inv_vol.items()}

    def _performance_weighted(self) -> dict[str, float]:
        """
        Sharpe-weighted allocation.
        Strategies with Sharpe < 0 receive the floor weight only.
        """
        sharpes = {
            name: max(stats.sharpe, 0.0)
            for name, stats in self._strategies.items()
        }
        total = sum(sharpes.values()) + 1e-10

        # If all sharpes are 0 (no history), fall back to equal weight
        if total < 1e-8:
            return self._equal_weight()

        return {k: v / total for k, v in sharpes.items()}

    def _kelly(self) -> dict[str, float]:
        """
        Fractional Kelly per strategy, normalised to sum = 1.
        Strategies with no trade history get equal share of residual capital.
        """
        kelly_fractions: dict[str, float] = {}
        no_history: list[str] = []

        for name, stats in self._strategies.items():
            if stats.n_trades < 5:
                no_history.append(name)
            else:
                kelly_fractions[name] = stats.kelly_fraction(self.kelly_fraction)

        total_kelly = sum(kelly_fractions.values())

        if not kelly_fractions:
            return self._equal_weight()

        # Reserve a slice for strategies without history
        history_share = 0.7 if no_history else 1.0
        no_hist_share = 1.0 - history_share

        weights: dict[str, float] = {}
        for name, f in kelly_fractions.items():
            weights[name] = (f / (total_kelly + 1e-10)) * history_share

        if no_history:
            share_each = no_hist_share / len(no_history)
            for name in no_history:
                weights[name] = share_each

        return weights

    def _custom(self) -> dict[str, float]:
        """Use user-supplied fixed weights; fall back to base_weight."""
        weights = {
            name: self._custom_weights.get(name, stats.base_weight)
            for name, stats in self._strategies.items()
        }
        return weights

    # ------------------------------------------------------------------
    # Status adjustments and normalisation
    # ------------------------------------------------------------------

    def _apply_status_adjustments(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply per-strategy status modifiers:
        - SUSPENDED  → weight = 0
        - REDUCED    → weight × 0.5
        - ACTIVE     → unchanged
        """
        adjusted: dict[str, float] = {}
        for name, w in weights.items():
            status = self._strategies[name].status
            if status == StrategyStatus.SUSPENDED:
                adjusted[name] = 0.0
            elif status == StrategyStatus.REDUCED:
                adjusted[name] = w * 0.5
            else:
                adjusted[name] = w
        return adjusted

    def _normalise(self, weights: dict[str, float]) -> dict[str, float]:
        """
        Normalise weights to sum = 1.
        Apply floor_weight to all active (non-zero) strategies.
        """
        active = {k: v for k, v in weights.items() if v > 0}
        if not active:
            # All suspended — equal weight among all (safety fallback)
            n = len(weights)
            return {k: 1.0 / n for k in weights}

        n_active = len(active)
        floor    = self.floor_weight
        max_floor_share = floor * n_active

        if max_floor_share >= 1.0:
            # Too many strategies, floor would exceed 100% — use equal weight
            share = 1.0 / n_active
            return {k: (share if v > 0 else 0.0) for k, v in weights.items()}

        # Apply floor: each active strategy gets at least `floor` fraction
        remaining = 1.0 - max_floor_share
        total_raw = sum(active.values()) + 1e-10
        normalised: dict[str, float] = {}

        for name, w in weights.items():
            if w <= 0:
                normalised[name] = 0.0
            else:
                normalised[name] = floor + (w / total_raw) * remaining

        # Re-normalise to handle floating point drift
        total = sum(normalised.values()) + 1e-10
        return {k: v / total for k, v in normalised.items()}

    # ------------------------------------------------------------------
    # Rebalance detection
    # ------------------------------------------------------------------

    def _detect_rebalance(self, new_weights: dict[str, float]) -> bool:
        if self._last_result is None:
            return True
        old = self._last_result.weights
        for name, w in new_weights.items():
            if abs(w - old.get(name, 0.0)) > 0.01:   # 1 % drift threshold
                return True
        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a complete per-strategy performance and allocation summary."""
        return {
            name: stats.summary()
            for name, stats in self._strategies.items()
        }

    def get_pnl_attribution(self) -> dict[str, float]:
        """Total PnL contribution per strategy."""
        return {
            name: round(stats.total_pnl, 2)
            for name, stats in self._strategies.items()
        }

    def total_pnl(self) -> float:
        return sum(s.total_pnl for s in self._strategies.values())

    def __repr__(self) -> str:
        active = sum(
            1 for s in self._strategies.values()
            if s.status == StrategyStatus.ACTIVE
        )
        return (
            f"PortfolioAllocator("
            f"capital={self.total_capital:,.0f}, "
            f"strategies={len(self._strategies)}, "
            f"active={active})"
        )
