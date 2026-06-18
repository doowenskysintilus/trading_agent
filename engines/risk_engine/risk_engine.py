"""
RiskEngine
==========
Hedge fund-grade global risk engine.

Evaluates proposed trades against live portfolio state across six
independent risk layers (plus Layer 0: economic calendar blackout).
Returns APPROVE / REDUCE / BLOCK.

Risk layers (evaluated in order — first block wins)
----------------------------------------------------
0. Event Blackout       emergency gate on major economic events
1. Kill Switch          hard circuit-breaker on daily loss / drawdown
2. Drawdown Control     portfolio-wide HWM drawdown ceiling
3. Exposure Limits      per-asset and per-strategy notional caps
4. Volatility Sizing    ATR-adjusted max position size (event-aware)
5. Correlation Risk     portfolio-level concentration check
6. Leverage Check       total gross exposure vs equity

Entry point
-----------
    engine = RiskEngine(config=RiskConfig(), calendar_provider=None)
    decision = engine.evaluate_portfolio_risk(trades, portfolio_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

try:
    from research.feature_store.calendar_provider import CalendarProvider, EventImportance
    _CALENDAR_AVAILABLE = True
except ImportError:
    try:
        # Fallback for relative imports
        from ..research.feature_store.calendar_provider import CalendarProvider, EventImportance
        _CALENDAR_AVAILABLE = True
    except ImportError:
        _CALENDAR_AVAILABLE = False
        CalendarProvider = None
        EventImportance = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskVerdict(str, Enum):
    APPROVE = "APPROVE"
    REDUCE  = "REDUCE"
    BLOCK   = "BLOCK"


class RejectReason(str, Enum):
    KILL_SWITCH_DAILY_LOSS     = "kill_switch_daily_loss"
    KILL_SWITCH_DRAWDOWN       = "kill_switch_drawdown"
    KILL_SWITCH_CONSECUTIVE    = "kill_switch_consecutive_losses"
    MAX_DRAWDOWN_EXCEEDED      = "max_drawdown_exceeded"
    ASSET_EXPOSURE_EXCEEDED    = "asset_exposure_exceeded"
    STRATEGY_EXPOSURE_EXCEEDED = "strategy_exposure_exceeded"
    VOLATILITY_SIZE_TOO_LARGE  = "volatility_size_too_large"
    CORRELATION_CONCENTRATION  = "correlation_concentration"
    LEVERAGE_EXCEEDED          = "leverage_exceeded"
    INSUFFICIENT_BALANCE       = "insufficient_balance"
    EVENT_BLACKOUT             = "event_blackout"  # Economic event blackout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """All configurable limits for the risk engine."""

    # --- Kill switch -------------------------------------------------------
    max_daily_loss_pct: float       = 0.03    # 3 % of equity → hard stop
    kill_drawdown_pct: float        = 0.15    # 15 % drawdown → kill switch
    max_consecutive_losses: int     = 6       # N losses in a row → suspend

    # --- Drawdown ----------------------------------------------------------
    max_portfolio_drawdown_pct: float = 0.20  # 20 % from HWM → BLOCK new trades

    # --- Exposure ----------------------------------------------------------
    max_asset_exposure_pct: float   = 0.25   # single asset ≤ 25 % of equity
    max_strategy_exposure_pct: float = 0.40  # single strategy ≤ 40 % of equity
    max_single_trade_pct: float     = 0.10   # single trade ≤ 10 % of equity

    # --- Volatility-adjusted sizing ----------------------------------------
    risk_per_trade_pct: float       = 0.01   # risk 1 % of equity per ATR unit
    atr_multiplier: float           = 2.0    # SL distance = ATR × multiplier
    max_position_size_pct: float    = 0.15   # hard cap per position

    # --- Correlation -------------------------------------------------------
    max_correlated_exposure_pct: float = 0.50  # correlated cluster ≤ 50 % of equity
    correlation_threshold: float       = 0.70  # |ρ| above this = correlated

    # --- Leverage ----------------------------------------------------------
    max_gross_leverage: float       = 3.0    # gross notional / equity

    # --- REDUCE sizing (when soft limit hit) --------------------------------
    reduce_factor: float            = 0.50   # reduce proposed size by this factor

    # --- Economic Calendar (Phase 1) ----------------------------------------
    event_blackout_enabled: bool    = True   # enable blackout around major events
    event_blackout_hours: float     = 0.5    # don't trade within 30 min of event
    event_vol_multiplier: float     = 1.5    # scale ATR by this during events


# ---------------------------------------------------------------------------
# Input / output data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeOrder:
    """A proposed trade to be evaluated."""

    symbol: str
    strategy: str
    direction: int              # +1 long, -1 short
    size: float                 # units
    entry_price: float
    atr: float                  # current ATR of the symbol
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def notional(self) -> float:
        return abs(self.size * self.entry_price)


@dataclass
class OpenPosition:
    """A currently open position in the portfolio."""

    symbol: str
    strategy: str
    direction: int
    size: float
    entry_price: float
    current_price: float

    @property
    def notional(self) -> float:
        return abs(self.size * self.current_price)

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.direction * self.size


@dataclass
class PortfolioState:
    """Complete snapshot of the portfolio at evaluation time."""

    equity: float
    balance: float
    peak_equity: float
    daily_start_equity: float
    open_positions: list[OpenPosition] = field(default_factory=list)
    consecutive_losses: int = 0
    returns_matrix: Optional[pd.DataFrame] = None  # columns = symbols, rows = bars

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def daily_loss_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return (self.daily_start_equity - self.equity) / self.daily_start_equity

    @property
    def gross_notional(self) -> float:
        return sum(p.notional for p in self.open_positions)

    @property
    def gross_leverage(self) -> float:
        return self.gross_notional / (self.equity + 1e-10)


@dataclass
class RiskCheckResult:
    """Result of an individual risk check."""

    name: str
    passed: bool
    verdict: RiskVerdict
    reason: Optional[RejectReason] = None
    message: str = ""
    suggested_size: Optional[float] = None


@dataclass
class RiskDecision:
    """
    Final output of evaluate_portfolio_risk().

    Attributes
    ----------
    verdict : RiskVerdict
        APPROVE / REDUCE / BLOCK
    approved_trades : list[TradeOrder]
        Trades approved as-is or with reduced size (REDUCE case).
    blocked_trades  : list[TradeOrder]
        Trades that were blocked.
    checks : list[RiskCheckResult]
        Full audit trail of every risk layer evaluated.
    """

    verdict: RiskVerdict
    approved_trades: list[TradeOrder] = field(default_factory=list)
    blocked_trades:  list[TradeOrder] = field(default_factory=list)
    checks: list[RiskCheckResult]     = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVE

    @property
    def blocking_reasons(self) -> list[str]:
        return [c.reason.value for c in self.checks if not c.passed and c.reason]

    def __repr__(self) -> str:
        return (
            f"RiskDecision(verdict={self.verdict.value}, "
            f"approved={len(self.approved_trades)}, "
            f"blocked={len(self.blocked_trades)}, "
            f"reasons={self.blocking_reasons})"
        )


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Evaluates one or more proposed TradeOrders against the live PortfolioState
    and returns a RiskDecision (APPROVE / REDUCE / BLOCK).

    Parameters
    ----------
    config : RiskConfig
        All configurable thresholds.
    calendar_provider : CalendarProvider, optional
        Economic calendar provider for event-aware sizing. If None, Layer 0 is skipped.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        calendar_provider: Optional[CalendarProvider] = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.calendar_provider = calendar_provider
        self._kill_switch_active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_portfolio_risk(
        self,
        trades: list[TradeOrder],
        portfolio_state: PortfolioState,
    ) -> RiskDecision:
        """
        Evaluate proposed trades against current portfolio risk state.

        Parameters
        ----------
        trades : list[TradeOrder]
            One or more proposed orders to evaluate.
        portfolio_state : PortfolioState
            Current live portfolio snapshot.

        Returns
        -------
        RiskDecision  with verdict APPROVE / REDUCE / BLOCK
        """
        checks: list[RiskCheckResult] = []
        approved: list[TradeOrder] = []
        blocked:  list[TradeOrder] = []

        # ---- Layer 0: Economic Event Blackout (pre-check) ----------------
        # If enabled and calendar available, check for nearby major events
        if self.config.event_blackout_enabled and self.calendar_provider and _CALENDAR_AVAILABLE:
            event_blocked = []
            for trade in trades:
                event_check = self._check_event_blackout(trade)
                checks.append(event_check)
                if not event_check.passed:
                    event_blocked.append(trade)
            
            if event_blocked:
                logger.info(
                    "EVENT BLACKOUT: %d trade(s) blocked due to nearby economic events",
                    len(event_blocked)
                )
                blocked.extend(event_blocked)
                trades = [t for t in trades if t not in event_blocked]
            
            # If all trades blocked, return early
            if not trades:
                return RiskDecision(
                    verdict=RiskVerdict.BLOCK,
                    blocked_trades=blocked,
                    checks=checks,
                    metadata={"event_blackout": True},
                )

        # ---- Layer 1: Kill switch (portfolio-wide) ----------------------
        ks_check = self._check_kill_switch(portfolio_state)
        checks.append(ks_check)

        if not ks_check.passed:
            self._kill_switch_active = True
            logger.warning("KILL SWITCH ACTIVATED: %s", ks_check.message)
            return RiskDecision(
                verdict=RiskVerdict.BLOCK,
                blocked_trades=list(trades),
                checks=checks,
                metadata={"kill_switch": True},
            )

        # Re-arm kill switch if conditions cleared
        self._kill_switch_active = False

        # ---- Layer 2: Portfolio drawdown --------------------------------
        dd_check = self._check_drawdown(portfolio_state)
        checks.append(dd_check)

        if not dd_check.passed:
            return RiskDecision(
                verdict=RiskVerdict.BLOCK,
                blocked_trades=list(trades),
                checks=checks,
            )

        # ---- Per-trade evaluation (layers 3-6) --------------------------
        for trade in trades:
            trade_checks: list[RiskCheckResult] = []
            current_size  = trade.size
            final_verdict = RiskVerdict.APPROVE

            # Layer 3: Exposure limits
            exp_check = self._check_exposure(trade, portfolio_state, current_size)
            trade_checks.append(exp_check)
            if not exp_check.passed:
                if exp_check.verdict == RiskVerdict.REDUCE and exp_check.suggested_size:
                    current_size  = exp_check.suggested_size
                    final_verdict = RiskVerdict.REDUCE
                else:
                    final_verdict = RiskVerdict.BLOCK

            # Layer 4: Volatility-adjusted sizing
            if final_verdict != RiskVerdict.BLOCK:
                vol_check = self._check_volatility_size(trade, portfolio_state, current_size)
                trade_checks.append(vol_check)
                if not vol_check.passed:
                    if vol_check.suggested_size:
                        current_size  = min(current_size, vol_check.suggested_size)
                        final_verdict = RiskVerdict.REDUCE
                    else:
                        final_verdict = RiskVerdict.BLOCK

            # Layer 5: Correlation concentration
            if final_verdict != RiskVerdict.BLOCK:
                corr_check = self._check_correlation(trade, portfolio_state)
                trade_checks.append(corr_check)
                if not corr_check.passed:
                    if corr_check.verdict == RiskVerdict.REDUCE:
                        current_size  = round(current_size * self.config.reduce_factor, 6)
                        final_verdict = RiskVerdict.REDUCE
                    else:
                        final_verdict = RiskVerdict.BLOCK

            # Layer 6: Leverage
            if final_verdict != RiskVerdict.BLOCK:
                lev_check = self._check_leverage(trade, portfolio_state, current_size)
                trade_checks.append(lev_check)
                if not lev_check.passed:
                    if lev_check.suggested_size:
                        current_size  = min(current_size, lev_check.suggested_size)
                        final_verdict = RiskVerdict.REDUCE
                    else:
                        final_verdict = RiskVerdict.BLOCK

            checks.extend(trade_checks)

            if final_verdict == RiskVerdict.BLOCK:
                blocked.append(trade)
            else:
                # Apply final approved size
                approved_trade = TradeOrder(
                    symbol       = trade.symbol,
                    strategy     = trade.strategy,
                    direction    = trade.direction,
                    size         = round(current_size, 6),
                    entry_price  = trade.entry_price,
                    atr          = trade.atr,
                    stop_loss    = trade.stop_loss,
                    take_profit  = trade.take_profit,
                )
                approved.append(approved_trade)

        # ---- Overall verdict -------------------------------------------
        if blocked and not approved:
            overall = RiskVerdict.BLOCK
        elif any(t.size < orig.size for t, orig in zip(approved, trades)):
            overall = RiskVerdict.REDUCE
        else:
            overall = RiskVerdict.APPROVE

        self._log_decision(overall, checks)

        return RiskDecision(
            verdict=overall,
            approved_trades=approved,
            blocked_trades=blocked,
            checks=checks,
            metadata={
                "portfolio_drawdown": round(portfolio_state.drawdown, 4),
                "gross_leverage":     round(portfolio_state.gross_leverage, 4),
                "equity":             portfolio_state.equity,
            },
        )

    # ------------------------------------------------------------------    # Layer 0 — Economic Event Blackout
    # ------------------------------------------------------------------

    def _check_event_blackout(self, trade: TradeOrder) -> RiskCheckResult:
        """
        Check if there's a high-impact economic event near the trade time.
        
        If event_blackout_enabled and calendar_provider is active:
        - BLOCK trades if event within blackout_hours
        - Returns reason EVENT_BLACKOUT on reject
        """
        cfg = self.config
        
        if not cfg.event_blackout_enabled or not self.calendar_provider:
            return RiskCheckResult(
                name="event_blackout",
                passed=True,
                verdict=RiskVerdict.APPROVE,
            )
        
        try:
            # Check if a high-impact event is happening or about to happen
            hours_until = self.calendar_provider.hours_until_next_event(
                trade.symbol,
                min_importance=EventImportance.HIGH if _CALENDAR_AVAILABLE else None,
            )
            
            if hours_until is None:
                # No upcoming events
                return RiskCheckResult(
                    name="event_blackout",
                    passed=True,
                    verdict=RiskVerdict.APPROVE,
                )
            
            if hours_until <= cfg.event_blackout_hours:
                return RiskCheckResult(
                    name="event_blackout",
                    passed=False,
                    verdict=RiskVerdict.BLOCK,
                    reason=RejectReason.EVENT_BLACKOUT,
                    message=(
                        f"High-impact event in {hours_until:.2f}h ≤ "
                        f"blackout {cfg.event_blackout_hours:.2f}h — "
                        f"trade blocked for {trade.symbol}"
                    ),
                )
        except Exception as e:
            logger.warning(f"Event blackout check failed: {e}. Allowing trade.")
            # Graceful fallback: don't block on provider errors
            pass
        
        return RiskCheckResult(
            name="event_blackout",
            passed=True,
            verdict=RiskVerdict.APPROVE,
        )

    # ------------------------------------------------------------------    # Layer 1 — Kill switch
    # ------------------------------------------------------------------

    def _check_kill_switch(self, ps: PortfolioState) -> RiskCheckResult:
        cfg = self.config

        if ps.daily_loss_pct >= cfg.max_daily_loss_pct:
            return RiskCheckResult(
                name="kill_switch",
                passed=False,
                verdict=RiskVerdict.BLOCK,
                reason=RejectReason.KILL_SWITCH_DAILY_LOSS,
                message=(
                    f"Daily loss {ps.daily_loss_pct*100:.2f}% ≥ "
                    f"limit {cfg.max_daily_loss_pct*100:.1f}%"
                ),
            )

        if ps.drawdown >= cfg.kill_drawdown_pct:
            return RiskCheckResult(
                name="kill_switch",
                passed=False,
                verdict=RiskVerdict.BLOCK,
                reason=RejectReason.KILL_SWITCH_DRAWDOWN,
                message=(
                    f"Drawdown {ps.drawdown*100:.2f}% ≥ "
                    f"kill threshold {cfg.kill_drawdown_pct*100:.1f}%"
                ),
            )

        if ps.consecutive_losses >= cfg.max_consecutive_losses:
            return RiskCheckResult(
                name="kill_switch",
                passed=False,
                verdict=RiskVerdict.BLOCK,
                reason=RejectReason.KILL_SWITCH_CONSECUTIVE,
                message=(
                    f"Consecutive losses {ps.consecutive_losses} ≥ "
                    f"limit {cfg.max_consecutive_losses}"
                ),
            )

        return RiskCheckResult(name="kill_switch", passed=True, verdict=RiskVerdict.APPROVE)

    # ------------------------------------------------------------------
    # Layer 2 — Drawdown
    # ------------------------------------------------------------------

    def _check_drawdown(self, ps: PortfolioState) -> RiskCheckResult:
        cfg = self.config
        if ps.drawdown >= cfg.max_portfolio_drawdown_pct:
            return RiskCheckResult(
                name="drawdown",
                passed=False,
                verdict=RiskVerdict.BLOCK,
                reason=RejectReason.MAX_DRAWDOWN_EXCEEDED,
                message=(
                    f"Portfolio drawdown {ps.drawdown*100:.2f}% ≥ "
                    f"max {cfg.max_portfolio_drawdown_pct*100:.1f}%"
                ),
            )
        return RiskCheckResult(name="drawdown", passed=True, verdict=RiskVerdict.APPROVE,
                               message=f"Drawdown {ps.drawdown*100:.2f}% OK")

    # ------------------------------------------------------------------
    # Layer 3 — Exposure limits
    # ------------------------------------------------------------------

    def _check_exposure(
        self,
        trade: TradeOrder,
        ps: PortfolioState,
        size: float,
    ) -> RiskCheckResult:
        cfg     = self.config
        equity  = ps.equity

        # Current notional already open for this asset
        asset_notional = sum(
            p.notional for p in ps.open_positions if p.symbol == trade.symbol
        )
        asset_notional += size * trade.entry_price
        asset_pct = asset_notional / (equity + 1e-10)

        if asset_pct > cfg.max_asset_exposure_pct:
            allowed_notional = equity * cfg.max_asset_exposure_pct
            existing         = sum(p.notional for p in ps.open_positions
                                   if p.symbol == trade.symbol)
            remaining        = max(allowed_notional - existing, 0.0)
            suggested_size   = remaining / (trade.entry_price + 1e-10)

            if suggested_size < 1e-8:
                return RiskCheckResult(
                    name="exposure_asset",
                    passed=False,
                    verdict=RiskVerdict.BLOCK,
                    reason=RejectReason.ASSET_EXPOSURE_EXCEEDED,
                    message=(
                        f"{trade.symbol} exposure {asset_pct*100:.1f}% > "
                        f"limit {cfg.max_asset_exposure_pct*100:.0f}% — no room"
                    ),
                )
            return RiskCheckResult(
                name="exposure_asset",
                passed=False,
                verdict=RiskVerdict.REDUCE,
                reason=RejectReason.ASSET_EXPOSURE_EXCEEDED,
                message=(
                    f"{trade.symbol} exposure capped → size {size:.4f} "
                    f"→ {suggested_size:.4f}"
                ),
                suggested_size=suggested_size,
            )

        # Strategy-level exposure
        strat_notional = sum(
            p.notional for p in ps.open_positions if p.strategy == trade.strategy
        )
        strat_notional += size * trade.entry_price
        strat_pct = strat_notional / (equity + 1e-10)

        if strat_pct > cfg.max_strategy_exposure_pct:
            return RiskCheckResult(
                name="exposure_strategy",
                passed=False,
                verdict=RiskVerdict.BLOCK,
                reason=RejectReason.STRATEGY_EXPOSURE_EXCEEDED,
                message=(
                    f"Strategy '{trade.strategy}' exposure {strat_pct*100:.1f}% > "
                    f"limit {cfg.max_strategy_exposure_pct*100:.0f}%"
                ),
            )

        # Single trade size cap
        trade_pct = (size * trade.entry_price) / (equity + 1e-10)
        if trade_pct > cfg.max_single_trade_pct:
            capped_size = (equity * cfg.max_single_trade_pct) / (trade.entry_price + 1e-10)
            return RiskCheckResult(
                name="exposure_single_trade",
                passed=False,
                verdict=RiskVerdict.REDUCE,
                reason=RejectReason.ASSET_EXPOSURE_EXCEEDED,
                message=f"Trade size capped: {size:.4f} → {capped_size:.4f}",
                suggested_size=capped_size,
            )

        return RiskCheckResult(name="exposure", passed=True, verdict=RiskVerdict.APPROVE)

    # ------------------------------------------------------------------
    # Layer 4 — Volatility-adjusted sizing
    # ------------------------------------------------------------------

    def _check_volatility_size(
        self,
        trade: TradeOrder,
        ps: PortfolioState,
        size: float,
    ) -> RiskCheckResult:
        cfg = self.config

        if trade.atr <= 0:
            return RiskCheckResult(
                name="volatility_size",
                passed=True,
                verdict=RiskVerdict.APPROVE,
                message="ATR not available, skipping vol check",
            )

        # ATR-based max size: risk_pct × equity / (ATR × multiplier)
        risk_budget   = ps.equity * cfg.risk_per_trade_pct
        
        # Event-aware ATR adjustment (Phase 1)
        atr_multiplier = cfg.atr_multiplier
        if cfg.event_blackout_enabled and self.calendar_provider and _CALENDAR_AVAILABLE:
            try:
                event_vol_mult = self.calendar_provider.expected_volatility_multiplier(
                    trade.symbol, window_hours=2.0
                )
                # Scale the ATR multiplier by event expectation
                # E.g., 1.5x event vol mult → 1.5x larger SL distance → smaller position
                atr_multiplier = cfg.atr_multiplier * event_vol_mult
            except Exception as e:
                logger.debug(f"Event vol multiplier failed: {e}. Using base atr_multiplier.")
        
        sl_distance   = trade.atr * atr_multiplier
        vol_max_size  = risk_budget / (sl_distance + 1e-10)

        # Hard cap
        hard_max_size = (ps.equity * cfg.max_position_size_pct) / (trade.entry_price + 1e-10)
        max_allowed   = min(vol_max_size, hard_max_size)

        if size > max_allowed:
            return RiskCheckResult(
                name="volatility_size",
                passed=False,
                verdict=RiskVerdict.REDUCE,
                reason=RejectReason.VOLATILITY_SIZE_TOO_LARGE,
                message=(
                    f"Size {size:.4f} > vol-adjusted max {max_allowed:.4f} "
                    f"(ATR={trade.atr:.5f}, atr_mult={atr_multiplier:.2f}, "
                    f"risk_budget={risk_budget:.2f})"
                ),
                suggested_size=round(max_allowed, 6),
            )

        return RiskCheckResult(name="volatility_size", passed=True, verdict=RiskVerdict.APPROVE)

    # ------------------------------------------------------------------
    # Layer 5 — Correlation risk
    # ------------------------------------------------------------------

    def _check_correlation(
        self,
        trade: TradeOrder,
        ps: PortfolioState,
    ) -> RiskCheckResult:
        cfg = self.config

        if ps.returns_matrix is None or ps.returns_matrix.empty:
            return RiskCheckResult(
                name="correlation",
                passed=True,
                verdict=RiskVerdict.APPROVE,
                message="No returns matrix — skipping correlation check",
            )

        if trade.symbol not in ps.returns_matrix.columns:
            return RiskCheckResult(
                name="correlation",
                passed=True,
                verdict=RiskVerdict.APPROVE,
                message=f"{trade.symbol} not in returns matrix",
            )

        corr = ps.returns_matrix.corr()
        trade_corr = corr.get(trade.symbol, pd.Series(dtype=float))

        # Find symbols in current portfolio highly correlated with the trade
        open_symbols = {p.symbol for p in ps.open_positions}
        correlated_symbols = [
            s for s in open_symbols
            if s in trade_corr.index and abs(trade_corr[s]) >= cfg.correlation_threshold
        ]

        if not correlated_symbols:
            return RiskCheckResult(name="correlation", passed=True, verdict=RiskVerdict.APPROVE)

        # Sum notional of correlated cluster
        cluster_notional = sum(
            p.notional for p in ps.open_positions if p.symbol in correlated_symbols
        )
        cluster_pct = cluster_notional / (ps.equity + 1e-10)

        if cluster_pct >= cfg.max_correlated_exposure_pct:
            return RiskCheckResult(
                name="correlation",
                passed=False,
                verdict=RiskVerdict.REDUCE,
                reason=RejectReason.CORRELATION_CONCENTRATION,
                message=(
                    f"Correlated cluster {correlated_symbols} = "
                    f"{cluster_pct*100:.1f}% ≥ limit "
                    f"{cfg.max_correlated_exposure_pct*100:.0f}%"
                ),
            )

        return RiskCheckResult(name="correlation", passed=True, verdict=RiskVerdict.APPROVE,
                               message=f"Correlated symbols: {correlated_symbols}")

    # ------------------------------------------------------------------
    # Layer 6 — Leverage
    # ------------------------------------------------------------------

    def _check_leverage(
        self,
        trade: TradeOrder,
        ps: PortfolioState,
        size: float,
    ) -> RiskCheckResult:
        cfg = self.config
        projected_gross = ps.gross_notional + size * trade.entry_price
        projected_lev   = projected_gross / (ps.equity + 1e-10)

        if projected_lev > cfg.max_gross_leverage:
            headroom_notional = max(
                ps.equity * cfg.max_gross_leverage - ps.gross_notional, 0.0
            )
            suggested_size = headroom_notional / (trade.entry_price + 1e-10)

            if suggested_size < 1e-8:
                return RiskCheckResult(
                    name="leverage",
                    passed=False,
                    verdict=RiskVerdict.BLOCK,
                    reason=RejectReason.LEVERAGE_EXCEEDED,
                    message=(
                        f"Projected leverage {projected_lev:.2f}x > "
                        f"max {cfg.max_gross_leverage:.1f}x — no headroom"
                    ),
                )

            return RiskCheckResult(
                name="leverage",
                passed=False,
                verdict=RiskVerdict.REDUCE,
                reason=RejectReason.LEVERAGE_EXCEEDED,
                message=(
                    f"Leverage capped: {projected_lev:.2f}x → "
                    f"size {size:.4f} → {suggested_size:.4f}"
                ),
                suggested_size=round(suggested_size, 6),
            )

        return RiskCheckResult(
            name="leverage",
            passed=True,
            verdict=RiskVerdict.APPROVE,
            message=f"Leverage {projected_lev:.2f}x OK",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _log_decision(verdict: RiskVerdict, checks: list[RiskCheckResult]) -> None:
        failed = [c for c in checks if not c.passed]
        if verdict == RiskVerdict.BLOCK:
            logger.warning(
                "RISK BLOCK — reasons: %s",
                [c.reason.value if c.reason else c.name for c in failed],
            )
        elif verdict == RiskVerdict.REDUCE:
            logger.info(
                "RISK REDUCE — adjustments: %s",
                [c.message for c in failed],
            )
        else:
            logger.debug("RISK APPROVE — all checks passed.")

    def get_risk_summary(self, ps: PortfolioState) -> dict:
        """Quick snapshot of current risk metrics without evaluating trades."""
        cfg = self.config
        return {
            "equity":             round(ps.equity, 2),
            "drawdown_pct":       round(ps.drawdown * 100, 2),
            "daily_loss_pct":     round(ps.daily_loss_pct * 100, 2),
            "gross_leverage":     round(ps.gross_leverage, 3),
            "consecutive_losses": ps.consecutive_losses,
            "kill_switch_active": self._kill_switch_active,
            "headroom": {
                "drawdown_remaining": round(
                    (cfg.max_portfolio_drawdown_pct - ps.drawdown) * 100, 2
                ),
                "daily_loss_remaining": round(
                    (cfg.max_daily_loss_pct - ps.daily_loss_pct) * 100, 2
                ),
                "leverage_remaining": round(
                    cfg.max_gross_leverage - ps.gross_leverage, 3
                ),
            },
        }
