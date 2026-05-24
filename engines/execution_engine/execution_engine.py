"""
MT5ExecutionEngine
==================
Smart order execution engine for MetaTrader 5.

Pipeline
--------
  AggregatedDecision + TradeOrder
          │
          ▼
  Pre-flight checks   (symbol info, market open, balance)
          │
          ▼
  Order construction  (market order with SL / TP)
          │
          ▼
  MT5 order_send()    with slippage guard + retry logic
          │
          ▼
  ExecutionResult     (filled price, slippage, ticket)
          │
          ▼
  TrailingStopManager (client-side trailing SL updates)

Dependencies
------------
    pip install MetaTrader5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# MT5 is an optional runtime dependency
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None          # type: ignore[assignment]
    _MT5_AVAILABLE = False

from engines.risk_engine.risk_engine import TradeOrder
from engines.signal_engine.signal_aggregator import AggregatedDecision
from research.alpha_models.base import SignalType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MT5 retcodes
# ---------------------------------------------------------------------------

_SUCCESS_CODES  = {10009}                   # TRADE_RETCODE_DONE
_REQUOTE_CODES  = {10004, 10020}            # REQUOTE, PRICE_CHANGED
_REJECT_CODES   = {10006, 10007, 10014,     # REJECT, CANCEL, INVALID_VOLUME
                   10017, 10018, 10019}     # MARKET_CLOSED, NO_MONEY, etc.


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExecutionStatus(str, Enum):
    FILLED       = "filled"
    PARTIAL      = "partial"
    REJECTED     = "rejected"
    CANCELLED    = "cancelled"
    ERROR        = "error"
    DISCONNECTED = "disconnected"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """All configurable parameters for MT5ExecutionEngine."""

    # Connection
    mt5_path: Optional[str]   = None     # path to terminal64.exe, or None for default
    login: Optional[int]      = None
    password: Optional[str]   = None
    server: Optional[str]     = None

    # Execution
    magic_number: int         = 20260524  # unique EA identifier
    comment: str              = "quant-fund-ai"
    deviation_points: int     = 20        # max slippage in points
    filling_mode: int         = 2         # ORDER_FILLING_IOC (broker-dependent)

    # Retry
    max_retries: int          = 3
    retry_delay_s: float      = 0.5       # seconds between retries
    requote_max_retries: int  = 5         # extra retries on requote

    # Slippage guard (independent of MT5 deviation)
    max_slippage_pct: float   = 0.0010    # 0.10 % of price

    # Trailing stop
    trailing_step_pct: float  = 0.0005   # move SL every 0.05 % of price
    trailing_activation_pct: float = 0.0010  # activate after 0.10 % profit


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Result of a single order execution attempt.

    Attributes
    ----------
    status : ExecutionStatus
    ticket : int
        MT5 order / position ticket (0 if not filled).
    requested_price : float
    fill_price : float
    slippage_points : float
    volume : float
        Actual filled volume.
    retcode : int
        Raw MT5 return code.
    retries : int
        Number of retry attempts made.
    message : str
        Human-readable status message.
    """

    status: ExecutionStatus
    ticket: int              = 0
    requested_price: float   = 0.0
    fill_price: float        = 0.0
    slippage_points: float   = 0.0
    volume: float            = 0.0
    retcode: int             = 0
    retries: int             = 0
    message: str             = ""

    @property
    def is_filled(self) -> bool:
        return self.status == ExecutionStatus.FILLED

    @property
    def slippage_pct(self) -> float:
        if self.requested_price <= 0:
            return 0.0
        return abs(self.fill_price - self.requested_price) / self.requested_price

    def __repr__(self) -> str:
        return (
            f"ExecutionResult(status={self.status.value}, "
            f"ticket={self.ticket}, fill={self.fill_price:.5f}, "
            f"slip={self.slippage_points:.1f}pts, retries={self.retries})"
        )


@dataclass
class PositionInfo:
    """Snapshot of an open MT5 position."""
    ticket: int
    symbol: str
    direction: int          # +1 long, -1 short
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    magic: int


# ---------------------------------------------------------------------------
# Trailing stop manager (client-side)
# ---------------------------------------------------------------------------

class TrailingStopManager:
    """
    Monitors open positions and updates SL as price moves in favour.
    Must be called on each new tick or bar via update().
    """

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        # ticket → last_sl
        self._tracked: dict[int, float] = {}

    def track(self, ticket: int, initial_sl: float) -> None:
        self._tracked[ticket] = initial_sl

    def untrack(self, ticket: int) -> None:
        self._tracked.pop(ticket, None)

    def update(self, positions: list[PositionInfo]) -> list[dict]:
        """
        Check each tracked position. Return list of SL modification requests
        that need to be sent to MT5.
        """
        if not _MT5_AVAILABLE:
            return []

        modifications: list[dict] = []
        cfg = self.config

        for pos in positions:
            if pos.ticket not in self._tracked:
                continue

            current_sl = self._tracked[pos.ticket]
            price      = pos.current_price

            # Activation threshold
            entry     = pos.open_price
            profit_pct = (price - entry) * pos.direction / (entry + 1e-10)

            if profit_pct < cfg.trailing_activation_pct:
                continue

            # Proposed new SL
            if pos.direction == +1:
                proposed_sl = price * (1.0 - cfg.trailing_step_pct)
                if proposed_sl > current_sl + 1e-10:
                    self._tracked[pos.ticket] = proposed_sl
                    modifications.append({
                        "ticket": pos.ticket,
                        "sl":     proposed_sl,
                        "tp":     pos.tp,
                    })
            else:
                proposed_sl = price * (1.0 + cfg.trailing_step_pct)
                if proposed_sl < current_sl - 1e-10:
                    self._tracked[pos.ticket] = proposed_sl
                    modifications.append({
                        "ticket": pos.ticket,
                        "sl":     proposed_sl,
                        "tp":     pos.tp,
                    })

        return modifications


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class MT5ExecutionEngine:
    """
    Smart execution engine for MetaTrader 5.

    Parameters
    ----------
    config : ExecutionConfig
    """

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config   = config or ExecutionConfig()
        self.trailing = TrailingStopManager(self.config)
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialise MT5 terminal connection."""
        if not _MT5_AVAILABLE:
            logger.error("MetaTrader5 package is not installed.")
            return False

        cfg = self.config
        kwargs: dict = {}
        if cfg.mt5_path:
            kwargs["path"] = cfg.mt5_path
        if cfg.login:
            kwargs.update(login=cfg.login, password=cfg.password, server=cfg.server)

        if not mt5.initialize(**kwargs):
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        info = mt5.terminal_info()
        logger.info(
            "MT5 connected — build %s, company: %s",
            info.build if info else "?",
            info.company if info else "?",
        )
        self._connected = True
        return True

    def disconnect(self) -> None:
        if _MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def __enter__(self) -> "MT5ExecutionEngine":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        decision: AggregatedDecision,
        trade: TradeOrder,
    ) -> ExecutionResult:
        """
        Execute a trade based on an AggregatedDecision and TradeOrder.

        Parameters
        ----------
        decision : AggregatedDecision
            Output of SignalAggregator — must be BUY or SELL.
        trade : TradeOrder
            Sized order from the risk/portfolio engine.

        Returns
        -------
        ExecutionResult
        """
        if not self._connected:
            return ExecutionResult(
                status=ExecutionStatus.DISCONNECTED,
                message="MT5 not connected. Call connect() first.",
            )

        if decision.signal == SignalType.HOLD:
            return ExecutionResult(
                status=ExecutionStatus.CANCELLED,
                message="Signal is HOLD — no order sent.",
            )

        if not _MT5_AVAILABLE:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                message="MetaTrader5 package not installed.",
            )

        # ---- Pre-flight checks -----------------------------------------
        preflight = self._preflight(trade)
        if preflight is not None:
            return preflight

        # ---- Build order request ----------------------------------------
        tick = mt5.symbol_info_tick(trade.symbol)
        if tick is None:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                message=f"Cannot get tick for {trade.symbol}",
            )

        direction    = trade.direction
        order_type   = mt5.ORDER_TYPE_BUY if direction == +1 else mt5.ORDER_TYPE_SELL
        price        = tick.ask if direction == +1 else tick.bid
        sl, tp       = self._compute_sl_tp(trade, price, direction)

        request = {
            "action":     mt5.TRADE_ACTION_DEAL,
            "symbol":     trade.symbol,
            "volume":     round(float(trade.size), 2),
            "type":       order_type,
            "price":      price,
            "sl":         sl,
            "tp":         tp,
            "deviation":  self.config.deviation_points,
            "magic":      self.config.magic_number,
            "comment":    self.config.comment,
            "type_time":  mt5.ORDER_TIME_GTC,
            "type_filling": self.config.filling_mode,
        }

        # ---- Send with retry --------------------------------------------
        result = self._send_with_retry(request, price)

        # ---- Track for trailing stop -----------------------------------
        if result.is_filled and trade.stop_loss is None:
            self.trailing.track(result.ticket, sl)

        return result

    # ------------------------------------------------------------------
    # Close / modify positions
    # ------------------------------------------------------------------

    def close_position(self, symbol: str, ticket: Optional[int] = None) -> ExecutionResult:
        """Close an open position (by symbol or specific ticket)."""
        if not self._connected or not _MT5_AVAILABLE:
            return ExecutionResult(status=ExecutionStatus.DISCONNECTED)

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return ExecutionResult(
                status=ExecutionStatus.CANCELLED,
                message=f"No open position for {symbol}",
            )

        pos = next((p for p in positions if p.ticket == ticket), positions[0]) \
              if ticket else positions[0]

        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick       = mt5.symbol_info_tick(symbol)
        if tick is None:
            return ExecutionResult(status=ExecutionStatus.ERROR, message="No tick data")

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    self.config.deviation_points,
            "magic":        self.config.magic_number,
            "comment":      f"close:{self.config.comment}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": self.config.filling_mode,
        }

        result = self._send_with_retry(request, price)
        if result.is_filled:
            self.trailing.untrack(pos.ticket)

        return result

    def apply_trailing_stops(self) -> int:
        """
        Evaluate and apply client-side trailing stop updates.
        Returns the number of SL modifications sent.
        """
        if not self._connected or not _MT5_AVAILABLE:
            return 0

        positions = self.get_open_positions()
        mods      = self.trailing.update(positions)
        applied   = 0

        for mod in mods:
            pos    = next((p for p in positions if p.ticket == mod["ticket"]), None)
            if pos is None:
                continue
            symbol = pos.symbol
            req    = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "symbol":   symbol,
                "position": mod["ticket"],
                "sl":       mod["sl"],
                "tp":       mod["tp"],
            }
            res = mt5.order_send(req)
            if res and res.retcode in _SUCCESS_CODES:
                applied += 1
                logger.debug(
                    "Trailing SL updated: ticket=%d sl=%.5f",
                    mod["ticket"], mod["sl"],
                )
            else:
                logger.warning(
                    "Trailing SL failed: ticket=%d retcode=%s",
                    mod["ticket"],
                    res.retcode if res else "None",
                )

        return applied

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[PositionInfo]:
        if not self._connected or not _MT5_AVAILABLE:
            return []

        raw = mt5.positions_get()
        if raw is None:
            return []

        result: list[PositionInfo] = []
        for p in raw:
            tick = mt5.symbol_info_tick(p.symbol)
            current_price = (tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask) \
                            if tick else p.price_open

            result.append(PositionInfo(
                ticket        = p.ticket,
                symbol        = p.symbol,
                direction     = +1 if p.type == mt5.ORDER_TYPE_BUY else -1,
                volume        = p.volume,
                open_price    = p.price_open,
                current_price = current_price,
                sl            = p.sl,
                tp            = p.tp,
                profit        = p.profit,
                magic         = p.magic,
            ))

        return result

    def get_account_info(self) -> Optional[dict]:
        if not self._connected or not _MT5_AVAILABLE:
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
            "currency": info.currency,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preflight(self, trade: TradeOrder) -> Optional[ExecutionResult]:
        """Return an error ExecutionResult if any pre-check fails, else None."""
        info = mt5.symbol_info(trade.symbol)
        if info is None:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                message=f"Symbol '{trade.symbol}' not found in MT5.",
            )

        if not info.trade_mode:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                message=f"Trading disabled for {trade.symbol}.",
            )

        acct = mt5.account_info()
        if acct is None or acct.free_margin <= 0:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                message="Insufficient free margin.",
            )

        return None

    def _compute_sl_tp(
        self,
        trade: TradeOrder,
        price: float,
        direction: int,
    ) -> tuple[float, float]:
        """Use trade's explicit SL/TP or fall back to ATR-based defaults."""
        if trade.stop_loss and trade.take_profit:
            return trade.stop_loss, trade.take_profit

        atr = trade.atr if trade.atr > 0 else price * 0.001
        sl  = price - direction * atr * 2.0
        tp  = price + direction * atr * 4.0
        return round(sl, 5), round(tp, 5)

    def _send_with_retry(
        self,
        request: dict,
        requested_price: float,
    ) -> ExecutionResult:
        """Send an MT5 order with retry logic for requotes and transient errors."""
        cfg      = self.config
        retries  = 0
        max_att  = cfg.max_retries + cfg.requote_max_retries

        for attempt in range(max_att):
            result = mt5.order_send(request)

            if result is None:
                logger.error("order_send() returned None — MT5 error: %s", mt5.last_error())
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    message=f"MT5 last_error: {mt5.last_error()}",
                    retries=retries,
                )

            retcode = result.retcode

            # ---- Success ------------------------------------------------
            if retcode in _SUCCESS_CODES:
                fill_price = result.price
                slippage   = (fill_price - requested_price) * (
                    1 if request.get("type") == mt5.ORDER_TYPE_BUY else -1
                )
                # Slippage guard
                if abs(fill_price - requested_price) / (requested_price + 1e-10) \
                        > cfg.max_slippage_pct:
                    logger.warning(
                        "Slippage %.5f exceeds limit %.5f — order filled but flagged.",
                        fill_price - requested_price,
                        cfg.max_slippage_pct * requested_price,
                    )

                return ExecutionResult(
                    status=ExecutionStatus.FILLED,
                    ticket=result.order,
                    requested_price=requested_price,
                    fill_price=fill_price,
                    slippage_points=abs(slippage) / (
                        mt5.symbol_info(request["symbol"]).point + 1e-10
                    ),
                    volume=result.volume,
                    retcode=retcode,
                    retries=retries,
                    message=f"Filled at {fill_price:.5f}",
                )

            # ---- Requote / price changed — refresh price and retry ------
            if retcode in _REQUOTE_CODES and attempt < max_att - 1:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick:
                    new_price       = tick.ask if request["type"] == mt5.ORDER_TYPE_BUY \
                                       else tick.bid
                    request["price"] = new_price
                    requested_price  = new_price
                retries += 1
                logger.debug("Requote on attempt %d — retrying at %.5f", attempt, request["price"])
                time.sleep(cfg.retry_delay_s)
                continue

            # ---- Hard reject — do not retry ----------------------------
            if retcode in _REJECT_CODES:
                msg = f"Order rejected — retcode {retcode}: {result.comment}"
                logger.error(msg)
                return ExecutionResult(
                    status=ExecutionStatus.REJECTED,
                    retcode=retcode,
                    retries=retries,
                    message=msg,
                )

            # ---- Other transient error — wait and retry ----------------
            if attempt < cfg.max_retries - 1:
                retries += 1
                logger.warning(
                    "Order attempt %d failed (retcode=%d) — retrying in %.1fs",
                    attempt + 1, retcode, cfg.retry_delay_s,
                )
                time.sleep(cfg.retry_delay_s * (2 ** attempt))  # exponential backoff
                continue

            # Max retries exhausted
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                retcode=retcode,
                retries=retries,
                message=f"Max retries exceeded — last retcode {retcode}: {result.comment}",
            )

        # Should not reach here
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            retries=retries,
            message="Unexpected exit from retry loop.",
        )
