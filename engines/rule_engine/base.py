"""
Rule engine — abstractions
==========================
Implements the contract of SCALPING_RULES.md as small, interchangeable rule
objects (SOLID):

  S  one class per rule family, one responsibility each
  O  adding a rule = adding a class; RuleEngine never changes
  L  every rule exposes the same evaluate(context) -> result contract
  I  entry rules (EntryRule) and position rules (PositionRule) are separate
  D  rules depend on the data handed to them (contexts / injected
     callables), never on MT5 or MySQL directly — testable offline

Entry rules return a RuleResult (pass / reject + optional size multiplier)
and are evaluated fail-fast in Stage order. Position rules return an
optional PositionAction (close / move stop-loss) for one open position.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional, Protocol

import pandas as pd


# ---------------------------------------------------------------------------
# Evaluation order (SCALPING_RULES.md §7)
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    HALT      = 0   # kill-switch / FTMO / persistence
    FREQUENCY = 1   # trade counts, cooldowns, daily target, one position
    MARKET    = 2   # session, spread, volatility, events, data freshness
    SIGNAL    = 3   # aggregated confidence vs regime
    ML        = 4   # WinClassifier P(win)
    ORDER     = 5   # net reward:risk of the built order


PRE_SIGNAL_STAGES  = frozenset({Stage.HALT, Stage.FREQUENCY, Stage.MARKET})
POST_SIGNAL_STAGES = frozenset({Stage.SIGNAL, Stage.ML})
ORDER_STAGES       = frozenset({Stage.ORDER})


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleResult:
    passed: bool
    rule_id: str
    reason: str = ""
    size_multiplier: float = 1.0
    metrics: dict = field(default_factory=dict)

    @classmethod
    def ok(cls, rule_id: str, size_multiplier: float = 1.0, **metrics: Any) -> "RuleResult":
        return cls(True, rule_id, "", size_multiplier, metrics)

    @classmethod
    def reject(cls, rule_id: str, reason: str, **metrics: Any) -> "RuleResult":
        return cls(False, rule_id, reason, 0.0, metrics)


@dataclass
class EvaluationResult:
    """Outcome of evaluating a set of stages for one entry candidate."""
    results: list[RuleResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> Optional[RuleResult]:
        return next((r for r in self.results if not r.passed), None)

    @property
    def size_multiplier(self) -> float:
        mult = 1.0
        for r in self.results:
            mult *= r.size_multiplier
        return mult

    @property
    def metrics(self) -> dict:
        out: dict = {}
        for r in self.results:
            out.update(r.metrics)
        return out


# ---------------------------------------------------------------------------
# Market data handed to the rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSnapshot:
    """Live quote + symbol properties (built from MT5 by the caller)."""
    symbol: str
    bid: float
    ask: float
    point: float
    server_time: Optional[datetime] = None   # broker clock (same clock as bars)
    volume_min: float = 0.01
    volume_step: float = 0.01

    @property
    def spread_price(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def spread_points(self) -> float:
        return self.spread_price / self.point if self.point > 0 else 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class EntryContext:
    """Everything an entry rule may look at. Fields not yet known at a
    given stage stay None (e.g. `order` before the order is built)."""
    symbol: str
    now: datetime                          # real UTC clock
    timeframe_s: int
    bars: Optional[pd.DataFrame] = None    # closed bars, broker clock index
    snapshot: Optional[MarketSnapshot] = None
    atr: float = 0.0
    portfolio_state: Any = None            # engines.risk_engine.PortfolioState
    open_symbols: list[str] = field(default_factory=list)
    decision: Any = None                   # AggregatedDecision
    regime: str = "neutral"
    ml_prediction: Any = None              # strategies.ml_agent.WinPrediction
    order: Any = None                      # engines.risk_engine.TradeOrder
    round_trip_cost: float = 0.0           # spread + commissions, price units


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManagedPosition:
    ticket: int
    symbol: str
    direction: int            # +1 long / -1 short
    open_price: float
    current_price: float
    sl: float
    tp: float
    entry_ts: datetime
    entry_atr: float
    round_trip_cost: float = 0.0

    @property
    def favourable_move(self) -> float:
        """Price distance in the position's favour (negative = losing)."""
        return (self.current_price - self.open_price) * self.direction


@dataclass(frozen=True)
class PositionContext:
    now: datetime
    timeframe_s: int
    position: ManagedPosition


@dataclass(frozen=True)
class PositionAction:
    kind: str                 # "close" | "modify_sl"
    ticket: int
    symbol: str
    rule_id: str
    reason: str
    new_sl: Optional[float] = None


# ---------------------------------------------------------------------------
# Rule interfaces
# ---------------------------------------------------------------------------

class EntryRule(ABC):
    """A rule deciding whether a new position may be opened."""
    rule_id: str = ""
    stage: Stage = Stage.MARKET

    @abstractmethod
    def evaluate(self, ctx: EntryContext) -> RuleResult: ...


class PositionRule(ABC):
    """A rule managing one already-open position."""
    rule_id: str = ""

    @abstractmethod
    def evaluate(self, ctx: PositionContext) -> Optional[PositionAction]: ...


class DecisionSink(Protocol):
    """Where decisions are journaled (TradingMonitor → MySQL / JSONL)."""
    def record_decision(self, row: dict) -> None: ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RuleEngine:
    """Composes entry and position rules. Knows nothing about any concrete
    rule — new rules are added by passing them in (open/closed)."""

    def __init__(
        self,
        entry_rules: list[EntryRule],
        position_rules: list[PositionRule],
        rules_version: str = "",
    ) -> None:
        # Stable sort keeps the registration order within a stage.
        self.entry_rules    = sorted(entry_rules, key=lambda r: r.stage)
        self.position_rules = list(position_rules)
        self.rules_version  = rules_version

    def evaluate_entry(self, ctx: EntryContext, stages: frozenset) -> EvaluationResult:
        """Evaluate the rules of `stages` in order; stop at the first reject."""
        out = EvaluationResult()
        for rule in self.entry_rules:
            if rule.stage not in stages:
                continue
            result = rule.evaluate(ctx)
            out.results.append(result)
            if not result.passed:
                break
        return out

    def evaluate_position(self, ctx: PositionContext) -> Optional[PositionAction]:
        """Return the single action to apply to this position.

        A close always wins. Among stop-loss moves, the most protective one
        (highest SL for a long, lowest for a short) is kept, and only if it
        actually tightens the current stop.
        """
        actions = [a for a in (r.evaluate(ctx) for r in self.position_rules) if a]
        closes = [a for a in actions if a.kind == "close"]
        if closes:
            return closes[0]
        pos = ctx.position
        moves = [a for a in actions if a.kind == "modify_sl" and a.new_sl is not None]
        if not moves:
            return None
        best = max(moves, key=lambda a: a.new_sl * pos.direction)
        if pos.sl and best.new_sl * pos.direction <= pos.sl * pos.direction:
            return None
        return best
