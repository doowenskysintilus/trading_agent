"""
SignalAggregator
================
Takes pre-computed AlphaSignal outputs from multiple strategies,
weights them by live performance scores, resolves conflicts, and
produces a final trading decision.

Architecture
------------
  list[AlphaSignal]
        │
        ▼
  PerformanceWeighter      ← adjusts weight per strategy based on PnL history
        │
        ▼
  ConflictResolver         ← handles BUY vs SELL disagreement
        │
        ▼
  AggregatedDecision       ← final signal + confidence + audit trail

Usage
-----
    agg = SignalAggregator()

    # Register initial static weights
    agg.set_weight("momentum",      1.0)
    agg.set_weight("mean_reversion", 1.0)
    agg.set_weight("rl_agent",       1.5)

    # Feed results of a closed trade to update performance weights
    agg.record_trade("momentum", pnl=120.0, duration_bars=8)

    # Aggregate a list of fresh signals
    signals = [momentum_signal, mr_signal, rl_signal]
    decision = agg.aggregate(signals)
    print(decision)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from research.alpha_models.base import AlphaSignal, SignalType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conflict resolution strategies
# ---------------------------------------------------------------------------

class ConflictStrategy(str, Enum):
    ABSTAIN             = "abstain"            # emit HOLD when BUY vs SELL disagree
    HIGHEST_CONFIDENCE  = "highest_confidence" # follow the side with greater total weight
    WEIGHTED_MAJORITY   = "weighted_majority"  # follow net signed weighted score


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

@dataclass
class AggregatedDecision:
    """
    Final output of the aggregation pipeline.

    Attributes
    ----------
    signal : SignalType
        Final trading direction.
    confidence : float
        Conviction score in [0, 1] derived from normalised vote margin.
    buy_score : float
        Sum of weighted buy votes.
    sell_score : float
        Sum of weighted sell votes.
    net_score : float
        buy_score - sell_score; positive = bullish bias.
    votes : list[dict]
        Per-strategy breakdown for audit / logging.
    conflict : bool
        True when BUY and SELL both received non-zero votes.
    conflict_resolution : str
        Which resolution path was taken.
    metadata : dict
        Arbitrary extra diagnostics.
    """

    signal: SignalType
    confidence: float
    buy_score: float
    sell_score: float
    net_score: float
    votes: list[dict] = field(default_factory=list)
    conflict: bool = False
    conflict_resolution: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return self.signal != SignalType.HOLD

    def __repr__(self) -> str:
        flag = " [CONFLICT]" if self.conflict else ""
        return (
            f"AggregatedDecision(signal={self.signal.value}, "
            f"confidence={self.confidence:.3f}, "
            f"buy={self.buy_score:.3f}, sell={self.sell_score:.3f}"
            f"{flag})"
        )


# ---------------------------------------------------------------------------
# Performance-based weighter
# ---------------------------------------------------------------------------

class PerformanceWeighter:
    """
    Tracks per-strategy recent trade outcomes and computes a dynamic
    performance multiplier applied on top of the base weight.

    The multiplier is derived from:
      - Recent win rate  (rolling window of last N trades)
      - Average PnL      (normalised)
      - Consistency      (1 / std of PnL)

    Score is clamped to [min_mult, max_mult] to avoid any single strategy
    dominating or being silenced completely.

    Parameters
    ----------
    window : int
        Number of recent trades used to compute the rolling score.
    min_mult : float
        Minimum weight multiplier (floor).
    max_mult : float
        Maximum weight multiplier (cap).
    """

    def __init__(
        self,
        window: int = 20,
        min_mult: float = 0.2,
        max_mult: float = 3.0,
    ) -> None:
        self._window   = window
        self._min_mult = min_mult
        self._max_mult = max_mult
        # strategy_name → deque of (pnl, duration_bars)
        self._history: dict[str, deque[tuple[float, int]]] = {}

    def record(self, strategy: str, pnl: float, duration_bars: int = 1) -> None:
        """Log the outcome of a closed trade for `strategy`."""
        if strategy not in self._history:
            self._history[strategy] = deque(maxlen=self._window)
        self._history[strategy].append((pnl, max(duration_bars, 1)))

    def multiplier(self, strategy: str) -> float:
        """
        Return the performance multiplier for `strategy`.
        Returns 1.0 (neutral) if fewer than 3 trades are recorded.
        """
        history = self._history.get(strategy)
        if not history or len(history) < 3:
            return 1.0

        pnls = np.array([p for p, _ in history], dtype=float)

        win_rate    = float(np.mean(pnls > 0))
        mean_pnl    = float(np.mean(pnls))
        std_pnl     = float(np.std(pnls)) + 1e-10
        consistency = 1.0 / (1.0 + std_pnl / (abs(mean_pnl) + 1e-10))

        # Composite score: blend of win rate and risk-adjusted mean
        score = 0.5 * win_rate + 0.3 * float(np.clip(mean_pnl / std_pnl, -1, 1)) \
                + 0.2 * consistency

        # Map [0, 1] score range onto [min_mult, max_mult]
        mult = self._min_mult + (self._max_mult - self._min_mult) * float(np.clip(score, 0, 1))
        return round(mult, 4)

    def summary(self) -> dict[str, dict]:
        """Return a performance summary dict for all tracked strategies."""
        out = {}
        for name, history in self._history.items():
            if not history:
                continue
            pnls = [p for p, _ in history]
            out[name] = {
                "n_trades":  len(pnls),
                "win_rate":  round(np.mean(np.array(pnls) > 0), 3),
                "mean_pnl":  round(float(np.mean(pnls)), 4),
                "total_pnl": round(float(np.sum(pnls)), 4),
                "multiplier": self.multiplier(name),
            }
        return out


# ---------------------------------------------------------------------------
# Conflict resolver
# ---------------------------------------------------------------------------

class ConflictResolver:
    """
    Decides the final signal when BUY and SELL forces both exist.

    Parameters
    ----------
    strategy : ConflictStrategy
        Resolution method (default WEIGHTED_MAJORITY).
    min_dominance : float
        For WEIGHTED_MAJORITY: net_score / total_score must exceed this
        threshold to avoid emitting HOLD. Prevents marginal wins.
    """

    def __init__(
        self,
        strategy: ConflictStrategy = ConflictStrategy.WEIGHTED_MAJORITY,
        min_dominance: float = 0.15,
    ) -> None:
        self.strategy       = strategy
        self.min_dominance  = min_dominance

    def resolve(
        self,
        buy_score: float,
        sell_score: float,
    ) -> tuple[SignalType, str]:
        """
        Returns (SignalType, resolution_label).
        """
        total = buy_score + sell_score

        if self.strategy == ConflictStrategy.ABSTAIN:
            return SignalType.HOLD, "conflict_abstain"

        if self.strategy == ConflictStrategy.HIGHEST_CONFIDENCE:
            if buy_score > sell_score:
                return SignalType.BUY,  "conflict_highest_confidence"
            if sell_score > buy_score:
                return SignalType.SELL, "conflict_highest_confidence"
            return SignalType.HOLD, "conflict_tie"

        # WEIGHTED_MAJORITY (default)
        if total < 1e-10:
            return SignalType.HOLD, "no_votes"

        dominance = abs(buy_score - sell_score) / total
        if dominance < self.min_dominance:
            return SignalType.HOLD, "conflict_insufficient_dominance"

        if buy_score > sell_score:
            return SignalType.BUY,  "conflict_weighted_majority"
        return SignalType.SELL, "conflict_weighted_majority"


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

class SignalAggregator:
    """
    Aggregates a list of pre-computed AlphaSignal objects into a single
    AggregatedDecision using dynamic performance-weighted voting.

    Parameters
    ----------
    conflict_strategy : ConflictStrategy
        How to resolve BUY ↔ SELL disagreements.
    min_confidence : float
        Signals with confidence below this are ignored.
    min_votes : int
        Minimum unweighted votes required on the winning side.
    performance_window : int
        Rolling window for PerformanceWeighter.
    """

    def __init__(
        self,
        conflict_strategy: ConflictStrategy = ConflictStrategy.WEIGHTED_MAJORITY,
        min_confidence: float = 0.05,
        min_votes: int = 1,
        performance_window: int = 20,
    ) -> None:
        self._base_weights: dict[str, float] = {}
        self._weighter  = PerformanceWeighter(window=performance_window)
        self._resolver  = ConflictResolver(strategy=conflict_strategy)
        self.min_confidence = min_confidence
        self.min_votes      = min_votes

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_weight(self, strategy: str, weight: float) -> None:
        """Set (or update) the static base weight for a strategy."""
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self._base_weights[strategy] = weight

    def record_trade(self, strategy: str, pnl: float, duration_bars: int = 1) -> None:
        """Feed a closed trade outcome to update dynamic weights."""
        self._weighter.record(strategy, pnl, duration_bars)

    def effective_weight(self, strategy: str) -> float:
        """base_weight × performance_multiplier for `strategy`."""
        base = self._base_weights.get(strategy, 1.0)
        mult = self._weighter.multiplier(strategy)
        return round(base * mult, 4)

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def aggregate(self, signals: Sequence[AlphaSignal]) -> AggregatedDecision:
        """
        Aggregate a list of AlphaSignal outputs.

        Parameters
        ----------
        signals : list[AlphaSignal]
            One signal per strategy, already computed.

        Returns
        -------
        AggregatedDecision
        """
        if not signals:
            return self._empty_decision("no_signals")

        buy_score  = 0.0
        sell_score = 0.0
        buy_votes  = 0
        sell_votes = 0
        votes: list[dict] = []

        for sig in signals:
            w = self.effective_weight(sig.strategy_name)

            vote_entry = {
                "strategy":   sig.strategy_name,
                "signal":     sig.signal.value,
                "confidence": round(sig.confidence, 4),
                "base_weight": self._base_weights.get(sig.strategy_name, 1.0),
                "perf_mult":   self._weighter.multiplier(sig.strategy_name),
                "eff_weight":  w,
                "counted":     False,
            }

            if sig.confidence < self.min_confidence:
                vote_entry["skip_reason"] = "low_confidence"
                votes.append(vote_entry)
                continue

            weighted = sig.confidence * w

            if sig.signal == SignalType.BUY:
                buy_score += weighted
                buy_votes += 1
                vote_entry["counted"] = True
            elif sig.signal == SignalType.SELL:
                sell_score += weighted
                sell_votes += 1
                vote_entry["counted"] = True
            # HOLD: counted=False, contributes nothing

            votes.append(vote_entry)

        return self._decide(buy_score, sell_score, buy_votes, sell_votes, votes)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        buy_score: float,
        sell_score: float,
        buy_votes: int,
        sell_votes: int,
        votes: list[dict],
    ) -> AggregatedDecision:

        total_score = buy_score + sell_score
        conflict    = buy_score > 0 and sell_score > 0
        resolution  = "none"

        # ---- No actionable votes at all --------------------------------
        if total_score < 1e-10:
            return AggregatedDecision(
                signal=SignalType.HOLD,
                confidence=0.0,
                buy_score=0.0,
                sell_score=0.0,
                net_score=0.0,
                votes=votes,
                conflict=False,
                conflict_resolution="no_actionable_votes",
            )

        # ---- Pure consensus (one side only) ----------------------------
        if not conflict:
            if buy_votes >= self.min_votes and buy_score > 0:
                signal     = SignalType.BUY
                confidence = self._confidence(buy_score, sell_score)
                resolution = "consensus_buy"
            elif sell_votes >= self.min_votes and sell_score > 0:
                signal     = SignalType.SELL
                confidence = self._confidence(sell_score, buy_score)
                resolution = "consensus_sell"
            else:
                signal, confidence = SignalType.HOLD, 0.0
                resolution = "insufficient_votes"
        else:
            # ---- Conflict: delegate to resolver -------------------------
            signal, resolution = self._resolver.resolve(buy_score, sell_score)
            if signal == SignalType.HOLD:
                confidence = 0.0
            elif signal == SignalType.BUY:
                confidence = self._confidence(buy_score, sell_score)
            else:
                confidence = self._confidence(sell_score, buy_score)

        return AggregatedDecision(
            signal=signal,
            confidence=round(confidence, 4),
            buy_score=round(buy_score, 4),
            sell_score=round(sell_score, 4),
            net_score=round(buy_score - sell_score, 4),
            votes=votes,
            conflict=conflict,
            conflict_resolution=resolution,
            metadata={
                "buy_votes":  buy_votes,
                "sell_votes": sell_votes,
                "total_score": round(total_score, 4),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence(winner_score: float, loser_score: float) -> float:
        """
        Confidence = margin / total, capped at 1.0.
        Pure consensus (loser = 0) gives confidence = winner normalised.
        """
        total = winner_score + loser_score
        if total < 1e-10:
            return 0.0
        return float(np.clip((winner_score - loser_score) / total, 0.0, 1.0))

    def _empty_decision(self, reason: str) -> AggregatedDecision:
        return AggregatedDecision(
            signal=SignalType.HOLD,
            confidence=0.0,
            buy_score=0.0,
            sell_score=0.0,
            net_score=0.0,
            conflict_resolution=reason,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def performance_summary(self) -> dict[str, dict]:
        """Current performance stats and effective weights per strategy."""
        return self._weighter.summary()

    def __repr__(self) -> str:
        return (
            f"SignalAggregator("
            f"strategies={list(self._base_weights.keys())}, "
            f"conflict={self._resolver.strategy.value})"
        )
