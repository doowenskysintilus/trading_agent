"""
SignalEngine
============
Aggregates signals from multiple AlphaModel instances into a single
consensus signal using confidence-weighted voting.

Usage
-----
    engine = SignalEngine()
    engine.register(MomentumAlpha())
    engine.register(MeanReversionAlpha())
    engine.register(RLAlpha(model_path="..."))

    result = engine.run(df)
    print(result.signal, result.confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from research.alpha_models.base import AlphaModel, AlphaSignal, SignalType


# ---------------------------------------------------------------------------
# Aggregated result
# ---------------------------------------------------------------------------

@dataclass
class AggregatedSignal:
    signal: SignalType
    confidence: float
    votes: dict[str, AlphaSignal] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AggregatedSignal(signal={self.signal.value}, "
            f"confidence={self.confidence:.3f}, "
            f"strategies={list(self.votes.keys())})"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Confidence-weighted voting aggregator over registered AlphaModels.

    Aggregation logic
    -----------------
    - Each active strategy votes BUY (+1), SELL (-1), or HOLD (0),
      weighted by its confidence score and an optional strategy weight.
    - A minimum confidence threshold filters weak individual signals.
    - Final direction is determined by the sign of the weighted sum.
    - Final confidence is the normalised magnitude of the weighted sum.
    - A quorum (min_votes) can be required before emitting a non-HOLD signal.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence an individual signal must have to be counted.
    min_votes : int
        Minimum number of agreeing votes required to emit BUY / SELL.
    """

    def __init__(
        self,
        min_confidence: float = 0.1,
        min_votes: int = 1,
    ) -> None:
        self._models: dict[str, AlphaModel] = {}
        self._weights: dict[str, float] = {}
        self.min_confidence = min_confidence
        self.min_votes      = min_votes

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model: AlphaModel, weight: float = 1.0) -> None:
        """Add an AlphaModel to the engine."""
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self._models[model.name]  = model
        self._weights[model.name] = weight

    def unregister(self, name: str) -> None:
        self._models.pop(name, None)
        self._weights.pop(name, None)

    @property
    def strategies(self) -> list[str]:
        return list(self._models.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> AggregatedSignal:
        """
        Compute signals from all registered models and aggregate them.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV + indicator data passed to each AlphaModel.

        Returns
        -------
        AggregatedSignal
        """
        if not self._models:
            return AggregatedSignal(
                signal=SignalType.HOLD,
                confidence=0.0,
                metadata={"reason": "no_strategies_registered"},
            )

        votes: dict[str, AlphaSignal] = {}
        weighted_score = 0.0
        total_weight   = 0.0
        buy_votes  = 0
        sell_votes = 0

        for name, model in self._models.items():
            try:
                signal = model(data)
            except Exception as exc:
                # Never let one strategy crash the engine
                signal = model._hold(error=str(exc))

            votes[name] = signal

            if not signal.is_actionable or signal.confidence < self.min_confidence:
                continue

            w     = self._weights[name]
            score = signal.confidence * w

            if signal.signal == SignalType.BUY:
                weighted_score += score
                buy_votes      += 1
            elif signal.signal == SignalType.SELL:
                weighted_score -= score
                sell_votes     += 1

            total_weight += w

        # ---- Aggregate ------------------------------------------------
        final_signal, final_confidence = self._aggregate(
            weighted_score, total_weight, buy_votes, sell_votes
        )

        return AggregatedSignal(
            signal=final_signal,
            confidence=final_confidence,
            votes=votes,
            metadata={
                "weighted_score": round(weighted_score, 4),
                "buy_votes":      buy_votes,
                "sell_votes":     sell_votes,
            },
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        weighted_score: float,
        total_weight: float,
        buy_votes: int,
        sell_votes: int,
    ) -> tuple[SignalType, float]:

        if total_weight == 0:
            return SignalType.HOLD, 0.0

        normalised = weighted_score / total_weight  # in [-1, 1]
        confidence = float(np.clip(abs(normalised), 0.0, 1.0))

        if normalised > 0 and buy_votes >= self.min_votes:
            return SignalType.BUY, confidence
        if normalised < 0 and sell_votes >= self.min_votes:
            return SignalType.SELL, confidence

        return SignalType.HOLD, confidence

    def __repr__(self) -> str:
        return (
            f"SignalEngine(strategies={self.strategies}, "
            f"min_confidence={self.min_confidence}, "
            f"min_votes={self.min_votes})"
        )
