"""
AlphaModel — Abstract base class for all alpha signal strategies.

Every concrete strategy must implement:
    - compute(data) -> AlphaSignal

AlphaSignal carries:
    - signal    : SignalType  (BUY / SELL / HOLD)
    - confidence: float       in [0.0, 1.0]
    - metadata  : dict        optional diagnostic info
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Signal container
# ---------------------------------------------------------------------------

@dataclass
class AlphaSignal:
    """
    Standardised output produced by every AlphaModel.

    Attributes
    ----------
    signal : SignalType
        Direction of the trade recommendation.
    confidence : float
        Conviction score in [0.0, 1.0].
        0.0 = no conviction, 1.0 = maximum conviction.
    strategy_name : str
        Name of the strategy that produced this signal.
    metadata : dict
        Optional diagnostic / debug information (indicator values, etc.).
    """

    signal: SignalType
    confidence: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    @property
    def is_actionable(self) -> bool:
        """True when the signal is not HOLD."""
        return self.signal != SignalType.HOLD

    def __repr__(self) -> str:
        return (
            f"AlphaSignal(strategy={self.strategy_name!r}, "
            f"signal={self.signal.value}, confidence={self.confidence:.3f})"
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class AlphaModel(ABC):
    """
    Abstract interface every alpha strategy must implement.

    Usage
    -----
    class MyStrategy(AlphaModel):
        def compute(self, data: pd.DataFrame) -> AlphaSignal:
            ...

    strategy = MyStrategy(name="my_strat")
    signal   = strategy.compute(df)
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self._name    = name
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self)  -> None: self._enabled = True
    def disable(self) -> None: self._enabled = False

    def __call__(self, data: pd.DataFrame) -> AlphaSignal:
        """
        Callable shorthand. Returns HOLD with 0 confidence when disabled.
        """
        if not self._enabled:
            return self._hold()
        return self.compute(data)

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        """
        Core signal logic. Must be implemented by every subclass.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV + indicator data. The strategy may use as many
            rows / columns as it needs.

        Returns
        -------
        AlphaSignal
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _hold(self, **meta: Any) -> AlphaSignal:
        return AlphaSignal(
            signal=SignalType.HOLD,
            confidence=0.0,
            strategy_name=self._name,
            metadata=meta,
        )

    def _buy(self, confidence: float, **meta: Any) -> AlphaSignal:
        return AlphaSignal(
            signal=SignalType.BUY,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strategy_name=self._name,
            metadata=meta,
        )

    def _sell(self, confidence: float, **meta: Any) -> AlphaSignal:
        return AlphaSignal(
            signal=SignalType.SELL,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            strategy_name=self._name,
            metadata=meta,
        )

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        return f"{self.__class__.__name__}(name={self._name!r}, {status})"
