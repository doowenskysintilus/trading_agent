"""
API Schemas
===========
Pydantic request and response models for all endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from config.settings import settings


# ---------------------------------------------------------------------------
# Shared envelope
# ---------------------------------------------------------------------------

class APIResponse(BaseModel):
    success: bool = True
    data:    Any  = None
    error:   Optional[str] = None
    ts:      str  = Field(default_factory=lambda: datetime.utcnow().isoformat())


def ok(data: Any = None) -> dict:
    return APIResponse(success=True, data=data).model_dump()

def err(msg: str) -> dict:
    return APIResponse(success=False, error=msg).model_dump()


# ---------------------------------------------------------------------------
# Trading control
# ---------------------------------------------------------------------------

class TradingStartRequest(BaseModel):
    # Defaults come from .env (config.settings.trading) so the pairs and
    # parameters can be changed without touching code.
    symbols:              list[str]   = Field(default_factory=lambda: list(settings.trading.symbols))
    timeframe:            str         = Field(default_factory=lambda: settings.trading.timeframe)
    cycle_interval_s:     int         = Field(default_factory=lambda: settings.trading.cycle_seconds, ge=60, le=86400)
    warmup_bars:          int         = Field(
        default_factory=lambda: settings.trading.warmup_bars,
        ge=0,
        le=500000,
        description="0 = use maximum available bars for the selected timeframe",
    )
    initial_balance:      float       = Field(default_factory=lambda: settings.trading.initial_balance, gt=0)
    allocation_method:    str         = Field(default_factory=lambda: settings.trading.allocation_method)
    htf_enabled:          bool        = Field(default_factory=lambda: settings.trading.htf_enabled)
    htf_timeframe:        str         = Field(default_factory=lambda: settings.trading.htf_timeframe)
    verbose_signals:      bool        = Field(default_factory=lambda: settings.trading.verbose_signals)
    ml_filter_enabled:    bool        = Field(default_factory=lambda: settings.trading.ml_filter_enabled)
    ml_min_win_proba:     float       = Field(default_factory=lambda: settings.trading.ml_min_win_proba, ge=0.0, le=1.0)
    sl_atr_multiplier:   float       = Field(default_factory=lambda: settings.trading.sl_atr_multiplier, ge=1.0, le=5.0)
    tp_atr_multiplier:   float       = Field(default_factory=lambda: settings.trading.tp_atr_multiplier, ge=1.0, le=5.0)
    # FTMO prop-firm compliance (Layer 1b in RiskEngine): hard-limits daily
    # loss / drawdown to FTMO's actual rules (soft-stopped before their real
    # 5% / 10% limits) instead of the generic kill-switch defaults, which are
    # looser on drawdown (15-20%) than what FTMO would actually enforce.
    ftmo_enabled:         bool        = False
    ftmo_initial_balance: float       = Field(default=0.0, ge=0.0)

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["EURUSD", "GBPUSD"],
                "timeframe": "H1",
                "cycle_interval_s": 3600,
                "sl_atr_multiplier": 2.0,
                "tp_atr_multiplier": 4.0,
            }
        }


class TradingStopRequest(BaseModel):
    reason: str = "operator"


class EmergencyStopRequest(BaseModel):
    reason: str = Field(default="manual", min_length=1, max_length=200)


class RetrainRequest(BaseModel):
    """Trigger a manual retraining of the learning models from results."""
    train_ml:        bool = Field(default=True)
    train_rl:        bool = Field(default=False)
    rl_timesteps:    int  = Field(default=50_000, ge=1_000, le=2_000_000)
    rl_history_bars: int  = Field(
        default=0,
        ge=0,
        le=500_000,
        description="0 = fetch ALL available history from broker (recommended)",
    )
    rl_continuous:   bool = Field(default=False)
    rl_interval_s:   int  = Field(default=1_800, ge=60, le=86_400)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    strategies:       list[str]       = Field(description="Strategy class names to include")
    symbol:           str             = Field(default="EURUSD")
    timeframe:        str             = Field(default="H1")
    initial_balance:  float           = Field(default=100_000.0, gt=0)
    commission:       float           = Field(default=0.0002, ge=0, le=0.01)
    spread:           float           = Field(default=0.0001, ge=0)
    sl_pct:           float           = Field(default=0.02, gt=0, le=0.5)
    tp_pct:           float           = Field(default=0.04, gt=0, le=1.0)
    window_size:      int             = Field(default=50, ge=10, le=500)
    seed:             int             = Field(default=42)

    class Config:
        json_schema_extra = {
            "example": {
                "strategies": ["MomentumAlpha", "MeanReversionAlpha"],
                "symbol": "EURUSD",
                "timeframe": "H1",
            }
        }


class BacktestJobStatus(str, Enum):
    QUEUED   = "queued"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"


# ---------------------------------------------------------------------------
# Manual trade override
# ---------------------------------------------------------------------------

class TradeDirection(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class ManualTradeRequest(BaseModel):
    symbol:    str         = Field(..., min_length=3, max_length=16)
    direction: TradeDirection
    size:      float       = Field(..., gt=0, le=100.0)
    sl_price:  Optional[float] = Field(default=None, gt=0)
    tp_price:  Optional[float] = Field(default=None, gt=0)
    comment:   str         = Field(default="manual_override", max_length=100)

    @field_validator("symbol")
    @classmethod
    def symbol_upper(cls, v: str) -> str:
        return v.upper().strip()


class ClosePositionRequest(BaseModel):
    symbol:  str   = Field(..., min_length=3, max_length=16)
    ticket:  Optional[int] = None     # close specific ticket; None = all for symbol
    comment: str   = Field(default="manual_close", max_length=100)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskConfigUpdate(BaseModel):
    """Partial update — only provided fields are applied."""
    max_daily_loss_pct:      Optional[float] = Field(default=None, ge=0.001, le=1.0)
    max_drawdown_pct:        Optional[float] = Field(default=None, ge=0.001, le=1.0)
    max_position_size_pct:   Optional[float] = Field(default=None, ge=0.001, le=1.0)
    max_leverage:            Optional[float] = Field(default=None, ge=0.1, le=100.0)
    max_consecutive_losses:  Optional[int]   = Field(default=None, ge=1, le=50)
    # FTMO prop-firm compliance (RiskEngine Layer 1b) — runtime-toggleable so
    # an already-running trader can be brought into FTMO compliance without
    # a restart.
    ftmo_enabled:                 Optional[bool]  = Field(default=None)
    ftmo_initial_balance:         Optional[float] = Field(default=None, gt=0)
    ftmo_daily_loss_pct:          Optional[float] = Field(default=None, ge=0.001, le=1.0)
    ftmo_max_drawdown_pct:        Optional[float] = Field(default=None, ge=0.001, le=1.0)
    ftmo_consistency_enabled:     Optional[bool]  = Field(default=None)
    ftmo_consistency_max_day_pct: Optional[float] = Field(default=None, ge=0.001, le=1.0)
