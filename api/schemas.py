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


class TradingStatusResponse(BaseModel):
    running:          bool
    cycle_id:         int
    emergency_active: bool
    retry_count:      int
    consecutive_ok:   int
    strategies:       list[str]
    symbols:          list[str]
    equity:           float
    balance:          float
    open_positions:   int


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class PositionResponse(BaseModel):
    symbol:        str
    direction:     str
    size:          float
    entry_price:   float
    unrealized_pnl: float
    strategy:      str


class PortfolioStatusResponse(BaseModel):
    equity:          float
    balance:         float
    open_pnl:        float
    daily_pnl:       float
    daily_pnl_pct:   float
    drawdown_pct:    float
    high_water_mark: float
    n_positions:     int
    positions:       list[PositionResponse]
    allocations:     dict[str, float]


class EquityPoint(BaseModel):
    ts:           str
    equity:       float
    balance:      float
    open_pnl:     float
    drawdown_pct: float
    n_positions:  int


class EconomicEventResponse(BaseModel):
    timestamp: str
    country:   str
    name:      str
    importance: str
    forecast:  Optional[float] = None
    previous:  Optional[float] = None
    actual:    Optional[float] = None
    revised:   Optional[float] = None
    units:     Optional[str] = None
    url:       Optional[str] = None


class CalendarEventsResponse(BaseModel):
    events: list[EconomicEventResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class StrategyPerformanceResponse(BaseModel):
    strategy:       str
    n_trades:       int
    win_rate:       float
    cumulative_pnl: float
    avg_pnl:        float
    sharpe:         float
    last_updated:   Optional[str]


class StrategyToggleRequest(BaseModel):
    enabled: bool


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestSymbolData(BaseModel):
    """Either a reference to a cached dataset or inline OHLCV records."""
    symbol:    str
    timeframe: str
    source:    str = Field(default="feature_store", description="'feature_store' or 'inline'")


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


class BacktestJobResponse(BaseModel):
    job_id:       str
    status:       BacktestJobStatus
    submitted_at: str
    started_at:   Optional[str]
    completed_at: Optional[str]
    duration_s:   Optional[float]
    error:        Optional[str]


class BacktestStrategyMetrics(BaseModel):
    strategy:         str
    total_return_pct: float
    cagr:             float
    sharpe:           float
    sortino:          float
    calmar:           float
    max_drawdown_pct: float
    n_trades:         int
    win_rate:         float
    profit_factor:    float
    var_95:           float


class BacktestResultResponse(BaseModel):
    job_id:            str
    status:            BacktestJobStatus
    strategy_metrics:  list[BacktestStrategyMetrics]
    portfolio_metrics: Optional[BacktestStrategyMetrics]
    n_trades_total:    int
    duration_s:        float


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


class TradeExecutionResponse(BaseModel):
    status:         str
    ticket:         Optional[str]
    fill_price:     Optional[float]
    slippage_points: Optional[float]
    retries:        int
    comment:        str


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class RiskStatusResponse(BaseModel):
    ts:             str
    total_exposure: float
    daily_pnl:      float
    daily_pnl_pct:  float
    leverage:       float
    n_positions:    int
    var_95:         float
    verdict:        str


class RiskConfigUpdate(BaseModel):
    """Partial update — only provided fields are applied."""
    max_daily_loss_pct:      Optional[float] = Field(default=None, ge=0.001, le=1.0)
    max_drawdown_pct:        Optional[float] = Field(default=None, ge=0.001, le=1.0)
    max_position_size_pct:   Optional[float] = Field(default=None, ge=0.001, le=1.0)
    max_leverage:            Optional[float] = Field(default=None, ge=0.1, le=100.0)
    max_consecutive_losses:  Optional[int]   = Field(default=None, ge=1, le=50)


class DrawdownPeriod(BaseModel):
    start_ts:      str
    end_ts:        Optional[str]
    depth_pct:     float
    duration_s:    int
    recovered:     bool


# ---------------------------------------------------------------------------
# Trade history
# ---------------------------------------------------------------------------

class TradeRecord(BaseModel):
    ts:           str
    symbol:       str
    strategy:     str
    direction:    str
    size:         float
    entry_price:  float
    exit_price:   float
    pnl:          float
    pnl_pct:      float
    duration_s:   int
    exit_reason:  str
