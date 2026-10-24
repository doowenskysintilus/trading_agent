"""
Centralised configuration
==========================
Single source of truth for every tunable parameter, loaded from a `.env`
file at the project root (falls back to process environment variables and
then to sensible defaults).

Usage
-----
    from config.settings import settings

    settings.api.host
    settings.mt5.login
    settings.cors.origins
    settings.mysql.as_dict()
    settings.scalping.min_confidence

Edit values by changing `.env` at the project root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    _DOTENV_OK = True
except ImportError:  # python-dotenv not installed — env vars still work
    _DOTENV_OK = False

    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


# ---------------------------------------------------------------------------
# Locate and load the .env file (project root = parent of this package)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = PROJECT_ROOT / ".env"

# override=False → real environment variables take precedence over .env,
# which is the expected behaviour in production / CI.
load_dotenv(ENV_PATH, override=False)


# ---------------------------------------------------------------------------
# Typed getters
# ---------------------------------------------------------------------------

def _get_str(key: str, default: str = "") -> str:
    val = os.environ.get(key)
    return val if val is not None and val != "" else default


def _get_int(key: str, default: int = 0) -> int:
    raw = os.environ.get(key, "")
    try:
        return int(raw) if raw != "" else default
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    raw = os.environ.get(key, "")
    try:
        return float(raw) if raw != "" else default
    except ValueError:
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _get_list(key: str, default: list[str] | None = None) -> list[str]:
    raw = os.environ.get(key, "")
    if raw == "":
        return list(default) if default else []
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Settings groups
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class APISettings:
    """REST/WebSocket API server settings."""
    host: str = field(default_factory=lambda: _get_str("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_int("API_PORT", 8000))
    api_key: str = field(default_factory=lambda: _get_str("QUANT_API_KEY", ""))
    reload: bool = field(default_factory=lambda: _get_bool("API_RELOAD", False))
    title: str = field(default_factory=lambda: _get_str("API_TITLE", "Quant Fund API"))
    version: str = field(default_factory=lambda: _get_str("API_VERSION", "1.0.0"))


@dataclass(frozen=True)
class CORSSettings:
    """Cross-Origin Resource Sharing settings for the dashboard/clients."""
    origins: list[str] = field(
        default_factory=lambda: _get_list("CORS_ORIGINS", ["*"])
    )
    allow_credentials: bool = field(
        default_factory=lambda: _get_bool("CORS_ALLOW_CREDENTIALS", True)
    )


@dataclass(frozen=True)
class MT5Settings:
    """MetaTrader 5 broker connection settings."""
    login: int = field(default_factory=lambda: _get_int("MT5_LOGIN", 0))
    password: str = field(default_factory=lambda: _get_str("MT5_PASSWORD", ""))
    server: str = field(default_factory=lambda: _get_str("MT5_SERVER", ""))
    path: str = field(default_factory=lambda: _get_str("MT5_PATH", ""))
    magic_number: int = field(
        default_factory=lambda: _get_int("MT5_MAGIC_NUMBER", 20260524)
    )


@dataclass(frozen=True)
class TradingSettings:
    """Default live-trading parameters (pairs, timeframe, sizing).

    These act as the defaults for the dashboard "Start Trading" panel and
    for the API when a request omits them.
    """
    symbols: list[str] = field(
        default_factory=lambda: _get_list("TRADING_SYMBOLS", ["EURUSD"])
    )
    timeframe: str = field(
        default_factory=lambda: _get_str("TRADING_TIMEFRAME", "H1")
    )
    cycle_seconds: int = field(
        default_factory=lambda: _get_int("TRADING_CYCLE_SECONDS", 3600)
    )
    warmup_bars: int = field(
        default_factory=lambda: _get_int("TRADING_WARMUP_BARS", 200)
    )
    initial_balance: float = field(
        default_factory=lambda: _get_float("TRADING_INITIAL_BALANCE", 100_000.0)
    )
    allocation_method: str = field(
        default_factory=lambda: _get_str("TRADING_ALLOCATION_METHOD", "RISK_PARITY")
    )

    # ---- Multi-timeframe trend filter -----------------------------------
    htf_enabled: bool = field(
        default_factory=lambda: _get_bool("TRADING_HTF_ENABLED", True)
    )
    htf_timeframe: str = field(
        default_factory=lambda: _get_str("TRADING_HTF_TIMEFRAME", "")
    )
    verbose_signals: bool = field(
        default_factory=lambda: _get_bool("TRADING_VERBOSE_SIGNALS", True)
    )

    # ---- ML win/loss confidence filter ----------------------------------
    ml_filter_enabled: bool = field(
        default_factory=lambda: _get_bool("TRADING_ML_FILTER_ENABLED", True)
    )
    ml_min_win_proba: float = field(
        default_factory=lambda: _get_float("TRADING_ML_MIN_WIN_PROBA", 0.50)
    )
    sl_atr_multiplier: float = field(
        default_factory=lambda: _get_float("TRADING_SL_ATR_MULTIPLIER", 2.0)
    )
    tp_atr_multiplier: float = field(
        default_factory=lambda: _get_float("TRADING_TP_ATR_MULTIPLIER", 4.0)
    )
    # Real-cost gate for every mode: the take-profit must pay the round-trip
    # cost (spread + commission measured per symbol). 0 disables it.
    min_net_reward_risk: float = field(
        default_factory=lambda: _get_float("TRADING_MIN_NET_RR", 1.0)
    )
    min_tp_cost_ratio: float = field(
        default_factory=lambda: _get_float("TRADING_MIN_TP_COST_RATIO", 3.0)
    )

    # ---- RL always-on background retraining -----------------------------
    # Keep RL retraining loop running as long as the API process is alive.
    rl_always_on: bool = field(
        default_factory=lambda: _get_bool("TRADING_RL_ALWAYS_ON", False)
    )
    rl_interval_s: int = field(
        default_factory=lambda: _get_int("TRADING_RL_INTERVAL_S", 300)
    )
    rl_timesteps: int = field(
        default_factory=lambda: _get_int("TRADING_RL_TIMESTEPS", 50_000)
    )
    # Timeframe used SPECIFICALLY for RL training history.
    # Decoupled from the live-trading timeframe so RL trains on longer bars
    # (H1 / H4) with years of history, while live inference runs on M1.
    # Set to "" to reuse the live timeframe (not recommended).
    rl_timeframe: str = field(
        default_factory=lambda: _get_str("TRADING_RL_TIMEFRAME", "H1")
    )
    # Number of bars to fetch for RL training history.
    # 0 = auto-detect the maximum available from the MT5 terminal.
    rl_history_bars: int = field(
        default_factory=lambda: _get_int("TRADING_RL_HISTORY_BARS", 0)
    )

    # ---- Economic Calendar (Phase 1) ------------------------------------
    include_calendar_features: bool = field(
        default_factory=lambda: _get_bool("TRADING_INCLUDE_CALENDAR_FEATURES", False)
    )
    calendar_source: str = field(
        default_factory=lambda: _get_str("TRADING_CALENDAR_SOURCE", "mock")
    )
    calendar_api_key: str = field(
        default_factory=lambda: _get_str("TRADING_CALENDAR_API_KEY", "")
    )
    fred_api_key: str = field(
        default_factory=lambda: _get_str("TRADING_FRED_API_KEY", "")
    )


@dataclass(frozen=True)
class AlertSettings:
    """Real-time alert delivery (drawdown / daily loss / emergency stop)."""
    telegram_bot_token: str = field(
        default_factory=lambda: _get_str("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: _get_str("TELEGRAM_CHAT_ID", "")
    )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@dataclass(frozen=True)
class RiskEngineSettings:
    """Risk engine configuration (event-aware sizing, blackout controls)."""
    
    event_blackout_enabled: bool = field(
        default_factory=lambda: _get_bool("RISK_EVENT_BLACKOUT_ENABLED", True)
    )
    event_blackout_hours: float = field(
        default_factory=lambda: _get_float("RISK_EVENT_BLACKOUT_HOURS", 0.5)
    )
    event_vol_multiplier: float = field(
        default_factory=lambda: _get_float("RISK_EVENT_VOL_MULTIPLIER", 1.5)
    )


def _get_float_map(key: str, default: str = "") -> dict[str, float]:
    """Parse "EURUSD:10,XAUUSD:30" into {"EURUSD": 10.0, "XAUUSD": 30.0}."""
    out: dict[str, float] = {}
    for item in _get_list(key, [s for s in default.split(",") if s]):
        name, _, value = item.partition(":")
        try:
            out[name.strip().upper()] = float(value)
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class ScalpingSettings:
    """Scalping rule set — every value of SCALPING_RULES.md (rule ids in
    comments). Enabled with SCALP_ENABLED=true; when disabled the live
    trader behaves exactly as before."""
    enabled: bool = field(default_factory=lambda: _get_bool("SCALP_ENABLED", False))
    rules_version: str = field(default_factory=lambda: _get_str("SCALP_RULES_VERSION", "SR-1.0"))

    # ---- Scheduling (cycle fast enough for position management) --------
    cycle_seconds: int = field(default_factory=lambda: _get_int("SCALP_CYCLE_SECONDS", 5))
    warmup_bars:   int = field(default_factory=lambda: _get_int("SCALP_WARMUP_BARS", 600))

    # ---- MK: market conditions -------------------------------------------
    # MK-01 / MK-02 — "HH:MM-HH:MM" UTC windows, comma separated
    fx_sessions:     list[str] = field(default_factory=lambda: _get_list(
        "SCALP_FX_SESSIONS", ["07:00-11:00", "12:30-16:30"]))
    crypto_sessions: list[str] = field(default_factory=lambda: _get_list(
        "SCALP_CRYPTO_SESSIONS", ["00:00-21:00"]))
    crypto_symbols:  list[str] = field(default_factory=lambda: _get_list(
        "SCALP_CRYPTO_SYMBOLS", ["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT"]))
    # MK-03
    friday_cutoff_utc:   str = field(default_factory=lambda: _get_str("SCALP_FRIDAY_CUTOFF_UTC", "19:00"))
    sunday_open_utc:     str = field(default_factory=lambda: _get_str("SCALP_SUNDAY_OPEN_UTC", "22:00"))
    sunday_open_wait_min: int = field(default_factory=lambda: _get_int("SCALP_SUNDAY_OPEN_WAIT_MIN", 30))
    # MK-04 — spread ≤ ratio × ATR and ≤ fixed cap (points) per symbol
    max_spread_atr_ratio: float = field(default_factory=lambda: _get_float("SCALP_MAX_SPREAD_ATR_RATIO", 0.25))
    max_spread_points: dict[str, float] = field(default_factory=lambda: _get_float_map(
        "SCALP_MAX_SPREAD_POINTS", "EURUSD:10,XAUUSD:30,BTCUSD:2000"))
    # MK-05 / MK-06
    min_atr_cost_ratio:   float = field(default_factory=lambda: _get_float("SCALP_MIN_ATR_COST_RATIO", 2.0))
    max_atr_median_ratio: float = field(default_factory=lambda: _get_float("SCALP_MAX_ATR_MEDIAN_RATIO", 3.0))
    atr_median_window:    int   = field(default_factory=lambda: _get_int("SCALP_ATR_MEDIAN_WINDOW", 500))
    # MK-07 — minutes before AND after a high-impact event
    event_window_min: int = field(default_factory=lambda: _get_int("SCALP_EVENT_WINDOW_MIN", 15))
    # MK-08 — max age of the last closed bar, in bars
    max_data_age_bars: float = field(default_factory=lambda: _get_float("SCALP_MAX_DATA_AGE_BARS", 2.0))

    # ---- SG / ML ----------------------------------------------------------
    min_confidence:          float = field(default_factory=lambda: _get_float("SCALP_MIN_CONFIDENCE", 0.55))   # SG-02
    neutral_min_confidence:  float = field(default_factory=lambda: _get_float("SCALP_NEUTRAL_MIN_CONFIDENCE", 0.65))  # SG-04
    ml_min_win_proba:        float = field(default_factory=lambda: _get_float("SCALP_ML_MIN_WIN_PROBA", 0.55))  # ML-01
    ml_unreliable_size_mult: float = field(default_factory=lambda: _get_float("SCALP_ML_UNRELIABLE_SIZE_MULT", 0.5))  # ML-03 / SZ-04
    ml_min_samples:          int   = field(default_factory=lambda: _get_int("SCALP_ML_MIN_SAMPLES", 300))   # ML-04
    ml_min_reliable_auc:     float = field(default_factory=lambda: _get_float("SCALP_ML_MIN_RELIABLE_AUC", 0.55))  # ML-07

    # ---- RL ---------------------------------------------------------------
    rl_timeframe:         str   = field(default_factory=lambda: _get_str("SCALP_RL_TIMEFRAME", "M1"))  # RL-03
    rl_overhold_penalty:  float = field(default_factory=lambda: _get_float("SCALP_RL_OVERHOLD_PENALTY", 0.05))  # RL-05

    # ---- SZ / EX ----------------------------------------------------------
    risk_per_trade_pct:   float = field(default_factory=lambda: _get_float("SCALP_RISK_PER_TRADE_PCT", 0.0025))  # SZ-01
    sl_atr_multiplier:    float = field(default_factory=lambda: _get_float("SCALP_SL_ATR_MULTIPLIER", 1.2))  # EX-01
    tp_atr_multiplier:    float = field(default_factory=lambda: _get_float("SCALP_TP_ATR_MULTIPLIER", 1.5))  # EX-02
    commission_pct_per_side: float = field(default_factory=lambda: _get_float("SCALP_COMMISSION_PCT", 0.000035))
    min_net_reward_risk:  float = field(default_factory=lambda: _get_float("SCALP_MIN_NET_RR", 1.0))   # EX-03
    min_tp_cost_ratio:    float = field(default_factory=lambda: _get_float("SCALP_MIN_TP_COST_RATIO", 3.0))  # EX-03
    max_holding_bars:     int   = field(default_factory=lambda: _get_int("SCALP_MAX_HOLDING_BARS", 15))  # EX-04
    breakeven_atr:        float = field(default_factory=lambda: _get_float("SCALP_BREAKEVEN_ATR", 0.8))  # EX-05
    trailing_activation_atr: float = field(default_factory=lambda: _get_float("SCALP_TRAIL_ACTIVATION_ATR", 1.0))  # EX-06
    trailing_step_atr:    float = field(default_factory=lambda: _get_float("SCALP_TRAIL_STEP_ATR", 0.5))  # EX-06
    max_slippage_spreads: float = field(default_factory=lambda: _get_float("SCALP_MAX_SLIPPAGE_SPREADS", 1.0))  # EX-07
    session_close_buffer_min: int = field(default_factory=lambda: _get_int("SCALP_SESSION_CLOSE_BUFFER_MIN", 10))  # EX-08

    # ---- FQ: frequency & capital protection ------------------------------
    max_trades_per_symbol_hour: int   = field(default_factory=lambda: _get_int("SCALP_MAX_TRADES_SYMBOL_HOUR", 4))  # FQ-01
    max_trades_per_day:         int   = field(default_factory=lambda: _get_int("SCALP_MAX_TRADES_DAY", 30))  # FQ-02
    symbol_cooldown_min:        int   = field(default_factory=lambda: _get_int("SCALP_SYMBOL_COOLDOWN_MIN", 5))  # FQ-03
    global_pause_losses:        int   = field(default_factory=lambda: _get_int("SCALP_GLOBAL_PAUSE_LOSSES", 3))  # FQ-04
    global_pause_min:           int   = field(default_factory=lambda: _get_int("SCALP_GLOBAL_PAUSE_MIN", 30))  # FQ-04
    max_daily_loss_pct:         float = field(default_factory=lambda: _get_float("SCALP_MAX_DAILY_LOSS_PCT", 0.02))  # FQ-06
    daily_profit_target_pct:    float = field(default_factory=lambda: _get_float("SCALP_DAILY_TARGET_PCT", 0.015))  # FQ-08
    ftmo_auto_detect:           bool  = field(default_factory=lambda: _get_bool("SCALP_FTMO_AUTO_DETECT", True))  # FQ-09


@dataclass(frozen=True)
class MySQLSettings:
    """MySQL metrics store settings (optional — falls back to JSONL logs)."""
    enabled: bool = field(default_factory=lambda: _get_bool("MYSQL_ENABLED", False))
    host: str = field(default_factory=lambda: _get_str("MYSQL_HOST", "localhost"))
    port: int = field(default_factory=lambda: _get_int("MYSQL_PORT", 3306))
    user: str = field(default_factory=lambda: _get_str("MYSQL_USER", "root"))
    password: str = field(default_factory=lambda: _get_str("MYSQL_PASSWORD", ""))
    database: str = field(default_factory=lambda: _get_str("MYSQL_DATABASE", "quant_fund"))

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }


@dataclass(frozen=True)
class DashboardSettings:
    """Dashboard build-time settings (informational; the Vite app reads its own)."""
    api_url: str = field(
        default_factory=lambda: _get_str("VITE_API_URL", "http://localhost:8000")
    )
    ws_url: str = field(
        default_factory=lambda: _get_str("VITE_WS_URL", "ws://localhost:8000/ws")
    )
    api_key: str = field(default_factory=lambda: _get_str("VITE_API_KEY", ""))


@dataclass(frozen=True)
class Settings:
    """Aggregate of every configuration group."""
    api: APISettings = field(default_factory=APISettings)
    cors: CORSSettings = field(default_factory=CORSSettings)
    mt5: MT5Settings = field(default_factory=MT5Settings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    mysql: MySQLSettings = field(default_factory=MySQLSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    risk_engine: RiskEngineSettings = field(default_factory=RiskEngineSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    scalping: ScalpingSettings = field(default_factory=ScalpingSettings)

    env_file: str = field(default_factory=lambda: str(ENV_PATH))
    env_file_loaded: bool = field(default_factory=lambda: _DOTENV_OK and ENV_PATH.exists())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

settings = Settings()
