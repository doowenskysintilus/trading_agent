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


@dataclass(frozen=True)
class MySQLSettings:
    """MySQL metrics store settings (optional — falls back to JSONL logs)."""
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

    env_file: str = field(default_factory=lambda: str(ENV_PATH))
    env_file_loaded: bool = field(default_factory=lambda: _DOTENV_OK and ENV_PATH.exists())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

settings = Settings()
