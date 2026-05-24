"""
MetricsStore
============
Dual-write persistence layer: MySQL (primary) + structured JSON logs (fallback/audit).

Schema
------
  equity_snapshots   — portfolio equity & drawdown over time
  strategy_pnl       — per-strategy cumulative PnL & stats
  trades             — individual trade records
  risk_snapshots     — risk exposure, leverage, VaR snapshots
  drawdown_periods   — start/end/depth of each drawdown episode

JSON logs (always written, regardless of MySQL state)
------------------------------------------------------
  <log_dir>/YYYYMMDD/equity.jsonl
  <log_dir>/YYYYMMDD/strategy_pnl.jsonl
  <log_dir>/YYYYMMDD/trades.jsonl
  <log_dir>/YYYYMMDD/risk.jsonl
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MySQL import
# ---------------------------------------------------------------------------

try:
    import mysql.connector
    from mysql.connector import pooling
    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False
    logger.warning(
        "mysql-connector-python not installed. "
        "Install with: pip install mysql-connector-python"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MySQLConfig:
    host:            str = "localhost"
    port:            int = 3306
    user:            str = "quant"
    password:        str = ""
    database:        str = "quant_fund"
    pool_size:       int = 5
    connect_timeout: int = 10
    autocommit:      bool = True


# ---------------------------------------------------------------------------
# DDL — table definitions
# ---------------------------------------------------------------------------

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS equity_snapshots (
        id            BIGINT AUTO_INCREMENT PRIMARY KEY,
        ts            DATETIME(3) NOT NULL,
        equity        DOUBLE      NOT NULL,
        balance       DOUBLE      NOT NULL,
        open_pnl      DOUBLE      DEFAULT 0.0,
        drawdown_pct  DOUBLE      DEFAULT 0.0,
        high_water    DOUBLE      DEFAULT 0.0,
        n_positions   INT         DEFAULT 0,
        INDEX idx_ts (ts)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_pnl (
        id             BIGINT AUTO_INCREMENT PRIMARY KEY,
        ts             DATETIME(3) NOT NULL,
        strategy_name  VARCHAR(64) NOT NULL,
        cycle_pnl      DOUBLE      DEFAULT 0.0,
        cumulative_pnl DOUBLE      DEFAULT 0.0,
        n_trades       INT         DEFAULT 0,
        win_rate       DOUBLE      DEFAULT 0.0,
        sharpe         DOUBLE      DEFAULT 0.0,
        INDEX idx_ts_strategy (ts, strategy_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id            BIGINT AUTO_INCREMENT PRIMARY KEY,
        ts            DATETIME(3) NOT NULL,
        symbol        VARCHAR(16) NOT NULL,
        strategy      VARCHAR(64),
        direction     VARCHAR(8),
        size          DOUBLE,
        entry_price   DOUBLE,
        exit_price    DOUBLE,
        pnl           DOUBLE,
        pnl_pct       DOUBLE,
        duration_s    INT,
        exit_reason   VARCHAR(32),
        ticket        VARCHAR(32),
        INDEX idx_ts        (ts),
        INDEX idx_symbol    (symbol),
        INDEX idx_strategy  (strategy)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_snapshots (
        id               BIGINT AUTO_INCREMENT PRIMARY KEY,
        ts               DATETIME(3) NOT NULL,
        total_exposure   DOUBLE      DEFAULT 0.0,
        daily_pnl        DOUBLE      DEFAULT 0.0,
        daily_pnl_pct    DOUBLE      DEFAULT 0.0,
        leverage         DOUBLE      DEFAULT 0.0,
        n_positions      INT         DEFAULT 0,
        var_95           DOUBLE      DEFAULT 0.0,
        risk_verdict     VARCHAR(16),
        INDEX idx_ts (ts)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS drawdown_periods (
        id            BIGINT AUTO_INCREMENT PRIMARY KEY,
        start_ts      DATETIME(3) NOT NULL,
        end_ts        DATETIME(3),
        peak_equity   DOUBLE,
        trough_equity DOUBLE,
        depth_pct     DOUBLE,
        duration_s    INT,
        recovered     TINYINT(1)  DEFAULT 0,
        INDEX idx_start (start_ts)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
]


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------

class MetricsStore:
    """
    Dual-write persistence layer.

    Write path:
      Every write method writes to JSON first (safe, atomic append),
      then to MySQL if connected.  MySQL failures are logged but never
      raise — trading must not stop due to monitoring failures.

    Read path:
      MySQL is preferred.  If unavailable, reads from JSON files.
    """

    def __init__(
        self,
        mysql_config: MySQLConfig | None = None,
        log_dir: str = "data/storage/logs/metrics",
    ) -> None:
        self.mysql_cfg  = mysql_config or MySQLConfig()
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._pool      = None
        self._lock      = threading.Lock()
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to MySQL and ensure tables exist. Returns True on success."""
        if not _MYSQL_AVAILABLE:
            logger.warning("MySQL unavailable — running in JSON-only mode.")
            return False

        cfg = self.mysql_cfg
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name          = "quant_pool",
                pool_size          = cfg.pool_size,
                host               = cfg.host,
                port               = cfg.port,
                user               = cfg.user,
                password           = cfg.password,
                database           = cfg.database,
                connection_timeout = cfg.connect_timeout,
                autocommit         = cfg.autocommit,
            )
            self._run_ddl()
            self._connected = True
            logger.info("MySQL connected — %s:%d/%s", cfg.host, cfg.port, cfg.database)
            return True

        except Exception as exc:
            logger.error("MySQL connection failed: %s — using JSON-only mode.", exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._pool = None

    def __enter__(self) -> "MetricsStore":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Write — equity snapshot
    # ------------------------------------------------------------------

    def write_equity_snapshot(
        self,
        ts:           datetime,
        equity:       float,
        balance:      float,
        open_pnl:     float     = 0.0,
        drawdown_pct: float     = 0.0,
        high_water:   float     = 0.0,
        n_positions:  int       = 0,
    ) -> None:
        row = {
            "ts":           _fmt_ts(ts),
            "equity":       round(equity, 4),
            "balance":      round(balance, 4),
            "open_pnl":     round(open_pnl, 4),
            "drawdown_pct": round(drawdown_pct, 6),
            "high_water":   round(high_water, 4),
            "n_positions":  n_positions,
        }
        self._json_append("equity", row)
        self._mysql_insert("equity_snapshots", row)

    # ------------------------------------------------------------------
    # Write — strategy PnL
    # ------------------------------------------------------------------

    def write_strategy_pnl(
        self,
        ts:             datetime,
        strategy_name:  str,
        cycle_pnl:      float = 0.0,
        cumulative_pnl: float = 0.0,
        n_trades:       int   = 0,
        win_rate:       float = 0.0,
        sharpe:         float = 0.0,
    ) -> None:
        row = {
            "ts":             _fmt_ts(ts),
            "strategy_name":  strategy_name,
            "cycle_pnl":      round(cycle_pnl, 4),
            "cumulative_pnl": round(cumulative_pnl, 4),
            "n_trades":       n_trades,
            "win_rate":       round(win_rate, 4),
            "sharpe":         round(sharpe, 4),
        }
        self._json_append("strategy_pnl", row)
        self._mysql_insert("strategy_pnl", row)

    # ------------------------------------------------------------------
    # Write — trade
    # ------------------------------------------------------------------

    def write_trade(
        self,
        ts:           datetime,
        symbol:       str,
        strategy:     str,
        direction:    str,
        size:         float,
        entry_price:  float,
        exit_price:   float,
        pnl:          float,
        duration_s:   int   = 0,
        exit_reason:  str   = "",
        ticket:       str   = "",
    ) -> None:
        pnl_pct = (exit_price - entry_price) / (entry_price + 1e-10)
        row = {
            "ts":           _fmt_ts(ts),
            "symbol":       symbol,
            "strategy":     strategy,
            "direction":    direction,
            "size":         round(size, 4),
            "entry_price":  round(entry_price, 5),
            "exit_price":   round(exit_price, 5),
            "pnl":          round(pnl, 4),
            "pnl_pct":      round(pnl_pct, 6),
            "duration_s":   duration_s,
            "exit_reason":  exit_reason,
            "ticket":       ticket,
        }
        self._json_append("trades", row)
        self._mysql_insert("trades", row)

    # ------------------------------------------------------------------
    # Write — risk snapshot
    # ------------------------------------------------------------------

    def write_risk_snapshot(
        self,
        ts:             datetime,
        total_exposure: float = 0.0,
        daily_pnl:      float = 0.0,
        daily_pnl_pct:  float = 0.0,
        leverage:       float = 0.0,
        n_positions:    int   = 0,
        var_95:         float = 0.0,
        risk_verdict:   str   = "",
    ) -> None:
        row = {
            "ts":             _fmt_ts(ts),
            "total_exposure": round(total_exposure, 4),
            "daily_pnl":      round(daily_pnl, 4),
            "daily_pnl_pct":  round(daily_pnl_pct, 6),
            "leverage":       round(leverage, 4),
            "n_positions":    n_positions,
            "var_95":         round(var_95, 4),
            "risk_verdict":   risk_verdict,
        }
        self._json_append("risk", row)
        self._mysql_insert("risk_snapshots", row)

    # ------------------------------------------------------------------
    # Write — drawdown period
    # ------------------------------------------------------------------

    def write_drawdown_period(
        self,
        start_ts:     datetime,
        peak_equity:  float,
        trough_equity: float,
        depth_pct:    float,
        end_ts:       Optional[datetime] = None,
        duration_s:   int   = 0,
        recovered:    bool  = False,
    ) -> None:
        row = {
            "start_ts":     _fmt_ts(start_ts),
            "end_ts":       _fmt_ts(end_ts) if end_ts else None,
            "peak_equity":  round(peak_equity, 4),
            "trough_equity": round(trough_equity, 4),
            "depth_pct":    round(depth_pct, 6),
            "duration_s":   duration_s,
            "recovered":    int(recovered),
        }
        self._json_append("drawdowns", row)
        self._mysql_insert("drawdown_periods", row)

    # ------------------------------------------------------------------
    # Query — equity curve
    # ------------------------------------------------------------------

    def query_equity_curve(
        self,
        start: Optional[datetime] = None,
        end:   Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        if self._connected:
            sql = "SELECT * FROM equity_snapshots"
            params, where = [], []
            if start:
                where.append("ts >= %s"); params.append(_fmt_ts(start))
            if end:
                where.append("ts <= %s"); params.append(_fmt_ts(end))
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += f" ORDER BY ts DESC LIMIT {limit}"
            rows = self._mysql_query(sql, params)
            if rows is not None:
                return rows
        # JSON fallback
        return self._json_query("equity", start, end, limit)

    # ------------------------------------------------------------------
    # Query — strategy PnL
    # ------------------------------------------------------------------

    def query_strategy_pnl(
        self,
        strategy: Optional[str] = None,
        start:    Optional[datetime] = None,
        end:      Optional[datetime] = None,
        limit:    int = 500,
    ) -> list[dict]:
        if self._connected:
            sql = "SELECT * FROM strategy_pnl"
            params, where = [], []
            if strategy:
                where.append("strategy_name = %s"); params.append(strategy)
            if start:
                where.append("ts >= %s"); params.append(_fmt_ts(start))
            if end:
                where.append("ts <= %s"); params.append(_fmt_ts(end))
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += f" ORDER BY ts DESC LIMIT {limit}"
            rows = self._mysql_query(sql, params)
            if rows is not None:
                return rows
        return self._json_query("strategy_pnl", start, end, limit)

    # ------------------------------------------------------------------
    # Query — trades
    # ------------------------------------------------------------------

    def query_trades(
        self,
        symbol:   Optional[str] = None,
        strategy: Optional[str] = None,
        start:    Optional[datetime] = None,
        end:      Optional[datetime] = None,
        limit:    int = 200,
    ) -> list[dict]:
        if self._connected:
            sql = "SELECT * FROM trades"
            params, where = [], []
            if symbol:
                where.append("symbol = %s"); params.append(symbol)
            if strategy:
                where.append("strategy = %s"); params.append(strategy)
            if start:
                where.append("ts >= %s"); params.append(_fmt_ts(start))
            if end:
                where.append("ts <= %s"); params.append(_fmt_ts(end))
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += f" ORDER BY ts DESC LIMIT {limit}"
            rows = self._mysql_query(sql, params)
            if rows is not None:
                return rows
        return self._json_query("trades", start, end, limit)

    # ------------------------------------------------------------------
    # Query — risk snapshots
    # ------------------------------------------------------------------

    def query_risk_snapshots(
        self,
        limit: int = 100,
        start: Optional[datetime] = None,
    ) -> list[dict]:
        if self._connected:
            sql = "SELECT * FROM risk_snapshots"
            params, where = [], []
            if start:
                where.append("ts >= %s"); params.append(_fmt_ts(start))
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += f" ORDER BY ts DESC LIMIT {limit}"
            rows = self._mysql_query(sql, params)
            if rows is not None:
                return rows
        return self._json_query("risk", start, None, limit)

    # ------------------------------------------------------------------
    # Query — drawdowns
    # ------------------------------------------------------------------

    def query_drawdowns(self, limit: int = 50) -> list[dict]:
        if self._connected:
            sql = f"SELECT * FROM drawdown_periods ORDER BY start_ts DESC LIMIT {limit}"
            rows = self._mysql_query(sql, [])
            if rows is not None:
                return rows
        return self._json_query("drawdowns", None, None, limit)

    # ------------------------------------------------------------------
    # Internal MySQL helpers
    # ------------------------------------------------------------------

    def _run_ddl(self) -> None:
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            for ddl in _DDL:
                cursor.execute(ddl)
            conn.commit()

    def _mysql_insert(self, table: str, row: dict) -> None:
        if not self._connected or self._pool is None:
            return
        try:
            cols   = ", ".join(f"`{k}`" for k in row)
            marks  = ", ".join(["%s"] * len(row))
            sql    = f"INSERT INTO `{table}` ({cols}) VALUES ({marks})"
            values = list(row.values())
            with self._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, values)
        except Exception as exc:
            logger.debug("MySQL insert failed (%s): %s", table, exc)

    def _mysql_query(self, sql: str, params: list) -> Optional[list[dict]]:
        if not self._connected or self._pool is None:
            return None
        try:
            with self._pool.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                # Convert datetime objects to ISO strings
                return [
                    {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                     for k, v in r.items()}
                    for r in rows
                ]
        except Exception as exc:
            logger.debug("MySQL query failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal JSON helpers
    # ------------------------------------------------------------------

    def _json_path(self, category: str, ts: Optional[datetime] = None) -> Path:
        date_str = (ts or datetime.now(timezone.utc)).strftime("%Y%m%d")
        dir_path = self.log_dir / date_str
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{category}.jsonl"

    def _json_append(self, category: str, row: dict) -> None:
        path = self._json_path(category)
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
        except OSError as exc:
            logger.error("JSON write failed (%s): %s", category, exc)

    def _json_query(
        self,
        category: str,
        start:    Optional[datetime],
        end:      Optional[datetime],
        limit:    int,
    ) -> list[dict]:
        """Scan today's (and yesterday's) JSON log for matching rows."""
        rows: list[dict] = []
        today = datetime.now(timezone.utc)
        for delta_days in (0, 1):
            from datetime import timedelta
            day   = today - timedelta(days=delta_days)
            path  = self._json_path(category, day)
            if not path.exists():
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            rows.append(obj)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

        # Most-recent first
        rows.reverse()
        return rows[:limit]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fmt_ts(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecond precision
