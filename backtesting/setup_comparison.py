"""
Setup comparison — which symbol × timeframe × strategy × mode is worth trading
==============================================================================
Walk-forward backtests of every combination with the account's REAL costs:

  * spread: the historical spread of each bar (MT5 `spread` column)
  * commission: the rate measured per symbol on the account's own closed
    trades (monitoring.account_history.commission_rates), else the default
  * exits as in live trading: ATR stop-loss / take-profit
      - normal mode   : SL/TP = TRADING_SL/TP_ATR_MULTIPLIER, no time stop
      - scalping mode : SL/TP = SCALP_SL/TP_ATR_MULTIPLIER + time stop
  * strategies fed the same FeatureEngineer columns as the live trader

Each row reports trades, win rate, net profit factor, net return, what the
costs ate, and the live-readiness verdict (VA-02..05).

Limits: each strategy is evaluated alone (no SignalAggregator vote, no
RiskEngine sizing, no scalping entry filters), so this measures the raw
edge of a signal net of costs — the first thing that must be positive.

Run (MT5 terminal open, .env credentials):
    python -m backtesting.setup_comparison
    python -m backtesting.setup_comparison --symbols BTCUSD EURUSD --timeframes M1 M5 M15
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from backtesting.backtest_engine import BacktestConfig
from backtesting.validation import check_live_readiness
from backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine
from research.alpha_models.base import AlphaModel

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES = ("M1", "M5", "M15", "H1")


@dataclass(frozen=True)
class SymbolCosts:
    point: float                 # price of one MT5 spread point
    commission_per_side: float   # fraction of notional
    measured: bool               # commission measured on the account (vs default)


@dataclass(frozen=True)
class ModeParams:
    name: str
    sl_atr: float
    tp_atr: float
    max_holding_bars: int


def mode_params() -> dict[str, ModeParams]:
    from config.settings import settings
    t, s = settings.trading, settings.scalping
    return {
        "normal":   ModeParams("normal", t.sl_atr_multiplier, t.tp_atr_multiplier, 0),
        "scalping": ModeParams("scalping", s.sl_atr_multiplier, s.tp_atr_multiplier, s.max_holding_bars),
    }


def default_strategies() -> dict[str, Callable[[], AlphaModel]]:
    """Strategies that run on historical features alone. CalendarAlpha needs
    historical event data and RLAlpha a trained model — not included."""
    from strategies.mean_reversion.mean_reversion_alpha import MeanReversionAlpha
    from strategies.momentum.momentum_alpha import MomentumAlpha
    return {"momentum": MomentumAlpha, "mean_reversion": MeanReversionAlpha}


def prepare_data(bars: pd.DataFrame) -> pd.DataFrame:
    """OHLCV (+ spread) joined with the live FeatureEngineer columns."""
    from research.feature_store.feature_engineer import FeatureEngineer
    features = FeatureEngineer().compute(bars).ffill().fillna(0.0)
    features = features.drop(columns=[c for c in features.columns if c in bars.columns])
    return bars.join(features, how="inner").ffill().fillna(0.0)


def evaluate(
    data: pd.DataFrame,
    strategy: AlphaModel,
    costs: SymbolCosts,
    mode: ModeParams,
    n_folds: int = 4,
) -> dict:
    cfg = BacktestConfig(
        spread            = 0.0,
        spread_point      = costs.point,
        slippage_std      = 0.0,
        commission        = costs.commission_per_side,
        sl_atr_multiplier = mode.sl_atr,
        tp_atr_multiplier = mode.tp_atr,
        max_holding_bars  = mode.max_holding_bars,
        n_workers         = 1,
    )
    wf = WalkForwardEngine(WalkForwardConfig(n_folds=n_folds, backtest_config=cfg)).run([strategy], data)
    logs = [f.result.trade_log for f in wf.folds if not f.result.trade_log.empty]
    trades = pd.concat(logs) if logs else pd.DataFrame(columns=["pnl", "costs"])
    pnl = trades["pnl"].astype(float)
    cost = trades["costs"].astype(float)
    gross = pnl + cost
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    readiness = check_live_readiness(wf)
    return {
        "trades":         len(trades),
        "win_rate":       round(float((pnl > 0).mean()), 3) if len(pnl) else None,
        "profit_factor":  round(float(wins / losses), 2) if losses > 0 else None,
        "net_return_pct": round(100 * float(pnl.sum()) / cfg.initial_balance, 2),
        "gross_return_pct": round(100 * float(gross.sum()) / cfg.initial_balance, 2),
        "costs_pct":      round(100 * float(cost.sum()) / cfg.initial_balance, 2),
        "profitable_folds": f"{readiness.profitable_folds}/{readiness.n_folds}",
        "max_dd_pct":     round(100 * readiness.max_drawdown, 2),
        "ready":          readiness.ready,
        "why_not":        "; ".join(readiness.failures),
    }


def compare(
    symbols: Iterable[str],
    timeframes: Iterable[str],
    load_bars: Callable[[str, str], pd.DataFrame],
    cost_lookup: Callable[[str], SymbolCosts],
    strategies: dict[str, Callable[[], AlphaModel]] | None = None,
    modes: Iterable[str] = ("normal", "scalping"),
    n_folds: int = 4,
) -> pd.DataFrame:
    strategies = strategies or default_strategies()
    params = mode_params()
    rows = []
    for symbol in symbols:
        costs = cost_lookup(symbol)
        for tf in timeframes:
            bars = load_bars(symbol, tf)
            if bars is None or len(bars) < n_folds * 150:
                logger.warning("%s %s: not enough history, skipped", symbol, tf)
                continue
            data = prepare_data(bars)
            for strat_name, factory in strategies.items():
                for mode in modes:
                    try:
                        res = evaluate(data, factory(), costs, params[mode], n_folds)
                    except Exception as exc:
                        logger.warning("%s %s %s %s failed: %s", symbol, tf, strat_name, mode, exc)
                        continue
                    rows.append({
                        "symbol": symbol, "timeframe": tf, "strategy": strat_name, "mode": mode,
                        "bars": len(data),
                        "from": str(data.index[0])[:10], "to": str(data.index[-1])[:10],
                        "commission_side_pct": round(100 * costs.commission_per_side, 4),
                        "commission_measured": costs.measured,
                        **res,
                    })
                    logger.info("%s %s %-14s %-8s → %s trades, PF %s, net %s %%",
                                symbol, tf, strat_name, mode, res["trades"],
                                res["profit_factor"], res["net_return_pct"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MT5-backed CLI
# ---------------------------------------------------------------------------

def _mt5_loaders(bars_per_tf: int):
    import MetaTrader5 as mt5

    from config.settings import settings
    from engines.execution_engine.execution_engine import ExecutionConfig, MT5ExecutionEngine
    from live_trading.live_trader import _get_mt5_tf
    from monitoring.account_history import AccountHistory

    m = settings.mt5
    engine = MT5ExecutionEngine(ExecutionConfig(
        login=m.login or None, password=m.password, server=m.server, mt5_path=m.path or None,
    ))
    if not engine.connect():
        raise RuntimeError("MT5 connection failed — is the terminal running?")
    measured = AccountHistory(
        engine.get_history_deals, engine.server_utc_offset_s, m.magic_number,
    ).commission_rates(engine.get_contract_size)
    default_rate = settings.scalping.commission_pct_per_side

    def load_bars(symbol: str, timeframe: str) -> pd.DataFrame | None:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, _get_mt5_tf(timeframe), 1, bars_per_tf)  # closed bars
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
        return df[["open", "high", "low", "close", "volume", "spread"]].astype(float)

    def cost_lookup(symbol: str) -> SymbolCosts:
        info = mt5.symbol_info(symbol)
        point = float(info.point) if info else 0.0
        rate = measured.get(symbol.upper(), {}).get("rate_per_side")
        return SymbolCosts(point, rate or default_rate, rate is not None)

    return engine, load_bars, cost_lookup


def _to_markdown(df: pd.DataFrame) -> str:
    cols = ["symbol", "timeframe", "strategy", "mode", "trades", "win_rate", "profit_factor",
            "gross_return_pct", "costs_pct", "net_return_pct", "profitable_folds", "max_dd_pct", "ready"]
    view = df[cols].sort_values(["net_return_pct"], ascending=False)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False))


def main() -> None:
    from config.settings import settings

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--symbols", nargs="*", default=list(settings.trading.symbols))
    parser.add_argument("--timeframes", nargs="*", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--modes", nargs="*", default=["normal", "scalping"])
    parser.add_argument("--bars", type=int, default=30_000, help="history bars per timeframe")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", default="data/storage/reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("backtesting.backtest_engine", "backtesting.walk_forward"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    engine, load_bars, cost_lookup = _mt5_loaders(args.bars)
    try:
        df = compare(args.symbols, args.timeframes, load_bars, cost_lookup,
                     modes=args.modes, n_folds=args.folds)
    finally:
        engine.disconnect()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df.to_csv(out / f"setup_comparison_{stamp}.csv", index=False)
    md = _to_markdown(df) if not df.empty else "(no result)"
    (out / f"setup_comparison_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved: {out / f'setup_comparison_{stamp}.csv'}")


if __name__ == "__main__":
    main()
