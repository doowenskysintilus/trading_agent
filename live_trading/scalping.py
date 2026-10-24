"""
ScalpingController
==================
Glue between LiveTrader and the scalping RuleEngine (SCALPING_RULES.md).

LiveTrader stays the orchestrator; this class owns everything specific to
the scalping rule set:

  * applying the scalping profile to LiveTraderConfig (SG-02, ML-01, SZ-01,
    EX-01/02, FQ-06, MK-09 closed bars, fast cycle for position management)
  * building EntryContexts from MT5 data and evaluating the rule stages
  * journaling every decision through the monitor (DB-01) and rebuilding
    the frequency journal from MySQL at startup (DB-03)
  * turning position rules into close / stop-loss actions (EX-04..06, EX-08)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from engines.rule_engine import (
    ORDER_STAGES, POST_SIGNAL_STAGES, PRE_SIGNAL_STAGES,
    EntryContext, EvaluationResult, JournalClose, JournalEntry,
    ManagedPosition, MarketSnapshot, PositionAction, PositionContext,
    TradeJournal, build_scalping_engine,
)
from engines.rule_engine.costs import CommissionModel
from engines.rule_engine.market_time import timeframe_seconds

logger = logging.getLogger(__name__)


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


class ScalpingController:
    def __init__(
        self,
        settings,                       # config.settings.ScalpingSettings
        *,
        timeframe: str,
        risk_engine,
        calendar_provider,
        exec_engine,
        sink_getter: Callable[[], object],
        commission_model: Optional[CommissionModel] = None,
    ) -> None:
        self.cfg = settings
        self.timeframe_s = timeframe_seconds(timeframe)
        self.journal = TradeJournal()
        self.commissions = commission_model or CommissionModel(settings.commission_pct_per_side)
        self._exec = exec_engine
        self._sink_getter = sink_getter
        self._last_reject: dict[str, str] = {}
        self._restored = False
        self.engine = build_scalping_engine(
            settings,
            risk_engine       = risk_engine,
            calendar_provider = calendar_provider,
            journal           = self.journal,
            is_persistence_ok = self._persistence_ok,
        )

    @property
    def rules_version(self) -> str:
        return self.engine.rules_version

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    @staticmethod
    def apply_profile(cfg, s) -> dict:
        """Override LiveTraderConfig with the scalping values. Returns the
        changed fields {name: (old, new)} so the caller can log them."""
        changes: dict = {}

        def _set(obj, name, value):
            old = getattr(obj, name)
            if old != value:
                setattr(obj, name, value)
                changes[name] = (old, value)

        _set(cfg, "min_confidence", s.min_confidence)                      # SG-02
        _set(cfg, "ml_min_win_proba", s.ml_min_win_proba)                  # ML-01
        _set(cfg, "sl_atr_multiplier", s.sl_atr_multiplier)                # EX-01
        _set(cfg, "tp_atr_multiplier", s.tp_atr_multiplier)                # EX-02
        _set(cfg, "closed_bars_only", True)                                # MK-09
        _set(cfg, "cycle_interval_seconds", min(cfg.cycle_interval_seconds, s.cycle_seconds))
        if cfg.warmup_bars > 0:
            _set(cfg, "warmup_bars", max(cfg.warmup_bars, s.warmup_bars))  # MK-06 median window
        _set(cfg.risk_config, "risk_per_trade_pct", s.risk_per_trade_pct)  # SZ-01
        _set(cfg.risk_config, "atr_multiplier", s.sl_atr_multiplier)       # same SL in Layer 4
        _set(cfg.risk_config, "max_daily_loss_pct", s.max_daily_loss_pct)  # FQ-06
        return changes

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        raw = self._exec.get_symbol_snapshot(symbol)
        if not raw:
            return None
        return MarketSnapshot(
            symbol      = symbol,
            bid         = raw["bid"],
            ask         = raw["ask"],
            point       = raw["point"],
            # Same convention as the bars index: broker clock labelled UTC.
            server_time = pd.Timestamp(raw["time"], unit="s", tz="UTC").to_pydatetime(),
            volume_min  = raw.get("volume_min", 0.01),
            volume_step = raw.get("volume_step", 0.01),
        )

    def round_trip_cost(self, snapshot: Optional[MarketSnapshot], price: float, symbol: str) -> float:
        """Spread + commission on both sides, in price units. The commission
        rate is the one measured on this symbol's own closed trades."""
        spread = snapshot.spread_price if snapshot else 0.0
        return spread + 2.0 * self.commissions.rate(symbol) * price

    def build_context(
        self,
        symbol: str,
        bars: pd.DataFrame,
        portfolio_state,
        open_symbols: list[str],
        atr: float,
    ) -> EntryContext:
        snap = self.snapshot(symbol)
        price = snap.mid if snap else float(bars["close"].iloc[-1])
        return EntryContext(
            symbol          = symbol,
            now             = datetime.now(timezone.utc),
            timeframe_s     = self.timeframe_s,
            bars            = bars,
            snapshot        = snap,
            atr             = atr,
            portfolio_state = portfolio_state,
            open_symbols    = open_symbols,
            round_trip_cost = self.round_trip_cost(snap, price, symbol),
        )

    # ------------------------------------------------------------------
    # Rule stages
    # ------------------------------------------------------------------

    def check_pre_signal(self, ctx: EntryContext) -> EvaluationResult:
        return self.engine.evaluate_entry(ctx, PRE_SIGNAL_STAGES)

    def check_signal(self, ctx: EntryContext) -> EvaluationResult:
        return self.engine.evaluate_entry(ctx, POST_SIGNAL_STAGES)

    def check_order(self, ctx: EntryContext) -> EvaluationResult:
        return self.engine.evaluate_entry(ctx, ORDER_STAGES)

    @staticmethod
    def rebase_order(order, snapshot: Optional[MarketSnapshot]) -> None:
        """Re-anchor entry / SL / TP on the live executable price (ask for a
        buy, bid for a sell) — the last closed bar's close can be stale."""
        if snapshot is None:
            return
        price = snapshot.ask if order.direction > 0 else snapshot.bid
        sl_dist = abs(order.entry_price - order.stop_loss)
        tp_dist = abs(order.take_profit - order.entry_price)
        order.entry_price = price
        order.stop_loss   = round(price - order.direction * sl_dist, 5)
        order.take_profit = round(price + order.direction * tp_dist, 5)

    # ------------------------------------------------------------------
    # Decision journal (DB-01 / DB-04)
    # ------------------------------------------------------------------

    def _persistence_ok(self) -> bool:
        sink = self._sink_getter()
        return bool(sink is not None and getattr(sink, "persistence_ok", False))

    def record(
        self,
        symbol: str,
        verdict: str,
        *,
        ctx: Optional[EntryContext] = None,
        evaluation: Optional[EvaluationResult] = None,
        rule_id: str = "",
        reason: str = "",
        ticket=None,
        extra: Optional[dict] = None,
        dedupe: bool = False,
    ) -> None:
        """Journal one decision. With dedupe=True, a rejection repeated by the
        same rule on consecutive bars (e.g. outside session) is written once."""
        failed = evaluation.failed if evaluation else None
        if failed is not None:
            rule_id = rule_id or failed.rule_id
            reason = reason or failed.reason
        if dedupe:
            if self._last_reject.get(symbol) == rule_id:
                return
            self._last_reject[symbol] = rule_id
        else:
            self._last_reject.pop(symbol, None)

        metrics = dict(evaluation.metrics) if evaluation else {}
        metrics.update(extra or {})
        decision = getattr(ctx, "decision", None)
        row = {
            "ts":            datetime.now(timezone.utc),
            "symbol":        symbol,
            "verdict":       verdict,
            "rule_id":       rule_id or None,
            "reason":        reason or None,
            "direction":     decision.signal.name if decision is not None else metrics.get("direction"),
            "confidence":    round(decision.confidence, 4) if decision is not None else None,
            "ml_win_proba":  metrics.get("ml_win_proba"),
            "rl_action":     metrics.get("rl_action"),
            "regime":        getattr(ctx, "regime", None) if ctx else metrics.get("regime"),
            "spread_points": (round(ctx.snapshot.spread_points, 2)
                              if ctx is not None and ctx.snapshot is not None else None),
            "atr":           ctx.atr if ctx is not None and ctx.atr else None,
            "size_mult":     metrics.get("size_mult"),
            "ticket":        str(ticket) if ticket else None,
            "rules_version": self.rules_version,
        }
        logger.info("    [%s] SCALP %s %s %s", symbol, verdict, rule_id or "", reason or "")
        sink = self._sink_getter()
        if sink is None:
            return
        try:
            sink.record_decision(row)
        except Exception as exc:
            logger.warning("Decision journal write failed: %s", exc)

    # ------------------------------------------------------------------
    # Frequency journal (DB-03)
    # ------------------------------------------------------------------

    def on_entry(self, symbol: str, ts: datetime) -> None:
        self.journal.record_entry(symbol, ts)

    def on_close(self, symbol: str, ts: datetime, pnl: float) -> None:
        self.journal.record_close(symbol, ts, pnl)

    def restore_once(self) -> None:
        """Rebuild trade counts / cooldowns from the persisted journal so a
        restart cannot reset them. Runs once, when a monitor is attached."""
        if self._restored:
            return
        sink = self._sink_getter()
        if sink is None:
            return
        self._restored = True
        since = datetime.now(timezone.utc) - timedelta(days=1)
        try:
            decisions = sink.query_decisions(verdict="EXECUTED", start=since, limit=1000)
            trades = sink.query_trades(start=since, limit=1000)
        except Exception as exc:
            logger.warning("Scalping journal restore failed: %s", exc)
            return
        entries = [JournalEntry(r["symbol"], ts) for r in decisions
                   if (ts := _parse_ts(r.get("ts"))) and ts >= since]
        closes = [JournalClose(r["symbol"], ts, float(r.get("pnl") or 0.0)) for r in trades
                  if (ts := _parse_ts(r.get("ts"))) and ts >= since]
        self.journal.load(entries, closes)
        logger.info("Scalping journal restored: %d entries, %d closed trades (last 24h).",
                    len(entries), len(closes))

    # ------------------------------------------------------------------
    # Position management (EX-04..06, EX-08)
    # ------------------------------------------------------------------

    def manage(self, positions: list, open_trades: dict[int, dict]) -> list[PositionAction]:
        """Actions for the positions this trader opened (tracked tickets)."""
        now = datetime.now(timezone.utc)
        actions: list[PositionAction] = []
        for pos in positions:
            info = open_trades.get(pos.ticket)
            if info is None:
                continue                 # manual / foreign position — not ours
            managed = ManagedPosition(
                ticket          = pos.ticket,
                symbol          = pos.symbol,
                direction       = pos.direction,
                open_price      = pos.open_price,
                current_price   = pos.current_price,
                sl              = pos.sl,
                tp              = pos.tp,
                entry_ts        = info["entry_ts"],
                entry_atr       = info.get("entry_atr", 0.0),
                round_trip_cost = info.get("round_trip_cost", 0.0),
            )
            action = self.engine.evaluate_position(
                PositionContext(now=now, timeframe_s=self.timeframe_s, position=managed)
            )
            if action is not None:
                actions.append(action)
        return actions
