#!/usr/bin/env python3
"""
Test script for the scalping rule set (SCALPING_RULES.md):
  1. Each entry rule in isolation (sessions, spread, volatility, freshness,
     events, frequency, cooldown, daily target, confidence, ML, R:R, halt)
  2. RuleEngine fail-fast order + size multipliers; position rule arbitration
  3. Journal restore (DB-03), decision persistence + JSONL fallback (DB-01/04)
  4. WinClassifier reliability (ML-07), RL over-hold penalty (RL-05),
     calendar window after an event (MK-07), live-readiness gate (VA-02..05)
  5. LiveTrader end-to-end with a fake MT5 + fake monitor: entry journaled,
     sized, re-anchored; time stop closes the position; FQ-05 latches
No real broker, database or order is touched.
"""

import dataclasses
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config.settings import settings
from engines.risk_engine.risk_engine import PortfolioState, RiskEngine, RiskConfig, TradeOrder
from engines.rule_engine import (
    POST_SIGNAL_STAGES, PRE_SIGNAL_STAGES,
    EntryContext, ManagedPosition, MarketSnapshot,
    PositionContext, RuleEngine, RuleResult, Stage, TradeJournal,
)
from engines.rule_engine.base import EntryRule
from engines.rule_engine.frequency_rules import (
    CooldownRule, DailyTargetRule, SinglePositionRule, TradeFrequencyRule,
)
from engines.rule_engine.halt_rules import PersistenceRule, RiskHaltRule
from engines.rule_engine.market_rules import (
    DataFreshnessRule, EventWindowRule, SessionRule, SpreadRule, VolatilityRule,
)
from engines.rule_engine.position_rules import (
    AtrTrailingRule, BreakevenRule, SessionCloseRule, TimeStopRule,
)
from engines.rule_engine.signal_rules import ConfidenceRule, MLFilterRule, RewardRiskRule
from engines.signal_engine.signal_aggregator import AggregatedDecision
from research.alpha_models.base import AlphaModel, SignalType
from strategies.ml_agent.win_classifier import WinClassifier, WinPrediction

PASS = "[OK]"
FAIL = "[FAIL]"
all_ok = True


def check(cond: bool, label: str) -> None:
    global all_ok
    print(f"  {PASS if cond else FAIL} {label}")
    all_ok &= bool(cond)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def make_bars(n=600, price=81_000.0, rng_width=100.0, end=None, freq="min"):
    end = end or utc(2026, 10, 21, 12, 0)
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    closes = price + np.cumsum(rng.normal(0, rng_width / 10, n))
    return pd.DataFrame({
        "open": closes, "high": closes + rng_width / 2, "low": closes - rng_width / 2,
        "close": closes, "volume": 100.0, "spread": 500.0,
    }, index=idx)


def decision(signal=SignalType.BUY, conf=0.9):
    return AggregatedDecision(signal=signal, confidence=conf, buy_score=conf,
                              sell_score=0.0, net_score=conf)


# ---------------------------------------------------------------------------
section("TEST 1: market rules (MK-01..08)")
sessions = SessionRule(["07:00-11:00", "12:30-16:30"], ["00:00-21:00"], ["BTCUSD"])
wed_in = utc(2026, 10, 21, 8, 0)       # Wednesday
wed_out = utc(2026, 10, 21, 11, 30)
sat = utc(2026, 10, 24, 8, 0)
fri_late = utc(2026, 10, 23, 19, 30)
check(sessions.evaluate(EntryContext("EURUSD", wed_in, 60)).passed, "EURUSD 08:00 Wed → inside London session")
r = sessions.evaluate(EntryContext("EURUSD", wed_out, 60))
check(not r.passed and r.rule_id == "MK-01", f"EURUSD 11:30 → rejected by MK-01 ({r.reason})")
check(sessions.evaluate(EntryContext("EURUSD", sat, 60)).rule_id == "MK-03", "Saturday FX → MK-03")
check(sessions.evaluate(EntryContext("EURUSD", fri_late, 60)).rule_id == "MK-03", "Friday 19:30 FX → MK-03")
check(sessions.evaluate(EntryContext("BTCUSD", sat, 60)).passed, "BTCUSD Saturday 08:00 → crypto allowed")
r = sessions.evaluate(EntryContext("BTCUSD", utc(2026, 10, 21, 22, 0), 60))
check(r.rule_id == "MK-02", "BTCUSD 22:00 → MK-02 (low-liquidity window)")

snap = MarketSnapshot("BTCUSD", 81_000.0, 81_005.0, 0.01, server_time=utc(2026, 10, 21, 12, 1, 30))
spread_rule = SpreadRule(0.25, {"BTCUSD": 2000})
check(spread_rule.evaluate(EntryContext("BTCUSD", wed_in, 60, snapshot=snap, atr=100)).passed,
      "spread 500 pts, 0.05×ATR → OK")
wide = MarketSnapshot("BTCUSD", 81_000.0, 81_030.0, 0.01)
r = spread_rule.evaluate(EntryContext("BTCUSD", wed_in, 60, snapshot=wide, atr=100))
check(not r.passed and "cap" in r.reason, f"3000 pts > 2000 cap → rejected ({r.reason})")
r = spread_rule.evaluate(EntryContext("BTCUSD", wed_in, 60, snapshot=snap, atr=10))
check(not r.passed and "ATR" in r.reason, "spread 5 > 0.25 × ATR 10 → rejected")
check(not spread_rule.evaluate(EntryContext("BTCUSD", wed_in, 60)).passed, "no quote → rejected")

bars = make_bars()
vol = VolatilityRule(2.0, 3.0, 500)
check(vol.evaluate(EntryContext("BTCUSD", wed_in, 60, bars=bars, atr=100, round_trip_cost=10)).passed,
      "ATR 100 vs cost 10 → MK-05 OK")
check(vol.evaluate(EntryContext("BTCUSD", wed_in, 60, bars=bars, atr=15, round_trip_cost=10)).rule_id == "MK-05",
      "ATR 15 < 2 × cost 10 → MK-05")
check(vol.evaluate(EntryContext("BTCUSD", wed_in, 60, bars=bars, atr=1_000, round_trip_cost=10)).rule_id == "MK-06",
      "ATR 10× its median → MK-06")

fresh = DataFreshnessRule(2.0)
check(fresh.evaluate(EntryContext("BTCUSD", wed_in, 60, bars=bars, snapshot=snap)).passed,
      "last closed bar 30 s old (broker clock) → fresh")
stale = MarketSnapshot("BTCUSD", 81_000.0, 81_005.0, 0.01, server_time=utc(2026, 10, 21, 12, 10))
check(not fresh.evaluate(EntryContext("BTCUSD", wed_in, 60, bars=bars, snapshot=stale)).passed,
      "last closed bar 9 min old → MK-08")


class _Cal:
    def __init__(self, active): self.active = active
    def event_is_active(self, symbol, window_minutes): return self.active


check(not EventWindowRule(_Cal(True), 15).evaluate(EntryContext("EURUSD", wed_in, 60)).passed,
      "high-impact event in window → MK-07")
check(EventWindowRule(None, 15).evaluate(EntryContext("EURUSD", wed_in, 60)).passed,
      "no calendar → MK-07 does not block")

# ---------------------------------------------------------------------------
section("TEST 2: frequency, halt and signal rules")
journal = TradeJournal()
now = utc(2026, 10, 21, 12, 0)
for minutes in (50, 40, 30, 20):
    journal.record_entry("BTCUSD", now - timedelta(minutes=minutes))
freq = TradeFrequencyRule(journal, 4, 30)
r = freq.evaluate(EntryContext("BTCUSD", now, 60))
check(r.rule_id == "FQ-01" and not r.passed, "4 BTCUSD trades in the last hour → FQ-01")
check(freq.evaluate(EntryContext("EURUSD", now, 60)).passed, "other symbol unaffected")
check(not SinglePositionRule().evaluate(EntryContext("BTCUSD", now, 60, open_symbols=["BTCUSD"])).passed,
      "position already open on the symbol → SC-04")

cool = CooldownRule(journal, 5, 3, 30)
journal.record_close("BTCUSD", now - timedelta(minutes=2), -10.0)
check(cool.evaluate(EntryContext("BTCUSD", now, 60)).rule_id == "FQ-03", "loss 2 min ago → FQ-03 cooldown")
check(cool.evaluate(EntryContext("BTCUSD", now + timedelta(minutes=6), 60)).passed, "after 5 min → allowed")
journal.record_close("EURUSD", now - timedelta(minutes=1), -5.0)
journal.record_close("XAUUSD", now - timedelta(seconds=30), -5.0)
r = cool.evaluate(EntryContext("ETHUSD", now, 60))
check(r.rule_id == "FQ-04", f"3 consecutive losses → global pause ({r.reason})")
check(cool.evaluate(EntryContext("ETHUSD", now + timedelta(minutes=31), 60)).passed, "pause over after 30 min")

ps = PortfolioState(equity=10_160, balance=10_160, peak_equity=10_160, daily_start_equity=10_000)
check(DailyTargetRule(0.015).evaluate(EntryContext("BTCUSD", now, 60, portfolio_state=ps)).rule_id == "FQ-08",
      "+1.6 % today ≥ +1.5 % target → FQ-08")

risk = RiskEngine(RiskConfig(max_daily_loss_pct=0.02))
ps_loss = PortfolioState(equity=9_750, balance=9_750, peak_equity=10_000, daily_start_equity=10_000)
r = RiskHaltRule(risk).evaluate(EntryContext("BTCUSD", now, 60, portfolio_state=ps_loss))
check(not r.passed and "daily_loss" in r.reason, f"-2.5 % today → halted ({r.reason[:40]}…)")
check(not PersistenceRule(lambda: False).evaluate(EntryContext("BTCUSD", now, 60)).passed,
      "journal down → DB-04 blocks entries")

ftmo = RiskEngine(RiskConfig(ftmo_enabled=True, ftmo_initial_balance=10_000))
ps_first_win = PortfolioState(equity=10_000.58, balance=10_000.58, peak_equity=10_000.58,
                              daily_start_equity=10_000, ftmo_initial_balance=10_000,
                              ftmo_total_profit=0.58, ftmo_today_profit=0.58)
check(ftmo.check_halt(ps_first_win) is None,
      "FTMO consistency: first +0.58 $ win (100 % of a tiny total) does not halt the day")
ps_big_day = dataclasses.replace(ps_first_win, equity=10_150, ftmo_total_profit=150, ftmo_today_profit=150)
r = ftmo.check_halt(ps_big_day)
check(r is not None and r.reason.value == "ftmo_consistency_rule",
      "…but once total profit ≥ 1 % of the challenge, a 100 % day still halts")

conf = ConfidenceRule(0.55, 0.65)
check(conf.evaluate(EntryContext("BTCUSD", now, 60, decision=decision(conf=0.6), regime="trending")).passed,
      "conf 0.60 trending ≥ 0.55 → OK")
check(conf.evaluate(EntryContext("BTCUSD", now, 60, decision=decision(conf=0.6), regime="neutral")).rule_id == "SG-04",
      "conf 0.60 neutral < 0.65 → SG-04")

ml = MLFilterRule(0.55, 0.5)
r = ml.evaluate(EntryContext("BTCUSD", now, 60, ml_prediction=WinPrediction(None, "no_model")))
check(r.passed and r.size_multiplier == 0.5, "no model → half size (ML-03)")
check(ml.evaluate(EntryContext("BTCUSD", now, 60, ml_prediction=WinPrediction(None, "feature_mismatch"))).rule_id == "ML-08",
      "feature mismatch → refused (ML-08)")
check(ml.evaluate(EntryContext("BTCUSD", now, 60, ml_prediction=WinPrediction(0.50, "ok"))).rule_id == "ML-01",
      "P(win) 0.50 < 0.55 → ML-01")

rr = RewardRiskRule(1.0, 3.0)
good = TradeOrder("BTCUSD", "agg", 1, 0.1, 81_000, 100, stop_loss=80_880, take_profit=81_150)
bad = TradeOrder("BTCUSD", "agg", 1, 0.1, 81_000, 100, stop_loss=80_880, take_profit=81_020)
check(rr.evaluate(EntryContext("BTCUSD", now, 60, order=good, round_trip_cost=10)).passed,
      "TP 150 / SL 120 / cost 10 → net R:R 1.07 OK")

from engines.rule_engine.costs import CommissionModel

cm = CommissionModel(0.000035, provider=lambda: {"BTCUSD": {"rate_per_side": 0.000326}})
check(cm.rate("BTCUSD") == 0.000326 and cm.rate("btcusd") == 0.000326, "measured BTCUSD commission used")
check(cm.rate("XAUUSD") == 0.000035, "symbol without history → configured default")
broken = CommissionModel(0.000035, provider=lambda: 1 / 0)
check(broken.rate("BTCUSD") == 0.000035, "history unavailable → default, no crash")
check(not rr.evaluate(EntryContext("BTCUSD", now, 60, order=bad, round_trip_cost=10)).passed,
      "TP 20 < 3 × cost → EX-03")

# ---------------------------------------------------------------------------
section("TEST 3: engine order, size multipliers, position arbitration")
calls = []


class _Probe(EntryRule):
    def __init__(self, rule_id, stage, passed, mult=1.0):
        self.rule_id, self.stage, self._passed, self._mult = rule_id, stage, passed, mult

    def evaluate(self, ctx):
        calls.append(self.rule_id)
        return RuleResult.ok(self.rule_id, self._mult) if self._passed else RuleResult.reject(self.rule_id, "x")


engine = RuleEngine([_Probe("C", Stage.MARKET, False), _Probe("A", Stage.HALT, True),
                     _Probe("B", Stage.FREQUENCY, True), _Probe("D", Stage.SIGNAL, True)], [])
res = engine.evaluate_entry(EntryContext("X", now, 60), PRE_SIGNAL_STAGES)
check(calls == ["A", "B", "C"] and res.failed.rule_id == "C", f"stage order + fail-fast: {calls}")
engine2 = RuleEngine([_Probe("M1", Stage.ML, True, 0.5), _Probe("S1", Stage.SIGNAL, True, 0.8)], [])
check(abs(engine2.evaluate_entry(EntryContext("X", now, 60), POST_SIGNAL_STAGES).size_multiplier - 0.4) < 1e-9,
      "size multipliers compound (0.8 × 0.5)")

entry_ts = now - timedelta(minutes=5)
pos = ManagedPosition(1, "BTCUSD", 1, 81_000, 81_120, sl=80_880, tp=81_150, entry_ts=entry_ts,
                      entry_atr=100, round_trip_cost=10)
pengine = RuleEngine([], [TimeStopRule(15), BreakevenRule(0.8), AtrTrailingRule(1.0, 0.5)])
act = pengine.evaluate_position(PositionContext(now, 60, pos))
check(act is not None and act.rule_id == "EX-06" and abs(act.new_sl - 81_070) < 1e-6,
      f"+1.2 ATR: trailing SL 81 070 beats breakeven 81 010 → {act and act.rule_id}")
pos_be = dataclasses.replace(pos, current_price=81_085)
act = pengine.evaluate_position(PositionContext(now, 60, pos_be))
check(act is not None and act.rule_id == "EX-05" and abs(act.new_sl - 81_010) < 1e-6,
      "+0.85 ATR: SL moved to breakeven + costs")
pos_old = dataclasses.replace(pos, entry_ts=now - timedelta(minutes=16))
act = pengine.evaluate_position(PositionContext(now, 60, pos_old))
check(act is not None and act.kind == "close" and act.rule_id == "EX-04", "held 16 min → time stop wins")
pos_tight = dataclasses.replace(pos_be, sl=81_050)
check(pengine.evaluate_position(PositionContext(now, 60, pos_tight)) is None,
      "never loosens a stop already tighter")
sclose = SessionCloseRule(sessions.windows_for, 10)
act = sclose.evaluate(PositionContext(utc(2026, 10, 21, 10, 55), 60,
                                      dataclasses.replace(pos, symbol="EURUSD")))
check(act is not None and act.rule_id == "EX-08", "EURUSD 10:55, session ends 11:00 → EX-08 close")

# ---------------------------------------------------------------------------
section("TEST 4: persistence, ML reliability, RL penalty, calendar, readiness")
from monitoring.metrics_store import MetricsStore

tmp = tempfile.mkdtemp()
try:
    store = MetricsStore(log_dir=tmp)
    check(store.connect() is False, "no MySQL config → never connects to the real database")
    store.write_decision({"ts": now, "symbol": "BTCUSD", "verdict": "EXECUTED", "rule_id": None,
                          "ticket": "42", "rules_version": "SR-1.0"})
    rows = store.query_decisions(verdict="EXECUTED")
    check(len(rows) == 1 and rows[0]["ticket"] == "42", "decision written to JSONL fallback and read back")
    check(store.persistence_ok, "persistence_ok with JSONL only (MySQL down)")
    store.write_trade(now, "BTCUSD", "agg", "BUY", 0.1, 81_000, 81_100, 10.0,
                      extra={"rl_action": "BUY", "regime": "trending", "bogus": 1})
    t = store.query_trades()[0]
    check(t.get("rl_action") == "BUY" and "bogus" not in t, "trade audit columns stored, unknown keys dropped")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

from live_trading.scalping import ScalpingController


class _Sink:
    persistence_ok = True
    def __init__(self, decisions, trades): self._d, self._t = decisions, trades
    def query_decisions(self, **kw): return self._d
    def query_trades(self, **kw): return self._t
    def record_decision(self, row): pass


recent = datetime.now(timezone.utc) - timedelta(minutes=10)
sink = _Sink([{"symbol": "BTCUSD", "ts": recent.isoformat()}] * 4,
             [{"symbol": "BTCUSD", "ts": recent.isoformat(), "pnl": -3.0}])
ctrl = ScalpingController(settings.scalping, timeframe="M1", risk_engine=risk, calendar_provider=None,
                          exec_engine=SimpleNamespace(), sink_getter=lambda: sink)
ctrl.restore_once()
check(ctrl.journal.count_entries(recent - timedelta(seconds=1), "BTCUSD") == 4,
      "DB-03: 4 entries restored from the journal after a restart")

ctrl_fees = ScalpingController(
    settings.scalping, timeframe="M1", risk_engine=risk, calendar_provider=None,
    exec_engine=SimpleNamespace(), sink_getter=lambda: sink,
    commission_model=CommissionModel(0.000035, provider=lambda: {"BTCUSD": {"rate_per_side": 0.000326}}),
)
btc_cost = ctrl_fees.round_trip_cost(snap, 81_000.0, "BTCUSD")
check(abs(btc_cost - (5.0 + 2 * 0.000326 * 81_000)) < 1e-6,
      f"BTCUSD round-trip cost uses the measured rate: {btc_cost:.2f} (was {5 + 2 * 0.000035 * 81_000:.2f})")

X = np.random.default_rng(1).normal(size=(400, 3))
y = (np.random.default_rng(2).random(400) > 0.5).astype(int)      # pure noise → AUC ≈ 0.5
tmp = tempfile.mkdtemp()
try:
    clf = WinClassifier(model_path=f"{tmp}/m.joblib")
    rep = clf.train(X, y, ["a", "b", "c"], min_samples=300, min_reliable_auc=0.55)
    pred = clf.predict({"a": 0, "b": 0, "c": 0})
    check(not rep.reliable and pred.status == "unreliable",
          f"noise model AUC {rep.auc:.2f} < 0.55 → unreliable (ML-07)")
    check(clf.predict({"a": 0}).status == "unreliable", "status reported before feature check")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

from strategies.rl_agent.rl_trainer import RewardConfig, RLTradingEnv

feats = np.random.default_rng(3).normal(0, 1e-4, size=(200, 4)).astype(np.float32)
prices = np.full(200, 1.10, dtype=np.float32)


def run_env(max_hold, alpha):
    env = RLTradingEnv(feats, RewardConfig(max_holding_bars=max_hold, alpha_overhold=alpha),
                       window_size=10, prices=prices, sl_pct=0.5, tp_pct=0.5)
    env.reset()
    total = 0.0
    env.step(2)                                     # open a long
    for _ in range(40):
        total += env.step(2)[1]                     # keep holding the same side
    return total


check(run_env(15, 0.05) < run_env(0, 0.0) - 1.0, "RL-05: holding 40 bars > 15 is penalised")

from research.feature_store.calendar_provider import CalendarProvider, EconomicEvent, EventImportance

cal = CalendarProvider(source="mock")
just_past = datetime.now(timezone.utc) - timedelta(minutes=5)
cal._fetch_events = lambda: [EconomicEvent(timestamp=just_past, country="USD", name="CPI",
                                           importance=EventImportance.HIGH)]
check(cal.event_is_active("EURUSD", window_minutes=15), "MK-07: event 5 min AGO still blocks")
check(not cal.event_is_active("EURUSD", window_minutes=3), "…but not outside the window")

from backtesting.validation import check_live_readiness, demo_matches_backtest
from backtesting.walk_forward import FoldResult, WalkForwardResult
from backtesting.metrics import PerformanceMetrics


def fold(i, ret, dd, pnls):
    pm = PerformanceMetrics(); pm.total_return_pct = ret; pm.max_drawdown_pct = dd
    return FoldResult(i, "", "", 0, SimpleNamespace(portfolio_metrics=pm,
                                                     trade_log=pd.DataFrame({"pnl": pnls})))


wf = WalkForwardResult(folds=[fold(i, 0.02, 0.03, [2.0] * 50 + [-1.0] * 30) for i in range(4)])
check(check_live_readiness(wf).ready, "4 profitable folds, 320 trades, PF 3.3, DD 3 % → ready")
wf_bad = WalkForwardResult(folds=[fold(0, -0.01, 0.08, [1.0, -2.0])])
rep = check_live_readiness(wf_bad)
check(not rep.ready and len(rep.failures) == 5, f"bad run → {len(rep.failures)} failures")
check(demo_matches_backtest(0.52, 0.58) and not demo_matches_backtest(0.40, 0.58), "VA-06 win-rate gap")

# ---------------------------------------------------------------------------
section("TEST 5: LiveTrader end-to-end (fake MT5, fake monitor)")
from engines.execution_engine.execution_engine import ExecutionResult, ExecutionStatus, PositionInfo
from live_trading.live_trader import CycleStatus, LiveTrader, LiveTraderConfig


class _BuyStrategy(AlphaModel):
    def __init__(self):
        super().__init__(name="momentum")

    def compute(self, data):
        sig = self._hold()
        return dataclasses.replace(sig, signal=SignalType.BUY, confidence=0.9) \
            if dataclasses.is_dataclass(sig) else sig


class _Monitor:
    persistence_ok = True
    def __init__(self): self.decisions, self.trades = [], []
    def __call__(self, result): pass
    def record_trade(self, **kw): self.trades.append(kw)
    def record_decision(self, row): self.decisions.append(row)
    def query_decisions(self, **kw): return []
    def query_trades(self, **kw): return []


class _FakeExec:
    def __init__(self):
        self.positions, self.closed, self.orders = [], [], []
        self.bar_end = None

    def get_open_positions(self): return list(self.positions)
    def get_account_info(self): return {"equity": 10_000.0, "balance": 10_000.0, "server": "Demo"}
    def ensure_connected(self): return True
    def get_closed_pnl(self, ticket): return {"pnl": -12.0, "exit_price": 80_990.0, "exit_reason": "client"}
    def modify_position_sl(self, *a): return True

    def get_symbol_snapshot(self, symbol):
        return {"bid": 81_000.0, "ask": 81_005.0, "point": 0.01,
                "time": int((self.bar_end + timedelta(seconds=90)).timestamp())}

    def execute(self, decision, order):
        self.orders.append(order)
        return ExecutionResult(ExecutionStatus.FILLED, ticket=777, requested_price=order.entry_price,
                               fill_price=order.entry_price, volume=0.01)

    def close_position(self, symbol, ticket):
        self.closed.append(ticket)
        self.positions = [p for p in self.positions if p.ticket != ticket]
        return ExecutionResult(ExecutionStatus.FILLED, ticket=ticket)


scalp = dataclasses.replace(settings.scalping, crypto_sessions=["00:00-23:59"])
trader = LiveTrader(LiveTraderConfig(symbols=["BTCUSD"], timeframe="M1", cycle_interval_seconds=60,
                                     scalping_enabled=True, scalping=scalp, htf_enabled=False,
                                     collect_experiences=False, ml_filter_enabled=False))
fake, mon = _FakeExec(), _Monitor()
trader._exec_engine = fake
trader._scalp._exec = fake
trader.register_strategy(_BuyStrategy(), weight=1.0)
trader.register_hook(mon)
trader._regime_enabled = False

check(trader.cfg.closed_bars_only and trader.cfg.risk_config.risk_per_trade_pct == 0.0025
      and trader.cfg.sl_atr_multiplier == 1.2, "scalping profile applied (MK-09, SZ-01, EX-01)")
check(trader._exec_engine is fake and trader.cfg.cycle_interval_seconds == scalp.cycle_seconds,
      "fast cycle for position management")

bars_now = make_bars(end=pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1))
fake.bar_end = bars_now.index[-1].to_pydatetime()
trader._fetch_data = lambda s: (trader._data_cache.__setitem__(s, bars_now), bars_now)[1]
trader._detect_regime = lambda s: "trending"
res = trader._run_cycle("BTCUSD")
verdicts = [d["verdict"] for d in mon.decisions]
print(f"  cycle status={res.status.name} decisions={verdicts} "
      f"{[(d['rule_id'], d['reason']) for d in mon.decisions if d['verdict'] != 'EXECUTED']}")
check(res.status == CycleStatus.OK and verdicts[-1] == "EXECUTED", "full cycle → EXECUTED and journaled")
order = fake.orders[-1] if fake.orders else None
check(order is not None and order.entry_price == 81_005.0, "order re-anchored on the live ask (rebase)")
check(order is not None and abs((order.entry_price - order.stop_loss) / (order.take_profit - order.entry_price) - 0.8) < 0.01,
      "SL/TP distances 1.2 / 1.5 ATR")
info = trader._open_trades.get(777, {})
check(info.get("entry_atr", 0) > 0 and info.get("round_trip_cost", 0) > 0 and info.get("size_mult") == 0.5,
      "entry ATR / costs / ML half-size stored with the ticket")
check(trader._scalp.journal.count_entries(datetime.now(timezone.utc) - timedelta(minutes=1), "BTCUSD") == 1,
      "entry counted for FQ-01/02")

res2 = trader._run_cycle("BTCUSD")
check(res2.status == CycleStatus.BAR_ALREADY_TRADED, "same bar → not re-analysed")

fake.positions = [PositionInfo(777, "BTCUSD", 1, 0.01, 81_005.0, 81_010.0, order.stop_loss,
                               order.take_profit, 0.0, 0)]
trader._open_trades[777]["entry_ts"] = datetime.now(timezone.utc) - timedelta(minutes=16)
trader._manage_scalp_positions()
check(fake.closed == [777], "time stop closed the position (EX-04)")
check(trader._open_trades[777].get("exit_reason_override") == "time_stop", "exit tagged time_stop")

trader._portfolio_state.consecutive_losses = trader.cfg.risk_config.max_consecutive_losses - 1
trader._reconcile_closed_trades()
check(mon.trades and mon.trades[-1]["exit_reason"] == "time_stop"
      and mon.trades[-1]["extra"]["rules_version"] == scalp.rules_version,
      "closed trade persisted with exit reason + audit columns (DB-02)")
check(trader.emergency.is_active, "6th consecutive loss latches the emergency stop (FQ-05)")
trader.emergency.clear()

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All scalping rule checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
