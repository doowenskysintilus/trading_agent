#!/usr/bin/env python3
"""
Test script for the SignalAggregator confidence fix.

Bug being fixed: SignalAggregator._confidence() computed
confidence = (winner_score - loser_score) / total, which equals 1.0
whenever only one side votes (pure consensus, loser_score = 0) —
regardless of how weak the underlying strategy's own confidence was.
This is what produced confidence=1.0 on every single AUDUSD SELL cycle
in data/storage/logs/live/cycles_20260623.jsonl (the live cycle log only
stores the final decision, not the raw per-strategy votes, so this test
reconstructs a plausible equivalent scenario rather than literally
replaying that log).

New formula: confidence = conviction x dominance
  - conviction = weighted-average confidence of the winning side's votes
  - dominance  = old margin/total formula, applied ONLY when there is a
    real conflict (both sides voted). Pure consensus -> dominance = 1.0.
"""

from research.alpha_models.base import AlphaSignal, SignalType
from engines.signal_engine.signal_aggregator import SignalAggregator

PASS = "✓"
FAIL = "✗"


def sig(name, signal, confidence):
    return AlphaSignal(signal=signal, confidence=confidence, strategy_name=name)


def check(label, value, expected, tol=1e-4):
    # tol=1e-4 because AggregatedDecision rounds confidence to 4 decimals.
    ok = abs(value - expected) <= tol
    print(f"  {PASS if ok else FAIL} {label}: got {value:.4f}, expected ~{expected:.4f}")
    return ok


all_ok = True

# ---------------------------------------------------------------------------
print("=" * 70)
print("TEST 1: single-strategy pure consensus at moderate confidence")
print("=" * 70)
# Reproduces the AUDUSD case: one active strategy, no opposing vote.
# Old formula: (0.55 - 0) / 0.55 = 1.0  <- the bug.
# New formula: conviction = 0.55, dominance = 1.0 (nothing to dominate) -> 0.55
agg = SignalAggregator(min_confidence=0.05)
agg.set_weight("momentum", 1.0)
decision = agg.aggregate([sig("momentum", SignalType.SELL, 0.55)])
all_ok &= check("confidence", decision.confidence, 0.55)
all_ok &= check("signal is SELL", 1.0 if decision.signal == SignalType.SELL else 0.0, 1.0)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 2: strong multi-strategy consensus")
print("=" * 70)
# Three strategies agree BUY at 0.8 confidence each, equal weight.
# conviction = 0.8, dominance = 1.0 (no opposing side) -> confidence ~0.8
agg2 = SignalAggregator(min_confidence=0.05)
for name in ("momentum", "mean_reversion", "calendar_alpha"):
    agg2.set_weight(name, 1.0)
decision2 = agg2.aggregate([
    sig("momentum", SignalType.BUY, 0.8),
    sig("mean_reversion", SignalType.BUY, 0.8),
    sig("calendar_alpha", SignalType.BUY, 0.8),
])
all_ok &= check("confidence", decision2.confidence, 0.8)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 3: conflict still discounts confidence via dominance")
print("=" * 70)
# One strong BUY (0.9) vs one strong SELL (0.9), equal weight -> total
# disagreement, dominance should crush confidence towards 0 even though
# each individual voter was very sure.
agg3 = SignalAggregator(min_confidence=0.05)
agg3.set_weight("momentum", 1.0)
agg3.set_weight("mean_reversion", 1.0)
decision3 = agg3.aggregate([
    sig("momentum", SignalType.BUY, 0.9),
    sig("mean_reversion", SignalType.SELL, 0.9),
])
# buy_score == sell_score == 0.9 -> dominance = 0 -> HOLD via resolver (min_dominance=0.15)
all_ok &= check(
    "conflict resolves to HOLD (insufficient dominance)",
    1.0 if decision3.signal == SignalType.HOLD else 0.0,
    1.0,
)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST 4: lopsided conflict — winner should be discounted, not pinned to 1.0")
print("=" * 70)
# BUY at 0.9 (weight 1.0) vs a weak SELL at 0.2 (weight 1.0).
# buy_score=0.9, sell_score=0.2, total=1.1
# conviction(buy) = 0.9/1.0 = 0.9
# dominance = (0.9-0.2)/1.1 = 0.6364
# expected confidence = 0.9 * 0.6364 = 0.5727
agg4 = SignalAggregator(min_confidence=0.05)
agg4.set_weight("momentum", 1.0)
agg4.set_weight("mean_reversion", 1.0)
decision4 = agg4.aggregate([
    sig("momentum", SignalType.BUY, 0.9),
    sig("mean_reversion", SignalType.SELL, 0.2),
])
all_ok &= check("signal is BUY", 1.0 if decision4.signal == SignalType.BUY else 0.0, 1.0)
all_ok &= check("confidence (conviction x dominance)", decision4.confidence, 0.9 * ((0.9 - 0.2) / 1.1))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All confidence-fix checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
