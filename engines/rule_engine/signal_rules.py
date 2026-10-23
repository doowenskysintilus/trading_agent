"""Stages 3–5 — aggregated signal, ML filter and the built order."""

from __future__ import annotations

from engines.rule_engine.base import EntryContext, EntryRule, RuleResult, Stage


class ConfidenceRule(EntryRule):
    """SG-02 — minimum aggregated confidence.
    SG-04 — stricter minimum when the regime is neutral."""
    rule_id = "SG-02/04"
    stage = Stage.SIGNAL

    def __init__(self, min_confidence: float, neutral_min_confidence: float) -> None:
        self._min = min_confidence
        self._neutral_min = neutral_min_confidence

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        if ctx.decision is None:
            return RuleResult.reject(self.rule_id, "no aggregated decision")
        conf = float(ctx.decision.confidence)
        threshold, rule = (
            (self._neutral_min, "SG-04") if ctx.regime == "neutral" else (self._min, "SG-02")
        )
        metrics = {"confidence": round(conf, 4), "regime": ctx.regime}
        if conf < threshold:
            return RuleResult.reject(
                rule, f"confidence {conf:.3f} < {threshold:.2f} (regime {ctx.regime})", **metrics,
            )
        return RuleResult.ok(self.rule_id, **metrics)


class MLFilterRule(EntryRule):
    """ML-01 — P(win) ≥ threshold.
    ML-03 / SZ-04 — model missing or not reliable (ML-07): trade at reduced
    size instead of letting everything through at full size.
    ML-08 — feature schema mismatch: refuse the trade."""
    rule_id = "ML-01/03/08"
    stage = Stage.ML

    def __init__(self, min_win_proba: float, unreliable_size_mult: float) -> None:
        self._min = min_win_proba
        self._fallback_mult = unreliable_size_mult

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        pred = ctx.ml_prediction
        status = getattr(pred, "status", "no_model")
        proba = getattr(pred, "proba", None)
        if status == "feature_mismatch":
            return RuleResult.reject("ML-08", "live features differ from the model's training schema",
                                     ml_status=status)
        if status != "ok" or proba is None:
            return RuleResult.ok("ML-03", size_multiplier=self._fallback_mult, ml_status=status)
        metrics = {"ml_win_proba": round(proba, 4), "ml_status": status}
        if proba < self._min:
            return RuleResult.reject("ML-01", f"P(win) {proba:.3f} < {self._min:.2f}", **metrics)
        return RuleResult.ok("ML-01", **metrics)


class RewardRiskRule(EntryRule):
    """EX-03 — net reward:risk after costs, and TP large enough to pay them."""
    rule_id = "EX-03"
    stage = Stage.ORDER

    def __init__(self, min_net_rr: float, min_tp_cost_ratio: float) -> None:
        self._min_rr = min_net_rr
        self._min_tp_cost = min_tp_cost_ratio

    def evaluate(self, ctx: EntryContext) -> RuleResult:
        order = ctx.order
        if order is None or order.stop_loss is None or order.take_profit is None:
            return RuleResult.reject(self.rule_id, "order without SL/TP (EX-01)")
        risk = abs(order.entry_price - order.stop_loss)
        reward = abs(order.take_profit - order.entry_price)
        cost = ctx.round_trip_cost
        net_rr = (reward - cost) / (risk + cost) if risk + cost > 0 else 0.0
        metrics = {"net_rr": round(net_rr, 3)}
        if cost > 0 and reward < self._min_tp_cost * cost:
            return RuleResult.reject(
                self.rule_id, f"TP distance {reward:.5f} < {self._min_tp_cost:g} × cost {cost:.5f}",
                **metrics,
            )
        if net_rr < self._min_rr:
            return RuleResult.reject(self.rule_id, f"net R:R {net_rr:.2f} < {self._min_rr:g}", **metrics)
        return RuleResult.ok(self.rule_id, **metrics)
