"""Scalping rule engine — see SCALPING_RULES.md and base.py for the design."""

from engines.rule_engine.base import (
    ORDER_STAGES, POST_SIGNAL_STAGES, PRE_SIGNAL_STAGES,
    EntryContext, EntryRule, EvaluationResult, ManagedPosition, MarketSnapshot,
    PositionAction, PositionContext, PositionRule, RuleEngine, RuleResult, Stage,
)
from engines.rule_engine.factory import build_scalping_engine
from engines.rule_engine.journal import JournalClose, JournalEntry, TradeJournal

__all__ = [
    "ORDER_STAGES", "POST_SIGNAL_STAGES", "PRE_SIGNAL_STAGES",
    "EntryContext", "EntryRule", "EvaluationResult", "ManagedPosition", "MarketSnapshot",
    "PositionAction", "PositionContext", "PositionRule", "RuleEngine", "RuleResult", "Stage",
    "build_scalping_engine", "JournalClose", "JournalEntry", "TradeJournal",
]
