"""Builds the scalping RuleEngine from ScalpingSettings (SCALPING_RULES.md)."""

from __future__ import annotations

from typing import Callable

from engines.rule_engine.base import RuleEngine
from engines.rule_engine.frequency_rules import (
    CooldownRule, DailyTargetRule, SinglePositionRule, TradeFrequencyRule,
)
from engines.rule_engine.halt_rules import PersistenceRule, RiskHaltRule
from engines.rule_engine.journal import TradeJournal
from engines.rule_engine.market_rules import (
    DataFreshnessRule, EventWindowRule, SessionRule, SpreadRule, VolatilityRule,
)
from engines.rule_engine.position_rules import (
    AtrTrailingRule, BreakevenRule, SessionCloseRule, TimeStopRule,
)
from engines.rule_engine.signal_rules import ConfidenceRule, MLFilterRule, RewardRiskRule


def build_scalping_engine(
    cfg,                                   # config.settings.ScalpingSettings
    *,
    risk_engine,
    calendar_provider,
    journal: TradeJournal,
    is_persistence_ok: Callable[[], bool],
) -> RuleEngine:
    sessions = SessionRule(
        fx_sessions          = cfg.fx_sessions,
        crypto_sessions      = cfg.crypto_sessions,
        crypto_symbols       = cfg.crypto_symbols,
        friday_cutoff        = cfg.friday_cutoff_utc,
        sunday_open          = cfg.sunday_open_utc,
        sunday_open_wait_min = cfg.sunday_open_wait_min,
    )
    entry_rules = [
        # Stage 0 — hard stops
        RiskHaltRule(risk_engine),
        PersistenceRule(is_persistence_ok),
        # Stage 1 — frequency / capital protection
        SinglePositionRule(),
        TradeFrequencyRule(journal, cfg.max_trades_per_symbol_hour, cfg.max_trades_per_day),
        CooldownRule(journal, cfg.symbol_cooldown_min, cfg.global_pause_losses, cfg.global_pause_min),
        DailyTargetRule(cfg.daily_profit_target_pct),
        # Stage 2 — market conditions
        sessions,
        DataFreshnessRule(cfg.max_data_age_bars),
        SpreadRule(cfg.max_spread_atr_ratio, cfg.max_spread_points),
        VolatilityRule(cfg.min_atr_cost_ratio, cfg.max_atr_median_ratio, cfg.atr_median_window),
        EventWindowRule(calendar_provider, cfg.event_window_min),
        # Stages 3–5 — signal, ML, order
        ConfidenceRule(cfg.min_confidence, cfg.neutral_min_confidence),
        MLFilterRule(cfg.ml_min_win_proba, cfg.ml_unreliable_size_mult),
        RewardRiskRule(cfg.min_net_reward_risk, cfg.min_tp_cost_ratio),
    ]
    position_rules = [
        TimeStopRule(cfg.max_holding_bars),
        SessionCloseRule(sessions.windows_for, cfg.session_close_buffer_min),
        BreakevenRule(cfg.breakeven_atr),
        AtrTrailingRule(cfg.trailing_activation_atr, cfg.trailing_step_atr),
    ]
    return RuleEngine(entry_rules, position_rules, rules_version=cfg.rules_version)
