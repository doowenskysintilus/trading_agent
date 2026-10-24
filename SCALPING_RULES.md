# Règles de trading — Scalping ML + RL

> **Statut : contrat obligatoire.** Toute décision d'ouverture, de gestion ou de
> fermeture de position en mode scalping DOIT respecter ces règles. Une règle ne
> peut être désactivée que par configuration (`.env`), jamais en dur dans le code,
> et chaque refus doit être tracé en base MySQL avec l'identifiant de la règle.
>
> Version des règles : **SR-1.0** (2026-10-23) — stockée avec chaque décision et chaque trade.

**Activation :** `SCALP_ENABLED=true` dans `.env` (ou `scalping_enabled: true` dans
`POST /trading/start`). Désactivé, le bot se comporte exactement comme avant.
Chaque valeur ci-dessous a sa variable `SCALP_*` dans [config/settings.py](config/settings.py)
(`ScalpingSettings`).

Toutes les règles sont **implémentées** (2026-10-23) et testées dans
[_test_scalping_rules.py](_test_scalping_rules.py). La colonne « Code » indique où.

Les valeurs ci-dessous sont des **valeurs de départ**. Elles doivent être validées
par backtest walk-forward (section 8) avant tout passage en réel.

---

## 1. Périmètre

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| SC-01 | Unité de temps d'exécution | M1 (entrée), M5 autorisé | `TRADING_TIMEFRAME` |
| SC-02 | Filtre de tendance sur une unité de temps supérieure | M15 (auto = +2 crans depuis M1) | `LiveTrader._htf_trend` (refus journalisé `SG-05`) |
| SC-03 | Symboles autorisés | EURUSD, XAUUSD, BTCUSD | `TRADING_SYMBOLS` |
| SC-04 | Une seule position ouverte par symbole | 1 | `SinglePositionRule` |
| SC-05 | Pas de martingale, pas de moyenne à la baisse, pas de hedge sur le même symbole | interdit | garanti par SC-04 + la taille ne fait que diminuer après une perte (SZ-03) |

## 2. Conditions de marché (avant tout signal)

Si une de ces règles échoue, **aucun signal n'est calculé** pour ce symbole sur ce cycle.

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| MK-01 | Sessions autorisées (heure UTC) — Forex/Or | 07:00–11:00 et 12:30–16:30 | `SessionRule` — `SCALP_FX_SESSIONS` |
| MK-02 | Sessions autorisées — Crypto | 00:00–21:00 UTC | `SessionRule` — `SCALP_CRYPTO_SESSIONS`, `SCALP_CRYPTO_SYMBOLS` |
| MK-03 | Pas d'entrée Forex le week-end, le vendredi après 19:00 UTC, ni dans les 30 min après l'ouverture du dimanche | — | `SessionRule` |
| MK-04 | Spread max | ≤ 0,25 × ATR(14) **et** ≤ plafond par symbole (`EURUSD` 10 pts, `XAUUSD` 30 pts, `BTCUSD` 2000 pts) | `SpreadRule` — `SCALP_MAX_SPREAD_POINTS` |
| MK-05 | Volatilité minimale | ATR ≥ 2 × coût aller-retour (spread + commission) | `VolatilityRule` |
| MK-05b | Commission = **mesurée par symbole** sur les derniers trades fermés du compte (recalcul horaire) ; `SCALP_COMMISSION_PCT` seulement pour un symbole sans historique (< 3 trades) | ex. FTMO : BTCUSD 0,0326 %/côté, EURUSD 0,0026 %/côté | `CommissionModel` (`engines/rule_engine/costs.py`), `GET /account/commissions` |
| MK-06 | Volatilité maximale | ATR ≤ 3 × médiane ATR des 500 dernières bougies | `VolatilityRule` |
| MK-07 | Blackout événement économique HIGH | ±15 min **avant et après** | `EventWindowRule` + `CalendarProvider.event_is_active` (corrigé : voyait seulement le futur) |
| MK-08 | Données fraîches | dernière bougie fermée ≤ 2 bougies d'âge (horloge du broker) | `DataFreshnessRule` |
| MK-09 | Signaux calculés uniquement sur bougies **fermées** | obligatoire | `LiveTraderConfig.closed_bars_only` (forcé par le profil scalping) |

## 3. Génération du signal (ML + RL)

### 3.1 Agrégation

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| SG-01 | Le signal final vient du `SignalAggregator` | — | existant |
| SG-02 | Confiance agrégée minimale | ≥ 0,55 | `ConfidenceRule` |
| SG-03 | Pondération par régime (Efficiency Ratio) | tendance → momentum ×1,5 ; range → mean reversion ×1,5 | existant (`_apply_regime_weighting`) |
| SG-04 | Régime « neutre » : confiance minimale relevée | ≥ 0,65 | `ConfidenceRule` |
| SG-05 | Le signal doit aller dans le sens de la tendance supérieure | obligatoire | existant, refus maintenant journalisé |

### 3.2 Filtre ML (`WinClassifier`)

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| ML-01 | P(gain) minimale pour entrer | ≥ 0,55 | `MLFilterRule` |
| ML-02 | Label « gain » = PnL **net** (spread, commission, swap) > 0 | obligatoire | existant : `get_closed_pnl` additionne profit + swap + commission |
| ML-03 | Modèle absent ou non fiable → **demi-taille** (plus « tout passe ») | ×0,5 | `MLFilterRule` + `WinClassifier.predict()` |
| ML-04 | Minimum d'échantillons pour entraîner | 300 trades | `RetrainService` (`SCALP_ML_MIN_SAMPLES`) |
| ML-05 | Découpage train/test chronologique | 75 / 25 | existant |
| ML-06 | Promotion seulement si AUC ≥ AUC déployé − 0,02 | — | existant |
| ML-07 | AUC minimale pour être `reliable` | ≥ 0,55 | `WinClassifier.train(min_reliable_auc=…)` |
| ML-08 | Schéma de features différent → trade **refusé** et tracé | refus | `MLFilterRule` (statut `feature_mismatch`) |

### 3.3 Agent RL (`RLAlpha`, RecurrentPPO-LSTM)

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| RL-01 | L'agent RL est **une voix** dans l'agrégateur ; il ne contourne ni le ML, ni le `RiskEngine`, ni ces règles | obligatoire | architecture existante |
| RL-02 | L'agent RL donne la direction, jamais la taille | obligatoire | architecture existante |
| RL-03 | Entraînement sur l'unité de temps du scalping | M1 | `_rl_timeframe()` dans `api/main.py` (`SCALP_RL_TIMEFRAME`) |
| RL-04 | Environnement d'entraînement avec le spread **mesuré** sur l'historique du broker | médiane du spread historique | `_make_rl_feature_provider` → `RetrainService._retrain_rl` |
| RL-05 | Pénalité si la position dépasse la durée max (EX-04) | 0,05 / bougie au-delà | `RewardConfig.max_holding_bars` / `alpha_overhold` |
| RL-06 | Bonus d'entrée, pénalités drawdown/risque conservés | — | existant |
| RL-07 | Promotion contre le modèle déployé, sauvegarde du précédent | — | existant |
| RL-08 | Rechargement à chaud après promotion | — | existant |
| RL-09 | Modèle RL absent ou en erreur → poids nul | — | existant : `RLAlpha` renvoie HOLD (confiance 0), non compté par l'agrégateur |

## 4. Taille de position

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| SZ-01 | Risque par trade | 0,25 % de l'equity | profil scalping → `risk_per_trade_pct` |
| SZ-02 | Multiplicateur de conviction 0,5× → 1,0× | — | existant |
| SZ-03 | Réduction après pertes : 1 − 0,15 × N, plancher 0,4× | — | existant |
| SZ-04 | Mode demi-taille si ML non fiable | ×0,5 | `MLFilterRule` |
| SZ-05 | Le `RiskEngine` (couches 3 à 6) a le dernier mot | obligatoire | existant, refus journalisé `RISK_BLOCKED` |
| SZ-06 | Taille sous le `volume_min` du broker → **refus** (plus d'arrondi vers le haut) | — | `ExecutionConfig.reject_below_min_volume` |

## 5. Entrée et sortie

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| EX-01 | Stop loss obligatoire, envoyé avec l'ordre, recalé sur le prix exécutable | SL = 1,2 × ATR(14) | profil scalping + `ScalpingController.rebase_order` |
| EX-02 | Take profit | TP = 1,5 × ATR(14) | profil scalping |
| EX-03 | (TP − coûts) / (SL + coûts) ≥ 1,0 et TP ≥ 3 × coût aller-retour | sinon refus | `RewardRiskRule` |
| EX-04 | Durée max d'une position | 15 bougies → fermeture | `TimeStopRule` |
| EX-05 | Breakeven : SL à entrée + coûts à +0,8 × ATR | — | `BreakevenRule` |
| EX-06 | Trailing basé sur l'ATR : activation à 1,0 × ATR, distance 0,5 × ATR | — | `AtrTrailingRule` (remplace l'ancien trailing en %, inactif en live) |
| EX-07 | Slippage max | ≤ 1 × spread courant | `ExecutionConfig.deviation_spread_mult` |
| EX-08 | Fermeture 10 min avant la fin de session et hors session | — | `SessionCloseRule` |
| EX-09 | Sortie sur signal opposé seulement si confiance ≥ SG-02 | — | `LiveTrader._scalp_exit_on_opposite_signal` |

## 6. Limites de fréquence et protection du capital

| ID | Règle | Valeur par défaut | Code |
|---|---|---|---|
| FQ-01 | Max trades par symbole et par heure | 4 | `TradeFrequencyRule` |
| FQ-02 | Max trades par jour (tous symboles) | 30 | `TradeFrequencyRule` |
| FQ-03 | Pause après une perte sur un symbole | 5 min | `CooldownRule` |
| FQ-04 | Pause globale après 3 pertes consécutives | 30 min | `CooldownRule` |
| FQ-05 | Kill-switch : 6 pertes consécutives → arrêt **jusqu'à reset manuel** | 6 | `RiskEngine` + arrêt d'urgence verrouillé dans `_reconcile_closed_trades` |
| FQ-06 | Perte journalière max | 2 % ; FTMO : 4,5 % | profil scalping → `max_daily_loss_pct` ; FTMO existant |
| FQ-07 | Drawdown max | FTMO 9 % du solde initial ; générique 15 % | existant (`RiskHaltRule` l'évalue dès l'étape 0) |
| FQ-08 | Objectif journalier atteint (+1,5 %) → plus de nouvelles entrées | — | `DailyTargetRule` |
| FQ-09 | Serveur FTMO détecté → limites FTMO activées | — | `LiveTrader._maybe_enable_ftmo` |
| FQ-10 | Arrêt d'urgence prioritaire sur toute règle | — | existant |

## 7. Ordre d'évaluation obligatoire

La première règle qui échoue arrête l'évaluation (fail-fast) et produit un refus tracé.
Implémenté par `RuleEngine.evaluate_entry` (tri par `Stage`).

```
0. HALT      kill-switch / FTMO / drawdown, journal disponible   (FQ-05..07, FQ-09, DB-04)
1. FREQUENCY une position par symbole, fréquence, pauses, objectif (SC-04, FQ-01..04, FQ-08)
2. MARKET    session, fraîcheur, spread, volatilité, événement     (MK-01..08)
   → signaux, régime, tendance supérieure                          (SG-01, SG-03, SG-05)
3. SIGNAL    confiance selon le régime                             (SG-02, SG-04)
4. ML        P(gain)                                               (ML-01, ML-03, ML-08)
5. ORDER     taille, prix exécutable, R:R net                      (SZ-01..04, EX-01..03)
6. RiskEngine couches 0–6                                          (SZ-05)
7. Exécution MT5 + déviation liée au spread                        (SZ-06, EX-07)
8. Gestion de position à chaque cycle                              (EX-04..06, EX-08, EX-09)
```

## 8. Validation avant passage en réel

Une version des règles ne passe du démo au réel que si
`backtesting.validation.check_live_readiness(walk_forward_result).ready` est vrai.

| ID | Critère | Seuil | Code |
|---|---|---|---|
| VA-01 | Backtest avec spread historique de chaque bougie | `BacktestConfig.spread_point` | `backtest_engine.py` |
| VA-02 | Walk-forward | ≥ 4 folds, rentable sur ≥ 3 | `check_live_readiness` |
| VA-03 | Trades out-of-sample | ≥ 300 | `check_live_readiness` |
| VA-04 | Profit factor net | ≥ 1,3 | `check_live_readiness` |
| VA-05 | Drawdown max out-of-sample | ≤ 5 % | `check_live_readiness` |
| VA-06 | ≥ 2 semaines en démo, écart de win rate ≤ 10 points avec le backtest | — | `demo_matches_backtest` |

## 9. Traçabilité MySQL

| ID | Règle | Code |
|---|---|---|
| DB-01 | Chaque décision (exécutée, refusée, fermeture, déplacement de SL) est enregistrée avec l'ID de la règle, les valeurs mesurées et la version des règles. Un refus répété par la même règle sur des bougies consécutives n'est écrit qu'une fois. | table `scalping_decisions`, `GET /scalping/decisions` |
| DB-02 | Table `trades` enrichie : `spread_at_entry`, `slippage`, `ml_win_proba`, `rl_action`, `regime`, `rules_version`, `exit_reason` précis (`time_stop`, `session_close`, `breakeven`, `trailing`, `signal`…) | migration automatique au démarrage (`MetricsStore._run_ddl`) |
| DB-03 | Compteurs et pauses reconstruits depuis la base au redémarrage | `ScalpingController.restore_once` |
| DB-04 | MySQL indisponible → repli JSONL ; si le repli échoue aussi, plus aucune entrée | `PersistenceRule` + `MetricsStore.persistence_ok` |

## 10. Contraintes d'architecture (SOLID)

Code dans [engines/rule_engine/](engines/rule_engine/) et [live_trading/scalping.py](live_trading/scalping.py).

- **S** — une classe par famille de règles (`SessionRule`, `SpreadRule`, `CooldownRule`, `MLFilterRule`…).
- **O** — ajouter une règle = ajouter une classe dans `factory.py` ; `RuleEngine` ne change pas.
- **L** — toutes les règles d'entrée implémentent `evaluate(EntryContext) -> RuleResult`.
- **I** — interfaces séparées `EntryRule` / `PositionRule`.
- **D** — les règles reçoivent leurs données (contexte, journal, fonctions injectées) : testables sans MT5 ni MySQL.
- Réutilisation de `RiskEngine` (nouvelle méthode `check_halt`), `WinClassifier`, `RLAlpha`, `SignalAggregator`, `MetricsStore`, `config/settings.py`.
- API : **FastAPI** (`GET /scalping/rules`, `GET /scalping/decisions`).
