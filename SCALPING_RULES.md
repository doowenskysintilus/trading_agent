# Règles de trading — Scalping ML + RL

> **Statut : contrat obligatoire.** Toute décision d'ouverture, de gestion ou de
> fermeture de position en mode scalping DOIT respecter ces règles. Une règle ne
> peut être désactivée que par configuration (`.env`), jamais en dur dans le code,
> et chaque refus doit être tracé en base MySQL avec l'identifiant de la règle.
>
> Version des règles : **SR-1.0** (2026-10-23) — à stocker avec chaque trade.

Légende du statut d'implémentation (vérifié dans le code au 2026-10-23) :

- ✅ **Existe** : déjà implémenté, à réutiliser tel quel
- 🔧 **À adapter** : existe mais valeur ou comportement à changer pour le scalping
- 🆕 **À créer** : n'existe pas encore

Les valeurs ci-dessous sont des **valeurs de départ**. Elles doivent être validées
par backtest walk-forward (section 8) avant tout passage en réel.

---

## 1. Périmètre

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| SC-01 | Unité de temps d'exécution | M1 (entrée), M5 autorisé | ✅ `TRADING_TIMEFRAME` |
| SC-02 | Filtre de tendance sur une unité de temps supérieure | M15 (auto = +2 crans depuis M1) | ✅ `htf_enabled` (`live_trader.py`) |
| SC-03 | Symboles autorisés | EURUSD, XAUUSD, BTCUSD (liste blanche) | ✅ `TRADING_SYMBOLS` |
| SC-04 | Une seule position ouverte par symbole | 1 | 🆕 |
| SC-05 | Pas de martingale, pas de moyenne à la baisse, pas de hedge sur le même symbole | interdit | 🆕 (garde-fou explicite) |

## 2. Conditions de marché (avant tout signal)

Si une de ces règles échoue, **aucun signal n'est calculé** pour ce symbole sur ce cycle.

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| MK-01 | Sessions autorisées (heure UTC) — Forex/Or | 07:00–11:00 et 12:30–16:30 (Londres + chevauchement NY) | 🆕 |
| MK-02 | Sessions autorisées — Crypto | 24/7 sauf 21:00–23:59 UTC (liquidité faible) | 🆕 |
| MK-03 | Pas d'entrée le vendredi après 19:00 UTC ni dans les 30 min après l'ouverture du dimanche | — | 🆕 |
| MK-04 | Spread max | spread ≤ 0,25 × ATR(14, M1) **et** ≤ plafond fixe par symbole (`EURUSD` 1,0 pip, `XAUUSD` 30 pts, `BTCUSD` 2000 pts) | 🆕 (`spread` déjà lu dans `data_feed.py`) |
| MK-05 | Volatilité minimale | ATR(14, M1) ≥ 2 × coût aller-retour (spread + commission) | 🆕 |
| MK-06 | Volatilité maximale | ATR(14, M1) ≤ 3 × médiane ATR des 500 dernières bougies | 🆕 |
| MK-07 | Blackout événement économique HIGH | ±15 min **avant et après** l'événement | 🔧 `RiskEngine` Layer 0 ne bloque qu'**avant** (30 min) → ajouter l'après-événement |
| MK-08 | Données fraîches | dernière bougie fermée ≤ 2 × durée de l'unité de temps, sinon pas de trade | 🆕 |
| MK-09 | Signaux calculés uniquement sur bougies **fermées** (jamais la bougie en cours) | obligatoire | ✅ à vérifier dans `_test_bar_dedup.py` |

## 3. Génération du signal (ML + RL)

### 3.1 Agrégation

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| SG-01 | Le signal final vient du `SignalAggregator` (momentum, mean reversion, RL, calendrier) | — | ✅ |
| SG-02 | Confiance agrégée minimale | ≥ 0,55 (scalping, plus strict que 0,45 actuel) | 🔧 `min_confidence` |
| SG-03 | Pondération par régime (Efficiency Ratio) : tendance → momentum ×1,5 ; range → mean reversion ×1,5 | — | ✅ `_apply_regime_weighting` |
| SG-04 | Régime « neutre » (0,25 < ER < 0,5) : confiance minimale relevée à 0,65 | — | 🆕 |
| SG-05 | Le signal doit aller dans le sens de la tendance de l'unité de temps supérieure (SC-02) | obligatoire | ✅ |

### 3.2 Filtre ML (`WinClassifier`)

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| ML-01 | P(gain) prédite minimale pour entrer | ≥ 0,55 | 🔧 `TRADING_ML_MIN_WIN_PROBA` (0,50 actuel) |
| ML-02 | Le label « gain » = PnL **net de spread et commission** > 0 | obligatoire | 🔧 à vérifier dans `research/training/dataset.py` |
| ML-03 | Le filtre n'est actif que si le modèle est marqué `reliable` ; sinon, mode **demi-taille** (pas « tout passe ») | demi-taille | 🔧 actuellement le filtre laisse tout passer sans modèle |
| ML-04 | Minimum d'échantillons pour entraîner | 300 trades scalping (60 actuel) | 🔧 |
| ML-05 | Découpage train/test chronologique, jamais aléatoire | 75 / 25 | ✅ |
| ML-06 | Promotion : nouveau modèle déployé seulement si AUC ≥ AUC déployé − 0,02 sur le même holdout | — | ✅ `win_classifier.py` |
| ML-07 | AUC minimale absolue pour être `reliable` | ≥ 0,55 | 🆕 |
| ML-08 | Les features d'entrée en live doivent avoir exactement le même schéma que l'entraînement ; sinon le trade est refusé | refus | ✅ (warning + `None`) → 🔧 transformer en refus explicite tracé |

### 3.3 Agent RL (`RLAlpha`, RecurrentPPO-LSTM)

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| RL-01 | L'agent RL est **une voix** dans l'agrégateur ; il ne peut jamais contourner le filtre ML, le `RiskEngine` ni ces règles | obligatoire | ✅ (architecture actuelle) |
| RL-02 | L'agent RL ne décide **pas** de la taille de position ; seulement la direction (BUY / SELL / HOLD) | obligatoire | ✅ |
| RL-03 | Entraînement sur la même unité de temps que l'exécution scalping (M1 ou M5) | M1/M5 | 🔧 `TRADING_RL_TIMEFRAME=H1` actuellement → modèle dédié scalping |
| RL-04 | L'environnement d'entraînement utilise le spread, la commission et le slippage **mesurés en live** (moyenne glissante en base) | — | 🔧 spread fixe 0,2 pip dans `rl_trainer.py` |
| RL-05 | Récompense : pénalité si la position dépasse la durée max (EX-04) | α = 0,05 / bougie au-delà | 🆕 |
| RL-06 | Récompense : bonus `entry_quality`, pénalités drawdown/risque déjà présents conservés | — | ✅ |
| RL-07 | Promotion : modèle candidat évalué contre le modèle déployé, rejeté en cas de régression nette ; sauvegarde du modèle précédent | — | ✅ `_should_promote_rl` |
| RL-08 | Rechargement à chaud du modèle après promotion | — | ✅ `maybe_reload()` |
| RL-09 | Si le modèle RL est absent ou en erreur → poids RL = 0 (les autres stratégies continuent) | — | 🆕 garde-fou explicite |

## 4. Taille de position

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| SZ-01 | Risque par trade (distance SL × taille) | 0,25 % de l'equity (1 % actuel trop élevé pour le scalping) | 🔧 `risk_per_trade_pct` |
| SZ-02 | Multiplicateur de conviction : 0,5× (confiance minimale) → 1,0× (confiance = 1) | — | ✅ `_build_trade_order` |
| SZ-03 | Réduction après pertes consécutives : 1 − 0,15 × N, plancher 0,4× | — | ✅ |
| SZ-04 | Mode demi-taille si ML non fiable (ML-03) | 0,5× | 🆕 |
| SZ-05 | Le `RiskEngine` (Layers 3–6 : exposition, volatilité, corrélation, levier) a toujours le dernier mot sur la taille | obligatoire | ✅ |
| SZ-06 | Taille arrondie au pas de volume du symbole (`volume_step`), refus si < `volume_min` | — | 🔧 à vérifier dans `execution_engine.py` |

## 5. Entrée et sortie

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| EX-01 | Stop loss obligatoire, envoyé **avec** l'ordre (jamais après) | SL = 1,2 × ATR(14, M1) | 🔧 `TRADING_SL_ATR_MULTIPLIER` (2,0 actuel) |
| EX-02 | Take profit | TP = 1,5 × ATR(14, M1) | 🔧 `TRADING_TP_ATR_MULTIPLIER` (4,0 actuel) |
| EX-03 | Rendement net minimal : (TP − coûts) / (SL + coûts) ≥ 1,0, et TP ≥ 3 × coût aller-retour | sinon refus | 🆕 |
| EX-04 | Durée max d'une position (time stop) | 15 bougies M1 → fermeture au marché | 🆕 |
| EX-05 | Breakeven : SL déplacé à entrée + coûts quand le gain latent atteint 0,8 × ATR | — | 🆕 |
| EX-06 | Trailing stop basé sur l'ATR (pas sur un % du prix) : activation à 1,0 × ATR, pas de 0,5 × ATR | — | 🔧 `TrailingStopManager` utilise des % fixes |
| EX-07 | Slippage max à l'exécution | ≤ 1 × spread courant (0,10 % du prix actuel trop large) | 🔧 `max_slippage_pct` |
| EX-08 | Fermeture forcée de toutes les positions scalping 10 min avant la fin de session autorisée et avant le week-end | — | 🆕 |
| EX-09 | Une sortie par signal opposé est autorisée seulement si la confiance opposée ≥ SG-02 | — | 🆕 |

## 6. Limites de fréquence et protection du capital

| ID | Règle | Valeur par défaut | Statut |
|---|---|---|---|
| FQ-01 | Max trades par symbole et par heure | 4 | 🆕 |
| FQ-02 | Max trades par jour (tous symboles) | 30 | 🆕 |
| FQ-03 | Pause après une perte sur un symbole (cooldown) | 5 min | 🆕 |
| FQ-04 | Pause globale après 3 pertes consécutives | 30 min | 🆕 |
| FQ-05 | Kill-switch : N pertes consécutives | 6 → arrêt jusqu'à reset manuel | ✅ `max_consecutive_losses` |
| FQ-06 | Perte journalière max | 2 % (scalping) ; FTMO : arrêt à 4,5 % | 🔧 `max_daily_loss_pct` (3 %) / ✅ FTMO |
| FQ-07 | Drawdown max | FTMO : arrêt à 9 % du solde initial ; générique : 15 % | ✅ |
| FQ-08 | Objectif journalier atteint (+1,5 %) → arrêt des nouvelles entrées pour la journée | — | 🆕 (aide aussi la règle de consistance FTMO) |
| FQ-09 | Compte FTMO détecté → `ftmo_enabled=True` obligatoire | — | 🔧 `False` par défaut |
| FQ-10 | Arrêt d'urgence (`EMERGENCY_STOP`) prioritaire sur toute règle | — | ✅ |

## 7. Ordre d'évaluation obligatoire

La première règle qui échoue arrête l'évaluation (fail-fast) et produit un refus tracé.

```
0. Arrêt d'urgence / kill-switch / limites FTMO          (FQ-05..07, FQ-09, FQ-10)
1. Limites de fréquence et cooldowns                      (FQ-01..04, FQ-08)
2. Conditions de marché : session, spread, volatilité,    (MK-01..09)
   événement, fraîcheur des données
3. Signal agrégé + régime + tendance supérieure           (SG-01..05, RL-01..09)
4. Filtre ML P(gain)                                      (ML-01..08)
5. Construction de l'ordre : taille, SL, TP, R:R net      (SZ-01..04, EX-01..03)
6. RiskEngine (Layers 0–6)                                (SZ-05)
7. Exécution MT5 + contrôle slippage                      (SZ-06, EX-07)
8. Gestion de position : breakeven, trailing, time stop,  (EX-04..06, EX-08..09)
   fermeture de session
```

## 8. Validation avant passage en réel

Une configuration de règles (version SR-x.y) ne peut passer du compte démo au réel que si :

| ID | Critère | Seuil |
|---|---|---|
| VA-01 | Backtest avec spread et commission réels | ✅ `backtest_engine.py` (spread fixe → 🔧 spread historique) |
| VA-02 | Walk-forward | ≥ 4 folds, rentable sur ≥ 3 folds — ✅ `backtesting/walk_forward.py` |
| VA-03 | Nombre de trades out-of-sample | ≥ 300 |
| VA-04 | Profit factor net | ≥ 1,3 |
| VA-05 | Drawdown max out-of-sample | ≤ 5 % |
| VA-06 | Période en démo (compte FTMO-Demo ou Binance Demo) | ≥ 2 semaines, résultats cohérents avec le backtest (écart de win rate ≤ 10 points) |

## 9. Traçabilité MySQL

| ID | Règle | Statut |
|---|---|---|
| DB-01 | Chaque décision (acceptée **ou** refusée) est enregistrée avec : horodatage, symbole, ID de la règle qui a refusé, valeurs mesurées (spread, ATR, confiance, P(gain), action RL, régime), version des règles | 🆕 table `scalping_decisions` |
| DB-02 | Table `trades` enrichie : `spread_at_entry`, `slippage`, `ml_win_proba`, `rl_action`, `regime`, `rules_version`, `exit_reason` (TP / SL / time_stop / breakeven / trailing / session_close / signal) | 🔧 `monitoring/metrics_store.py` |
| DB-03 | Les compteurs de fréquence et cooldowns (FQ-01..04) sont reconstruits depuis MySQL au redémarrage (pas seulement en mémoire) | 🆕 |
| DB-04 | Si MySQL est indisponible : repli sur les logs JSONL existants, **et** le trading continue uniquement si le repli fonctionne | ✅ repli JSONL existant → 🔧 condition de blocage |

## 10. Contraintes d'architecture (SOLID)

- **S** — une classe par famille de règles (`SessionRule`, `SpreadRule`, `CooldownRule`, `MLFilterRule`…), chacune avec une seule responsabilité.
- **O** — ajouter une règle = ajouter une classe ; le moteur de règles n'est jamais modifié.
- **L** — toutes les règles implémentent la même interface `evaluate(context) -> RuleResult` (`passed`, `rule_id`, `reason`, `size_multiplier`) et sont interchangeables.
- **I** — interfaces séparées pour les règles d'entrée (`EntryRule`) et de gestion de position (`PositionRule`).
- **D** — le moteur dépend d'abstractions : fournisseurs de prix, de spread, d'historique de trades (MySQL) et de modèles injectés, pour pouvoir tester sans MT5 ni MySQL.
- Les règles **réutilisent** `RiskEngine`, `WinClassifier`, `RLAlpha`, `SignalAggregator`, `MetricsStore` et `config/settings.py` ; elles ne dupliquent aucune logique existante.
- Toutes les valeurs de ce document sont lues depuis `config/settings.py` (nouveau groupe `ScalpingSettings`, préfixe `.env` `SCALP_`).
- API : le projet utilise **FastAPI** (`api/main.py`), pas Flask. Les endpoints de lecture et de modification des règles seront ajoutés dans FastAPI, sauf décision contraire.
