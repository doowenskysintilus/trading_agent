"""
RetrainService
==============
Manual, on-demand retraining of the learning models from the system's own
collected results.

Two models, run in parallel with the rule-based strategies:

  * **WinClassifier** (sklearn) — learns P(win) from the entry features of
    every closed trade. Trains directly on the collected experiences.
  * **RecurrentPPO** (sb3-contrib) — the RL agent, retrained on a continuous
    feature series via the existing :class:`RLTrainer`. (RecurrentPPO is the
    practical, Windows-friendly alternative to Ray/IMPALA.)

Retraining runs in a background thread so the API request returns instantly;
progress/results are exposed via :meth:`status`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from research.training.dataset import build_supervised_dataset, load_experiences
from strategies.ml_agent.win_classifier import WinClassifier

logger = logging.getLogger(__name__)


@dataclass
class RetrainStatus:
    state:     str = "idle"          # idle | running | done | error
    started:   Optional[str] = None
    finished:  Optional[str] = None
    ml:        dict = field(default_factory=dict)
    rl:        dict = field(default_factory=dict)
    message:   str = ""
    continuous: bool = False
    run_count: int = 0
    rl_timesteps_done: int = 0
    rl_timesteps_total: int = 0

    def as_dict(self) -> dict:
        return {
            "state":    self.state,
            "started":  self.started,
            "finished": self.finished,
            "ml":       self.ml,
            "rl":       self.rl,
            "message":  self.message,
            "continuous": self.continuous,
            "run_count": self.run_count,
            "rl_timesteps_done": self.rl_timesteps_done,
            "rl_timesteps_total": self.rl_timesteps_total,
        }


class RetrainService:
    """Coordinates background retraining of the ML and RL models."""

    def __init__(
        self,
        dataset_dir: str = "data/storage/datasets",
        model_dir:   str = "data/storage/models",
    ) -> None:
        self.dataset_dir = dataset_dir
        self.model_dir   = model_dir
        self._status     = RetrainStatus()
        self._lock       = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return self._status.as_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._status.state == "running"

    # ------------------------------------------------------------------

    def start(
        self,
        train_ml: bool = True,
        train_rl: bool = False,
        rl_timesteps: int = 50_000,
        rl_continuous: bool = False,
        rl_interval_s: int = 1800,
        rl_symbol: Optional[str] = None,
        rl_timeframe: Optional[str] = None,
        rl_feature_provider: Optional[Callable[[], pd.DataFrame]] = None,
    ) -> dict:
        """Kick off a background retraining run. Returns the initial status.

        Raises RuntimeError if a run is already in progress.
        """
        with self._lock:
            if self._status.state == "running":
                raise RuntimeError("A retraining run is already in progress.")
            self._stop_event.clear()
            self._status = RetrainStatus(
                state="running",
                started=datetime.now(timezone.utc).isoformat(),
                message="Retraining started.",
                continuous=bool(rl_continuous),
                run_count=0,
            )

        self._thread = threading.Thread(
            target=self._run,
            kwargs=dict(
                train_ml=train_ml, train_rl=train_rl,
                rl_timesteps=rl_timesteps, rl_symbol=rl_symbol,
                rl_continuous=rl_continuous, rl_interval_s=rl_interval_s,
                rl_timeframe=rl_timeframe, rl_feature_provider=rl_feature_provider,
            ),
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def stop(self) -> dict:
        """Request graceful stop for a running retraining loop."""
        with self._lock:
            if self._status.state != "running":
                return self._status.as_dict()
            self._status.message = "Stop requested. Finishing current run..."
        self._stop_event.set()
        return self.status()

    # ------------------------------------------------------------------

    def _run(
        self,
        train_ml: bool,
        train_rl: bool,
        rl_timesteps: int,
        rl_continuous: bool,
        rl_interval_s: int,
        rl_symbol: Optional[str],
        rl_timeframe: Optional[str],
        rl_feature_provider: Optional[Callable[[], pd.DataFrame]],
    ) -> None:
        ml_result: dict = {}
        rl_result: dict = {}
        runs = 0
        try:
            if train_ml:
                print(
                    f"\n{'▶'*3} ML RETRAIN — WinClassifier (données : {self.dataset_dir}) {'▶'*3}",
                    flush=True,
                )
                ml_result = self._retrain_ml()
            if train_rl:
                while not self._stop_event.is_set():
                    runs += 1
                    print(
                        f"\n{'▶'*3} RL RETRAIN — Run #{runs} | "
                        f"{rl_timesteps:,} steps | "
                        f"{'continu' if rl_continuous else 'one-shot'} {'▶'*3}",
                        flush=True,
                    )
                    # Signal "en cours" AVANT le run (peut durer plusieurs minutes)
                    with self._lock:
                        self._status.run_count = runs
                        self._status.rl_timesteps_done = 0
                        self._status.rl_timesteps_total = rl_timesteps
                        self._status.message = f"RL run {runs} en cours…"
                    rl_result = self._retrain_rl(
                        rl_timesteps, rl_symbol, rl_timeframe, rl_feature_provider,
                        progress_callback=self._on_rl_progress,
                    )
                    with self._lock:
                        self._status.rl = rl_result
                        self._status.run_count = runs
                        if rl_result.get("trained", False):
                            self._status.message = f"RL run {runs} complete."
                        else:
                            self._status.message = (
                                f"RL run {runs} failed: {rl_result.get('message', 'unknown error')}"
                            )

                    # Stop if this is a one-shot run.
                    if not rl_continuous:
                        break

                    # Wait until the next loop or a stop request.
                    if self._stop_event.wait(timeout=max(60, int(rl_interval_s))):
                        break

            with self._lock:
                self._status.state    = "done"
                self._status.finished = datetime.now(timezone.utc).isoformat()
                self._status.ml       = ml_result
                self._status.rl       = rl_result
                self._status.run_count = runs
                if self._stop_event.is_set() and train_rl:
                    self._status.message = f"Retraining stopped by user after {runs} RL run(s)."
                else:
                    self._status.message = "Retraining complete."
        except Exception as exc:
            logger.exception("Retraining failed")
            with self._lock:
                self._status.state    = "error"
                self._status.finished = datetime.now(timezone.utc).isoformat()
                self._status.ml       = ml_result
                self._status.rl       = rl_result
                self._status.run_count = runs
                self._status.message  = f"{type(exc).__name__}: {exc}"

    def _on_rl_progress(self, timesteps_done: int) -> None:
        """Called periodically during RL training to update timestep counter."""
        with self._lock:
            self._status.rl_timesteps_done = timesteps_done
            total = self._status.rl_timesteps_total or 1
            run = self._status.run_count
            pct = int(100 * timesteps_done / total)
            self._status.message = (
                f"RL run {run} en cours… {timesteps_done:,}/{total:,} steps ({pct}%)"
            )

    # ------------------------------------------------------------------
    # ML — win/loss classifier
    # ------------------------------------------------------------------

    def _retrain_ml(self) -> dict:
        X, y, names = build_supervised_dataset(self.dataset_dir)
        clf    = WinClassifier(model_path=f"{self.model_dir}/win_classifier.joblib")
        report = clf.train(X, y, names)
        result = report.as_dict()
        self._print_ml_report(result)
        return result

    @staticmethod
    def _print_ml_report(r: dict) -> None:
        """Print a formatted ML training report to the terminal."""
        line = "─" * 58
        if not r.get("trained"):
            print(
                f"\n{line}\n"
                f"  ML WIN CLASSIFIER — NON ENTRAÎNÉ\n"
                f"{line}\n"
                f"  Raison : {r.get('message', '?')}\n"
                f"{line}\n",
                flush=True,
            )
            return

        n        = r.get("n_samples", 0)
        n_wins   = r.get("n_wins",    0)
        n_losses = r.get("n_losses",  0)
        acc      = r.get("accuracy",  0.0) * 100
        auc      = r.get("auc",       0.0)
        reliable = r.get("reliable",  True)
        warning  = r.get("warning",   "")

        if acc >= 70:
            acc_sym = "✓ BON"
        elif acc >= 60:
            acc_sym = "~ ACCEPTABLE"
        else:
            acc_sym = "✗ FAIBLE"

        if auc >= 0.75:
            auc_sym = "✓ BON"
        elif auc >= 0.60:
            auc_sym = "~ ACCEPTABLE"
        else:
            auc_sym = "✗ FAIBLE"

        win_rate = (n_wins / n * 100) if n > 0 else 0.0
        fiab = "✓ FIABLE" if reliable else "⚠ DONNÉES INSUFFISANTES"

        print(
            f"\n{'═'*58}\n"
            f"  ML WIN CLASSIFIER — RÉSULTATS D'ENTRAÎNEMENT\n"
            f"{'═'*58}\n"
            f"  Trades analysés     : {n:>6,}\n"
            f"  Wins / Losses       : {n_wins:>4,} W  /  {n_losses:>4,} L  "
            f"(win rate réel : {win_rate:.1f}%)\n"
            f"  Features utilisées  : {r.get('n_features', 0):>6,}\n"
            f"{'─'*58}\n"
            f"  Accuracy            : {acc:>6.1f}%   {acc_sym}\n"
            f"  AUC-ROC             : {auc:>6.3f}    {auc_sym}\n"
            f"  Fiabilité           : {fiab}\n",
            end="",
            flush=True,
        )
        if warning:
            print(f"  ⚠  {warning}", flush=True)
        print(f"{'═'*58}\n", flush=True)

    # ------------------------------------------------------------------
    # RL — RecurrentPPO via the existing RLTrainer
    # ------------------------------------------------------------------

    def _retrain_rl(
        self,
        timesteps: int,
        symbol: Optional[str],
        timeframe: Optional[str],
        feature_provider: Optional[Callable[[], pd.DataFrame]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
        try:
            from strategies.rl_agent.rl_trainer import RLTrainer, RLTrainerConfig
        except Exception as exc:
            return {"trained": False, "message": f"RL stack unavailable: {exc}"}

        # Obtain a continuous feature series. RL needs price-series history,
        # not the sparse per-trade experiences.
        feature_df: Optional[pd.DataFrame] = None
        if feature_provider is not None:
            try:
                feature_df = feature_provider()
            except Exception as exc:
                logger.warning("RL feature provider failed: %s", exc)

        cfg = RLTrainerConfig(
            symbol          = symbol or "EURUSD",
            timeframe       = timeframe or "H1",
            total_timesteps = timesteps,
            model_dir       = self.model_dir,
        )

        try:
            trainer = RLTrainer(cfg)
            train_f, val_f, test_f, train_p, val_p, test_p = trainer.prepare_data(custom_data=feature_df)
            if len(train_f) <= cfg.window_size + 2 or len(val_f) <= cfg.window_size + 2:
                msg = ("Not enough market-data history to retrain the RL "
                       "agent. Provide more bars / populate the FeatureStore.")
                print(f"\n[RL] ✗ {msg}\n", flush=True)
                return {"trained": False, "message": msg}
            if progress_callback is not None:
                trainer.set_progress_callback(progress_callback)
            trainer.train(train_f, val_f, train_prices=train_p, val_prices=val_p)

            # Évaluation sur le jeu de test (si assez de données)
            eval_results: dict = {}
            if len(test_f) > cfg.window_size + 2:
                try:
                    eval_results = trainer.evaluate(test_f, n_episodes=5)
                except Exception as exc:
                    logger.warning("RL evaluation failed: %s", exc)

            result = {
                "trained":    True,
                "timesteps":  timesteps,
                "train_bars": int(len(train_f)),
                "val_bars":   int(len(val_f)),
                "test_bars":  int(len(test_f)),
                "message":    "RL agent retrained (RecurrentPPO).",
                **({
                    "eval_mean_reward": round(eval_results["mean_reward"], 4),
                    "eval_std_reward":  round(eval_results["std_reward"],  4),
                } if eval_results else {}),
            }
            return result
        except FileNotFoundError:
            msg = ("No FeatureStore data found and no feature provider "
                   "supplied — cannot retrain RL agent yet.")
            print(f"\n[RL] ✗ {msg}\n", flush=True)
            return {"trained": False, "message": msg}
        except Exception as exc:
            print(f"\n[RL] ✗ Erreur : {type(exc).__name__}: {exc}\n", flush=True)
            return {"trained": False, "message": f"{type(exc).__name__}: {exc}"}
