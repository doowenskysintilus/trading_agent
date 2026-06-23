"""
WinClassifier
=============
Supervised win/loss model that learns *directly from the system's own
trading results*. It maps the entry observation (the same features the
strategies see) to the probability that a trade will be profitable.

Used in parallel with the RL agent as a confidence filter: trades whose
predicted win-probability is too low can be skipped or down-weighted.

Backed by scikit-learn (GradientBoostingClassifier). Persisted with joblib.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


@dataclass
class TrainReport:
    """Summary of a training run, returned to the API/dashboard."""

    trained:      bool          = False
    n_samples:    int           = 0
    n_wins:       int           = 0
    n_losses:     int           = 0
    accuracy:     float         = 0.0
    auc:          float         = 0.0
    feature_names: list[str]    = field(default_factory=list)
    message:      str           = ""
    warning:      str           = ""
    reliable:     bool          = True

    def as_dict(self) -> dict:
        return {
            "trained":       self.trained,
            "n_samples":     self.n_samples,
            "n_wins":        self.n_wins,
            "n_losses":      self.n_losses,
            "accuracy":      round(self.accuracy, 4),
            "auc":           round(self.auc, 4),
            "n_features":    len(self.feature_names),
            "message":       self.message,
            "warning":       self.warning,
            "reliable":      self.reliable,
        }


class WinClassifier:
    """Predicts P(win) from a trade's entry features."""

    def __init__(self, model_path: str = "data/storage/models/win_classifier.joblib") -> None:
        self.model_path     = Path(model_path)
        self._model         = None
        self._feature_names: list[str] = []
        self._reliable: bool = True  # False when trained on < 60 samples (bypass filter)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X, y, feature_names: list[str], min_samples: int = 30) -> TrainReport:
        """Fit the classifier on (features → win) data.

        Requires at least ``min_samples`` trades and both classes present;
        otherwise returns an untrained report explaining why.
        """
        if not _SKLEARN_OK:
            return TrainReport(message="scikit-learn not installed.")

        n = len(y)
        n_wins   = int(np.sum(np.asarray(y) == 1))
        n_losses = n - n_wins

        if n < min_samples:
            return TrainReport(
                n_samples=n, n_wins=n_wins, n_losses=n_losses,
                message=f"Not enough trades to train ({n} < {min_samples}). "
                        f"Keep trading to collect more data.",
            )
        if n_wins == 0 or n_losses == 0:
            return TrainReport(
                n_samples=n, n_wins=n_wins, n_losses=n_losses,
                message="Need both winning and losing trades to learn.",
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        # Temporal (chronological) split — never mix future data into training.
        # Random splitting would leak future trades into the training set.
        if n >= 60:
            split      = int(n * 0.75)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]
            # Ensure both splits have both classes
            if len(np.unique(y_te)) < 2:
                # Fall back to training-set evaluation with a warning
                X_te, y_te = X_tr, y_tr
        else:
            X_tr, X_te, y_tr, y_te = X, X, y, y

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,          # stochastic gradient boosting — reduces overfit
            min_samples_leaf=5,     # prevents learning single-sample patterns
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        acc   = float(accuracy_score(y_te, preds))
        try:
            auc = float(roc_auc_score(y_te, proba))
        except ValueError:
            auc = 0.0

        warning_parts: list[str] = []
        reliable = True

        # Tiny datasets often look "perfect" but do not generalize.
        if n < 60:
            reliable = False
            warning_parts.append(
                "Evaluation done on training data (<60 samples): metrics likely optimistic"
            )

        # Suspiciously high scores on small samples indicate overfit risk.
        if n < 120 and (acc >= 0.95 or auc >= 0.98):
            reliable = False
            warning_parts.append(
                "Overfit risk: very high metrics on a small dataset"
            )

        if n < 200:
            warning_parts.append(
                "Small sample: collect more closed trades for stable ML filter"
            )

        warning = "; ".join(warning_parts)

        self._model = model
        self._feature_names = list(feature_names)
        self._reliable = reliable
        self._save()

        logger.info(
            "WinClassifier trained — n=%d acc=%.3f auc=%.3f", n, acc, auc,
        )
        return TrainReport(
            trained=True, n_samples=n, n_wins=n_wins, n_losses=n_losses,
            accuracy=acc, auc=auc, feature_names=self._feature_names,
            message="Model trained and saved.",
            warning=warning,
            reliable=reliable,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_win_proba(self, features: dict[str, float]) -> Optional[float]:
        """Return P(win) for one trade's feature dict, or None if no model.

        Returns None (filter passes through) when the incoming feature set
        doesn't match the model's training features, so a changed feature
        config never silently corrupts predictions.
        """
        if self._model is None and not self.load():
            return None

        if not self._reliable:
            logger.debug(
                "WinClassifier: model trained on <60 samples — bypassing filter "
                "to avoid overfit-based blocking. Collect more trade data."
            )
            return None

        if not self._feature_names:
            return None

        incoming_keys = set(features.keys())
        trained_keys  = set(self._feature_names)
        missing = trained_keys - incoming_keys
        if missing:
            logger.warning(
                "WinClassifier: %d feature(s) missing from observation "
                "(e.g. %s) — model was trained on a different feature set. "
                "Skipping filter (model needs retraining).",
                len(missing),
                next(iter(missing)),
            )
            return None

        x = np.array(
            [[float(features.get(name, 0.0)) for name in self._feature_names]],
            dtype=np.float64,
        )
        try:
            return float(self._model.predict_proba(x)[0, 1])
        except Exception as exc:
            logger.debug("WinClassifier predict failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Atomically write the model file using a temp-file + os.replace().

        Without this, a live trader thread loading the model mid-write would
        receive a corrupt joblib file and silently predict garbage probabilities.
        """
        if not _SKLEARN_OK or self._model is None:
            return
        dest = self.model_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=dest.parent, suffix=".joblib.tmp"
        )
        try:
            os.close(fd)
            joblib.dump(
                {
                    "model":         self._model,
                    "feature_names": self._feature_names,
                    "reliable":      self._reliable,
                },
                tmp_path,
            )
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> bool:
        """Load a previously trained model. Returns True on success."""
        if not _SKLEARN_OK or not self.model_path.exists():
            return False
        try:
            blob = joblib.load(self.model_path)
            self._model         = blob["model"]
            self._feature_names = blob["feature_names"]
            self._reliable      = bool(blob.get("reliable", True))
            return True
        except Exception as exc:
            logger.warning("WinClassifier load failed: %s", exc)
            return False
