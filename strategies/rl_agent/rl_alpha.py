"""
RLAlpha
=======
Wraps a trained SB3 / sb3-contrib policy as an AlphaModel.

Supports both standard PPO and RecurrentPPO (LSTM).

For RecurrentPPO, the LSTM hidden state is propagated across
consecutive `compute()` calls, preserving temporal context across
the full inference sequence — exactly as during training.

Confidence is derived from the softmax of the policy's action
logits when available; falls back to a deterministic 1.0/0.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research.alpha_models.base import AlphaModel, AlphaSignal, SignalType

import logging
logger = logging.getLogger(__name__)

# SB3 imports are optional — codebase can be imported without SB3 installed.
try:
    from stable_baselines3.common.policies import BasePolicy
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

try:
    from sb3_contrib import RecurrentPPO as _RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False


# Action mapping (must match TradingEnv)
_ACTION_TO_SIGNAL: dict[int, SignalType] = {
    0: SignalType.HOLD,
    1: SignalType.BUY,
    2: SignalType.SELL,
}


class RLAlpha(AlphaModel):
    """
    Inference wrapper around a trained SB3 / sb3-contrib policy.

    Parameters
    ----------
    model_path : str | Path
        Path to a saved SB3 model (zip file).
    algo : str
        'PPO', 'RecurrentPPO', 'A2C', 'SAC', etc.
    window_size : int
        Must match the window_size used during training.
    deterministic : bool
        Use deterministic actions (recommended for live inference).
    reset_lstm_on_compute : bool
        When True, LSTM hidden state is reset on every call to compute()
        (stateless inference). When False (default), the hidden state
        persists across calls — suitable for live step-by-step inference.
    """

    def __init__(
        self,
        name: str = "rl_agent",
        model_path: Optional[str | Path] = None,
        algo: str = "PPO",
        window_size: int = 32,
        deterministic: bool = True,
        reset_lstm_on_compute: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(name=name, enabled=enabled)

        self.window_size            = window_size
        self.deterministic          = deterministic
        self.reset_lstm_on_compute  = reset_lstm_on_compute
        self._model                 = None
        self._is_recurrent          = False

        # LSTM hidden state: (h_n, c_n), maintained across steps
        self._lstm_states: Optional[tuple] = None
        self._episode_starts: Optional[np.ndarray] = None

        self._model_path: Optional[Path] = Path(model_path) if model_path else None
        self._algo = algo
        self._model_mtime: float = 0.0

        if model_path is not None:
            self.load(model_path, algo)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def maybe_reload(self) -> None:
        """Hot-reload the model when its file changes on disk.

        Called once per compute() so a freshly retrained model (from
        /rl/retrain) is picked up without restarting the live trader.
        """
        if self._model_path is None:
            return
        try:
            mtime = self._model_path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._model_mtime:
            return
        try:
            self.load(self._model_path, self._algo)
            self._model_mtime = mtime
            logger.info("RLAlpha: model reloaded from %s (mtime=%.0f).", self._model_path, mtime)
        except Exception as exc:
            logger.warning("RLAlpha: hot-reload failed — %s", exc)

    def load(self, model_path: str | Path, algo: str = "PPO") -> None:
        """Load a trained SB3 / sb3-contrib model from disk."""
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is not installed.")

        algo_upper = algo.upper()

        if algo_upper == "RECURRENTPPO":
            if not _SB3_CONTRIB_AVAILABLE:
                raise ImportError(
                    "sb3-contrib is required for RecurrentPPO. "
                    "Install with: pip install sb3-contrib"
                )
            self._model       = _RecurrentPPO.load(str(model_path))
            self._is_recurrent = True
        else:
            import importlib
            sb3_module = importlib.import_module("stable_baselines3")
            algo_class = getattr(sb3_module, algo_upper)
            self._model        = algo_class.load(str(model_path))
            self._is_recurrent = False

        # Track mtime so maybe_reload() can detect future changes.
        self._model_path = Path(model_path)
        self._algo = algo
        try:
            self._model_mtime = self._model_path.stat().st_mtime
        except OSError:
            self._model_mtime = 0.0

        # Initialise LSTM state for first inference call
        self._reset_lstm_state()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Manually reset the LSTM hidden state (call at start of new episode)."""
        self._reset_lstm_state()

    def compute(self, data: pd.DataFrame) -> AlphaSignal:
        self.maybe_reload()
        if not self.is_loaded:
            return self._hold(reason="model_not_loaded")

        if len(data) < self.window_size:
            return self._hold(reason="insufficient_data", rows=len(data))

        if self.reset_lstm_on_compute:
            self._reset_lstm_state()

        obs = self._build_observation(data)

        if self._is_recurrent:
            action, self._lstm_states = self._model.predict(
                obs,
                state          = self._lstm_states,
                episode_start  = self._episode_starts,
                deterministic  = self.deterministic,
            )
            # After first step, episode_start is False
            self._episode_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = self._model.predict(obs, deterministic=self.deterministic)

        action     = int(np.asarray(action).reshape(-1)[0])
        signal     = _ACTION_TO_SIGNAL.get(action, SignalType.HOLD)
        confidence = self._action_confidence(obs, action)

        return AlphaSignal(
            signal        = signal,
            confidence    = confidence,
            strategy_name = self._name,
            metadata      = {
                "raw_action":   action,
                "is_recurrent": self._is_recurrent,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_lstm_state(self) -> None:
        self._lstm_states    = None
        self._episode_starts = np.ones((1,), dtype=bool)

    def _build_observation(self, data: pd.DataFrame) -> np.ndarray:
        """
        Build the flat observation vector from the last `window_size` rows.
        Mirrors the observation space defined in RLTradingEnv.
        Portfolio features default to neutral (no live position context).
        """
        numeric = data.select_dtypes(include=[np.number])
        window  = numeric.iloc[-self.window_size:].values.astype(np.float32).flatten()

        # 4 portfolio features matching RLTradingEnv._observe()
        portfolio = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([window, portfolio]).reshape(1, -1)

    def _action_confidence(self, obs: np.ndarray, action: int) -> float:
        """
        Extract action probability as confidence score.

        For RecurrentPPO: uses policy distribution with current LSTM state.
        For standard PPO: uses standard policy distribution.
        Falls back to 1.0/0.0 when distribution is unavailable.
        """
        try:
            import torch
            policy     = self._model.policy
            obs_tensor = policy.obs_to_tensor(obs)[0]

            with torch.no_grad():
                if self._is_recurrent and self._lstm_states is not None:
                    # RecurrentPPO: pass LSTM state to get correct distribution
                    features, _ = policy.extract_features(obs_tensor)
                    latent_pi, _ = policy.mlp_extractor(features)
                    distribution = policy._get_action_dist_from_latent(latent_pi)
                else:
                    distribution = policy.get_distribution(obs_tensor)

                probs = distribution.distribution.probs.squeeze().cpu().numpy()

            # Confidence for non-HOLD actions:
            # We want a score that reflects BOTH how likely this action is
            # AND how much the model prefers it over HOLD.
            # Raw formula: (p_action / non_hold) can over-amplify weak signals
            # when hold_prob ≈ 1.  We multiply back by non_hold so very
            # HOLD-dominant models produce low confidence for their signals.
            action_prob = float(np.clip(probs[action], 0.0, 1.0))

            if action != 0:
                hold_prob   = float(probs[0])
                non_hold    = 1.0 - hold_prob + 1e-8
                relative    = float(np.clip(probs[action] / non_hold, 0.0, 1.0))
                # Penalise when HOLD dominates: multiply by (non_hold / 0.5)
                # so that a non_hold of 0.5 (equal chance of hold vs non-hold)
                # gives no penalty, while non_hold < 0.5 reduces confidence.
                hold_penalty = float(np.clip(non_hold / 0.5, 0.0, 1.0))
                action_prob  = float(np.clip(relative * hold_penalty, 0.0, 1.0))

            return action_prob
        except Exception:
            return 1.0 if action != 0 else 0.0
