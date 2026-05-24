"""
PPO LSTM Policy
===============
Custom LSTM-based actor-critic architecture for trading.

Architecture
------------
  Observation (flat vector)
        │
        ▼
  InputEncoder              Linear → LayerNorm → GELU
        │
        ▼
  LSTMCore                  nn.LSTM (n_layers, hidden_size)
        │
   ┌────┴────┐
   ▼         ▼
 Actor     Critic
 head      head
(softmax)  (V(s))

Compatibility
-------------
- Works as a `policy_kwargs` feature extractor for SB3 PPO.
- Used as the full policy by RecurrentPPO (sb3-contrib) when
  passed as `policy_class`.

Usage with sb3-contrib RecurrentPPO
-------------------------------------
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO
    from strategies.rl_agent.ppo_lstm_policy import build_recurrent_policy

    model = RecurrentPPO(
        policy       = "MlpLstmPolicy",
        env          = DummyVecEnv([lambda: env]),
        policy_kwargs = build_recurrent_policy(lstm_hidden=256, n_lstm_layers=2),
        ...
    )

Usage with standard SB3 PPO (feature extractor mode)
------------------------------------------------------
    from stable_baselines3 import PPO
    from strategies.rl_agent.ppo_lstm_policy import LSTMFeaturesExtractor

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        policy_kwargs = {
            "features_extractor_class":  LSTMFeaturesExtractor,
            "features_extractor_kwargs": {"lstm_hidden": 128},
        },
    )
"""

from __future__ import annotations

from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy
    import gymnasium as gym
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Input encoder
# ---------------------------------------------------------------------------

class InputEncoder(nn.Module):
    """
    Lightweight feedforward encoder that projects the raw observation
    into a latent embedding before the LSTM.

    Parameters
    ----------
    input_dim : int
    hidden_dim : int
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# LSTM core
# ---------------------------------------------------------------------------

class LSTMCore(nn.Module):
    """
    Stateful LSTM with optional layer normalisation on the output.

    Parameters
    ----------
    input_dim    : int
    hidden_size  : int
    n_layers     : int
    dropout      : float  (applied between layers when n_layers > 1)
    layer_norm   : bool
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        n_layers: int   = 2,
        dropout: float  = 0.1,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x      : (batch, seq_len, input_dim)
        hidden : (h_0, c_0) each (n_layers, batch, hidden_size), or None

        Returns
        -------
        out    : (batch, hidden_size)   — last time-step only
        hidden : (h_n, c_n)
        """
        out, hidden = self.lstm(x, hidden)
        last = out[:, -1, :]            # last timestep
        return self.norm(last), hidden


# ---------------------------------------------------------------------------
# Full actor-critic network
# ---------------------------------------------------------------------------

class LSTMActorCriticNet(nn.Module):
    """
    Combined actor-critic network with shared LSTM backbone.

    Actor  → Categorical distribution over {HOLD, BUY, SELL}
    Critic → Scalar state value V(s)

    Parameters
    ----------
    obs_dim      : int    flat observation size
    n_actions    : int    3 for trading (HOLD / BUY / SELL)
    encoder_dim  : int    encoder hidden size
    lstm_hidden  : int    LSTM hidden size
    n_lstm_layers: int
    head_hidden  : int    actor/critic head hidden size
    dropout      : float
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int       = 3,
        encoder_dim: int     = 128,
        lstm_hidden: int     = 256,
        n_lstm_layers: int   = 2,
        head_hidden: int     = 64,
        dropout: float       = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = InputEncoder(obs_dim, encoder_dim, dropout)
        self.lstm    = LSTMCore(encoder_dim, lstm_hidden, n_lstm_layers, dropout)

        self.actor_head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, n_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor head: small init for better initial exploration
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns
        -------
        action_logits : (batch, n_actions)
        value         : (batch, 1)
        hidden        : updated LSTM state
        """
        # obs shape: (batch, obs_dim) → add seq_len=1 for LSTM
        encoded = self.encoder(obs)              # (batch, encoder_dim)
        encoded = encoded.unsqueeze(1)           # (batch, 1, encoder_dim)

        lstm_out, hidden = self.lstm(encoded, hidden)  # (batch, lstm_hidden)

        action_logits = self.actor_head(lstm_out)      # (batch, n_actions)
        value         = self.critic_head(lstm_out)     # (batch, 1)

        return action_logits, value, hidden

    def get_action_probs(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        """Return action probabilities (softmax) and updated hidden state."""
        logits, _, hidden = self.forward(obs, hidden)
        return torch.softmax(logits, dim=-1), hidden

    def get_value(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> torch.Tensor:
        _, value, _ = self.forward(obs, hidden)
        return value


# ---------------------------------------------------------------------------
# SB3 features extractor wrapper
# ---------------------------------------------------------------------------

if _SB3_AVAILABLE:
    class LSTMFeaturesExtractor(BaseFeaturesExtractor):
        """
        SB3 BaseFeaturesExtractor wrapper around LSTMCore.

        Use with standard PPO via policy_kwargs:
            policy_kwargs = {
                "features_extractor_class":  LSTMFeaturesExtractor,
                "features_extractor_kwargs": {
                    "lstm_hidden":   256,
                    "n_lstm_layers": 2,
                    "encoder_dim":   128,
                },
            }
        """

        def __init__(
            self,
            observation_space: gym.Space,
            lstm_hidden: int    = 256,
            n_lstm_layers: int  = 2,
            encoder_dim: int    = 128,
            dropout: float      = 0.1,
        ) -> None:
            super().__init__(observation_space, features_dim=lstm_hidden)

            obs_dim = int(np.prod(observation_space.shape))
            self.encoder = InputEncoder(obs_dim, encoder_dim, dropout)
            self.lstm    = LSTMCore(encoder_dim, lstm_hidden, n_lstm_layers, dropout)
            self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            encoded  = self.encoder(observations)          # (batch, encoder_dim)
            encoded  = encoded.unsqueeze(1)                # (batch, 1, encoder_dim)
            out, self._hidden = self.lstm(encoded, None)   # reset hidden each call
            return out


# ---------------------------------------------------------------------------
# Policy kwargs builders
# ---------------------------------------------------------------------------

def build_ppo_policy_kwargs(
    lstm_hidden: int   = 256,
    n_lstm_layers: int = 2,
    encoder_dim: int   = 128,
    net_arch: Optional[list] = None,
) -> dict:
    """
    Returns policy_kwargs for SB3 PPO with LSTM feature extractor.

    Note: standard PPO resets hidden state between rollout episodes.
    For true sequential hidden state across steps, use RecurrentPPO.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required.")
    return {
        "features_extractor_class":  LSTMFeaturesExtractor,
        "features_extractor_kwargs": {
            "lstm_hidden":   lstm_hidden,
            "n_lstm_layers": n_lstm_layers,
            "encoder_dim":   encoder_dim,
        },
        "net_arch": net_arch or [dict(pi=[64], vf=[64])],
    }


def build_recurrent_policy_kwargs(
    lstm_hidden_size: int = 256,
    n_lstm_layers: int    = 2,
    net_arch: Optional[list] = None,
) -> dict:
    """
    Returns policy_kwargs for sb3-contrib RecurrentPPO.
    RecurrentPPO natively propagates LSTM hidden state across steps.
    """
    return {
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers":    n_lstm_layers,
        "net_arch":         net_arch or [dict(pi=[64, 64], vf=[64, 64])],
        "enable_critic_lstm": True,
    }
