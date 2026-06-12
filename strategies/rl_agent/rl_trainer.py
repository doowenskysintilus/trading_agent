"""
RLTrainer
=========
End-to-end training pipeline for the PPO-LSTM trading agent.

Pipeline
--------
  FeatureStore.load(symbol, timeframe)
        │
        ▼
  train / val / test split
        │
        ▼
  TradingEnv  (reward: profit + drawdown penalty + risk penalty)
        │
        ▼
  RecurrentPPO  (sb3-contrib)  or  PPO  (SB3 fallback)
  with LSTMFeaturesExtractor
        │
        ▼
  EvalCallback  + CheckpointCallback
        │
        ▼
  trained model  →  RLAlpha  (inference-ready AlphaModel)

Reward function
---------------
  r = pnl_reward
      - alpha_dd  × drawdown_penalty
      - alpha_risk × risk_penalty
      + alpha_sharp × sharpe_component

  pnl_reward      = ΔV / V₀   (normalised P&L per step)
  drawdown_penalty= max(0, max_equity - equity) / max_equity
  risk_penalty    = |position| × σ_recent  (volatility-adjusted exposure)
  sharpe_component= rolling_mean_return / (rolling_std_return + ε)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research.feature_store.feature_store import FeatureStore
from strategies.rl_agent.ppo_lstm_policy import (
    build_ppo_policy_kwargs,
    build_recurrent_policy_kwargs,
    _SB3_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Optional SB3 / sb3-contrib imports
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        BaseCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

try:
    from sb3_contrib import RecurrentPPO
    _SB3_CONTRIB_AVAILABLE = True
except ImportError:
    _SB3_CONTRIB_AVAILABLE = False

# TensorBoard is optional: SB3 raises ImportError at training time if a
# tensorboard_log directory is set but the package is missing.
try:
    import tensorboard  # noqa: F401
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Weights for the composite reward function."""

    alpha_drawdown:   float = 2.0   # drawdown penalty coefficient
    alpha_risk:       float = 0.5   # volatility-adjusted exposure penalty
    alpha_sharpe:     float = 0.3   # Sharpe component weight
    sharpe_window:    int   = 20    # bars for rolling Sharpe estimate
    pnl_scale:        float = 100.0 # scale P&L component (keeps reward ~[-1, 1])


# ---------------------------------------------------------------------------
# Trading environment with composite reward
# ---------------------------------------------------------------------------

class RLTradingEnv(gym.Env if _SB3_AVAILABLE else object):
    """
    Gymnasium trading environment with composite reward.

    Observation: flat vector of feature_store columns over `window_size` bars
                 + 4 portfolio features [position, pnl_pct, drawdown, vol]

    Action space: Discrete(3)  — 0=HOLD, 1=BUY, 2=SELL
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,           # (n_bars, n_features)
        reward_cfg: RewardConfig,
        window_size: int   = 32,
        initial_balance: float = 100_000.0,
        spread: float      = 0.0001,
        sl_pct: float      = 0.02,
        tp_pct: float      = 0.04,
        position_size_pct: float = 0.10,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("gymnasium and stable-baselines3 are required.")

        self.features       = features.astype(np.float32)
        self.reward_cfg     = reward_cfg
        self.window_size    = window_size
        self.initial_balance = initial_balance
        self.spread         = spread
        self.sl_pct         = sl_pct
        self.tp_pct         = tp_pct
        self.position_size_pct = position_size_pct

        n_features = features.shape[1]
        obs_dim    = window_size * n_features + 4   # +4 portfolio features

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        # Internal state (initialised in reset)
        self._bar          = window_size
        self._balance      = initial_balance
        self._position     = 0      # -1, 0, +1
        self._entry_price  = 0.0
        self._peak_equity  = initial_balance
        self._equity       = initial_balance
        self._returns: list[float] = []

    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._bar          = self.window_size
        self._balance      = self.initial_balance
        self._position     = 0
        self._entry_price  = 0.0
        self._peak_equity  = self.initial_balance
        self._equity       = self.initial_balance
        self._returns      = []
        return self._observe(), {}

    def step(self, action: int):
        bar    = self._bar
        close  = self._get_close(bar)
        prev_equity = self._equity

        # ---- Execute action --------------------------------------------
        reward_components = self._execute(action, close)

        # ---- Advance bar -----------------------------------------------
        self._bar += 1
        terminated = self._bar >= len(self.features) - 1
        truncated  = False

        # ---- Compute composite reward ----------------------------------
        reward = self._compute_reward(
            prev_equity, reward_components, close, terminated
        )

        return self._observe(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, action: int, close: float) -> dict:
        """Open / close / hold position. Returns reward components."""
        direction = action - 1          # 0→-1(SELL), 1→0(HOLD), 2→+1(BUY)
        spread    = self.spread

        pnl = 0.0
        if self._position != 0 and (action == 0 or direction != self._position):
            # Close
            exec_price = close - spread * self._position
            raw_pnl    = (exec_price - self._entry_price) * self._position
            pnl        = raw_pnl * self._get_size()
            self._balance += pnl
            self._position    = 0
            self._entry_price = 0.0

        if action != 1 and self._position == 0:
            # Open
            exec_price       = close + spread * direction
            self._position   = direction
            self._entry_price = exec_price

        # Mark-to-market
        unrealized = 0.0
        if self._position != 0:
            unrealized = (
                (close - self._entry_price) * self._position * self._get_size()
            )
        self._equity = self._balance + unrealized
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        return {"pnl": pnl}

    def _get_size(self) -> float:
        notional = self._equity * self.position_size_pct
        close    = self._get_close(self._bar)
        return notional / (close + 1e-10)

    def _get_close(self, bar: int) -> float:
        """Assumes last feature column is (or includes) a price proxy."""
        # Use the first feature column as close price proxy
        return float(self.features[bar, 0])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_equity: float,
        components: dict,
        close: float,
        done: bool,
    ) -> float:
        cfg = self.reward_cfg

        # 1. P&L component
        delta_equity = self._equity - prev_equity
        pnl_reward   = (delta_equity / (self.initial_balance + 1e-10)) * cfg.pnl_scale

        # Track returns for Sharpe
        ret = delta_equity / (prev_equity + 1e-10)
        self._returns.append(ret)

        # 2. Drawdown penalty
        dd = max(0.0, (self._peak_equity - self._equity) / (self._peak_equity + 1e-10))
        drawdown_penalty = cfg.alpha_drawdown * dd

        # 3. Volatility-adjusted exposure penalty
        recent_vols = self.features[max(0, self._bar - 20): self._bar, :]
        vol_proxy   = float(np.std(recent_vols[:, 0])) if len(recent_vols) > 1 else 0.0
        risk_penalty = cfg.alpha_risk * abs(self._position) * vol_proxy * cfg.pnl_scale

        # 4. Sharpe component
        sharpe_bonus = 0.0
        if len(self._returns) >= cfg.sharpe_window:
            window_rets  = np.array(self._returns[-cfg.sharpe_window:])
            mean_r, std_r = float(np.mean(window_rets)), float(np.std(window_rets))
            sharpe_bonus  = cfg.alpha_sharpe * (mean_r / (std_r + 1e-8))

        reward = pnl_reward - drawdown_penalty - risk_penalty + sharpe_bonus

        # Terminal penalty if deep drawdown
        if done and dd > 0.25:
            reward -= 10.0

        return float(np.clip(reward, -10.0, 10.0))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        bar     = min(self._bar, len(self.features) - 1)
        window  = self.features[bar - self.window_size: bar].flatten()

        # Portfolio features
        pnl_pct  = (self._equity - self.initial_balance) / (self.initial_balance + 1e-10)
        dd       = (self._peak_equity - self._equity) / (self._peak_equity + 1e-10)
        recent   = self.features[max(0, bar - 20): bar, 0]
        vol      = float(np.std(recent)) if len(recent) > 1 else 0.0

        portfolio = np.array([
            float(self._position),
            float(np.clip(pnl_pct, -1, 1)),
            float(np.clip(dd, 0, 1)),
            float(vol),
        ], dtype=np.float32)

        return np.concatenate([window, portfolio]).astype(np.float32)

    def render(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

@dataclass
class RLTrainerConfig:
    """Hyperparameters and training settings."""

    # ---- Feature store --------------------------------------------------
    feature_store_root: str  = "data/storage"
    symbol: str              = "EURUSD"
    timeframe: str           = "H1"
    feature_version: str     = "v1"

    # ---- Data splits (fraction of total) --------------------------------
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test is the remainder

    # ---- Environment ----------------------------------------------------
    window_size: int     = 32
    initial_balance: float = 100_000.0
    spread: float        = 0.0001
    sl_pct: float        = 0.02
    tp_pct: float        = 0.04
    position_size_pct: float = 0.10
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # ---- LSTM architecture ----------------------------------------------
    lstm_hidden: int     = 256
    n_lstm_layers: int   = 2
    encoder_dim: int     = 128
    net_arch: list       = field(default_factory=lambda: [dict(pi=[64, 64], vf=[64, 64])])

    # ---- PPO hyperparameters --------------------------------------------
    use_recurrent_ppo: bool  = True    # use sb3-contrib RecurrentPPO if available
    learning_rate: float     = 3e-4
    n_steps: int             = 512     # rollout length
    batch_size: int          = 64
    n_epochs: int            = 10
    gamma: float             = 0.99
    gae_lambda: float        = 0.95
    clip_range: float        = 0.2
    ent_coef: float          = 0.01    # entropy regularisation (exploration)
    vf_coef: float           = 0.5
    max_grad_norm: float     = 0.5
    normalize_advantage: bool = True

    # ---- Training schedule ----------------------------------------------
    total_timesteps: int     = 500_000
    eval_freq: int           = 10_000
    n_eval_episodes: int     = 5
    checkpoint_freq: int     = 50_000

    # ---- Resume / warm-start --------------------------------------------
    # When True, train() continues from the last saved model (if present)
    # instead of starting from scratch, so stopping and relaunching keeps the
    # knowledge acquired in previous runs and the timestep counter accumulates.
    warm_start: bool = True

    # ---- I/O -----------------------------------------------------------
    model_dir: str   = "data/storage/models"
    model_name: str  = "ppo_lstm_trading"
    log_dir: str     = "data/storage/logs"
    # Append-only reward progress (one line per episode) so the dashboard can
    # plot the reward curve in real time. Cumulative across warm-started runs.
    progress_path: str = "data/storage/logs/rl_progress.jsonl"
    verbose: int     = 1


# ---------------------------------------------------------------------------
# Reward-tracking callback
# ---------------------------------------------------------------------------

class RewardTrackingCallback(BaseCallback if _SB3_AVAILABLE else object):
    """Logs episode rewards and streams reward progress to a JSONL file.

    Each completed episode appends one line {ts, timestep, episode,
    last_reward, mean_reward_10} so the dashboard can plot the reward curve
    in real time and confirm the reward trends upward over training.
    """

    def __init__(self, verbose: int = 0, progress_path: str | None = None) -> None:
        if _SB3_AVAILABLE:
            super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0
        self._progress_path = Path(progress_path) if progress_path else None
        if self._progress_path is not None:
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_progress(self, last_reward: float, mean_reward: float) -> None:
        if self._progress_path is None:
            return
        import json
        from datetime import datetime, timezone
        record = {
            "ts":          datetime.now(timezone.utc).isoformat(),
            "timestep":    int(getattr(self, "num_timesteps", 0)),
            "episode":     len(self.episode_rewards),
            "last_reward": round(float(last_reward), 6),
            "mean_reward": round(float(mean_reward), 6),
        }
        try:
            with self._progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.debug("RL progress write failed: %s", exc)

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            recent = self.episode_rewards[-10:]
            mean_recent = float(np.mean(recent))
            self._write_progress(self.episode_rewards[-1], mean_recent)
            if self.verbose >= 1 and len(self.episode_rewards) % 10 == 0:
                logger.info(
                    "Episode %d | mean_reward=%.4f | last=%.4f",
                    len(self.episode_rewards),
                    mean_recent,
                    self.episode_rewards[-1],
                )
            self._ep_reward = 0.0
        return True


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class RLTrainer:
    """
    Full training pipeline for the LSTM-PPO trading agent.

    Parameters
    ----------
    config : RLTrainerConfig
    """

    def __init__(self, config: RLTrainerConfig | None = None) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 and gymnasium are required for RLTrainer. "
                "Install with: pip install stable-baselines3 gymnasium"
            )
        self.cfg   = config or RLTrainerConfig()
        self.model = None
        self._train_env = None
        self._val_env   = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        custom_data: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load features from FeatureStore (or use custom_data).

        Returns
        -------
        train_features, val_features, test_features
            Each is a float32 numpy array of shape (n_bars, n_features).
        """
        cfg = self.cfg

        if custom_data is not None:
            features = custom_data.select_dtypes(include=[np.number]).values.astype(np.float32)
            logger.info("Using custom data: %d bars × %d features", *features.shape)
        else:
            store = FeatureStore(root_dir=cfg.feature_store_root)
            df = store.load(
                symbol=cfg.symbol,
                timeframe=cfg.timeframe,
                version=cfg.feature_version,
            )
            # Remove OHLCV raw columns — keep only engineered features
            drop_cols = {"open", "high", "low", "close", "volume"}
            feature_cols = [c for c in df.columns if c not in drop_cols]
            features = df[feature_cols].ffill().fillna(0).values.astype(np.float32)
            logger.info(
                "Loaded %d bars × %d features from FeatureStore",
                *features.shape,
            )

        n = len(features)
        n_train = int(n * cfg.train_frac)
        n_val   = int(n * cfg.val_frac)

        train = features[:n_train]
        val   = features[n_train: n_train + n_val]
        test  = features[n_train + n_val:]

        logger.info(
            "Split — train=%d  val=%d  test=%d",
            len(train), len(val), len(test),
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Environment builders
    # ------------------------------------------------------------------

    def _make_env(self, features: np.ndarray, seed: int = 0):
        cfg = self.cfg

        def _init():
            env = RLTradingEnv(
                features         = features,
                reward_cfg       = cfg.reward_config,
                window_size      = cfg.window_size,
                initial_balance  = cfg.initial_balance,
                spread           = cfg.spread,
                sl_pct           = cfg.sl_pct,
                tp_pct           = cfg.tp_pct,
                position_size_pct= cfg.position_size_pct,
            )
            return Monitor(env)

        return DummyVecEnv([_init])

    # ------------------------------------------------------------------
    # Model builder
    # ------------------------------------------------------------------

    def _build_model(self, train_env):
        cfg = self.cfg

        use_recurrent = cfg.use_recurrent_ppo and _SB3_CONTRIB_AVAILABLE
        AlgoClass = RecurrentPPO if use_recurrent else PPO

        if use_recurrent:
            policy_kwargs = build_recurrent_policy_kwargs(
                lstm_hidden_size = cfg.lstm_hidden,
                n_lstm_layers    = cfg.n_lstm_layers,
                net_arch         = cfg.net_arch,
            )
            policy = "MlpLstmPolicy"
            logger.info("Using RecurrentPPO with MlpLstmPolicy (sb3-contrib)")
        else:
            policy_kwargs = build_ppo_policy_kwargs(
                lstm_hidden   = cfg.lstm_hidden,
                n_lstm_layers = cfg.n_lstm_layers,
                encoder_dim   = cfg.encoder_dim,
                net_arch      = cfg.net_arch,
            )
            policy = "MlpPolicy"
            logger.info("Using PPO with LSTM feature extractor (SB3 fallback)")

        common_kwargs = dict(
            learning_rate      = cfg.learning_rate,
            n_steps            = cfg.n_steps,
            batch_size         = cfg.batch_size,
            n_epochs           = cfg.n_epochs,
            gamma              = cfg.gamma,
            gae_lambda         = cfg.gae_lambda,
            clip_range         = cfg.clip_range,
            ent_coef           = cfg.ent_coef,
            vf_coef            = cfg.vf_coef,
            max_grad_norm      = cfg.max_grad_norm,
            normalize_advantage= cfg.normalize_advantage,
            policy_kwargs      = policy_kwargs,
            tensorboard_log    = cfg.log_dir if _TENSORBOARD_AVAILABLE else None,
            verbose            = cfg.verbose,
        )

        return AlgoClass(policy, train_env, **common_kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
    ) -> "RLTrainer":
        """
        Train the LSTM-PPO agent.

        Parameters
        ----------
        train_features : np.ndarray  (n_train_bars, n_features)
        val_features   : np.ndarray  (n_val_bars, n_features)

        Returns self for chaining.
        """
        cfg = self.cfg
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

        train_env = self._make_env(train_features)
        val_env   = self._make_env(val_features)

        self._train_env = train_env
        self._val_env   = val_env

        # Warm-start: resume from the last saved model so a stop/relaunch keeps
        # the knowledge from previous runs instead of training from scratch.
        resumed = False
        model_path = Path(cfg.model_dir) / f"{cfg.model_name}.zip"
        if cfg.warm_start and model_path.exists():
            try:
                AlgoClass = RecurrentPPO if (cfg.use_recurrent_ppo and _SB3_CONTRIB_AVAILABLE) else PPO
                self.model = AlgoClass.load(str(model_path), env=train_env)
                resumed = True
                logger.info(
                    "Warm-start — resumed from %s (%d timesteps so far)",
                    model_path, getattr(self.model, "num_timesteps", 0),
                )
            except Exception as exc:
                logger.warning(
                    "Warm-start failed (%s) — training a fresh model.", exc,
                )
                self.model = None

        if self.model is None:
            self.model = self._build_model(train_env)

        callbacks = CallbackList([
            EvalCallback(
                eval_env         = val_env,
                best_model_save_path = cfg.model_dir,
                log_path         = cfg.log_dir,
                eval_freq        = cfg.eval_freq,
                n_eval_episodes  = cfg.n_eval_episodes,
                deterministic    = True,
                verbose          = cfg.verbose,
            ),
            CheckpointCallback(
                save_freq    = cfg.checkpoint_freq,
                save_path    = cfg.model_dir,
                name_prefix  = cfg.model_name,
                verbose      = cfg.verbose,
            ),
            RewardTrackingCallback(verbose=cfg.verbose, progress_path=cfg.progress_path),
        ])

        logger.info("Training started — total_timesteps=%d (resumed=%s)",
                    cfg.total_timesteps, resumed)
        self.model.learn(
            total_timesteps = cfg.total_timesteps,
            callback        = callbacks,
            reset_num_timesteps = not resumed,
        )
        logger.info("Training complete.")
        # Persist the up-to-date model so the next run can resume from it
        # and live inference (RLAlpha) picks up the latest weights.
        self.save()
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_features: np.ndarray,
        n_episodes: int = 10,
    ) -> dict[str, float]:
        """
        Evaluate the trained model on test data.

        Returns
        -------
        dict with mean_reward, std_reward, mean_episode_len, win_rate
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call .train() first.")

        test_env = self._make_env(test_features)

        all_rewards = []
        all_lengths = []

        for _ in range(n_episodes):
            obs = test_env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            ep_reward, ep_len = 0.0, 0
            done = False

            while not done:
                if _SB3_CONTRIB_AVAILABLE and hasattr(self.model, "predict"):
                    action, lstm_states = self.model.predict(
                        obs,
                        state          = lstm_states,
                        episode_start  = episode_starts,
                        deterministic  = True,
                    )
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                obs, reward, done_arr, _ = test_env.step(action)
                done = bool(done_arr[0])
                ep_reward += float(reward[0])
                ep_len    += 1

            all_rewards.append(ep_reward)
            all_lengths.append(ep_len)

        results = {
            "mean_reward":     float(np.mean(all_rewards)),
            "std_reward":      float(np.std(all_rewards)),
            "mean_episode_len": float(np.mean(all_lengths)),
            "n_episodes":      n_episodes,
        }
        logger.info("Evaluation — %s", results)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Save model to disk. Returns the saved path."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        save_path = Path(path or f"{self.cfg.model_dir}/{self.cfg.model_name}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info("Model saved → %s.zip", save_path)
        return save_path

    def load(self, path: str | Path) -> "RLTrainer":
        """Load a previously saved model."""
        path = Path(path)
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"Model not found: {path}")

        algo_class = RecurrentPPO if _SB3_CONTRIB_AVAILABLE else PPO
        self.model = algo_class.load(str(path))
        logger.info("Model loaded ← %s", path)
        return self

    # ------------------------------------------------------------------
    # High-level convenience
    # ------------------------------------------------------------------

    def fit(
        self,
        custom_data: Optional[pd.DataFrame] = None,
    ) -> "RLAlpha":
        """
        Full pipeline: load data → train → save → return RLAlpha.

        Returns
        -------
        RLAlpha  — inference-ready AlphaModel
        """
        train_f, val_f, _ = self.prepare_data(custom_data)
        self.train(train_f, val_f)
        model_path = self.save()

        # Import here to avoid circular imports at module level
        from strategies.rl_agent.rl_alpha import RLAlpha

        algo = "RecurrentPPO" if (_SB3_CONTRIB_AVAILABLE and self.cfg.use_recurrent_ppo) else "PPO"
        return RLAlpha(
            model_path   = model_path,
            algo         = algo,
            window_size  = self.cfg.window_size,
        )


# ---------------------------------------------------------------------------
# Quick-start function
# ---------------------------------------------------------------------------

def train_ppo_lstm(
    data: pd.DataFrame,
    symbol: str          = "EURUSD",
    timeframe: str       = "H1",
    total_timesteps: int = 500_000,
    **kwargs,
) -> "RLAlpha":
    """
    Convenience function: train and return a ready-to-use RLAlpha.

    Parameters
    ----------
    data            : pd.DataFrame  OHLCV + engineered features
    symbol          : str
    timeframe       : str
    total_timesteps : int
    **kwargs        : override any RLTrainerConfig field

    Returns
    -------
    RLAlpha  — integrates directly into BacktestEngine / LiveTrader

    Example
    -------
    >>> from strategies.rl_agent.rl_trainer import train_ppo_lstm
    >>> rl_signal = train_ppo_lstm(df, total_timesteps=200_000)
    >>> result = engine.run([rl_signal, momentum, mean_rev], df)
    """
    cfg = RLTrainerConfig(
        symbol=symbol,
        timeframe=timeframe,
        total_timesteps=total_timesteps,
        **{k: v for k, v in kwargs.items() if hasattr(RLTrainerConfig, k)},
    )
    trainer = RLTrainer(cfg)
    return trainer.fit(custom_data=data)
