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
    """
    Weights for the composite reward function.

    The reward has two categories:

    EXTRINSIC (from the market — realised outcomes):
      pnl_reward      : normalised equity change per step
      drawdown_penalty: penalises drawdowns from the equity high-water mark
      risk_penalty    : penalises holding through high volatility
      sharpe_bonus    : rewards consistent risk-adjusted returns

    INTRINSIC (from the agent itself — shape exploration behaviour):
      anti_hold_penalty: penalises excessive inaction when the market moves.
                         Without this the agent learns HOLD=safe and never
                         takes BUY positions (root cause of 291 SELL / 0 BUY).
      entry_quality_bonus: rewards entering a position aligned with short-term
                           momentum (buy when recent return > 0, sell when < 0).
                           Intrinsic because it fires at entry, before any P&L.
      novelty_bonus   : small reward for acting in high-volatility states the
                        agent has rarely visited (exploration drive).
    """

    # ---- Extrinsic weights -----------------------------------------------
    alpha_drawdown:   float = 2.0   # drawdown penalty coefficient
    alpha_risk:       float = 0.5   # volatility-adjusted exposure penalty
    alpha_sharpe:     float = 0.3   # Sharpe component weight
    sharpe_window:    int   = 20    # bars for rolling Sharpe estimate
    pnl_scale:        float = 100.0 # scale P&L component (keeps reward ~[-1, 1])

    # ---- Intrinsic weights -----------------------------------------------
    # Anti-HOLD: penalty per bar of idle position when market is moving.
    # Kicks in after `hold_threshold` consecutive HOLD bars.
    # Prevents the agent from learning that "do nothing = no loss = reward 0".
    alpha_anti_hold:  float = 0.05
    hold_threshold:   int   = 5     # bars of HOLD before penalty activates

    # Entry quality: bonus when opening a position aligned with short-term
    # momentum (log_return × direction > 0). Fired once at entry, not per-bar.
    alpha_entry:      float = 0.15

    # Novelty: small bonus for acting (non-HOLD) when market volatility is
    # above the session median — encourages exploration in uncertain states.
    alpha_novelty:    float = 0.08


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
        features: np.ndarray,            # (n_bars, n_features) — observation features only
        reward_cfg: RewardConfig,
        window_size: int        = 64,
        initial_balance: float  = 100_000.0,
        spread: float           = 0.00002,   # 0.2 pip ECN
        commission_pct: float   = 0.000035,  # ~3.5 USD/lot one-way
        sl_pct: float           = 0.02,
        tp_pct: float           = 0.04,
        position_size_pct: float = 0.02,
        prices: Optional[np.ndarray] = None,  # (n_bars,) actual close prices for PnL calc
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("gymnasium and stable-baselines3 are required.")

        self.features       = features.astype(np.float32)
        # prices: actual close prices used for PnL / SL / TP calculations.
        # Falls back to features[:, 0] when not provided (legacy behaviour),
        # but that only works correctly when the first feature column is a price.
        self._prices         = prices.astype(np.float32) if prices is not None else None
        self.reward_cfg      = reward_cfg
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.spread          = spread
        self.commission_pct  = commission_pct
        self.sl_pct          = sl_pct
        self.tp_pct          = tp_pct
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
        self._sl_price     = 0.0    # live stop-loss price
        self._tp_price     = 0.0    # live take-profit price
        self._peak_equity  = initial_balance
        self._equity       = initial_balance
        self._returns: list[float] = []
        # --- Intrinsic reward state ---
        self._hold_streak: int = 0         # consecutive HOLD bars (no position)
        self._prev_position: int = 0       # position at previous step
        self._vol_history: list[float] = []  # recent vol readings for median

    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._bar           = self.window_size
        self._balance       = self.initial_balance
        self._position      = 0
        self._entry_price   = 0.0
        self._sl_price      = 0.0
        self._tp_price      = 0.0
        self._peak_equity   = self.initial_balance
        self._equity        = self.initial_balance
        self._returns       = []
        self._hold_streak   = 0
        self._prev_position = 0
        self._vol_history   = []
        return self._observe(), {}

    def step(self, action: int):
        bar    = self._bar
        close  = self._get_close(bar)
        prev_equity   = self._equity
        prev_position = self._position   # capture BEFORE execute (for intrinsic)

        # ---- SL/TP auto-close (applied before agent's action) ----------
        sl_tp_closed = self._check_sl_tp(close)
        if sl_tp_closed:
            action = 1  # position already closed — treat as HOLD

        # ---- Execute action --------------------------------------------
        reward_components = self._execute(action, close)
        reward_components["sl_tp_closed"] = sl_tp_closed

        # ---- Advance bar -----------------------------------------------
        self._bar += 1
        terminated = self._bar >= len(self.features) - 1
        truncated  = False

        # ---- Compute composite reward (extrinsic + intrinsic) ----------
        reward = self._compute_reward(
            prev_equity, reward_components, close, terminated,
            action=action, prev_position=prev_position,
        )

        return self._observe(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _check_sl_tp(self, close: float) -> bool:
        """Auto-close the open position if SL or TP is triggered. Returns True if closed."""
        if self._position == 0 or self._entry_price <= 0:
            return False

        # For LONG: SL fires when close <= sl_price, TP when close >= tp_price
        # For SHORT: SL fires when close >= sl_price, TP when close <= tp_price
        size = self._get_size()
        if self._position == 1:
            if close <= self._sl_price or close >= self._tp_price:
                exec_price = close - self.spread
                pnl  = (exec_price - self._entry_price) * size
                pnl -= exec_price * size * self.commission_pct  # close commission
                self._balance    += pnl
                self._position    = 0
                self._entry_price = 0.0
                self._sl_price    = 0.0
                self._tp_price    = 0.0
                self._equity      = self._balance
                return True
        elif self._position == -1:
            if close >= self._sl_price or close <= self._tp_price:
                exec_price = close + self.spread
                pnl  = (self._entry_price - exec_price) * size
                pnl -= exec_price * size * self.commission_pct  # close commission
                self._balance    += pnl
                self._position    = 0
                self._entry_price = 0.0
                self._sl_price    = 0.0
                self._tp_price    = 0.0
                self._equity      = self._balance
                return True
        return False

    def _execute(self, action: int, close: float) -> dict:
        """Open / close / hold position. Returns reward components."""
        direction = action - 1          # 0→-1(SELL), 1→0(HOLD), 2→+1(BUY)
        spread    = self.spread

        pnl = 0.0
        if self._position != 0 and (action == 0 or direction != self._position):
            # Close by agent signal (spread + commission)
            exec_price = close - spread * self._position
            size       = self._get_size()
            raw_pnl    = (exec_price - self._entry_price) * self._position * size
            commission = exec_price * size * self.commission_pct
            pnl        = raw_pnl - commission
            self._balance += pnl
            self._position    = 0
            self._entry_price = 0.0
            self._sl_price    = 0.0
            self._tp_price    = 0.0

        if action != 1 and self._position == 0:
            # Open new position: pay spread + entry commission, set SL/TP
            exec_price        = close + spread * direction
            size              = self._get_size()
            open_commission   = exec_price * size * self.commission_pct
            self._balance    -= open_commission   # deduct immediately on open
            self._position    = direction
            self._entry_price = exec_price
            if direction == 1:
                self._sl_price = exec_price * (1.0 - self.sl_pct)
                self._tp_price = exec_price * (1.0 + self.tp_pct)
            else:
                self._sl_price = exec_price * (1.0 + self.sl_pct)
                self._tp_price = exec_price * (1.0 - self.tp_pct)

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
        price = self._entry_price if self._position != 0 and self._entry_price > 0 \
                else self._get_close(self._bar)
        return (self._equity * self.position_size_pct) / (price + 1e-10)

    def _get_close(self, bar: int) -> float:
        """Return the actual close price for bar `bar`.

        Uses the dedicated `_prices` array when provided (recommended).
        Falls back to features[:, 0] only when no price array was passed.
        """
        if self._prices is not None:
            idx = min(bar, len(self._prices) - 1)
            return float(self._prices[idx])
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
        action: int = 1,
        prev_position: int = 0,
    ) -> float:
        cfg = self.reward_cfg
        bar = max(0, self._bar - 1)  # bar just processed

        # ------------------------------------------------------------------
        # EXTRINSIC REWARDS (from the market)
        # ------------------------------------------------------------------

        # E1. P&L component — normalised equity change
        delta_equity = self._equity - prev_equity
        pnl_reward   = (delta_equity / (self.initial_balance + 1e-10)) * cfg.pnl_scale
        ret = delta_equity / (prev_equity + 1e-10)
        self._returns.append(ret)

        # E2. Drawdown penalty
        dd = max(0.0, (self._peak_equity - self._equity) / (self._peak_equity + 1e-10))
        drawdown_penalty = cfg.alpha_drawdown * dd

        # E3. Volatility-adjusted exposure penalty
        recent_slice = self.features[max(0, bar - 20): bar + 1, :]
        vol_proxy    = float(np.std(recent_slice[:, 0])) if len(recent_slice) > 1 else 0.0
        risk_penalty = cfg.alpha_risk * abs(self._position) * vol_proxy * cfg.pnl_scale

        # E4. Sharpe bonus — rolling risk-adjusted return
        sharpe_bonus = 0.0
        if len(self._returns) >= cfg.sharpe_window:
            window_rets = np.array(self._returns[-cfg.sharpe_window:])
            mean_r = float(np.mean(window_rets))
            std_r  = float(np.std(window_rets))
            sharpe_bonus = cfg.alpha_sharpe * (mean_r / (std_r + 1e-8))

        # E5. Terminal penalty — deep drawdown = failed episode
        terminal_penalty = 10.0 if (done and dd > 0.25) else 0.0

        # ------------------------------------------------------------------
        # INTRINSIC REWARDS (from the agent — shape exploration behaviour)
        # ------------------------------------------------------------------

        # Track vol history for novelty detection (rolling median)
        self._vol_history.append(vol_proxy)
        if len(self._vol_history) > 100:
            self._vol_history.pop(0)

        # I1. Anti-HOLD penalty
        # Without this the agent defaults to HOLD=0 loss, learning to never
        # trade. Penalty grows with consecutive HOLD bars when market moves.
        if self._position == 0 and action == 1:  # idle (HOLD, no position)
            self._hold_streak += 1
        else:
            self._hold_streak = 0

        anti_hold_penalty = 0.0
        if self._hold_streak > cfg.hold_threshold and vol_proxy > 1e-5:
            # Penalty proportional to how long the agent has been idle and
            # how much the market is moving (vol_proxy ≈ std of log_returns).
            idle_excess      = self._hold_streak - cfg.hold_threshold
            anti_hold_penalty = cfg.alpha_anti_hold * vol_proxy * min(idle_excess, 20)

        # I2. Entry quality bonus
        # Fires ONCE when the agent opens a new position.
        # Rewards momentum alignment: BUY when log_return > 0, SELL when < 0.
        # This is intrinsic: it fires before any P&L is realised.
        entry_bonus = 0.0
        just_opened = (prev_position == 0 and self._position != 0)
        if just_opened and bar < len(self.features):
            log_ret   = float(self.features[bar, 0])  # col 0 = log_return
            alignment = log_ret * self._position       # +1 = aligned with momentum
            if alignment > 0:
                entry_bonus = cfg.alpha_entry * abs(alignment) * 100

        # I3. Novelty bonus
        # Small reward for taking action (non-HOLD) when the current volatility
        # is above the agent's recent median — encourages exploring unusual states.
        novelty_bonus = 0.0
        if action != 1 and len(self._vol_history) >= 10:
            median_vol = float(np.median(self._vol_history))
            if vol_proxy > median_vol * 1.2:   # 20% above median = "unusual"
                novelty_bonus = cfg.alpha_novelty * (vol_proxy - median_vol)

        # ------------------------------------------------------------------
        # Composite reward
        # ------------------------------------------------------------------
        reward = (
            pnl_reward
            - drawdown_penalty
            - risk_penalty
            + sharpe_bonus
            - terminal_penalty
            # intrinsic
            - anti_hold_penalty
            + entry_bonus
            + novelty_bonus
        )

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
    feature_store_root: str  = "data/storage/features"
    symbol: str              = "EURUSD"
    timeframe: str           = "H1"
    feature_version: str     = "v1"
    # Optional hard floor for historical bars. 0 means "no extra floor"
    # beyond the structural minimum implied by the window/splits.
    min_history_bars: int    = 0

    # ---- Data splits (fraction of total) --------------------------------
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test is the remainder

    # ---- Environment ----------------------------------------------------
    window_size: int          = 64       # 64h ≈ 3 trading days context (was 32)
    initial_balance: float    = 100_000.0
    spread: float             = 0.00002  # 0.2 pip — realistic ECN spread (was 0.0001)
    commission_pct: float     = 0.000035 # ~3.5 USD/lot one-way commission
    sl_pct: float             = 0.02
    tp_pct: float             = 0.04
    position_size_pct: float  = 0.02     # 2% per trade (was 10% — dangerously high)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # ---- LSTM architecture ----------------------------------------------
    lstm_hidden: int     = 256
    n_lstm_layers: int   = 2
    encoder_dim: int     = 128
    net_arch: list       = field(default_factory=lambda: [dict(pi=[64, 64], vf=[64, 64])])

    # ---- PPO hyperparameters --------------------------------------------
    use_recurrent_ppo: bool   = True
    learning_rate: float      = 2e-4    # slightly lower for stability
    n_steps: int              = 1024    # longer rollout = better value estimates (was 512)
    batch_size: int           = 64
    n_epochs: int             = 10
    gamma: float              = 0.99
    gae_lambda: float         = 0.95
    clip_range: float         = 0.2
    ent_coef: float           = 0.03    # higher entropy = more exploration (was 0.01)
    vf_coef: float            = 0.5
    max_grad_norm: float      = 0.5
    normalize_advantage: bool = True

    # ---- Training schedule ----------------------------------------------
    total_timesteps: int      = 3_000_000   # 3M steps (was 500k — far too few)
    eval_freq: int            = 25_000
    n_eval_episodes: int      = 5
    checkpoint_freq: int      = 100_000

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

    def __init__(self, verbose: int = 0, progress_path: str | None = None,
                 progress_callback=None) -> None:
        if _SB3_AVAILABLE:
            super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0
        self._progress_path = Path(progress_path) if progress_path else None
        self._progress_callback = progress_callback   # callable(timesteps_done)
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
            ep_num   = len(self.episode_rewards) + 1
            last_r   = self._ep_reward
            self.episode_rewards.append(last_r)
            recent       = self.episode_rewards[-10:]
            mean_recent  = float(np.mean(recent))
            steps        = int(getattr(self, "num_timesteps", 0))
            self._write_progress(last_r, mean_recent)
            # Fire external progress callback every completed episode.
            if self._progress_callback is not None:
                try:
                    self._progress_callback(steps)
                except Exception:
                    pass
            # Print a live progress line every 5 episodes so the operator
            # can see reward trending upward (or flag a problem) in real time.
            if ep_num % 5 == 0:
                trend = "↑" if len(self.episode_rewards) >= 2 and last_r > self.episode_rewards[-2] else "↓"
                print(
                    f"  [RL] Ep {ep_num:>5d} | "
                    f"récompense={last_r:+.4f} {trend} | "
                    f"moy(10)={mean_recent:+.4f} | "
                    f"steps={steps:>8,}",
                    flush=True,
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
        self._progress_callback = None

    def set_progress_callback(self, fn) -> None:
        """Register a callable(timesteps_done) called during training."""
        self._progress_callback = fn

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

        close_prices: Optional[np.ndarray] = None

        if custom_data is not None:
            numeric = custom_data.select_dtypes(include=[np.number])
            if "close" in numeric.columns:
                close_prices = numeric["close"].values.astype(np.float32)
            features_df = numeric.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            # Remove raw OHLCV to avoid price-level leakage into the agent's obs
            drop_cols = {"open", "high", "low", "close", "volume"}
            obs_cols = [c for c in features_df.columns if c not in drop_cols]
            features = features_df[obs_cols].values.astype(np.float32)
            logger.info("Using custom data: %d bars × %d features", *features.shape)
        else:
            store = FeatureStore(root=cfg.feature_store_root)
            try:
                df = store.load(
                    symbol=cfg.symbol,
                    timeframe=cfg.timeframe,
                    version=cfg.feature_version,
                )
            except FileNotFoundError:
                logger.warning(
                    "FeatureStore miss for %s/%s/%s — rebuilding from MT5.",
                    cfg.symbol,
                    cfg.timeframe,
                    cfg.feature_version,
                )
                df = store.refresh_from_market_data(
                    symbol=cfg.symbol,
                    timeframe=cfg.timeframe,
                    n_bars=max(cfg.min_history_bars, cfg.window_size * 40),
                    version=cfg.feature_version,
                    overwrite=True,
                )
            # Separate close price BEFORE dropping OHLCV — used for PnL/SL/TP
            if "close" in df.columns:
                close_prices = df["close"].values.astype(np.float32)
            # Keep only engineered features in the agent's observation
            drop_cols = {"open", "high", "low", "close", "volume"}
            feature_cols = [c for c in df.columns if c not in drop_cols]
            features_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            features = features_df.values.astype(np.float32)
            logger.info(
                "Loaded %d bars × %d features from FeatureStore",
                *features.shape,
            )

        if not np.isfinite(features).all():
            raise ValueError("Non-finite values found in RL feature matrix.")

        # Structural minimum: enough bars for a proper train/val/test split where
        # each split has at least window_size+2 usable steps.
        # window_size*4 ensures: 70% train ≥ window_size, 15% val ≥ window_size.
        min_required = max(cfg.window_size * 4, int(cfg.min_history_bars))
        if len(features) < min_required:
            raise ValueError(
                f"RL history too short: got {len(features)} bars, need at least "
                f"{min_required} (window_size={cfg.window_size} × 4). "
                f"Fix: set TRADING_RL_TIMEFRAME=H1 in .env so RL trains on H1 bars "
                f"(years of history), not {cfg.timeframe} (few hundred bars)."
            )

        n = len(features)
        n_train = int(n * cfg.train_frac)
        n_val   = int(n * cfg.val_frac)

        min_split = cfg.window_size + 2
        if n_train <= min_split or n_val <= min_split:
            # Adaptive fallback for limited history: preserve a usable
            # validation set and allow retraining instead of hard-failing.
            n_train = max(min_split + 1, int(n * 0.80))
            n_val = n - n_train
            if n_val <= min_split:
                need = (min_split + 1) * 2
                raise ValueError(
                    f"RL history too short for window={cfg.window_size}: "
                    f"got {n} bars, need at least {need}."
                )

        train = features[:n_train]
        val   = features[n_train: n_train + n_val]
        test  = features[n_train + n_val:]

        train_prices = val_prices = test_prices = None
        if close_prices is not None:
            train_prices = close_prices[:n_train]
            val_prices   = close_prices[n_train: n_train + n_val]
            test_prices  = close_prices[n_train + n_val:]

        logger.info(
            "Split — train=%d  val=%d  test=%d  (prices=%s)",
            len(train), len(val), len(test),
            "yes" if close_prices is not None else "no",
        )
        return train, val, test, train_prices, val_prices, test_prices

    # ------------------------------------------------------------------
    # Environment builders
    # ------------------------------------------------------------------

    def _make_env(
        self,
        features: np.ndarray,
        seed: int = 0,
        prices: Optional[np.ndarray] = None,
    ):
        cfg = self.cfg

        def _init():
            env = RLTradingEnv(
                features          = features,
                reward_cfg        = cfg.reward_config,
                window_size       = cfg.window_size,
                initial_balance   = cfg.initial_balance,
                spread            = cfg.spread,
                commission_pct    = cfg.commission_pct,
                sl_pct            = cfg.sl_pct,
                tp_pct            = cfg.tp_pct,
                position_size_pct = cfg.position_size_pct,
                prices            = prices,
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
        train_prices: Optional[np.ndarray] = None,
        val_prices: Optional[np.ndarray] = None,
    ) -> "RLTrainer":
        """
        Train the LSTM-PPO agent.

        Parameters
        ----------
        train_features : np.ndarray  (n_train_bars, n_features)  — observation features only
        val_features   : np.ndarray  (n_val_bars, n_features)
        train_prices   : np.ndarray or None  (n_train_bars,) — actual close prices for PnL calc
        val_prices     : np.ndarray or None  (n_val_bars,)

        Returns self for chaining.
        """
        cfg = self.cfg
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

        train_env = self._make_env(train_features, prices=train_prices)
        val_env   = self._make_env(val_features, prices=val_prices)

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
            RewardTrackingCallback(verbose=cfg.verbose, progress_path=cfg.progress_path,
                                   progress_callback=self._progress_callback),
        ])

        algo_name = "RecurrentPPO" if (cfg.use_recurrent_ppo and _SB3_CONTRIB_AVAILABLE) else "PPO"
        print(
            f"\n{'─'*58}\n"
            f"  RL AGENT ({algo_name}) — ENTRAÎNEMENT\n"
            f"{'─'*58}\n"
            f"  Timesteps total  : {cfg.total_timesteps:>10,}\n"
            f"  Bars train / val : {len(train_features):>7,} / {len(val_features):,}\n"
            f"  Warm-start       : {'OUI (reprise)' if resumed else 'NON (fresh)'}\n"
            f"{'─'*58}",
            flush=True,
        )
        logger.info("Training started — total_timesteps=%d (resumed=%s)",
                    cfg.total_timesteps, resumed)
        self.model.learn(
            total_timesteps = cfg.total_timesteps,
            callback        = callbacks,
            reset_num_timesteps = not resumed,
        )

        # Final summary after training
        reward_cb = callbacks.callbacks[-1]   # RewardTrackingCallback is last
        all_ep    = getattr(reward_cb, "episode_rewards", [])
        if all_ep:
            mean_last20 = float(np.mean(all_ep[-20:]))
            mean_first5 = float(np.mean(all_ep[:5])) if len(all_ep) >= 5 else all_ep[0]
            improvement = mean_last20 - mean_first5
            trend_sym   = "↑ AMÉLIORATION" if improvement > 0 else "↓ DÉTÉRIORATION"
            print(
                f"\n{'═'*58}\n"
                f"  RL AGENT — RÉSULTATS FINAUX\n"
                f"{'═'*58}\n"
                f"  Épisodes complétés   : {len(all_ep):>6,}\n"
                f"  Récompense moy(20)   : {mean_last20:>+10.4f}\n"
                f"  Récompense initiale  : {mean_first5:>+10.4f}\n"
                f"  Progression          : {improvement:>+10.4f}  {trend_sym}\n"
                f"  Modèle sauvegardé → {cfg.model_dir}/{cfg.model_name}.zip\n"
                f"{'═'*58}\n",
                flush=True,
            )
        else:
            print(f"\n[RL] Entraînement terminé ({cfg.total_timesteps:,} steps).\n", flush=True)

        logger.info("Training complete.")
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

        mean_r = float(np.mean(all_rewards))
        std_r  = float(np.std(all_rewards))
        mean_l = float(np.mean(all_lengths))
        results = {
            "mean_reward":      mean_r,
            "std_reward":       std_r,
            "mean_episode_len": mean_l,
            "n_episodes":       n_episodes,
        }
        verdict = "✓ BON" if mean_r > 0 else "✗ NÉGATIF — continuer l'entraînement"
        print(
            f"\n{'─'*58}\n"
            f"  RL AGENT — ÉVALUATION ({n_episodes} épisodes test)\n"
            f"{'─'*58}\n"
            f"  Récompense moyenne   : {mean_r:>+10.4f}  {verdict}\n"
            f"  Écart-type           : {std_r:>10.4f}\n"
            f"  Durée moy. épisode   : {mean_l:>10.0f} bars\n"
            f"{'─'*58}\n",
            flush=True,
        )
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


# ---------------------------------------------------------------------------
# Episode replay (for environment visualisation)
# ---------------------------------------------------------------------------

def replay_episode(
    features: np.ndarray,
    model_path: str,
    *,
    algo: str = "RecurrentPPO",
    window_size: int = 32,
    reward_cfg: Optional["RewardConfig"] = None,
    initial_balance: float = 100_000.0,
    spread: float = 0.0001,
    sl_pct: float = 0.02,
    tp_pct: float = 0.04,
    position_size_pct: float = 0.10,
    close_prices: Optional[np.ndarray] = None,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
) -> list[dict]:
    """Run one full episode of a trained agent in :class:`RLTradingEnv`,
    recording the trajectory for visualisation.

    Each recorded point holds the bar index, price (the real close when
    ``close_prices`` is given, else the env's close proxy), the action taken
    (0=HOLD, 1=BUY, 2=SELL), the resulting position (-1/0/+1), equity and the
    step reward. Returns the trajectory as a list of dicts.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("gymnasium and stable-baselines3 are required.")

    env = RLTradingEnv(
        features          = features,
        reward_cfg        = reward_cfg or RewardConfig(),
        window_size       = window_size,
        initial_balance   = initial_balance,
        spread            = spread,
        sl_pct            = sl_pct,
        tp_pct            = tp_pct,
        position_size_pct = position_size_pct,
    )

    AlgoClass = RecurrentPPO if (algo == "RecurrentPPO" and _SB3_CONTRIB_AVAILABLE) else PPO
    model = AlgoClass.load(str(model_path))

    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    trajectory: list[dict] = []
    done = False
    step_i = 0

    while not done:
        action, lstm_states = model.predict(
            obs,
            state         = lstm_states,
            episode_start = episode_starts,
            deterministic = deterministic,
        )
        episode_starts = np.zeros((1,), dtype=bool)
        act = int(np.asarray(action).reshape(-1)[0])

        bar = int(env._bar)
        obs, reward, terminated, truncated, _ = env.step(act)
        done = bool(terminated or truncated)

        if close_prices is not None and 0 <= bar < len(close_prices):
            price = float(close_prices[bar])
        else:
            price = float(env._get_close(bar))

        trajectory.append({
            "step":     step_i,
            "bar":      bar,
            "price":    price,
            "action":   act,
            "position": int(env._position),
            "equity":   float(env._equity),
            "reward":   float(reward),
        })

        step_i += 1
        if max_steps is not None and step_i >= max_steps:
            break

    return trajectory

