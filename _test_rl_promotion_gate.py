#!/usr/bin/env python3
"""
Test script for the RL retrain promotion gate (Phase C, RL side).

Before this fix, RetrainService._retrain_rl() -> RLTrainer.train() always
called self.save() unconditionally, overwriting the deployed
model_dir/model_name.zip (the exact file RLAlpha hot-reloads into the live
trader) even when the freshly retrained candidate was worse.

Two parts:
  1. Deterministic table test of the pure decision function
     _should_promote_rl() (no baseline -> always promote; small
     fluctuations within the noise margin -> promote; clear regression ->
     reject).
  2. A real (tiny) end-to-end smoke run through RetrainService._retrain_rl()
     with a synthetic feature series and a handful of timesteps, just to
     confirm the backup/evaluate/restore/cleanup plumbing doesn't crash and
     leaves the model directory in a consistent state (no leftover
     .pre_retrain_backup.zip, model file always loadable afterward).
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.training.retrain_service import RetrainService, _should_promote_rl

PASS = "✓"
FAIL = "✗"
all_ok = True

# ---------------------------------------------------------------------------
print("=" * 70)
print("PART 1: _should_promote_rl() decision table")
print("=" * 70)

cases = [
    # (baseline, candidate, expected_promoted, label)
    (None,  1.0,  True,  "no baseline (nothing deployed yet) -> always promote"),
    (1.0,   None, True,  "candidate eval failed -> fail open, promote"),
    (2.0,   2.4,  True,  "candidate clearly better -> promote"),
    (2.0,   1.9,  True,  "tiny dip within noise margin -> promote"),
    (2.0,   0.5,  False, "clear regression beyond margin -> reject"),
    (-1.0, -3.0,  False, "negative baseline, candidate much worse -> reject"),
    (-1.0, -1.1,  True,  "negative baseline, small dip within margin -> promote"),
]

for baseline, candidate, expected, label in cases:
    got = _should_promote_rl(baseline, candidate)
    ok = got == expected
    print(f"  {PASS if ok else FAIL} {label}: baseline={baseline} candidate={candidate} "
          f"-> promoted={got} (expected {expected})")
    all_ok &= ok

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: end-to-end smoke run through RetrainService._retrain_rl()")
print("=" * 70)

tmpdir = Path(tempfile.mkdtemp(prefix="rl_promotion_test_"))
try:
    rng = np.random.default_rng(7)
    n_bars = 1200
    close = 1.10 + np.cumsum(rng.normal(0, 0.0003, n_bars))
    df = pd.DataFrame({
        "open":   close + rng.normal(0, 0.0001, n_bars),
        "high":   close + np.abs(rng.normal(0, 0.0002, n_bars)),
        "low":    close - np.abs(rng.normal(0, 0.0002, n_bars)),
        "close":  close,
        "volume": rng.integers(100, 1000, n_bars).astype(float),
        "feat_a": rng.normal(0, 1, n_bars),
        "feat_b": np.sin(np.linspace(0, 40, n_bars)),
    })

    svc = RetrainService(dataset_dir=str(tmpdir / "datasets"), model_dir=str(tmpdir / "models"))

    # Monkeypatch RLTrainerConfig defaults via kwargs isn't exposed by
    # _retrain_rl's signature, so patch the tiny-training knobs directly on
    # the RLTrainerConfig class default via a temporary override.
    from strategies.rl_agent import rl_trainer as _rl_trainer_mod
    _orig_defaults = dict(
        window_size=_rl_trainer_mod.RLTrainerConfig.window_size,
        n_steps=_rl_trainer_mod.RLTrainerConfig.n_steps,
        batch_size=_rl_trainer_mod.RLTrainerConfig.batch_size,
        n_epochs=_rl_trainer_mod.RLTrainerConfig.n_epochs,
        eval_freq=_rl_trainer_mod.RLTrainerConfig.eval_freq,
        checkpoint_freq=_rl_trainer_mod.RLTrainerConfig.checkpoint_freq,
        min_history_bars=_rl_trainer_mod.RLTrainerConfig.min_history_bars,
    )
    _rl_trainer_mod.RLTrainerConfig.window_size = 8
    _rl_trainer_mod.RLTrainerConfig.n_steps = 64
    _rl_trainer_mod.RLTrainerConfig.batch_size = 32
    _rl_trainer_mod.RLTrainerConfig.n_epochs = 2
    _rl_trainer_mod.RLTrainerConfig.eval_freq = 10_000_000  # skip mid-train eval saves
    _rl_trainer_mod.RLTrainerConfig.checkpoint_freq = 10_000_000
    _rl_trainer_mod.RLTrainerConfig.min_history_bars = 0

    try:
        print("  -- run 1 (fresh, nothing deployed yet) --")
        r1 = svc._retrain_rl(
            timesteps=128, symbol="EURUSD", timeframe="H1",
            feature_provider=lambda: df,
        )
        model_path = Path(svc.model_dir) / "ppo_lstm_trading.zip"
        backup_path = model_path.with_suffix(".pre_retrain_backup.zip")
        ok = (r1.get("trained") is True and r1.get("promoted") is True
              and r1.get("deployed_eval_mean_reward") is None
              and model_path.exists() and not backup_path.exists())
        print(f"  {PASS if ok else FAIL} run1: trained={r1.get('trained')} "
              f"promoted={r1.get('promoted')} deployed_eval={r1.get('deployed_eval_mean_reward')} "
              f"eval_mean_reward={r1.get('eval_mean_reward')} model_exists={model_path.exists()} "
              f"no_leftover_backup={not backup_path.exists()}")
        all_ok &= ok

        print("  -- run 2 (retrain against a now-deployed baseline) --")
        r2 = svc._retrain_rl(
            timesteps=128, symbol="EURUSD", timeframe="H1",
            feature_provider=lambda: df,
        )
        ok2 = (r2.get("trained") is True
               and r2.get("deployed_eval_mean_reward") is not None
               and isinstance(r2.get("promoted"), bool)
               and model_path.exists() and not backup_path.exists())
        print(f"  {PASS if ok2 else FAIL} run2: trained={r2.get('trained')} "
              f"promoted={r2.get('promoted')} deployed_eval={r2.get('deployed_eval_mean_reward')} "
              f"eval_mean_reward={r2.get('eval_mean_reward')} model_exists={model_path.exists()} "
              f"no_leftover_backup={not backup_path.exists()}")
        print(f"    message: {r2.get('message')}")
        all_ok &= ok2

        # The saved file must still be a loadable SB3 model either way.
        from strategies.rl_agent.rl_trainer import RLTrainer, RLTrainerConfig
        reload_check = RLTrainer(RLTrainerConfig(model_dir=svc.model_dir))
        reload_check.load(str(model_path))
        ok3 = reload_check.model is not None
        print(f"  {PASS if ok3 else FAIL} model file is loadable after run2")
        all_ok &= ok3

    finally:
        for k, v in _orig_defaults.items():
            setattr(_rl_trainer_mod.RLTrainerConfig, k, v)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
if all_ok:
    print(f"{PASS} All RL promotion-gate checks passed.")
else:
    print(f"{FAIL} Some checks failed — see above.")
    raise SystemExit(1)
