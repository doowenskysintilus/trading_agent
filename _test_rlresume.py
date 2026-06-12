import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from strategies.rl_agent.rl_trainer import RLTrainer, RLTrainerConfig

mdir = "data/storage/models/_test_rl"
shutil.rmtree(mdir, ignore_errors=True)

rng = np.random.default_rng(0)
n = 400
close = 100 + np.cumsum(rng.normal(0, 0.5, n))
df = pd.DataFrame({
    "open": close, "high": close + 0.5, "low": close - 0.5,
    "close": close, "volume": rng.uniform(1, 5, n),
    "feat1": rng.normal(size=n), "feat2": rng.normal(size=n),
})

cfg = RLTrainerConfig(
    window_size=8, n_steps=64, batch_size=32, n_epochs=2,
    total_timesteps=128, eval_freq=1000, checkpoint_freq=10000,
    model_dir=mdir, log_dir=mdir + "/logs", verbose=0,
    warm_start=True,
)

t1 = RLTrainer(cfg)
tr, val, _ = t1.prepare_data(custom_data=df)
t1.train(tr, val)
ts1 = int(getattr(t1.model, "num_timesteps", 0))
zip_path = Path(mdir) / "ppo_lstm_trading.zip"
print("run1 num_timesteps:", ts1, "| saved:", zip_path.exists())
assert zip_path.exists()

t2 = RLTrainer(cfg)
tr2, val2, _ = t2.prepare_data(custom_data=df)
t2.train(tr2, val2)
ts2 = int(getattr(t2.model, "num_timesteps", 0))
print("run2 num_timesteps:", ts2)
assert ts2 > ts1, f"warm-start did not accumulate ({ts2} <= {ts1})"

print("RL WARM-START OK")
shutil.rmtree(mdir, ignore_errors=True)
