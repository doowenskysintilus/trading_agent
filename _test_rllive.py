import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from strategies.rl_agent.rl_trainer import RLTrainer, RLTrainerConfig

mdir = "data/storage/models"
model_zip = Path(mdir) / "ppo_lstm_trading.zip"
for p in Path(mdir).glob("ppo_lstm_trading*"):
    p.unlink(missing_ok=True)

rng = np.random.default_rng(0)
n = 300
close = 100 + np.cumsum(rng.normal(0, 0.5, n))
df = pd.DataFrame({
    "open": close, "high": close + 0.5, "low": close - 0.5,
    "close": close, "volume": rng.uniform(1, 5, n),
    "feat1": rng.normal(size=n), "feat2": rng.normal(size=n),
})

cfg = RLTrainerConfig(
    window_size=8, n_steps=64, batch_size=32, n_epochs=2,
    total_timesteps=128, eval_freq=1000, checkpoint_freq=10000,
    model_dir=mdir, log_dir=mdir + "/_tblogs", verbose=0,
)
trainer = RLTrainer(cfg)
tr, val, _ = trainer.prepare_data(custom_data=df)
trainer.train(tr, val)
assert model_zip.exists()
print("RL trained OK")

from strategies.rl_agent.rl_alpha import RLAlpha
alpha = RLAlpha(model_path=str(model_zip), algo="RecurrentPPO", window_size=8)
feat = df[["feat1", "feat2", "close", "high", "low", "open", "volume"]]
sig = alpha.compute(feat)
print("RLAlpha signal:", sig.signal.name, "conf", round(sig.confidence, 3))
assert sig.strategy_name == "rl_agent"

from fastapi.testclient import TestClient
from api.main import create_app
app = create_app()
with TestClient(app) as c:
    pass
from api.dependencies import get_app_state
state = get_app_state()
print("strategies after startup:", list(state.strategies.keys()))
assert "rl_agent" in state.strategies, "RL agent not auto-registered"

print("RL LIVE WIRING OK")

shutil.rmtree(mdir + "/_tblogs", ignore_errors=True)
for p in Path(mdir).glob("ppo_lstm_trading*"):
    p.unlink(missing_ok=True)
