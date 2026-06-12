# Quant Fund AI — Institutional-Grade Algorithmic Trading System

A full-stack quantitative trading system combining multi-strategy alpha generation,
reinforcement learning, real-time execution, risk management, and live dashboard monitoring.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           quant-fund-ai                                     │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────────────┐ │
│  │  Research     │   │  Strategies  │   │  Engines                        │ │
│  │  ─────────── │   │  ─────────── │   │  ──────────────────────────     │ │
│  │  FeatureStore │──▶│  Momentum    │   │  SignalAggregator               │ │
│  │  FeatureEngineer  │  MeanReversion│──▶│  RiskEngine (6-layer)          │ │
│  │  AlphaModel  │   │  RLAlpha     │   │  PortfolioAllocator (5 methods) │ │
│  │  (ABC)       │   │  Ensemble    │   │  ExecutionEngine (MT5)          │ │
│  └──────────────┘   └──────────────┘   └─────────────────────────────────┘ │
│           │                 │                          │                    │
│           └─────────────────┴──────────────────────────▼                   │
│                                                 ┌────────────┐             │
│                                                 │ LiveTrader │             │
│                                                 │ (8-stage   │             │
│                                                 │  pipeline) │             │
│                                                 └─────┬──────┘             │
│                    ┌─────────────────────────────────┘                     │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Monitoring                                                         │   │
│  │  TradingMonitor ──▶ MetricsStore ──▶ MySQL (primary) + JSONL (audit)│   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  WebSocket broadcast_sync ──▶ FastAPI /ws ──▶ React Dashboard      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────┐   ┌────────────────────────────────────────┐  │
│  │  Trading API (port 8000) │   │  Dashboard (port 3000)                │  │
│  │  /trading /portfolio     │   │  PnL Chart · Strategy Comparison      │  │
│  │  /strategies /backtest   │   │  Positions · Risk Status · Trade Feed │  │
│  │  /trades /risk           │   │  WebSocket real-time updates (2s)     │  │
│  └─────────────────────────┘   └────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
quant-fund-ai/
├── api/                         # Trading system REST API + WebSocket
│   ├── main.py                  # FastAPI app factory, all routers
│   ├── ws.py                    # WebSocket connection manager
│   ├── auth.py                  # API key auth + rate limiter
│   ├── schemas.py               # Pydantic request/response models
│   └── dependencies.py          # AppState singleton + DI factories
│
├── monitoring/                  # Observability layer
│   ├── monitor.py               # TradingMonitor — LiveTrader hook
│   ├── metrics_store.py         # MySQL + JSONL dual-write persistence
│   └── api.py                   # Monitoring-only FastAPI sub-app
│
├── live_trading/                # Live execution
│   ├── live_trader.py           # 8-stage trading loop orchestrator
│   └── data_feed.py             # MT5 OHLCV + streaming market data
│
├── engines/
│   ├── signal_engine/
│   │   ├── signal_engine.py     # Weighted signal voting
│   │   └── signal_aggregator.py # PerformanceWeighter + ConflictResolver
│   ├── risk_engine/
│   │   └── risk_engine.py       # 6-layer risk evaluation
│   ├── portfolio_engine/
│   │   └── portfolio_engine.py  # 5 allocation methods (Kelly, Risk Parity…)
│   └── execution_engine/
│       └── execution_engine.py  # MT5 order management + trailing stops
│
├── strategies/
│   ├── momentum/
│   │   └── momentum_alpha.py    # EMA crossover + RSI filter
│   ├── mean_reversion/
│   │   └── mean_reversion_alpha.py  # Bollinger Bands + Z-score
│   ├── rl_agent/
│   │   ├── rl_alpha.py          # Inference wrapper (stateful LSTM)
│   │   ├── rl_trainer.py        # PPO-LSTM training pipeline
│   │   └── ppo_lstm_policy.py   # Custom LSTM actor-critic architecture
│   └── ensemble/                # (planned) Combined strategy
│
├── backtesting/
│   ├── backtest_engine.py       # Parallel multi-strategy backtester
│   └── metrics.py               # Institutional performance statistics
│
├── research/
│   ├── alpha_models/
│   │   └── base.py              # AlphaModel ABC + SignalType + AlphaSignal
│   └── feature_store/
│       ├── feature_engineer.py  # 20+ technical indicators
│       └── feature_store.py     # Versioned Parquet cache
│
├── dashboard/                   # React real-time dashboard
│   ├── src/
│   │   ├── App.jsx              # WS orchestrator + layout
│   │   ├── hooks/useWebSocket.js # Auto-reconnect WS hook
│   │   ├── components/
│   │   │   ├── PnLChart.jsx     # Area chart with custom tooltip
│   │   │   ├── StrategyComparison.jsx  # Bar chart + stats table
│   │   │   ├── PositionsTable.jsx      # Open positions table
│   │   │   ├── RiskStatus.jsx          # Verdict + gauges
│   │   │   └── TradesFeed.jsx          # Live feed with flash animation
│   │   └── index.css            # Dark terminal theme
│   ├── package.json
│   └── vite.config.js           # Proxy /api and /ws to :8000
│
├── data/
│   ├── market_data/             # Raw OHLCV data (Parquet)
│   └── storage/
│       └── logs/metrics/        # JSONL audit logs (auto-created)
│
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (dashboard)
- MetaTrader5 terminal (Windows, optional — system runs without it)
- MySQL 8.0+ (optional — falls back to JSONL logs)

### Python

```bash
# Clone / navigate to the project
cd "d:\projet startup\agent trading\quant-fund-ai"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# PyTorch with CUDA (optional, adjust for your GPU)
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### React Dashboard

```bash
cd dashboard
npm install
```

---

## Configuration

All settings are centralised in a single **`.env`** file at the project root and
loaded via [`config/settings.py`](config/settings.py). Edit your values directly
in `.env` — no code changes required:

```python
from config.settings import settings
settings.api.port        # 8000
settings.mt5.login       # MT5 account
settings.cors.origins    # list of allowed origins
```

### Environment Variables (`.env`)

| Variable | Description | Default |
|---|---|---|
| `API_HOST` / `API_PORT` | API bind host / port | `0.0.0.0` / `8000` |
| `API_RELOAD` | Auto-reload on code change | `false` |
| `QUANT_API_KEY` | API key for all protected endpoints | *(empty = open)* |
| `CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000,...` |
| `CORS_ALLOW_CREDENTIALS` | Allow credentials in CORS | `true` |
| `MT5_LOGIN` | MetaTrader5 account number | *(empty = use logged-in terminal)* |
| `MT5_PASSWORD` | MT5 password | — |
| `MT5_SERVER` | MT5 broker server | — |
| `MT5_PATH` | Path to `terminal64.exe` | *(default terminal)* |
| `MT5_MAGIC_NUMBER` | EA identifier on orders | `20260524` |
| `MYSQL_ENABLED` | Enable MySQL metrics store | `false` |
| `MYSQL_HOST` / `MYSQL_PORT` | MySQL host / port | `localhost` / `3306` |
| `MYSQL_USER` / `MYSQL_PASSWORD` | MySQL credentials | `root` / *(empty)* |
| `MYSQL_DATABASE` | MySQL database name | `quant_fund` |
| `VITE_API_URL` / `VITE_WS_URL` | Dashboard API / WebSocket URLs | `http://localhost:8000` |
| `VITE_API_KEY` | API key injected into dashboard build | — |

> `.env` is git-ignored and never committed. Real environment
> variables always take precedence over `.env` values.

---

## Running the System

### 1 — Start the API server

```bash
# Configure everything in .env (host, port, QUANT_API_KEY, CORS, MT5…)
python -m api.main

# CLI flags still override .env if needed
python -m api.main --host 0.0.0.0 --port 8000

# Or via uvicorn directly (reads .env through config/settings.py)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Interactive docs: http://localhost:8000/docs

### 2 — Start the dashboard

```bash
cd dashboard
npm run dev   # → http://localhost:3000
```

### 3 — Start trading (programmatic)

```python
from api.main import create_app
from api.dependencies import get_app_state
from monitoring.monitor import TradingMonitor, MonitorConfig
from monitoring.metrics_store import MySQLConfig
from live_trading.live_trader import LiveTrader, LiveTraderConfig
from strategies.momentum.momentum_alpha import MomentumAlpha
from strategies.mean_reversion.mean_reversion_alpha import MeanReversionAlpha
import uvicorn

# ── Subsystems ──────────────────────────────────────────────────
monitor = TradingMonitor(MonitorConfig(
    mysql_config=MySQLConfig(
        host="localhost", user="quant", password="pass", database="quant_fund"
    )
))

trader = LiveTrader(LiveTraderConfig(
    symbols=["EURUSD", "GBPUSD"],
    timeframe="H1",
    cycle_interval_seconds=3600,
))

trader.register_strategy(MomentumAlpha(symbol="EURUSD"), weight=1.0)
trader.register_strategy(MeanReversionAlpha(symbol="EURUSD"), weight=0.8)
trader.register_hook(monitor)

# ── Wire into API AppState ──────────────────────────────────────
state = get_app_state()
state.attach_trader(trader)
state.attach_monitor(monitor)

# ── Start ────────────────────────────────────────────────────────
app = create_app(api_key="your_secret_key")
trader.start_background()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## API Endpoints

All endpoints require `Authorization: Bearer <key>` or `X-API-Key: <key>`  
(unless `QUANT_API_KEY` is empty).

### Trading Control

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe (no auth) |
| `POST` | `/trading/start` | Start live trading loop |
| `POST` | `/trading/stop` | Graceful stop |
| `POST` | `/trading/emergency_stop` | Immediate halt + cancel orders |
| `DELETE` | `/trading/emergency_stop` | Clear emergency flag |
| `GET` | `/trading/status` | Full runtime status |

### Portfolio

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/portfolio/status` | Equity, balance, PnL, drawdown, positions |
| `GET` | `/portfolio/equity_curve` | Time-series equity curve |
| `GET` | `/portfolio/allocations` | Capital per strategy |
| `GET` | `/portfolio/drawdowns` | Drawdown history |

### Strategies

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/strategies` | All strategies + stats |
| `GET` | `/strategies/{name}/performance` | Detailed metrics |
| `GET` | `/strategies/{name}/trades` | Trade history |
| `POST` | `/strategies/{name}/enable` | Enable (with weight) |
| `POST` | `/strategies/{name}/disable` | Set weight to 0 |

### Backtesting

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/backtest/run` | Submit async backtest job |
| `GET` | `/backtest/{job_id}/status` | Poll job status |
| `GET` | `/backtest/{job_id}/results` | Fetch completed results |
| `GET` | `/backtest/jobs` | List recent jobs |

### Trades

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/trades/history` | Trade history (filterable) |
| `GET` | `/trades/open` | Open MT5 positions |
| `POST` | `/trades/override` | Manual trade (bypass signals) |
| `POST` | `/trades/close` | Close position(s) |
| `GET` | `/trades/stats` | Win rate, profit factor, avg P&L |

### Risk

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/risk/status` | Current risk snapshot |
| `GET` | `/risk/history` | Historical snapshots |
| `GET` | `/risk/drawdowns` | Drawdown stats + history |
| `PATCH` | `/risk/config` | Update risk limits at runtime |
| `GET` | `/risk/config` | Current risk configuration |

### WebSocket  `/ws`

```js
const ws = new WebSocket("ws://localhost:8000/ws?api_key=your_key")

ws.onmessage = (e) => {
  const { type, payload, ts } = JSON.parse(e.data)
  // type: "snapshot" | "tick" | "trade" | "positions" | "strategies" | "alert" | "heartbeat"
}
```

| Type | Frequency | Payload |
|---|---|---|
| `snapshot` | On connect | Full state: equity curve, trades, risk, strategies, positions |
| `tick` | Every 2 s | Latest equity points + portfolio summary + risk |
| `trade` | On close | Single trade record |
| `positions` | Every 10 s | Open MT5 positions |
| `strategies` | Every 10 s | Per-strategy stats |
| `alert` | On threshold breach | Risk / emergency alert |

---

## Strategies

### Momentum Alpha
- **Signal**: EMA crossover (fast/slow) filtered by RSI
- **Long**: fast EMA > slow EMA AND RSI not overbought
- **Short**: fast EMA < slow EMA AND RSI not oversold

### Mean Reversion Alpha
- **Signal**: Bollinger Bands + Z-score
- **Long**: price below lower band (Z-score < -2)
- **Short**: price above upper band (Z-score > 2)

### RL Agent (PPO-LSTM)
- **Architecture**: `InputEncoder → LSTMCore → Actor/Critic heads`
- **Training**: `RLTrainer.fit()` with composite reward:  
  `r = pnl_reward − α·drawdown − β·risk_volatility + γ·sharpe`
- **Inference**: Stateful LSTM hidden state persisted across bars

```python
# Train
from strategies.rl_agent.rl_trainer import train_ppo_lstm
model = train_ppo_lstm(data, symbol="EURUSD", timeframe="H1", total_timesteps=500_000)

# Load for live trading
from strategies.rl_agent.rl_alpha import RLAlpha
alpha = RLAlpha.load("models/rl_eurusd.zip", algo="RecurrentPPO")
```

---

## Risk Management (6 Layers)

1. **Daily loss limit** — halt if daily PnL < -X%
2. **Drawdown limit** — halt if drawdown > X%
3. **Position size** — cap individual position as % of equity
4. **Leverage** — cap total exposure / equity ratio
5. **Consecutive losses** — halt after N consecutive losing trades
6. **Correlation** — limit correlated position concentration

Update limits at runtime via API:
```bash
curl -X PATCH http://localhost:8000/risk/config \
  -H "Authorization: Bearer your_key" \
  -H "Content-Type: application/json" \
  -d '{"max_daily_loss_pct": 0.03, "max_drawdown_pct": 0.15}'
```

---

## Monitoring & Persistence

### MySQL Schema

| Table | Purpose |
|---|---|
| `equity_snapshots` | Portfolio equity curve with drawdown |
| `strategy_pnl` | Per-strategy cumulative PnL + Sharpe |
| `trades` | Complete trade history |
| `risk_snapshots` | Risk exposure snapshots |
| `drawdown_periods` | Drawdown episode records |

### JSON Fallback

All writes go to `data/storage/logs/metrics/YYYYMMDD/<category>.jsonl` as a  
primary audit trail even when MySQL is unavailable.

### Custom Alerts (Telegram, Slack, etc.)

```python
def send_telegram(alert_type: str, data: dict) -> None:
    requests.post(TELEGRAM_URL, json={"text": f"[{alert_type}] {data}"})

monitor.register_alert(send_telegram)
```

---

## Backtesting

```python
from backtesting.backtest_engine import BacktestEngine, BacktestConfig

engine = BacktestEngine(BacktestConfig(
    initial_balance = 100_000,
    commission      = 0.0002,
    spread          = 0.0001,
))

result = engine.run(
    strategies = [MomentumAlpha("EURUSD"), MeanReversionAlpha("EURUSD")],
    data       = feature_store.load("EURUSD", "H1"),
)

print(result.portfolio_metrics.sharpe)        # Portfolio Sharpe ratio
print(result.strategy_results["Momentum"].metrics.max_drawdown_pct)
```

Or via API (async):
```bash
curl -X POST http://localhost:8000/backtest/run \
  -H "Authorization: Bearer your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["MomentumAlpha", "MeanReversionAlpha"],
    "symbol": "EURUSD",
    "timeframe": "H1",
    "initial_balance": 100000
  }'
# → { "job_id": "a3f9c12b8e4d", "status": "queued" }
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | REST API + WebSocket server |
| `pydantic v2` | Request / response validation |
| `torch` | Neural network (PPO-LSTM policy) |
| `gymnasium` | RL environment |
| `stable-baselines3` | PPO training loop |
| `sb3-contrib` | RecurrentPPO (LSTM-stateful) |
| `pandas` + `numpy` | Data manipulation |
| `pyarrow` | Parquet feature cache |
| `mysql-connector-python` | Metrics persistence |
| `MetaTrader5` | Live broker connection (Windows) |
| `React 18` + `Recharts` | Real-time dashboard |

---

## License

Proprietary — all rights reserved.
