import { useState, useEffect, useCallback } from 'react'
import PnLChart           from './components/PnLChart.jsx'
import StrategyComparison from './components/StrategyComparison.jsx'
import PositionsTable     from './components/PositionsTable.jsx'
import RiskStatus         from './components/RiskStatus.jsx'
import TradesFeed         from './components/TradesFeed.jsx'
import TradingControls    from './components/TradingControls.jsx'
import RetrainControls    from './components/RetrainControls.jsx'
import useWebSocket       from './hooks/useWebSocket.js'

// ---------------------------------------------------------------------------
// Config — override with VITE_API_KEY env var if needed
// ---------------------------------------------------------------------------
const API_KEY = import.meta.env.VITE_API_KEY ?? ''
const WS_URL  = (() => {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host  = window.location.host
  const qs    = API_KEY ? `?api_key=${encodeURIComponent(API_KEY)}` : ''
  return `${proto}//${host}/ws${qs}`
})()

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [equity,     setEquity]     = useState([])          // equity curve points
  const [portfolio,  setPortfolio]  = useState(null)        // { equity, balance, daily_pnl … }
  const [strategies, setStrategies] = useState([])          // [{ strategy, cumulative_pnl … }]
  const [positions,  setPositions]  = useState([])          // open MT5 positions
  const [risk,       setRisk]       = useState(null)        // latest risk snapshot
  const [trades,     setTrades]     = useState([])          // recent trades (newest first)
  const [alerts,     setAlerts]     = useState([])          // risk / emergency alerts
  const [account,    setAccount]    = useState(null)        // real MT5 account { balance, equity, currency … }
  const [running,    setRunning]    = useState(false)       // trading loop active
  const [emergency,  setEmergency]  = useState(false)       // emergency stop engaged

  const { lastMessage, status } = useWebSocket(WS_URL)

  // ---- Deduplicate equity points by ts --------------------------------
  const mergeEquity = useCallback((incoming = []) => {
    setEquity((prev) => {
      if (!incoming.length) return prev
      const seen  = new Set(prev.map((p) => p.ts))
      const fresh = incoming.filter((p) => !seen.has(p.ts))
      if (!fresh.length) return prev
      return [...prev, ...fresh].slice(-500)
    })
  }, [])

  // ---- Merge trades (deduplicate by ts+symbol) ------------------------
  const mergeTrades = useCallback((incoming = []) => {
    if (!incoming.length) return
    setTrades((prev) => {
      const seen  = new Set(prev.map((t) => `${t.ts}|${t.symbol}`))
      const fresh = incoming.filter((t) => !seen.has(`${t.ts}|${t.symbol}`))
      if (!fresh.length) return prev
      return [...fresh, ...prev].slice(0, 200)
    })
  }, [])

  // ---- Dispatch WebSocket messages ------------------------------------
  useEffect(() => {
    if (!lastMessage) return
    const { type, payload } = lastMessage

    switch (type) {
      case 'snapshot':
        if (payload.equity_curve)  mergeEquity(payload.equity_curve)
        if (payload.recent_trades) mergeTrades(payload.recent_trades)
        if (payload.portfolio)     setPortfolio(payload.portfolio)
        if (payload.strategies?.strategies) setStrategies(payload.strategies.strategies)
        else if (Array.isArray(payload.strategies)) setStrategies(payload.strategies)
        if (payload.positions?.positions) setPositions(payload.positions.positions)
        else if (Array.isArray(payload.positions)) setPositions(payload.positions)
        if (payload.risk)          setRisk(payload.risk)
        if (payload.account && Object.keys(payload.account).length) setAccount(payload.account)
        if (payload.status) {
          if (typeof payload.status.running === 'boolean') setRunning(payload.status.running)
          if (typeof payload.status.emergency_active === 'boolean') setEmergency(payload.status.emergency_active)
        }
        break

      case 'tick':
        if (payload.equity_curve) mergeEquity(payload.equity_curve)
        if (payload.portfolio)    setPortfolio(payload.portfolio)
        if (payload.risk)         setRisk(payload.risk)
        if (payload.trades)       mergeTrades(payload.trades)
        if (payload.account && Object.keys(payload.account).length) setAccount(payload.account)
        break

      case 'positions':
        setPositions(payload.positions ?? [])
        break

      case 'strategies':
        setStrategies(payload.strategies ?? [])
        break

      case 'risk_update':
        setRisk(payload)
        break

      case 'equity_update':
        mergeEquity(Array.isArray(payload) ? payload : [payload])
        break

      case 'trade':
        mergeTrades([payload])
        break

      case 'alert':
        setAlerts((prev) => [payload, ...prev].slice(0, 10))
        break

      default:
        break
    }
  }, [lastMessage, mergeEquity, mergeTrades])

  // ---- Derived values -------------------------------------------------
  // Prefer the REAL MT5 account figures when connected; otherwise fall back
  // to the internal monitor's portfolio tracking.
  const ccy         = account?.currency ?? 'USD'
  const equity_val  = account?.equity  ?? portfolio?.equity  ?? 0
  const balance     = account?.balance ?? portfolio?.balance ?? 0
  const free_margin = account?.free_margin ?? null
  const daily_pnl   = portfolio?.daily_pnl     ?? 0
  const daily_pct   = portfolio?.daily_pnl_pct ?? 0
  const drawdown    = portfolio?.drawdown_pct  ?? 0
  const n_positions = positions.length

  return (
    <div className="app">

      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-brand">
          <span className="brand-glyph">◆</span>
          <span className="brand-name">QUANT FUND</span>
        </div>

        <div className="header-metrics">
          <Metric label={`EQUITY (${ccy})`}  value={fmt(equity_val)} />
          <Metric label={`BALANCE (${ccy})`} value={fmt(balance)} />
          {free_margin != null && (
            <Metric label={`FREE MARGIN (${ccy})`} value={fmt(free_margin)} />
          )}
          <Metric
            label="DAY P&L"
            value={`${daily_pnl >= 0 ? '+' : ''}$${fmt(Math.abs(daily_pnl))}`}
            color={daily_pnl >= 0 ? 'var(--green)' : 'var(--red)'}
          />
          <Metric
            label="DAY %"
            value={`${daily_pct >= 0 ? '+' : ''}${daily_pct.toFixed(2)}%`}
            color={daily_pct >= 0 ? 'var(--green)' : 'var(--red)'}
          />
          <Metric
            label="DRAWDOWN"
            value={`${drawdown.toFixed(2)}%`}
            color={drawdown > 10 ? 'var(--red)' : drawdown > 5 ? 'var(--yellow)' : 'var(--text-dim)'}
          />
          <Metric label="POSITIONS" value={n_positions} />
        </div>

        <div className="header-right">
          <TradingControls running={running} emergency={emergency} />
          <WsStatusBadge status={status} />
          {alerts.length > 0 && (
            <button className="alert-badge" onClick={() => setAlerts([])}>
              ⚠ {alerts.length} ALERT{alerts.length > 1 ? 'S' : ''}
            </button>
          )}
        </div>
      </header>

      {/* ── Alert banner ────────────────────────────────────────────── */}
      {alerts.length > 0 && (
        <div className="alert-banner">
          <span className="alert-icon">⚠</span>
          <span className="alert-text">{alerts[0].message ?? alerts[0].type ?? 'Alert'}</span>
          <button className="alert-close" onClick={() => setAlerts([])}>✕</button>
        </div>
      )}

      {/* ── Main grid ───────────────────────────────────────────────── */}
      <main className="grid">

        <section className="panel panel-chart">
          <PanelHeader title="Portfolio Equity Curve" />
          <div className="panel-body">
            <PnLChart data={equity} />
          </div>
        </section>

        <section className="panel panel-risk">
          <PanelHeader title="Risk Status" />
          <div className="panel-body">
            <RiskStatus risk={risk} portfolio={portfolio} />
          </div>
        </section>

        <section className="panel panel-strategy">
          <PanelHeader title="Strategy Performance" />
          <div className="panel-body">
            <RetrainControls />
            <StrategyComparison strategies={strategies} />
          </div>
        </section>

        <section className="panel panel-positions">
          <PanelHeader title={`Open Positions (${n_positions})`} />
          <div className="panel-body panel-body--scroll">
            <PositionsTable positions={positions} />
          </div>
        </section>

        <section className="panel panel-trades">
          <PanelHeader title="Live Trades Feed" badge={trades.length} />
          <div className="panel-body panel-body--scroll">
            <TradesFeed trades={trades} />
          </div>
        </section>

      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Metric({ label, value, color }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  )
}

function PanelHeader({ title, badge }) {
  return (
    <div className="panel-header">
      <span className="panel-title">{title}</span>
      {badge != null && <span className="panel-badge">{badge}</span>}
    </div>
  )
}

const STATUS_COLOR = {
  connected:    'var(--green)',
  connecting:   'var(--yellow)',
  reconnecting: 'var(--yellow)',
  disconnected: 'var(--red)',
}

function WsStatusBadge({ status }) {
  return (
    <div className="ws-badge">
      <span className="ws-dot" style={{ background: STATUS_COLOR[status] ?? 'var(--yellow)' }} />
      <span className="ws-label">{status.toUpperCase()}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Number formatter
// ---------------------------------------------------------------------------

function fmt(n) {
  if (n == null || isNaN(n)) return '—'
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M'
  if (Math.abs(n) >= 10_000)    return (n / 1_000).toFixed(1) + 'K'
  return Number(n).toFixed(2)
}
