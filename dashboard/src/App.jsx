import { useState, useEffect, useCallback } from 'react'
import PnLChart           from './components/PnLChart.jsx'
import StrategyComparison from './components/StrategyComparison.jsx'
import PositionsTable     from './components/PositionsTable.jsx'
import RiskStatus         from './components/RiskStatus.jsx'
import TradesFeed         from './components/TradesFeed.jsx'
import EconomicCalendar   from './components/EconomicCalendar.jsx'
import TradingControls    from './components/TradingControls.jsx'
import RetrainControls    from './components/RetrainControls.jsx'
import useWebSocket       from './hooks/useWebSocket.js'
import { AUTH_HEADERS, WS_URL, buildApiPath } from './apiConfig.js'

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

  // ---- Normalise positions payloads from WS/REST ---------------------
  const setPositionsSafe = useCallback((incoming) => {
    const raw = Array.isArray(incoming)
      ? incoming
      : Array.isArray(incoming?.positions)
        ? incoming.positions
        : []

    const normalized = raw
      .filter(Boolean)
      .map((p) => ({
        ...p,
        // Preserve both formats used by backend structs and ad-hoc payloads.
        size: p.size ?? p.volume ?? p.lot ?? p.quantity ?? null,
        entry_price: p.entry_price ?? p.open_price ?? p.price_open ?? p.entry ?? null,
        current_price: p.current_price ?? p.price_current ?? p.price ?? p.current ?? null,
        unrealized_pnl: p.unrealized_pnl ?? p.profit ?? p.pnl ?? null,
      }))

    // Merge with previous snapshot so a partial payload does not blank cells.
    // Also keep a stable order (ticket/symbol) to prevent row remount flicker.
    setPositions((prev) => {
      const next = mergePositions(prev, normalized)
      if (samePositions(prev, next)) return prev
      return next
    })
  }, [])

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
      const tradeKey = (t) => `${t.ticket ?? ''}|${t.ts}|${t.symbol}|${t.strategy ?? ''}|${t.pnl ?? ''}`
      const seen  = new Set(prev.map((t) => tradeKey(t)))
      const fresh = incoming.filter((t) => !seen.has(tradeKey(t)))
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
        if (Object.prototype.hasOwnProperty.call(payload, 'positions')) {
          setPositionsSafe(payload.positions)
        }
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
        if (Object.prototype.hasOwnProperty.call(payload, 'positions')) {
          setPositionsSafe(payload.positions)
        }
        if (Array.isArray(payload.strategies)) setStrategies(payload.strategies)
        if (payload.account && Object.keys(payload.account).length) setAccount(payload.account)
        if (payload.status) {
          if (typeof payload.status.running === 'boolean') setRunning(payload.status.running)
          if (typeof payload.status.emergency_active === 'boolean') setEmergency(payload.status.emergency_active)
        }
        break

      case 'positions':
        setPositionsSafe(payload.positions)
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
  }, [lastMessage, mergeEquity, mergeTrades, setPositionsSafe])

  // ---- REST fallback for real-time open positions --------------------
  // WebSocket remains the primary channel; this lightweight poll keeps the
  // positions panel fresh if tick cadence is low or one frame is missed.
  useEffect(() => {
    let timer = null
    let active = true

    const refreshOpenPositions = async () => {
      try {
        const r = await fetch(buildApiPath('/api/trades/open'), { headers: AUTH_HEADERS })
        const d = await r.json()
        if (!active) return
        if (r.ok && d?.success !== false) {
          setPositionsSafe(d?.data ?? [])
        }
      } catch {
        // Keep last known snapshot; WS updates continue independently.
      }
    }

    refreshOpenPositions()
    timer = setInterval(refreshOpenPositions, status === 'connected' ? 3000 : 8000)

    return () => {
      active = false
      if (timer) clearInterval(timer)
    }
  }, [status, setPositionsSafe])

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

        <section className="panel panel-calendar">
          <PanelHeader title="Economic Calendar" />
          <div className="panel-body panel-body--scroll">
            <EconomicCalendar />
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

function samePositions(a = [], b = []) {
  if (a.length !== b.length) return false
  const key = (p) => [
    p.ticket ?? '',
    p.symbol ?? '',
    p.direction ?? p.side ?? '',
    p.size ?? p.volume ?? '',
    p.entry_price ?? p.open_price ?? '',
    p.current_price ?? p.price_current ?? p.price ?? '',
    p.unrealized_pnl ?? p.profit ?? p.pnl ?? '',
  ].join('|')

  for (let i = 0; i < a.length; i += 1) {
    if (key(a[i]) !== key(b[i])) return false
  }
  return true
}

function mergePositions(prev = [], incoming = []) {
  const byKey = new Map()

  for (const p of prev) {
    byKey.set(positionKey(p), p)
  }

  for (const p of incoming) {
    const key = positionKey(p)
    const old = byKey.get(key)
    byKey.set(key, {
      ...old,
      ...p,
      // Keep previous non-null values when new frame omits a field.
      size: coalesce(p.size, old?.size),
      entry_price: coalesce(p.entry_price, old?.entry_price),
      current_price: coalesce(p.current_price, old?.current_price),
      unrealized_pnl: coalesce(p.unrealized_pnl, old?.unrealized_pnl),
    })
  }

  return Array.from(byKey.values()).sort((x, y) => {
    const tx = Number(x.ticket ?? Number.MAX_SAFE_INTEGER)
    const ty = Number(y.ticket ?? Number.MAX_SAFE_INTEGER)
    if (tx !== ty) return tx - ty
    return String(x.symbol ?? '').localeCompare(String(y.symbol ?? ''))
  })
}

function positionKey(p) {
  if (p?.ticket != null) return `t:${p.ticket}`
  return `s:${p?.symbol ?? ''}|d:${p?.direction ?? p?.side ?? ''}`
}

function coalesce(nextVal, prevVal) {
  return nextVal == null ? prevVal ?? null : nextVal
}
