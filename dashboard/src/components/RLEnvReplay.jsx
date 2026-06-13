import { useState, useCallback } from 'react'
import {
  ComposedChart, Line, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API_KEY = import.meta.env.VITE_API_KEY ?? ''
const AUTH_HEADERS = API_KEY ? { 'X-API-Key': API_KEY } : {}

// ---------------------------------------------------------------------------
// RLEnvReplay
// Replays one episode of the trained RL agent inside its trading environment:
// the price series with the agent's BUY / SELL actions and the resulting
// equity curve. Lets you SEE what the agent actually does, bar by bar.
// ---------------------------------------------------------------------------
export default function RLEnvReplay() {
  const [data,  setData]  = useState(null)
  const [meta,  setMeta]  = useState(null)
  const [busy,  setBusy]  = useState(false)
  const [error, setError] = useState('')
  const [msg,   setMsg]   = useState('')

  const fetchEpisode = useCallback(async () => {
    setBusy(true)
    setError('')
    setMsg('')
    try {
      const r = await fetch('/api/rl/episode?bars=600', { headers: AUTH_HEADERS })
      const d = await r.json()
      if (!r.ok || d?.success === false) {
        throw new Error(d?.error || d?.detail || `HTTP ${r.status}`)
      }
      const traj = d?.data?.trajectory ?? []
      if (!traj.length) {
        setData(null)
        setMeta(null)
        setMsg(d?.data?.message || 'No episode data.')
        return
      }
      // Mark the bars where the agent OPENS a position. Use the resulting
      // position (truth) rather than the raw action, which is ambiguous in
      // the env's action→direction mapping.
      let prevPos = 0
      const chart = traj.map((p) => {
        const opened = p.position !== prevPos ? p.position : 0
        prevPos = p.position
        return {
          step:   p.step,
          price:  p.price,
          equity: p.equity,
          long:   opened === 1 ? p.price : null,
          short:  opened === -1 ? p.price : null,
        }
      })
      setData(chart)
      setMeta(d.data)
    } catch (e) {
      setError(e.message || 'Replay failed')
    } finally {
      setBusy(false)
    }
  }, [])

  const pnlPct = meta
    ? ((meta.final_equity - meta.initial_balance) / meta.initial_balance) * 100
    : 0

  const nLong  = data ? data.filter((d) => d.long != null).length : 0
  const nShort = data ? data.filter((d) => d.short != null).length : 0

  return (
    <div className="rle">
      <div className="rle-head">
        <button className="btn btn-accent" onClick={fetchEpisode} disabled={busy}>
          {busy ? 'Replaying…' : 'Visualise environment'}
        </button>
        {meta && (
          <span className="rle-stat">
            {meta.symbol} {meta.timeframe} · {meta.n} bars ·{' '}
            <span style={{ color: 'var(--green)' }}>{nLong} long</span> ·{' '}
            <span style={{ color: 'var(--red)' }}>{nShort} short</span> · PnL{' '}
            <span style={{ color: pnlPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
            </span>
          </span>
        )}
      </div>

      {error && <div className="rle-err">{error}</div>}
      {msg && !data && <div className="rle-empty">{msg}</div>}

      {data && (
        <div className="rle-chart">
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis
                dataKey="step"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
                axisLine={false}
                tickLine={false}
                minTickGap={40}
              />
              <YAxis
                yAxisId="price"
                domain={['auto', 'auto']}
                tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
                axisLine={false}
                tickLine={false}
                width={56}
                tickFormatter={(v) => Number(v).toFixed(4)}
              />
              <YAxis
                yAxisId="equity"
                orientation="right"
                domain={['auto', 'auto']}
                tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
                axisLine={false}
                tickLine={false}
                width={56}
                tickFormatter={fmtK}
              />
              <Tooltip content={<ReplayTooltip />} />
              <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'var(--font)' }} />
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="price"
                name="Price"
                stroke="var(--text-dim)"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="equity"
                type="monotone"
                dataKey="equity"
                name="Equity"
                stroke="var(--accent)"
                strokeWidth={1.8}
                dot={false}
                isAnimationActive={false}
              />
              <Scatter
                yAxisId="price"
                dataKey="long"
                name="Open long"
                fill="var(--green)"
                shape="triangle"
                isAnimationActive={false}
              />
              <Scatter
                yAxisId="price"
                dataKey="short"
                name="Open short"
                fill="var(--red)"
                shape="triangle"
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tooltip + helpers
// ---------------------------------------------------------------------------
function ReplayTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const evt = d.long != null ? 'Open long' : d.short != null ? 'Open short' : '—'
  return (
    <div className="chart-tooltip">
      <div className="ct-time">bar {d.step} · {evt}</div>
      <div className="ct-row">
        <span className="ct-label">Price</span>
        <span className="ct-val">{Number(d.price).toFixed(5)}</span>
      </div>
      <div className="ct-row">
        <span className="ct-label">Equity</span>
        <span className="ct-val">{fmtK(d.equity)}</span>
      </div>
    </div>
  )
}

function fmtK(v) {
  const n = Number(v) || 0
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (Math.abs(n) >= 1000)      return `${(n / 1000).toFixed(1)}K`
  return `${n.toFixed(0)}`
}
