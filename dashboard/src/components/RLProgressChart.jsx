import { useState, useEffect, useRef, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { AUTH_HEADERS, buildApiPath } from '../apiConfig.js'

// ---------------------------------------------------------------------------
// RLProgressChart
// Plots the RL agent's reward over training time (one point per episode) so
// you can confirm the reward trends upward. Polls /api/rl/progress; polls
// faster while a retrain is running.
// ---------------------------------------------------------------------------
export default function RLProgressChart({ running = false }) {
  const [points, setPoints] = useState([])
  const pollRef = useRef(null)

  const fetchProgress = useCallback(async () => {
    try {
      const r = await fetch(buildApiPath('/api/rl/progress?limit=1000'), { headers: AUTH_HEADERS })
      const d = await r.json()
      const pts = d?.data?.points
      if (Array.isArray(pts)) setPoints(pts)
    } catch { /* keep last data */ }
  }, [])

  useEffect(() => {
    fetchProgress()
  }, [fetchProgress])

  // Poll every 2s while running, every 15s otherwise.
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(fetchProgress, running ? 2000 : 15000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [running, fetchProgress])

  if (!points.length) {
    return (
      <div className="rlp-empty">
        No RL training history yet. Train the agent (Retrain · incl. RL) to see
        the reward curve.
      </div>
    )
  }

  const last      = points[points.length - 1]
  const rewards   = points.map((p) => p.mean_reward ?? 0)
  const minR      = Math.min(...rewards)
  const maxR      = Math.max(...rewards)
  const pad       = (maxR - minR) * 0.08 || 0.01
  const yDomain   = [minR - pad, maxR + pad]
  const trendUp   = rewards[rewards.length - 1] >= rewards[0]

  return (
    <div className="rlp">
      <div className="rlp-head">
        <span className="rlp-title">RL reward over time</span>
        <span className="rlp-stat">
          ep {last.episode} · step {fmtK(last.timestep)} · mean{' '}
          <span style={{ color: trendUp ? 'var(--green)' : 'var(--red)' }}>
            {Number(last.mean_reward).toFixed(4)}
          </span>
        </span>
      </div>
      <div className="rlp-chart">
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={points} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="timestep"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={fmtK}
              tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
              axisLine={false}
              tickLine={false}
              minTickGap={40}
            />
            <YAxis
              domain={yDomain}
              tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
              axisLine={false}
              tickLine={false}
              width={46}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip content={<RLTooltip />} />
            <Line
              type="monotone"
              dataKey="last_reward"
              name="Episode"
              stroke="var(--text-dim)"
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="mean_reward"
              name="Mean (10)"
              stroke={trendUp ? 'var(--green)' : 'var(--red)'}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tooltip + helpers
// ---------------------------------------------------------------------------
function RLTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="ct-time">step {fmtK(d.timestep)} · ep {d.episode}</div>
      <div className="ct-row">
        <span className="ct-label">Mean reward</span>
        <span className="ct-val">{Number(d.mean_reward).toFixed(4)}</span>
      </div>
      <div className="ct-row">
        <span className="ct-label">Episode reward</span>
        <span className="ct-val">{Number(d.last_reward).toFixed(4)}</span>
      </div>
    </div>
  )
}

function fmtK(v) {
  const n = Number(v) || 0
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1000)      return `${(n / 1000).toFixed(0)}K`
  return `${n}`
}
