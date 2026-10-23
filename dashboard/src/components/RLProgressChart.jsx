import { useState, useEffect, useRef, useCallback } from 'react'
import {
  LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AUTH_HEADERS, buildApiPath } from '../apiConfig.js'

// ---------------------------------------------------------------------------
// RLProgressChart
// Two readable views of the RL agent's learning:
//   1. the current / latest training run — mean reward vs training steps
//      (every run restarts at step 0, so runs are never overlaid)
//   2. one bar per run — its final mean reward, to see whether successive
//      retrainings improve the agent.
// Polls /api/rl/progress; faster while a retrain is running.
// ---------------------------------------------------------------------------
export default function RLProgressChart({ running = false }) {
  const [points, setPoints] = useState([])
  const [runs, setRuns] = useState([])
  const pollRef = useRef(null)

  const fetchProgress = useCallback(async () => {
    try {
      const r = await fetch(buildApiPath('/api/rl/progress?limit=1000'), { headers: AUTH_HEADERS })
      const d = await r.json()
      if (Array.isArray(d?.data?.points)) setPoints(d.data.points)
      if (Array.isArray(d?.data?.runs)) setRuns(d.data.runs)
    } catch { /* keep last data */ }
  }, [])

  useEffect(() => { fetchProgress() }, [fetchProgress])

  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(fetchProgress, running ? 3000 : 20000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [running, fetchProgress])

  if (!points.length) {
    return (
      <div className="rlp-empty">
        Aucun entraînement RL enregistré. Cochez « agent RL inclus » puis
        « Réentraîner les modèles » pour voir la courbe d'apprentissage.
      </div>
    )
  }

  const first = points[0]
  const last = points[points.length - 1]
  const delta = (last.mean_reward ?? 0) - (first.mean_reward ?? 0)
  const improving = delta >= 0
  const runBars = runs.map((r, i) => ({
    ...r,
    label: `#${i + 1}`,
    date: fmtDate(r.started),
  }))

  return (
    <div className="rlp">
      <div className="rlp-head">
        <span className="rlp-title">
          {running ? 'Entraînement RL en cours' : 'Dernier entraînement RL'}
          {' · '}{fmtDate(first.run || first.ts)}
        </span>
        <span className="rlp-stat">
          {fmtK(last.timestep)} étapes · récompense moyenne{' '}
          <b style={{ color: improving ? 'var(--green)' : 'var(--red)' }}>
            {fmtNum(last.mean_reward)}
          </b>
          {' '}({improving ? '+' : ''}{fmtNum(delta)} depuis le début du run)
        </span>
      </div>

      <div className="rlp-grid">
        <div className="rlp-chart">
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={points} margin={{ top: 8, right: 12, left: 4, bottom: 14 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis
                dataKey="timestep" type="number" domain={['dataMin', 'dataMax']}
                tickFormatter={fmtK} minTickGap={40}
                tick={{ fill: 'var(--text-dim)', fontSize: 10 }} axisLine={false} tickLine={false}
                label={{ value: "étapes d'entraînement", position: 'insideBottom', offset: -8,
                         fill: 'var(--text-dim)', fontSize: 10 }}
              />
              <YAxis
                tickFormatter={fmtNum} width={56}
                tick={{ fill: 'var(--text-dim)', fontSize: 10 }} axisLine={false} tickLine={false}
              />
              <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="4 4" />
              <Tooltip content={<RunTooltip />} />
              <Line type="monotone" dataKey="last_reward" name="Épisode" stroke="var(--text-dim)"
                    strokeWidth={1} strokeOpacity={0.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="mean_reward" name="Moyenne" strokeWidth={2.5}
                    stroke={improving ? 'var(--green)' : 'var(--red)'} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="rlp-legend">
            <span><i style={{ background: improving ? 'var(--green)' : 'var(--red)' }} /> moyenne glissante</span>
            <span><i style={{ background: 'var(--text-dim)' }} /> récompense d'un épisode</span>
            <span>plus haut = meilleur</span>
          </div>
        </div>

        {runBars.length > 1 && (
          <div className="rlp-chart">
            <div className="rlp-subtitle">Récompense finale par run ({runBars.length} runs)</div>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={runBars} margin={{ top: 8, right: 8, left: 4, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: 'var(--text-dim)', fontSize: 10 }}
                       axisLine={false} tickLine={false} />
                <YAxis tickFormatter={fmtNum} width={56}
                       tick={{ fill: 'var(--text-dim)', fontSize: 10 }} axisLine={false} tickLine={false} />
                <ReferenceLine y={0} stroke="var(--text-dim)" />
                <Tooltip content={<RunBarTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                <Bar dataKey="final_reward" isAnimationActive={false}>
                  {runBars.map((r, i) => (
                    <Cell key={i} fill={i === runBars.length - 1 ? 'var(--accent)' : 'var(--text-dim)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tooltips + helpers
// ---------------------------------------------------------------------------
function RunTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="ct-time">étape {fmtK(d.timestep)} · {fmtDate(d.ts)}</div>
      <div className="ct-row"><span className="ct-label">Moyenne</span>
        <span className="ct-val">{fmtNum(d.mean_reward)}</span></div>
      <div className="ct-row"><span className="ct-label">Épisode</span>
        <span className="ct-val">{fmtNum(d.last_reward)}</span></div>
    </div>
  )
}

function RunBarTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="ct-time">Run {r.label} · {r.date}</div>
      <div className="ct-row"><span className="ct-label">Récompense finale</span>
        <span className="ct-val">{fmtNum(r.final_reward)}</span></div>
      <div className="ct-row"><span className="ct-label">Étapes</span>
        <span className="ct-val">{fmtK(r.steps)}</span></div>
    </div>
  )
}

function fmtK(v) {
  const n = Number(v) || 0
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(0)}K`
  return `${n}`
}

function fmtNum(v) {
  const n = Number(v) || 0
  if (Math.abs(n) >= 1000) return n.toLocaleString('fr-FR', { maximumFractionDigits: 0 })
  return n.toLocaleString('fr-FR', { maximumFractionDigits: 2 })
}

function fmtDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString('fr-FR', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  })
}
