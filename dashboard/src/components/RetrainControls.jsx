import { useState, useCallback, useEffect, useRef } from 'react'
import RLProgressChart from './RLProgressChart.jsx'
import RLEnvReplay from './RLEnvReplay.jsx'
import { AUTH_HEADERS, buildApiPath } from '../apiConfig.js'

// ---------------------------------------------------------------------------
// RetrainControls
// Manual trigger to retrain the learning models from the system's own trade
// results: the ML win/loss classifier (always) and, optionally, the RL agent.
// ---------------------------------------------------------------------------
export default function RetrainControls() {
  const [status,  setStatus]  = useState(null)
  const [busy,    setBusy]    = useState(false)
  const [error,   setError]   = useState('')
  const [trainRl, setTrainRl] = useState(false)
  const [continuous, setContinuous] = useState(false)
  const [intervalS, setIntervalS] = useState(1800)
  const [useMaxHistory, setUseMaxHistory] = useState(true)
  const [historyBars, setHistoryBars] = useState(5000)
  const pollRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(buildApiPath('/api/rl/status'), { headers: AUTH_HEADERS })
      const d = await r.json()
      if (d?.data) setStatus(d.data)
      return d?.data
    } catch { return null }
  }, [])

  // Poll while a run is in progress.
  useEffect(() => {
    fetchStatus()
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [fetchStatus])

  useEffect(() => {
    if (status?.state === 'running' && !pollRef.current) {
      pollRef.current = setInterval(fetchStatus, 1500)
    } else if (status?.state !== 'running' && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [status?.state, fetchStatus])

  const handleRetrain = useCallback(async () => {
    setBusy(true)
    setError('')
    try {
      const body = {
        train_ml: true,
        train_rl: trainRl,
      }
      if (trainRl) {
        body.rl_continuous = continuous
        body.rl_interval_s = intervalS
        body.rl_history_bars = useMaxHistory ? 0 : historyBars
      }
      const r = await fetch(buildApiPath('/api/rl/retrain'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
        body:    JSON.stringify(body),
      })
      const d = await r.json()
      if (!r.ok || d?.success === false) {
        const raw = String(d?.error || d?.detail || `HTTP ${r.status}`)
        if (r.status === 409 || raw.toLowerCase().includes('already in progress')) {
          // Sync UI with active run when backend reports a lock.
          await fetchStatus()
          throw new Error('A retraining run is already active. Click Stop or wait for completion.')
        }
        throw new Error(raw)
      }
      setStatus(d.data)
    } catch (e) {
      setError(e.message || 'Retrain failed')
    } finally {
      setBusy(false)
    }
  }, [trainRl, continuous, intervalS, useMaxHistory, historyBars])

  const handleStop = useCallback(async () => {
    setBusy(true)
    setError('')
    try {
      const r = await fetch(buildApiPath('/api/rl/stop'), {
        method:  'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
      })
      const d = await r.json()
      if (!r.ok || d?.success === false) {
        throw new Error(d?.error || d?.detail || `HTTP ${r.status}`)
      }
      setStatus(d.data)
    } catch (e) {
      setError(e.message || 'Stop failed')
    } finally {
      setBusy(false)
    }
  }, [])

  const running = status?.state === 'running'
  const ml = status?.ml || {}
  const rl = status?.rl || {}

  return (
    <div className="retrain">
      <div className="retrain-row">
        <button
          className="btn btn-accent"
          onClick={handleRetrain}
          disabled={busy || running}
        >
          {running ? 'Training…' : 'Retrain models'}
        </button>
        <button
          className="btn"
          onClick={handleStop}
          disabled={busy || !running}
        >
          Stop
        </button>
        <label className="retrain-check">
          <input
            type="checkbox"
            checked={trainRl}
            onChange={(e) => setTrainRl(e.target.checked)}
            disabled={busy || running}
          />
          incl. RL agent
        </label>
        <label className="retrain-check">
          <input
            type="checkbox"
            checked={continuous}
            onChange={(e) => setContinuous(e.target.checked)}
            disabled={busy || running || !trainRl}
          />
          RL en continu
        </label>
        <label className="retrain-check">
          <input
            type="checkbox"
            checked={useMaxHistory}
            onChange={(e) => setUseMaxHistory(e.target.checked)}
            disabled={busy || running || !trainRl}
          />
          Max bougies (timeframe)
        </label>
        <label className="retrain-check">
          history bars
          <input
            type="number"
            min={200}
            max={500000}
            value={historyBars}
            onChange={(e) => setHistoryBars(Math.max(200, Math.min(500000, Number(e.target.value || 5000))))}
            disabled={busy || running || !trainRl || useMaxHistory}
          />
        </label>
        <label className="retrain-check">
          interval (s)
          <input
            type="number"
            min={60}
            max={86400}
            value={intervalS}
            onChange={(e) => setIntervalS(Math.max(60, Math.min(86400, Number(e.target.value || 1800))))}
            disabled={busy || running || !trainRl || !continuous}
          />
        </label>
      </div>

      {error && <div className="retrain-err">{error}</div>}

      {status && (
        <div className="retrain-status">
          <span className={`retrain-state retrain-state--${status.state}`}>
            {status.state}
          </span>
          {status.run_count != null && (
            <span className="retrain-metric">
              runs: {status.run_count}
              {status.state === 'running' && status.rl_timesteps_total > 0 && (
                <>
                  {' · '}
                  {(status.rl_timesteps_done ?? 0).toLocaleString()}
                  {'/'}
                  {status.rl_timesteps_total.toLocaleString()} steps
                  {' ('}
                  {Math.round(100 * (status.rl_timesteps_done ?? 0) / status.rl_timesteps_total)}
                  {'%)'}
                </>
              )}
            </span>
          )}
          {ml.trained != null && (
            <span className="retrain-metric">
              ML: {ml.trained
                ? `${ml.n_samples} trades · acc ${(ml.accuracy * 100).toFixed(0)}% · auc ${ml.auc}`
                : (ml.message || 'not trained')}
            </span>
          )}
          {ml.warning && (
            <span className="retrain-metric" style={{ color: 'var(--yellow)' }}>
              ML warning: {ml.warning}
            </span>
          )}
          {trainRl && rl.message && (
            <span className="retrain-metric">RL: {rl.message}</span>
          )}
        </div>
      )}

      <RLProgressChart running={running} />
      <RLEnvReplay />
    </div>
  )
}
