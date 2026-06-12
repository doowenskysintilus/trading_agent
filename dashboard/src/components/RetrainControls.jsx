import { useState, useCallback, useEffect, useRef } from 'react'
import RLProgressChart from './RLProgressChart.jsx'

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API_KEY = import.meta.env.VITE_API_KEY ?? ''

const AUTH_HEADERS = {
  'Content-Type': 'application/json',
  ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
}

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
  const pollRef = useRef(null)

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/rl/status', { headers: AUTH_HEADERS })
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
      const r = await fetch('/api/rl/retrain', {
        method:  'POST',
        headers: AUTH_HEADERS,
        body:    JSON.stringify({ train_ml: true, train_rl: trainRl }),
      })
      const d = await r.json()
      if (!r.ok || d?.success === false) {
        throw new Error(d?.error || d?.detail || `HTTP ${r.status}`)
      }
      setStatus(d.data)
    } catch (e) {
      setError(e.message || 'Retrain failed')
    } finally {
      setBusy(false)
    }
  }, [trainRl])

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
        <label className="retrain-check">
          <input
            type="checkbox"
            checked={trainRl}
            onChange={(e) => setTrainRl(e.target.checked)}
            disabled={busy || running}
          />
          incl. RL agent
        </label>
      </div>

      {error && <div className="retrain-err">{error}</div>}

      {status && (
        <div className="retrain-status">
          <span className={`retrain-state retrain-state--${status.state}`}>
            {status.state}
          </span>
          {ml.trained != null && (
            <span className="retrain-metric">
              ML: {ml.trained
                ? `${ml.n_samples} trades · acc ${(ml.accuracy * 100).toFixed(0)}% · auc ${ml.auc}`
                : (ml.message || 'not trained')}
            </span>
          )}
          {trainRl && rl.message && (
            <span className="retrain-metric">RL: {rl.message}</span>
          )}
        </div>
      )}

      <RLProgressChart running={running} />
    </div>
  )
}
