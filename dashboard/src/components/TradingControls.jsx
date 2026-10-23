import { useState, useCallback, useEffect } from 'react'
import { AUTH_HEADERS, buildApiPath } from '../apiConfig.js'

// Common FX / metal pairs offered as quick presets. The user can also type
// any symbol manually (must match the broker's exact symbol name in MT5).
const PRESET_SYMBOLS = [
  'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
  'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY',
  'GBPJPY', 'XAUUSD', 'XAGUSD', 'BTCUSD',
]

const TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']

// Higher timeframes available for the trend filter. '' = auto-derive server-side.
const HTF_TIMEFRAMES = ['', 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']

const ALLOCATION_METHODS = [
  { value: 'RISK_PARITY',          label: 'Parité des risques' },
  { value: 'EQUAL_WEIGHT',         label: 'Poids égaux' },
  { value: 'PERFORMANCE_WEIGHTED', label: 'Pondéré par la performance' },
  { value: 'KELLY',                label: 'Critère de Kelly' },
]

// ---------------------------------------------------------------------------
// TradingControls
// ===========================================================================
// Header control panel to start / stop the autonomous trading loop and pick
// which pairs the system trades. Once started, the backend LiveTrader places
// and manages all orders automatically (entries, SL/TP, trailing stops).
// ---------------------------------------------------------------------------
export default function TradingControls({ running, emergency }) {
  const [open,    setOpen]    = useState(false)
  const [busy,    setBusy]    = useState(false)
  const [error,   setError]   = useState('')

  // Form state
  const [symbols,    setSymbols]    = useState(['EURUSD'])
  const [custom,     setCustom]     = useState('')
  const [timeframe,  setTimeframe]  = useState('H1')
  const [interval,   setInterval]   = useState(3600)
  const [warmupBars, setWarmupBars] = useState(200)
  const [useMaxBars, setUseMaxBars] = useState(false)
  const [balance,    setBalance]    = useState(100000)
  const [allocation, setAllocation] = useState('RISK_PARITY')
  const [htfEnabled,   setHtfEnabled]   = useState(true)
  const [htfTimeframe, setHtfTimeframe] = useState('')
  const [mlEnabled,    setMlEnabled]    = useState(true)
  const [mlMinWin,     setMlMinWin]     = useState(0.50)
  const [slMultiplier, setSlMultiplier] = useState(2.0)
  const [tpMultiplier, setTpMultiplier] = useState(4.0)
  const [ftmoEnabled,  setFtmoEnabled]  = useState(false)
  const [scalpEnabled, setScalpEnabled] = useState(false)

  // Pre-fill the panel with the defaults configured in .env (server-side).
  useEffect(() => {
    let cancelled = false
    fetch(buildApiPath('/api/trading/config'), { headers: AUTH_HEADERS })
      .then((r) => (r.ok ? r.json() : null))
      .then((res) => {
        const cfg = res?.data
        if (cancelled || !cfg) return
        if (Array.isArray(cfg.symbols) && cfg.symbols.length) setSymbols(cfg.symbols)
        if (cfg.timeframe)         setTimeframe(cfg.timeframe)
        if (cfg.cycle_interval_s)  setInterval(cfg.cycle_interval_s)
        if (cfg.warmup_bars)       setWarmupBars(cfg.warmup_bars)
        if (cfg.initial_balance)   setBalance(cfg.initial_balance)
        if (cfg.allocation_method) setAllocation(cfg.allocation_method)
        if (typeof cfg.htf_enabled === 'boolean') setHtfEnabled(cfg.htf_enabled)
        if (typeof cfg.htf_timeframe === 'string') setHtfTimeframe(cfg.htf_timeframe)
        if (typeof cfg.ml_filter_enabled === 'boolean') setMlEnabled(cfg.ml_filter_enabled)
        if (typeof cfg.ml_min_win_proba === 'number') setMlMinWin(cfg.ml_min_win_proba)
        if (typeof cfg.sl_atr_multiplier === 'number') setSlMultiplier(cfg.sl_atr_multiplier)
        if (typeof cfg.tp_atr_multiplier === 'number') setTpMultiplier(cfg.tp_atr_multiplier)
        if (typeof cfg.scalping_enabled === 'boolean') setScalpEnabled(cfg.scalping_enabled)
      })
      .catch(() => { /* keep hardcoded fallbacks */ })
    return () => { cancelled = true }
  }, [])

  const toggleSymbol = useCallback((sym) => {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym],
    )
  }, [])

  const addCustom = useCallback(() => {
    const sym = custom.trim().toUpperCase()
    if (sym && !symbols.includes(sym)) setSymbols((prev) => [...prev, sym])
    setCustom('')
  }, [custom, symbols])

  // ---- API calls ------------------------------------------------------
  const callApi = useCallback(async (path, body) => {
    setBusy(true)
    setError('')
    try {
      const res = await fetch(buildApiPath(`/api${path}`), {
        method:  'POST',
        headers: AUTH_HEADERS,
        body:    body ? JSON.stringify(body) : undefined,
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok || data?.success === false) {
        throw new Error(data?.error || data?.detail || `HTTP ${res.status}`)
      }
      return data
    } catch (e) {
      setError(e.message || 'Requête impossible')
      throw e
    } finally {
      setBusy(false)
    }
  }, [])

  const handleStart = useCallback(async () => {
    if (!symbols.length) {
      setError('Choisissez au moins une paire.')
      return
    }
    // Guard against empty / invalid numeric fields, which would otherwise be
    // sent as 0 and rejected by the API (cycle ≥ 1, balance > 0 → 422).
    const cycle = Math.min(Math.max(Number(interval) || 0, 1), 86400)
    const bal   = Number(balance) > 0 ? Number(balance) : 100000
    const bars  = useMaxBars ? 0 : Math.min(Math.max(Number(warmupBars) || 0, 50), 500000)
    try {
      await callApi('/trading/start', {
        symbols,
        timeframe,
        cycle_interval_s: cycle,
        warmup_bars:      bars,
        initial_balance:  bal,
        allocation_method: allocation,
        htf_enabled:   htfEnabled,
        htf_timeframe: htfTimeframe,
        ml_filter_enabled: mlEnabled,
        ml_min_win_proba:  Math.min(Math.max(Number(mlMinWin) || 0, 0), 1),
        sl_atr_multiplier: Math.min(Math.max(Number(slMultiplier) || 2.0, 1.0), 5.0),
        tp_atr_multiplier: Math.min(Math.max(Number(tpMultiplier) || 4.0, 1.0), 5.0),
        // FTMO mode reuses the same balance as both portfolio sizing seed
        // and the immutable challenge reference (RiskEngine Layer 1b).
        ftmo_enabled:         ftmoEnabled,
        ftmo_initial_balance: ftmoEnabled ? bal : 0,
        // Scalping rule set (SCALPING_RULES.md) — overrides SL/TP, sizing
        // and thresholds server-side.
        scalping_enabled: scalpEnabled,
      })
      setOpen(false)
    } catch { /* error already surfaced */ }
  }, [callApi, symbols, timeframe, interval, warmupBars, useMaxBars, balance, allocation, htfEnabled, htfTimeframe, mlEnabled, mlMinWin, slMultiplier, tpMultiplier, ftmoEnabled, scalpEnabled])

  const handleStop = useCallback(async () => {
    try {
      await callApi('/trading/stop', { reason: 'operator (dashboard)' })
    } catch { /* error already surfaced */ }
  }, [callApi])

  const handleEmergency = useCallback(async () => {
    try {
      await callApi('/trading/emergency_stop', { reason: 'manual (dashboard)' })
    } catch { /* error already surfaced */ }
  }, [callApi])

  // ---- Render ---------------------------------------------------------
  return (
    <div className="tc">
      <div className="tc-status">
        <span
          className="tc-dot"
          style={{
            background: emergency ? 'var(--red)' : running ? 'var(--green)' : 'var(--text-dim)',
          }}
        />
        <span className="tc-label">
          {emergency ? 'URGENCE' : running ? 'EN TRADING' : 'ARRÊTÉ'}
        </span>
      </div>

      {running ? (
        <>
          <button className="tc-btn tc-btn--stop" disabled={busy} onClick={handleStop}>
            ■ ARRÊTER
          </button>
          <button className="tc-btn tc-btn--emergency" disabled={busy} onClick={handleEmergency}>
            ⚠ ARRÊT D'URGENCE
          </button>
        </>
      ) : (
        <button
          className="tc-btn tc-btn--start"
          disabled={busy || emergency}
          onClick={() => setOpen((v) => !v)}
        >
          ▶ DÉMARRER
        </button>
      )}

      {/* Config popover */}
      {open && !running && (
        <div className="tc-popover">
          <div className="tc-popover-head">
            <span>Démarrer le trading</span>
            <button className="tc-x" onClick={() => setOpen(false)}>✕</button>
          </div>

          <div className="tc-field">
            <label>Paires</label>
            <div className="tc-chips">
              {PRESET_SYMBOLS.map((sym) => (
                <button
                  key={sym}
                  className={`tc-chip ${symbols.includes(sym) ? 'tc-chip--on' : ''}`}
                  onClick={() => toggleSymbol(sym)}
                >
                  {sym}
                </button>
              ))}
            </div>
            <div className="tc-custom">
              <input
                type="text"
                placeholder="Autre symbole (ex. EURGBP)"
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addCustom()}
              />
              <button onClick={addCustom}>+ Ajouter</button>
            </div>
            {symbols.length > 0 && (
              <div className="tc-selected">
                {symbols.map((s) => (
                  <span key={s} className="tc-selected-chip" onClick={() => toggleSymbol(s)}>
                    {s} ✕
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="tc-row">
            <div className="tc-field">
              <label>Unité de temps</label>
              <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
                {TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
              </select>
            </div>
            <div className="tc-field">
              <label>Cycle (s)</label>
              <input
                type="number"
                min={60}
                max={86400}
                value={interval}
                onChange={(e) => setInterval(e.target.value)}
              />
            </div>
            <div className="tc-field">
              <label>Bougies chargées</label>
              <input
                type="number"
                min={50}
                max={500000}
                value={warmupBars}
                disabled={useMaxBars}
                onChange={(e) => setWarmupBars(e.target.value)}
              />
              <label style={{ marginTop: 6, display: 'block' }}>
                <input
                  type="checkbox"
                  checked={useMaxBars}
                  onChange={(e) => setUseMaxBars(e.target.checked)}
                  style={{ marginRight: 6, verticalAlign: 'middle' }}
                />
                Maximum disponible
              </label>
            </div>
          </div>

          <div className="tc-row">
            <div className="tc-field">
              <label>Solde initial</label>
              <input
                type="number"
                min={1}
                value={balance}
                onChange={(e) => setBalance(e.target.value)}
              />
            </div>
            <div className="tc-field">
              <label>Allocation</label>
              <select value={allocation} onChange={(e) => setAllocation(e.target.value)}>
                {ALLOCATION_METHODS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="tc-row">
            <div className="tc-field">
              <label>
                <input
                  type="checkbox"
                  checked={htfEnabled}
                  onChange={(e) => setHtfEnabled(e.target.checked)}
                  style={{ marginRight: 6, verticalAlign: 'middle' }}
                />
                Filtre de tendance (UT supérieure)
              </label>
            </div>
            <div className="tc-field">
              <label>UT de la tendance</label>
              <select
                value={htfTimeframe}
                disabled={!htfEnabled}
                onChange={(e) => setHtfTimeframe(e.target.value)}
              >
                {HTF_TIMEFRAMES.map((tf) => (
                  <option key={tf || 'auto'} value={tf}>{tf || 'Auto'}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="tc-row">
            <div className="tc-field">
              <label>
                <input
                  type="checkbox"
                  checked={mlEnabled}
                  onChange={(e) => setMlEnabled(e.target.checked)}
                  style={{ marginRight: 6, verticalAlign: 'middle' }}
                />
                Filtre ML de probabilité de gain
              </label>
            </div>
            <div className="tc-field">
              <label>Probabilité min. de gain</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={mlMinWin}
                disabled={!mlEnabled}
                onChange={(e) => setMlMinWin(e.target.value)}
              />
            </div>
          </div>

          <div className="tc-row">
            <div className="tc-field">
              <label>SL (× ATR)</label>
              <input
                type="number"
                min={1}
                max={5}
                step={0.1}
                value={slMultiplier}
                onChange={(e) => setSlMultiplier(e.target.value)}
              />
            </div>
            <div className="tc-field">
              <label>TP (× ATR)</label>
              <input
                type="number"
                min={1}
                max={5}
                step={0.1}
                value={tpMultiplier}
                onChange={(e) => setTpMultiplier(e.target.value)}
              />
            </div>
          </div>

          <div className="tc-row">
            <div className="tc-field">
              <label>
                <input
                  type="checkbox"
                  checked={ftmoEnabled}
                  onChange={(e) => setFtmoEnabled(e.target.checked)}
                  style={{ marginRight: 6, verticalAlign: 'middle' }}
                />
                Mode conformité FTMO
              </label>
            </div>
          </div>
          {ftmoEnabled && (
            <p className="tc-hint">
              Applique les vraies limites FTMO (perte journalière 5 %, drawdown 10 %,
              avec un arrêt un peu avant) calculées sur le solde initial ci-dessus,
              plus strictes que les limites de risque génériques.
            </p>
          )}

          <div className="tc-row">
            <div className="tc-field">
              <label>
                <input
                  type="checkbox"
                  checked={scalpEnabled}
                  onChange={(e) => setScalpEnabled(e.target.checked)}
                  style={{ marginRight: 6, verticalAlign: 'middle' }}
                />
                Règles de scalping (SCALPING_RULES.md)
              </label>
            </div>
          </div>
          {scalpEnabled && (
            <p className="tc-hint">
              Chaque entrée et sortie passe par le moteur de règles de scalping : sessions,
              spread, pauses, filtre ML, durée max, breakeven et trailing ATR.
              Ses valeurs (SL 1,2 / TP 1,5 ATR, risque 0,25 %, confiance min. 0,55…)
              remplacent les champs SL/TP, ML et cycle ci-dessus. Elles se règlent
              avec les variables SCALP_* du fichier .env.
            </p>
          )}

          {error && <div className="tc-error">{error}</div>}

          <button className="tc-btn tc-btn--start tc-btn--full" disabled={busy} onClick={handleStart}>
            {busy ? 'Démarrage…' : '▶ Démarrer le trading autonome'}
          </button>
          <p className="tc-hint">
            Le système passera et gérera tous les ordres automatiquement
            (entrées, stop-loss, take-profit, stops suiveurs).
          </p>
        </div>
      )}

      {error && !open && <span className="tc-error tc-error--inline">{error}</span>}
    </div>
  )
}
