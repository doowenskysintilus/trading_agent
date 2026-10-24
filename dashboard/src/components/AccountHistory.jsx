import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  ComposedChart, Bar, Line, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AUTH_HEADERS, buildApiPath } from '../apiConfig.js'

// ---------------------------------------------------------------------------
// AccountHistory
// Full MT5 account history: realised P&L since the account opened and per
// hour / day / week / month / year over any date range, plus every closed
// position (bot and manual). Data: /api/account/{summary,pnl,history}.
// Periods are cut on the browser's local calendar (tz_offset_min).
// ---------------------------------------------------------------------------

const GROUPS = [
  { value: 'hour',  label: 'Heure' },
  { value: 'day',   label: 'Jour' },
  { value: 'week',  label: 'Semaine' },
  { value: 'month', label: 'Mois' },
  { value: 'year',  label: 'Année' },
]

const SOURCES = [
  { value: 'all',    label: 'Tous les trades' },
  { value: 'bot',    label: 'Bot uniquement' },
  { value: 'manual', label: 'Manuels uniquement' },
]

const QUICK_RANGES = [
  { label: "Aujourd'hui", days: 0,    group: 'hour' },
  { label: '7 jours',     days: 7,    group: 'day' },
  { label: '30 jours',    days: 30,   group: 'day' },
  { label: '1 an',        days: 365,  group: 'month' },
  { label: 'Tout',        days: null, group: 'month' },
]

const TZ_OFFSET_MIN = -new Date().getTimezoneOffset()
const PAGE = 50

export default function AccountHistory() {
  const [summary, setSummary] = useState(null)
  const [fees,    setFees]    = useState(null)
  const [series,  setSeries]  = useState([])
  const [trades,  setTrades]  = useState([])
  const [total,   setTotal]   = useState(0)
  const [group,   setGroup]   = useState('day')
  const [source,  setSource]  = useState('all')
  const [start,   setStart]   = useState(isoDay(-30))
  const [end,     setEnd]     = useState(isoDay(0))
  const [shown,   setShown]   = useState(PAGE)
  const [error,   setError]   = useState('')
  const [loading, setLoading] = useState(false)

  const query = useCallback((extra = {}) => {
    const p = new URLSearchParams({ source, tz_offset_min: String(TZ_OFFSET_MIN), ...extra })
    if (start) p.set('start', start)
    if (end) p.set('end', end)
    return p.toString()
  }, [source, start, end])

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const get = async (path) => {
        const r = await fetch(buildApiPath(path), { headers: AUTH_HEADERS })
        const d = await r.json().catch(() => ({}))
        if (!r.ok || d?.success === false) throw new Error(d?.detail || d?.error || `HTTP ${r.status}`)
        return d.data
      }
      const [s, p, h, f] = await Promise.all([
        get(`/api/account/summary?source=${source}&tz_offset_min=${TZ_OFFSET_MIN}`),
        get(`/api/account/pnl?${query({ group })}`),
        get(`/api/account/history?${query({ limit: '10000' })}`),
        get('/api/account/commissions').catch(() => null),
      ])
      setSummary(s)
      setFees(f)
      setSeries(p || [])
      setTrades(h?.trades || [])
      setTotal(h?.total || 0)
      setShown(PAGE)
    } catch (e) {
      setError(e.message || 'Chargement impossible')
    } finally {
      setLoading(false)
    }
  }, [source, group, query])

  useEffect(() => { load() }, [load])

  // Refresh every minute so new closed trades appear without reloading.
  useEffect(() => {
    const t = setInterval(load, 60000)
    return () => clearInterval(t)
  }, [load])

  const applyQuick = (q) => {
    setEnd(isoDay(0))
    setStart(q.days == null ? '' : isoDay(-q.days))
    setGroup(q.group)
  }

  const range = useMemo(() => {
    const net = trades.reduce((a, t) => a + t.net, 0)
    const wins = trades.filter((t) => t.net > 0).length
    return { net, n: trades.length, wins }
  }, [trades])

  return (
    <div className="ah">
      {/* ── Totals ─────────────────────────────────────────────────────── */}
      <div className="ah-tiles">
        <Tile label="Depuis l'ouverture" data={summary?.all}
              sub={summary?.account_opened ? `compte ouvert le ${fmtDay(summary.account_opened)}` : ''} />
        <Tile label="Cette année" data={summary?.year} />
        <Tile label="Ce mois" data={summary?.month} />
        <Tile label="Cette semaine" data={summary?.week} />
        <Tile label="Aujourd'hui" data={summary?.today} />
        <div className="ah-tile ah-tile--stats">
          <span className="ah-tile-label">Statistiques (tout)</span>
          <span>Taux de gain <b>{pct(summary?.win_rate)}</b></span>
          <span>Profit factor <b>{summary?.profit_factor ?? '—'}</b></span>
          <span>Rendement <b className={cls(summary?.return_pct)}>
            {summary?.return_pct != null ? `${sign(summary.return_pct)}${Math.abs(summary.return_pct).toFixed(2)} %` : '—'}
          </b> sur {money(summary?.deposits)} déposés</span>
        </div>
      </div>

      {/* ── Filters ────────────────────────────────────────────────────── */}
      <div className="ah-filters">
        <div className="ah-quick">
          {QUICK_RANGES.map((q) => (
            <button key={q.label} className="ah-chip" onClick={() => applyQuick(q)}>{q.label}</button>
          ))}
        </div>
        <label>Du <input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label>
        <label>au <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} /></label>
        <label>Regrouper par
          <select value={group} onChange={(e) => setGroup(e.target.value)}>
            {GROUPS.map((g) => <option key={g.value} value={g.value}>{g.label}</option>)}
          </select>
        </label>
        <label>Trades
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            {SOURCES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </label>
        <button className="ah-chip" onClick={load} disabled={loading}>
          {loading ? 'Chargement…' : '↻ Actualiser'}
        </button>
      </div>
      {error && <div className="ah-error">{error}</div>}

      {/* ── P&L per period ─────────────────────────────────────────────── */}
      <div className="ah-card">
        <div className="ah-card-head">
          <span>Gain / perte par {GROUPS.find((g) => g.value === group)?.label.toLowerCase()}</span>
          <span>
            Sur la période : <b className={cls(range.net)}>{sign(range.net)}{money(Math.abs(range.net))}</b>
            {' · '}{range.n} trades · {range.n ? Math.round(100 * range.wins / range.n) : 0} % gagnants
          </span>
        </div>
        {series.length ? (
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={series} margin={{ top: 10, right: 16, left: 8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="period" tickFormatter={(v) => fmtPeriod(v, group)} minTickGap={24}
                     tick={{ fill: 'var(--text-dim)', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis yAxisId="net" tickFormatter={(v) => `${v} $`} width={60}
                     tick={{ fill: 'var(--text-dim)', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis yAxisId="cum" orientation="right" tickFormatter={(v) => `${v} $`} width={60}
                     tick={{ fill: 'var(--accent)', fontSize: 10 }} axisLine={false} tickLine={false} />
              <ReferenceLine yAxisId="net" y={0} stroke="var(--text-dim)" />
              <Tooltip content={<PeriodTooltip group={group} />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Bar yAxisId="net" dataKey="net" isAnimationActive={false} maxBarSize={40}>
                {series.map((p, i) => (
                  <Cell key={i} fill={p.net >= 0 ? 'var(--green)' : 'var(--red)'} />
                ))}
              </Bar>
              <Line yAxisId="cum" dataKey="cumulative" stroke="var(--accent)" strokeWidth={2}
                    dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="ah-empty">Aucun trade fermé sur cette période.</div>
        )}
        <div className="ah-legend">
          <span><i style={{ background: 'var(--green)' }} /> gain de la période</span>
          <span><i style={{ background: 'var(--red)' }} /> perte de la période</span>
          <span><i style={{ background: 'var(--accent)' }} /> cumul sur la période (axe de droite)</span>
        </div>
      </div>

      {/* ── Real fees per symbol ───────────────────────────────────────── */}
      {fees?.measured && Object.keys(fees.measured).length > 0 && (
        <div className="ah-card">
          <div className="ah-card-head">
            <span>Commissions réelles par symbole</span>
            <span>
              mesurées sur les derniers trades fermés — utilisées par les règles de scalping
              (défaut {(fees.default_rate_per_side * 100).toFixed(4)} %/côté sans historique)
            </span>
          </div>
          <div className="ah-table-wrap">
            <table className="ah-table">
              <thead>
                <tr><th>Symbole</th><th>Par côté (% du notionnel)</th><th>Aller-retour par lot</th><th>Trades mesurés</th></tr>
              </thead>
              <tbody>
                {Object.entries(fees.measured).sort().map(([sym, f]) => (
                  <tr key={sym}>
                    <td>{sym}</td>
                    <td>{(f.rate_per_side * 100).toFixed(4)} %</td>
                    <td>{money(f.per_lot_round_trip)}</td>
                    <td>{f.trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Closed trades ──────────────────────────────────────────────── */}
      <div className="ah-card">
        <div className="ah-card-head">
          <span>Historique des trades MT5 ({total})</span>
          <span>gains nets : commissions et swaps déduits</span>
        </div>
        <div className="ah-table-wrap">
          <table className="ah-table">
            <thead>
              <tr>
                <th>Fermé le</th><th>Symbole</th><th>Sens</th><th>Lots</th>
                <th>Entrée</th><th>Sortie</th><th>Durée</th>
                <th>Brut</th><th>Frais</th><th>Net</th><th>Sortie par</th><th>Origine</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, shown).map((t) => (
                <tr key={t.position_id}>
                  <td title={`ouvert le ${fmtDateTime(t.open_time)}`}>{fmtDateTime(t.close_time)}</td>
                  <td>{t.symbol}</td>
                  <td className={t.direction === 'BUY' ? 'pos' : 'neg'}>{t.direction === 'BUY' ? 'Achat' : 'Vente'}</td>
                  <td>{t.volume}</td>
                  <td>{t.open_price}</td>
                  <td>{t.close_price}</td>
                  <td>{fmtDuration(t.duration_s)}</td>
                  <td className={cls(t.profit)}>{t.profit.toFixed(2)}</td>
                  <td>{(t.commission + t.swap).toFixed(2)}</td>
                  <td className={cls(t.net)}><b>{sign(t.net)}{t.net.toFixed(2)}</b></td>
                  <td>{t.exit_reason}</td>
                  <td>{t.source === 'bot' ? 'Bot' : 'Manuel'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!trades.length && <div className="ah-empty">Aucun trade fermé sur cette période.</div>}
        </div>
        {shown < trades.length && (
          <button className="ah-chip ah-more" onClick={() => setShown((n) => n + PAGE)}>
            Afficher {Math.min(PAGE, trades.length - shown)} de plus ({trades.length - shown} restants)
          </button>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components & helpers
// ---------------------------------------------------------------------------

function Tile({ label, data, sub }) {
  const net = data?.net
  return (
    <div className="ah-tile">
      <span className="ah-tile-label">{label}</span>
      <span className={`ah-tile-value ${cls(net)}`}>
        {net == null ? '—' : `${sign(net)}${money(Math.abs(net))}`}
      </span>
      <span className="ah-tile-sub">{data ? `${data.trades} trades` : ''}{sub ? ` · ${sub}` : ''}</span>
    </div>
  )
}

function PeriodTooltip({ active, payload, group }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="ct-time">{fmtPeriod(p.period, group, true)}</div>
      <div className="ct-row"><span className="ct-label">Résultat</span>
        <span className="ct-val" style={{ color: p.net >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {sign(p.net)}{money(Math.abs(p.net))}</span></div>
      <div className="ct-row"><span className="ct-label">Trades</span>
        <span className="ct-val">{p.trades} ({p.wins} gagnants)</span></div>
      <div className="ct-row"><span className="ct-label">Cumul</span>
        <span className="ct-val">{sign(p.cumulative)}{money(Math.abs(p.cumulative))}</span></div>
    </div>
  )
}

function isoDay(offsetDays) {
  const d = new Date()
  d.setDate(d.getDate() + offsetDays)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

// API periods are local wall-clock times without timezone: parse as local.
function localDate(v) {
  const [date, time = '00:00:00'] = String(v).split('T')
  const [y, m, d] = date.split('-').map(Number)
  const [hh, mm] = time.split(':').map(Number)
  return new Date(y, (m || 1) - 1, d || 1, hh || 0, mm || 0)
}

function fmtPeriod(v, group, long = false) {
  const d = localDate(v)
  switch (group) {
    case 'hour':  return d.toLocaleString('fr-FR', long
      ? { weekday: 'short', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }
      : { day: '2-digit', month: '2-digit', hour: '2-digit' })
    case 'week':  return `sem. du ${d.toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' })}`
    case 'month': return d.toLocaleDateString('fr-FR', { month: long ? 'long' : 'short', year: 'numeric' })
    case 'year':  return String(d.getFullYear())
    default:      return d.toLocaleDateString('fr-FR', long
      ? { weekday: 'long', day: '2-digit', month: 'long', year: 'numeric' }
      : { day: '2-digit', month: 'short' })
  }
}

function fmtDateTime(iso) {
  return new Date(iso).toLocaleString('fr-FR', {
    day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit',
  })
}

function fmtDay(iso) {
  return new Date(iso).toLocaleDateString('fr-FR', { day: '2-digit', month: 'long', year: 'numeric' })
}

function fmtDuration(s) {
  if (s < 60) return `${s} s`
  if (s < 3600) return `${Math.round(s / 60)} min`
  if (s < 86400) return `${(s / 3600).toFixed(1)} h`
  return `${(s / 86400).toFixed(1)} j`
}

function money(v) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return `${Number(v).toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} $`
}

const sign = (v) => (v > 0 ? '+' : v < 0 ? '−' : '')
const cls = (v) => (v > 0 ? 'pos' : v < 0 ? 'neg' : '')
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)} %`)
