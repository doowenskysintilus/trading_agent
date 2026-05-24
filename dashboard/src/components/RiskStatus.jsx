// ---------------------------------------------------------------------------
// Risk Status Panel
// ---------------------------------------------------------------------------

export default function RiskStatus({ risk, portfolio }) {
  const dailyPct  = portfolio?.daily_pnl_pct  ?? risk?.daily_pnl_pct  ?? 0
  const drawdown  = portfolio?.drawdown_pct    ?? risk?.drawdown_pct   ?? 0
  const leverage  = risk?.leverage             ?? 0
  const var95     = risk?.var_95               ?? 0
  const exposure  = risk?.total_exposure       ?? 0
  const nPos      = risk?.n_positions          ?? portfolio?.n_positions ?? 0

  // Normalise verdict string
  const rawVerdict = (risk?.verdict ?? '').toUpperCase().trim()
  const verdict    = rawVerdict || (risk ? 'OK' : 'NO DATA')

  const verdictColor = {
    'OK':      'var(--green)',
    'PASS':    'var(--green)',
    'WARN':    'var(--yellow)',
    'WARNING': 'var(--yellow)',
    'BLOCK':   'var(--red)',
    'FAIL':    'var(--red)',
    'NO DATA': 'var(--text-dim)',
  }[verdict] ?? 'var(--yellow)'

  return (
    <div className="risk-wrap">

      {/* Overall verdict */}
      <div className="risk-verdict" style={{ borderColor: verdictColor, color: verdictColor }}>
        <span className="verdict-dot" style={{ background: verdictColor }} />
        {verdict}
      </div>

      {/* Gauges */}
      <div className="risk-gauges">
        <Gauge
          label="Daily P&L"
          value={dailyPct}
          max={10}
          unit="%"
          warnAt={-2}
          dangerAt={-5}
          invertDirection
        />
        <Gauge
          label="Drawdown"
          value={drawdown}
          max={25}
          unit="%"
          warnAt={5}
          dangerAt={10}
        />
        <Gauge
          label="Leverage"
          value={leverage}
          max={10}
          unit="×"
          warnAt={3}
          dangerAt={6}
        />
      </div>

      {/* Key stats */}
      <div className="risk-stats">
        <RiskStat label="VaR 95%"   value={`$${Math.abs(var95).toFixed(0)}`} />
        <RiskStat label="Exposure"  value={exposure >= 1000 ? `$${(exposure / 1000).toFixed(1)}K` : `$${exposure.toFixed(0)}`} />
        <RiskStat label="Positions" value={nPos} />
        <RiskStat label="Daily PnL" value={`${dailyPct >= 0 ? '+' : ''}${dailyPct.toFixed(2)}%`}
                  color={dailyPct >= 0 ? 'var(--green)' : 'var(--red)'} />
      </div>

    </div>
  )
}

// ---------------------------------------------------------------------------
// Gauge bar
// ---------------------------------------------------------------------------

function Gauge({ label, value, max, unit, warnAt, dangerAt, invertDirection = false }) {
  const abs      = Math.abs(value)
  const fillPct  = Math.min((abs / max) * 100, 100)

  // For "daily P&L": danger when negative, OK when positive
  const isWarn   = invertDirection ? value < warnAt   : value > warnAt
  const isDanger = invertDirection ? value < dangerAt : value > dangerAt

  const color = isDanger ? 'var(--red)'
              : isWarn   ? 'var(--yellow)'
              : 'var(--green)'

  const sign = value >= 0 ? (invertDirection ? '+' : '') : '-'

  return (
    <div className="gauge">
      <div className="gauge-header">
        <span className="gauge-label">{label}</span>
        <span className="gauge-value" style={{ color }}>
          {sign}{abs.toFixed(1)}{unit}
        </span>
      </div>
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${fillPct}%`, background: color }} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Stat cell
// ---------------------------------------------------------------------------

function RiskStat({ label, value, color }) {
  return (
    <div className="risk-stat">
      <span className="rs-label">{label}</span>
      <span className="rs-value" style={color ? { color } : undefined}>{value}</span>
    </div>
  )
}
