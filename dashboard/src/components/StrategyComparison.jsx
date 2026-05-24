import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const PALETTE = [
  '#00c9a7', '#0096ff', '#f59e0b', '#c084fc',
  '#fb7185', '#34d399', '#60a5fa', '#fbbf24',
]

// ---------------------------------------------------------------------------
// Strategy Performance Comparison
// ---------------------------------------------------------------------------

export default function StrategyComparison({ strategies }) {
  if (!strategies?.length) return <Empty />

  const data = strategies.map((s, i) => ({
    name:   shortName(s.strategy ?? s.name ?? `S${i}`),
    full:   s.strategy ?? s.name ?? `S${i}`,
    pnl:    +(s.cumulative_pnl ?? s.pnl ?? 0),
    win:    +((s.win_rate ?? 0) * 100),
    sharpe: +(s.sharpe ?? 0),
    trades: s.n_trades ?? 0,
    color:  PALETTE[i % PALETTE.length],
  }))

  return (
    <div className="strat-wrap">
      {/* PnL bar chart */}
      <div className="strat-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 4, right: 10, left: 0, bottom: 0 }}
            barSize={28}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: 'var(--text-dim)', fontSize: 11, fontFamily: 'var(--font)' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
              axisLine={false}
              tickLine={false}
              width={52}
              tickFormatter={(v) =>
                v >= 0
                  ? `+$${Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + 'K' : v.toFixed(0)}`
                  : `-$${Math.abs(v) >= 1000 ? (Math.abs(v) / 1000).toFixed(1) + 'K' : Math.abs(v).toFixed(0)}`
              }
            />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.04)' }}
              content={<BarTooltip />}
            />
            <Bar dataKey="pnl" radius={[4, 4, 0, 0]} isAnimationActive={false}>
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.pnl >= 0 ? entry.color : 'var(--red)'}
                  opacity={0.9}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stats table */}
      <div className="strat-table">
        <div className="st-row st-head">
          <span>STRATEGY</span>
          <span className="align-right">TRADES</span>
          <span className="align-right">WIN %</span>
          <span className="align-right">SHARPE</span>
        </div>
        {data.map((s, i) => (
          <div key={i} className="st-row">
            <span className="st-name" style={{ color: s.color }}>{s.full}</span>
            <span className="align-right">{s.trades}</span>
            <span
              className="align-right"
              style={{ color: s.win >= 50 ? 'var(--green)' : 'var(--red)' }}
            >
              {s.win.toFixed(1)}%
            </span>
            <span
              className="align-right"
              style={{
                color: s.sharpe >= 1.5 ? 'var(--green)'
                     : s.sharpe >= 0   ? 'var(--text)'
                     : 'var(--red)',
              }}
            >
              {s.sharpe.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function BarTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="ct-time">{d.full}</div>
      <div className="ct-row">
        <span className="ct-label">Cum. PnL</span>
        <span className="ct-val" style={{ color: d.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {d.pnl >= 0 ? '+' : ''}${Math.abs(d.pnl).toFixed(2)}
        </span>
      </div>
      <div className="ct-row">
        <span className="ct-label">Win Rate</span>
        <span className="ct-val">{d.win.toFixed(1)}%</span>
      </div>
      <div className="ct-row">
        <span className="ct-label">Trades</span>
        <span className="ct-val">{d.trades}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shortName(name) {
  // MomentumAlpha → Momentum, mean_reversion_alpha → MeanRev
  return name.replace(/alpha$/i, '').replace(/_/g, '').trim().slice(0, 10)
}

function Empty() {
  return <div className="empty">No strategy data yet…</div>
}
