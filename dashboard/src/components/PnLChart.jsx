import {
  AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { format, parseISO, isValid } from 'date-fns'

// ---------------------------------------------------------------------------
// PnL / Equity Curve chart
// ---------------------------------------------------------------------------

export default function PnLChart({ data }) {
  if (!data?.length) return <Empty />

  const equity  = data.map((d) => d.equity  ?? 0)
  const minEq   = Math.min(...equity)
  const maxEq   = Math.max(...equity)
  const yPad    = (maxEq - minEq) * 0.05 || 100
  const yDomain = [minEq - yPad, maxEq + yPad]

  // Reference line at the starting equity
  const startEquity = equity[0]

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="grad-up" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="var(--accent)" stopOpacity={0.35} />
            <stop offset="95%" stopColor="var(--accent)" stopOpacity={0}    />
          </linearGradient>
          <linearGradient id="grad-down" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="var(--red)" stopOpacity={0.30} />
            <stop offset="95%" stopColor="var(--red)" stopOpacity={0}    />
          </linearGradient>
        </defs>

        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--border)"
          vertical={false}
        />

        <XAxis
          dataKey="ts"
          tickFormatter={fmtTime}
          tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
          minTickGap={60}
        />

        <YAxis
          domain={yDomain}
          tick={{ fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'var(--font)' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
          width={52}
        />

        <Tooltip content={<CustomTooltip startEquity={startEquity} />} />

        <ReferenceLine
          y={startEquity}
          stroke="var(--text-dim)"
          strokeDasharray="4 4"
          strokeWidth={1}
        />

        <Area
          type="monotone"
          dataKey="equity"
          stroke="var(--accent)"
          strokeWidth={2}
          fill={`url(#${equity[equity.length - 1] >= startEquity ? 'grad-up' : 'grad-down'})`}
          dot={false}
          activeDot={{ r: 4, fill: 'var(--accent)', strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

function CustomTooltip({ active, payload, startEquity }) {
  if (!active || !payload?.length) return null
  const d      = payload[0].payload
  const change = d.equity - (startEquity ?? d.equity)
  const chPct  = startEquity ? (change / startEquity) * 100 : 0

  return (
    <div className="chart-tooltip">
      <div className="ct-time">{fmtTimeFull(d.ts)}</div>
      <div className="ct-row">
        <span className="ct-label">Equity</span>
        <span className="ct-val">${d.equity?.toLocaleString()}</span>
      </div>
      <div className="ct-row">
        <span className="ct-label">Change</span>
        <span className="ct-val" style={{ color: change >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {change >= 0 ? '+' : ''}${Math.abs(change).toFixed(2)} ({chPct.toFixed(2)}%)
        </span>
      </div>
      {d.drawdown_pct != null && (
        <div className="ct-row">
          <span className="ct-label">Drawdown</span>
          <span className="ct-val" style={{ color: d.drawdown_pct > 5 ? 'var(--red)' : 'var(--text-dim)' }}>
            {d.drawdown_pct.toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtTime(ts) {
  try {
    const d = typeof ts === 'string' ? parseISO(ts) : new Date(ts)
    return isValid(d) ? format(d, 'HH:mm') : ts
  } catch { return ts }
}

function fmtTimeFull(ts) {
  try {
    const d = typeof ts === 'string' ? parseISO(ts) : new Date(ts)
    return isValid(d) ? format(d, 'MMM d HH:mm:ss') : ts
  } catch { return ts }
}

function Empty() {
  return <div className="empty">Waiting for equity data…</div>
}
