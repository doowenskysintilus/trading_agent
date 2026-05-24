import { useRef, useEffect } from 'react'
import { format, parseISO, isValid } from 'date-fns'

// ---------------------------------------------------------------------------
// Live Trades Feed
// ---------------------------------------------------------------------------

export default function TradesFeed({ trades }) {
  if (!trades?.length) return <div className="empty">Awaiting trades…</div>

  return (
    <div className="trades-feed">
      <div className="tf-head tf-row">
        <span>TIME</span>
        <span>SYMBOL</span>
        <span>DIR</span>
        <span>SIZE</span>
        <span>EXIT</span>
        <span className="align-right">P&L</span>
        <span>STRATEGY</span>
      </div>
      {trades.map((t, i) => (
        <TradeRow key={`${t.ts ?? i}|${t.symbol ?? i}|${i}`} trade={t} isNew={i === 0} />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Single trade row
// ---------------------------------------------------------------------------

function TradeRow({ trade: t, isNew }) {
  const rowRef  = useRef(null)
  const isBuy   = (t.direction ?? '').toUpperCase() === 'BUY'
  const pnl     = t.pnl ?? 0
  const isProfit = pnl >= 0

  // Flash animation on new trade
  useEffect(() => {
    if (!isNew) return
    const el = rowRef.current
    if (!el) return
    el.classList.add('tf-flash')
    const id = setTimeout(() => el.classList.remove('tf-flash'), 1200)
    return () => clearTimeout(id)
  }, [isNew])

  return (
    <div ref={rowRef} className="tf-row tf-data">
      <span className="tf-time">{fmtTime(t.ts)}</span>
      <span className="tf-symbol">{t.symbol ?? '—'}</span>
      <span className={`tf-dir ${isBuy ? 'long' : 'short'}`}>
        {isBuy ? '▲ B' : '▼ S'}
      </span>
      <span className="tf-size num">{t.size ?? t.volume ?? '—'}</span>
      <span className="tf-exit">{t.exit_reason ?? '—'}</span>
      <span className={`tf-pnl num align-right ${isProfit ? 'profit' : 'loss'}`}>
        {isProfit ? '+' : ''}${Math.abs(pnl).toFixed(2)}
      </span>
      <span className="tf-strategy">{t.strategy ?? '—'}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtTime(ts) {
  if (!ts) return '—'
  try {
    const d = typeof ts === 'string' ? parseISO(ts) : new Date(ts)
    return isValid(d) ? format(d, 'HH:mm:ss') : String(ts)
  } catch { return String(ts) }
}
