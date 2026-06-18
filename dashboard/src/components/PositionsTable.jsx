// ---------------------------------------------------------------------------
// Open Positions Table
// ---------------------------------------------------------------------------

export default function PositionsTable({ positions }) {
  if (!positions?.length) {
    return <div className="empty">No open positions</div>
  }

  return (
    <div className="table-wrap">
      <table className="pos-table">
        <thead>
          <tr>
            <th>SYMBOL</th>
            <th>DIR</th>
            <th>SIZE</th>
            <th className="align-right">ENTRY</th>
            <th className="align-right">CURRENT</th>
            <th className="align-right">UNREALISED P&L</th>
            <th>STRATEGY</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((p, i) => <PositionRow key={p.ticket ?? i} pos={p} />)}
        </tbody>
      </table>
    </div>
  )
}

function PositionRow({ pos: p }) {
  const dir     = toDirection(p.direction ?? p.side ?? p.signal)
  const isBuy   = dir === 'BUY'
  const size    = pickNum(p.size, p.volume, p.lot, p.quantity)
  const entry   = pickNum(p.entry_price, p.open_price, p.price_open, p.entry)
  const current = pickNum(p.current_price, p.price_current, p.price)
  const pnl     = pickNum(p.unrealized_pnl, p.profit, p.pnl) ?? 0
  const pnlPct  = p.pnl_pct ?? 0
  const isProfit = pnl >= 0
  const strategy = Array.isArray(p.strategy)
    ? p.strategy.join(', ')
    : p.strategy

  return (
    <tr className="pos-row">
      <td className="pos-symbol">{p.symbol}</td>

      <td>
        <span className={`dir-chip ${isBuy ? 'long' : 'short'}`}>
          {isBuy ? '▲ BUY' : '▼ SELL'}
        </span>
      </td>

      <td className="num">{size != null ? size : '—'}</td>

      <td className="num align-right">{fmtPrice(entry)}</td>

      <td className="num align-right">{fmtPrice(current)}</td>

      <td className={`num align-right pnl-cell ${isProfit ? 'profit' : 'loss'}`}>
        {isProfit ? '+' : ''}${Math.abs(pnl).toFixed(2)}
        {pnlPct !== 0 && (
          <span className="pnl-pct">
            {' '}({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
          </span>
        )}
      </td>

      <td className="pos-strategy">{strategy ?? '—'}</td>
    </tr>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtPrice(p) {
  if (p == null) return '—'
  return p >= 100 ? p.toFixed(2) : p.toFixed(5)
}

function pickNum(...vals) {
  for (const v of vals) {
    if (v == null || v === '') continue
    const n = Number(v)
    if (!Number.isNaN(n) && Number.isFinite(n)) return n
  }
  return null
}

function toDirection(v) {
  if (typeof v === 'number') return v >= 0 ? 'BUY' : 'SELL'
  const s = String(v ?? '').trim().toUpperCase()
  if (s === '1' || s === '+1' || s === 'BUY' || s === 'LONG') return 'BUY'
  if (s === '-1' || s === 'SELL' || s === 'SHORT') return 'SELL'
  return 'SELL'
}
