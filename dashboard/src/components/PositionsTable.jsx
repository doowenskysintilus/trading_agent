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
  const isBuy   = (p.direction ?? '').toUpperCase() === 'BUY'
  const pnl     = p.unrealized_pnl ?? p.pnl ?? 0
  const pnlPct  = p.pnl_pct ?? 0
  const isProfit = pnl >= 0

  return (
    <tr className="pos-row">
      <td className="pos-symbol">{p.symbol}</td>

      <td>
        <span className={`dir-chip ${isBuy ? 'long' : 'short'}`}>
          {isBuy ? '▲ BUY' : '▼ SELL'}
        </span>
      </td>

      <td className="num">{p.size ?? p.volume ?? '—'}</td>

      <td className="num align-right">{fmtPrice(p.entry_price ?? p.open_price)}</td>

      <td className="num align-right">{fmtPrice(p.current_price ?? p.price)}</td>

      <td className={`num align-right pnl-cell ${isProfit ? 'profit' : 'loss'}`}>
        {isProfit ? '+' : ''}${Math.abs(pnl).toFixed(2)}
        {pnlPct !== 0 && (
          <span className="pnl-pct">
            {' '}({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
          </span>
        )}
      </td>

      <td className="pos-strategy">{p.strategy ?? '—'}</td>
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
