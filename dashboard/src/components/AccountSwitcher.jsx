import { useState, useCallback } from 'react'
import { getActivePort, setActivePort } from '../apiConfig.js'

const ACCOUNTS_KEY = 'qf_accounts'   // [{ label, port }] saved account list

function loadSavedAccounts() {
  try {
    const raw = JSON.parse(localStorage.getItem(ACCOUNTS_KEY) || '[]')
    return Array.isArray(raw) ? raw : []
  } catch {
    return []
  }
}

function saveAccounts(list) {
  localStorage.setItem(ACCOUNTS_KEY, JSON.stringify(list))
}

// ---------------------------------------------------------------------------
// AccountSwitcher
// ===========================================================================
// Multi-account support: each account is a fully separate API instance on
// its own port (run_all_accounts.py). This dashboard only ever talks to
// ONE port at a time — switching accounts means picking a different port
// and reloading, not merging data from several accounts into one view.
// ---------------------------------------------------------------------------
export default function AccountSwitcher() {
  const [open,     setOpen]     = useState(false)
  const [accounts, setAccounts] = useState(loadSavedAccounts)
  const [newLabel, setNewLabel] = useState('')
  const [newPort,  setNewPort]  = useState('')

  const activePort = getActivePort()
  const active = accounts.find((a) => String(a.port) === String(activePort))
  const activeLabel = active ? active.label : (activePort ? `Port ${activePort}` : 'Default (.env)')

  const switchTo = useCallback((port) => {
    setActivePort(port)
    window.location.reload()
  }, [])

  const addAccount = useCallback(() => {
    const port = Number(newPort)
    if (!newLabel.trim() || !port) return
    const next = [...accounts.filter((a) => String(a.port) !== String(port)),
                   { label: newLabel.trim(), port }]
    setAccounts(next)
    saveAccounts(next)
    setNewLabel('')
    setNewPort('')
  }, [accounts, newLabel, newPort])

  const removeAccount = useCallback((port) => {
    const next = accounts.filter((a) => String(a.port) !== String(port))
    setAccounts(next)
    saveAccounts(next)
  }, [accounts])

  return (
    <div className="tc" style={{ position: 'relative' }}>
      <button className="tc-btn" onClick={() => setOpen((v) => !v)} title="Switch MT5 account">
        ◇ {activeLabel}
      </button>

      {open && (
        <div className="tc-popover">
          <div className="tc-popover-head">
            <span>Accounts</span>
            <button className="tc-x" onClick={() => setOpen(false)}>✕</button>
          </div>

          <div className="tc-field">
            <label>Saved accounts (one API instance per port)</label>
            {accounts.length === 0 && (
              <p className="tc-hint">No saved accounts yet — add one below. Ports come from
                accounts.json / run_all_accounts.py on the backend.</p>
            )}
            {accounts.map((a) => (
              <div key={a.port} className="tc-row" style={{ alignItems: 'center', marginBottom: 6 }}>
                <button
                  className="tc-chip"
                  style={String(a.port) === String(activePort) ? { borderColor: 'var(--green)', color: 'var(--green)' } : undefined}
                  onClick={() => switchTo(a.port)}
                >
                  {a.label} :{a.port}
                </button>
                <button className="tc-x" onClick={() => removeAccount(a.port)}>✕</button>
              </div>
            ))}
            {activePort && (
              <button className="tc-chip" onClick={() => switchTo('')}>
                Use default (.env / build) port
              </button>
            )}
          </div>

          <div className="tc-field">
            <label>Add account</label>
            <div className="tc-row">
              <input
                type="text" placeholder="Label (e.g. FTMO 10k #1)"
                value={newLabel} onChange={(e) => setNewLabel(e.target.value)}
              />
              <input
                type="number" placeholder="Port" style={{ maxWidth: 90 }}
                value={newPort} onChange={(e) => setNewPort(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addAccount()}
              />
              <button className="tc-btn" onClick={addAccount}>+ Add</button>
            </div>
          </div>

          <p className="tc-hint">
            Switching reloads the page and points every REST/WebSocket call at the
            selected port — one account's full view at a time, not merged.
          </p>
        </div>
      )}
    </div>
  )
}
