// ---------------------------------------------------------------------------
// Multi-account support: the API base can be switched at runtime (picked in
// AccountSwitcher, persisted to localStorage) instead of being frozen at
// build time — each account is a separate full API instance on its own port
// (see run_all_accounts.py), so "switching accounts" just means pointing
// REST/WebSocket calls at a different port and reloading.
// ---------------------------------------------------------------------------

const ACTIVE_PORT_KEY = 'qf_active_port'

const BUILD_API_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const BUILD_WS_URL  = import.meta.env.VITE_WS_URL ?? ''

function getActivePort() {
  return localStorage.getItem(ACTIVE_PORT_KEY) || ''
}

function setActivePort(port) {
  if (port) localStorage.setItem(ACTIVE_PORT_KEY, String(port))
  else localStorage.removeItem(ACTIVE_PORT_KEY)
}

const API_KEY = import.meta.env.VITE_API_KEY ?? ''

// A saved port overrides the build-time host:port; the build-time value
// (including any deployed hostname) still wins for everything else, so this
// only ever changes which local port dev/desktop users are pointed at.
const _activePort = getActivePort()

const API_BASE = _activePort
  ? `http://localhost:${_activePort}`
  : BUILD_API_URL

const WS_URL = _activePort
  ? `ws://localhost:${_activePort}/ws${API_KEY ? `?api_key=${encodeURIComponent(API_KEY)}` : ''}`
  : BUILD_WS_URL || (() => {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.host
      const qs = API_KEY ? `?api_key=${encodeURIComponent(API_KEY)}` : ''
      return `${proto}//${host}/ws${qs}`
    })()

const AUTH_HEADERS = {
  'Content-Type': 'application/json',
  ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
}

const buildApiPath = (path) => API_BASE ? `${API_BASE}${path}` : path

export {
  API_KEY,
  API_BASE,
  AUTH_HEADERS,
  WS_URL,
  buildApiPath,
  getActivePort,
  setActivePort,
}
