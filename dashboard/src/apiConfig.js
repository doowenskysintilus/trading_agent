const API_KEY = import.meta.env.VITE_API_KEY ?? ''
const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const WS_URL = import.meta.env.VITE_WS_URL || (() => {
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
}
