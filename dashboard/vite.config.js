import { defineConfig, createLogger } from 'vite'
import react from '@vitejs/plugin-react'

// ---------------------------------------------------------------------------
// Custom logger.
//
// Vite's internal proxy attaches its own `socket.on('error')` handler to the
// WebSocket upgrade socket and logs raw stack traces such as:
//
//   [vite] ws proxy socket error:
//   Error: write ECONNABORTED ...
//
// These are harmless: they happen whenever the dashboard's auto-reconnecting
// WebSocket retries while the backend is restarting / momentarily unavailable.
// We can't unhook Vite's internal listener from the config, so instead we wrap
// the logger and collapse this noise into a single throttled, readable warning.
// ---------------------------------------------------------------------------
const NOISY_PROXY_RE = /ws proxy socket error|ECONNABORTED|ECONNRESET|ECONNREFUSED/i
let lastWarn = 0

const baseLogger = createLogger()
const quietLogger = {
  ...baseLogger,
  error(msg, options) {
    if (typeof msg === 'string' && NOISY_PROXY_RE.test(msg)) {
      const now = Date.now()
      if (now - lastWarn > 5000) {
        lastWarn = now
        baseLogger.warn(
          '[vite proxy] backend connection dropped (auto-reconnecting). ' +
          'If this persists, make sure the API is running on http://localhost:8000 ' +
          '(start it with: python -m api.main).',
        )
      }
      return
    }
    baseLogger.error(msg, options)
  },
}

export default defineConfig({
  plugins: [react()],
  customLogger: quietLogger,
  server: {
    port: 3000,
    proxy: {
      // REST API calls
      '/api': {
        target:       'http://localhost:8000',
        changeOrigin: true,
        rewrite:      (path) => path.replace(/^\/api/, ''),
      },
      // WebSocket
      '/ws': {
        target:       'ws://localhost:8000',
        ws:           true,
        changeOrigin: true,
      },
    },
  },
})
