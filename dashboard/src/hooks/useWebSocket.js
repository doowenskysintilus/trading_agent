import { useState, useEffect, useRef, useCallback } from 'react'

const BASE_DELAY_MS = 2_000
const MAX_DELAY_MS  = 30_000

/**
 * useWebSocket
 * ============
 * Connects to a WebSocket URL and returns the latest parsed message.
 * Automatically reconnects with exponential back-off on disconnect.
 *
 * @param {string} url  - Full WS URL (ws:// or wss://)
 * @returns {{ lastMessage: object|null, status: string, send: Function }}
 *   status: 'connecting' | 'connected' | 'reconnecting' | 'disconnected'
 */
export default function useWebSocket(url) {
  const [lastMessage, setLastMessage] = useState(null)
  const [status,      setStatus]      = useState('connecting')

  const wsRef     = useRef(null)
  const delayRef  = useRef(BASE_DELAY_MS)
  const mountRef  = useRef(true)
  const timerRef  = useRef(null)

  const connect = useCallback(() => {
    if (!mountRef.current || !url) {
      setStatus('disconnected')
      return
    }

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountRef.current) return
        setStatus('connected')
        delayRef.current = BASE_DELAY_MS   // reset back-off
      }

      ws.onmessage = (e) => {
        if (!mountRef.current) return
        try {
          setLastMessage(JSON.parse(e.data))
        } catch {
          // ignore malformed frames
        }
      }

      ws.onclose = () => {
        if (!mountRef.current) return
        setStatus('reconnecting')
        const delay      = delayRef.current
        delayRef.current = Math.min(delay * 2, MAX_DELAY_MS)
        timerRef.current = setTimeout(connect, delay)
      }

      ws.onerror = () => {
        ws.close()   // triggers onclose → reconnect
      }
    } catch {
      setStatus('disconnected')
    }
  }, [url])

  useEffect(() => {
    mountRef.current = true
    connect()
    return () => {
      mountRef.current = false
      clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        typeof data === 'string' ? data : JSON.stringify(data)
      )
    }
  }, [])

  return { lastMessage, status, send }
}
