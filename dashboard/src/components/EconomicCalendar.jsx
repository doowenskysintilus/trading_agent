import { useEffect, useState } from 'react'
import { format, parseISO, isValid } from 'date-fns'

import './EconomicCalendar.css'

const API_KEY = import.meta.env.VITE_API_KEY ?? ''
const AUTH_HEADERS = API_KEY ? { 'X-API-Key': API_KEY } : {}

export default function EconomicCalendar() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let active = true

    const fetchEvents = async () => {
      setLoading(true)
      setError(null)

      try {
        const response = await fetch('/api/calendar/events?next_n_hours=72', {
          headers: AUTH_HEADERS,
        })
        const payload = await response.json()

        if (!response.ok || payload?.success === false) {
          throw new Error(payload?.error || response.statusText)
        }

        setEvents(Array.isArray(payload.data?.events) ? payload.data.events : [])
      } catch (err) {
        setError(err?.message ?? 'Erreur lors du chargement du calendrier')
        setEvents([])
      } finally {
        if (active) setLoading(false)
      }
    }

    fetchEvents()
    const interval = setInterval(fetchEvents, 5 * 60 * 1000)
    return () => {
      active = false
      clearInterval(interval)
    }
  }, [])

  return (
    <div className="economic-calendar">
      {loading && <div className="calendar-empty">Chargement du calendrier…</div>}
      {error && <div className="calendar-error">{error}</div>}
      {!loading && !error && !events.length && (
        <div className="calendar-empty">Aucun événement économique trouvé sur les prochaines 72h.</div>
      )}

      {!loading && !error && events.length > 0 && (
        <div className="calendar-table">
          <div className="calendar-row calendar-row--header">
            <span>UTC</span>
            <span>Pays</span>
            <span>Événement</span>
            <span>Impact</span>
            <span>Forecast</span>
            <span>Préc.</span>
            <span>Réel</span>
          </div>
          {events.map((event, i) => (
            <div key={`${event.timestamp}-${event.name}-${i}`} className="calendar-row">
              <span>{fmtDate(event.timestamp)}</span>
              <span>{event.country ?? '—'}</span>
              <span className="calendar-event-name">{event.name ?? '—'}</span>
              <span className={`importance importance--${importanceClass(event.importance)}`}>
                {event.importance ?? '—'}
              </span>
              <span>{formatValue(event.forecast, event.units)}</span>
              <span>{formatValue(event.previous, event.units)}</span>
              <span>{formatValue(event.actual, event.units)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function formatValue(value, units) {
  if (value == null || value === '') return '—'
  return `${value}${units || ''}`
}

function fmtDate(ts) {
  if (!ts) return '—'
  try {
    const d = typeof ts === 'string' ? parseISO(ts) : new Date(ts)
    return isValid(d) ? format(d, 'yyyy-MM-dd HH:mm') : String(ts)
  } catch {
    return String(ts)
  }
}

function importanceClass(importance) {
  if (!importance) return 'low'
  const norm = String(importance).toLowerCase()
  if (norm.includes('high')) return 'high'
  if (norm.includes('medium')) return 'medium'
  return 'low'
}
