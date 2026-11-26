import { useCallback, useEffect, useRef, useState } from 'react'
import { playAlertSound, unlockAudio } from '../alertSound'
import { emptyEpisode, evaluateEpisode } from '../alertLogic'
import {
  addDetectionStats,
  combineDashboardStats,
  initialDashboardStats,
} from '../dashboardStats'
import { countViolatingPeople } from '../personViolations'

const SAMPLE_MS = 3000
const MAX_HISTORY = 20

export function useAlerts() {
  const [muted, setMuted] = useState(false)
  const [history, setHistory] = useState([])
  const [rawStats, setRawStats] = useState(initialDashboardStats)
  const mutedRef = useRef(false)
  const episodeRef = useRef(emptyEpisode())
  const lastSampleRef = useRef(0)

  useEffect(() => {
    const events = ['pointerdown', 'keydown', 'drop']
    events.forEach((event) => window.addEventListener(event, unlockAudio))
    return () =>
      events.forEach((event) => window.removeEventListener(event, unlockAudio))
  }, [])

  const toggleMute = useCallback(() => {
    mutedRef.current = !mutedRef.current
    setMuted(mutedRef.current)
    if (!mutedRef.current) playAlertSound()
  }, [])

  const notify = useCallback((detections, source, personViolationCount = null) => {
    const now = Date.now()
    const missing = detections.filter((detection) =>
      detection.class_name.startsWith('NO-'),
    )
    const violationPeople =
      source === 'Webcam' && Number.isInteger(personViolationCount)
        ? personViolationCount
        : (countViolatingPeople(detections) ?? (missing.length > 0 ? 1 : 0))

    let newViolations = violationPeople
    if (source === 'Webcam') {
      const counts =
        violationPeople > 0 ? { 'Personne en infraction': violationPeople } : {}
      const result = evaluateEpisode(episodeRef.current, counts, now)
      episodeRef.current = result.episode
      newViolations = result.newViolations
    }

    const sampled =
      source !== 'Webcam' || now - lastSampleRef.current >= SAMPLE_MS
    if (sampled) lastSampleRef.current = now

    // L'image ajoute ses résultats. La webcam ne conserve que son pic de frame
    // pour qu'une personne présente pendant plusieurs secondes ne soit pas recomptée.
    if (sampled || newViolations > 0) {
      setRawStats((previous) =>
        addDetectionStats(previous, detections, source, violationPeople),
      )
    }

    if (newViolations === 0) return
    if (!mutedRef.current) playAlertSound()

    const classes = [...new Set(missing.map((detection) => detection.class_name))]
    const entry = {
      id: `${now}-${Math.random()}`,
      time: new Date(now),
      source,
      count: source === 'Webcam' ? newViolations : violationPeople,
      unit: 'personne',
      classes,
    }
    setHistory((previous) => [entry, ...previous].slice(0, MAX_HISTORY))
  }, [])

  const clearHistory = useCallback(() => setHistory([]), [])
  const stats = combineDashboardStats(rawStats)

  return { muted, toggleMute, history, notify, clearHistory, stats }
}
