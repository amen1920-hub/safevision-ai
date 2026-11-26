import { useCallback, useEffect, useRef, useState } from 'react'
import { playAlertSound, unlockAudio } from '../alertSound'

const COOLDOWN_MS = 3000 // webcam : au plus une alerte toutes les 3 s
const MAX_HISTORY = 20

export function useAlerts() {
  const [muted, setMuted] = useState(false)
  const [history, setHistory] = useState([])
  const [stats, setStats] = useState({
    analyses: 0,
    objects: 0,
    violations: 0,
    nonCompliant: 0,
    byClass: {},
  })
  const mutedRef = useRef(false)
  const lastWebcamAlertRef = useRef(0)

  // Autorise le son dès la première interaction avec la page
  useEffect(() => {
    const events = ['pointerdown', 'keydown', 'drop']
    events.forEach((e) => window.addEventListener(e, unlockAudio))
    return () =>
      events.forEach((e) => window.removeEventListener(e, unlockAudio))
  }, [])

  const toggleMute = useCallback(() => {
    mutedRef.current = !mutedRef.current
    setMuted(mutedRef.current)
    if (!mutedRef.current) playAlertSound() // bip de test à la réactivation
  }, [])

  // Appelée après chaque détection. 'source' vaut 'Image' ou 'Webcam'.
  const notify = useCallback((detections, source) => {
    const missing = detections.filter((d) => d.class_name.startsWith('NO-'))
    setStats((prev) => {
      const byClass = { ...prev.byClass }
      for (const d of detections) {
        byClass[d.class_name] = (byClass[d.class_name] || 0) + 1
      }
      return {
        analyses: prev.analyses + 1,
        objects: prev.objects + detections.length,
        violations: prev.violations + missing.length,
        nonCompliant: prev.nonCompliant + (missing.length > 0 ? 1 : 0),
        byClass,
      }
    })
    if (missing.length === 0) return

    const now = Date.now()
    if (source === 'Webcam') {
      if (now - lastWebcamAlertRef.current < COOLDOWN_MS) return
      lastWebcamAlertRef.current = now
    }

    if (!mutedRef.current) playAlertSound()

    const classes = [...new Set(missing.map((d) => d.class_name))]
    const entry = {
      id: `${now}-${Math.random()}`,
      time: new Date(now),
      source,
      count: missing.length,
      classes,
    }
    setHistory((prev) => [entry, ...prev].slice(0, MAX_HISTORY))
  }, [])

  const clearHistory = useCallback(() => setHistory([]), [])

  return { muted, toggleMute, history, notify, clearHistory, stats }
}
