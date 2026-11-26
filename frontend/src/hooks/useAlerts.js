import { useCallback, useEffect, useRef, useState } from 'react'
import { playAlertSound, unlockAudio } from '../alertSound'
import { emptyEpisode, evaluateEpisode } from '../alertLogic'

const SAMPLE_MS = 3000 // webcam : une image sur 3 s compte dans les statistiques
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
  const episodeRef = useRef(emptyEpisode())
  const lastSampleRef = useRef(0)

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
  // Image : chaque non-conformité déclenche une alerte.
  // Webcam : une seule alerte par situation ; une nouvelle alerte seulement si
  // une nouvelle non-conformité apparaît (autre personne, autre équipement).
  const notify = useCallback((detections, source) => {
    const now = Date.now()
    const missing = detections.filter((d) => d.class_name.startsWith('NO-'))

    let newViolations = missing.length
    if (source === 'Webcam') {
      const counts = {}
      for (const d of missing) {
        counts[d.class_name] = (counts[d.class_name] || 0) + 1
      }
      const result = evaluateEpisode(episodeRef.current, counts, now)
      episodeRef.current = result.episode
      newViolations = result.newViolations
    }

    // Statistiques : toutes les images, mais la webcam seulement 1 fois / 3 s
    // (sinon chaque image du flux est comptée comme une analyse).
    const sampled =
      source !== 'Webcam' || now - lastSampleRef.current >= SAMPLE_MS
    if (sampled) lastSampleRef.current = now
    if (sampled || newViolations > 0) {
      setStats((prev) => {
        const byClass = { ...prev.byClass }
        if (sampled) {
          for (const d of detections) {
            byClass[d.class_name] = (byClass[d.class_name] || 0) + 1
          }
        }
        return {
          analyses: prev.analyses + (sampled ? 1 : 0),
          objects: prev.objects + (sampled ? detections.length : 0),
          violations: prev.violations + newViolations,
          nonCompliant:
            prev.nonCompliant + (sampled && missing.length > 0 ? 1 : 0),
          byClass,
        }
      })
    }

    if (newViolations === 0) return

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
