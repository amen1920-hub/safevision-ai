// Bip d'alerte généré avec l'API Web Audio : aucun fichier audio à charger.
let ctx = null

function getContext() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext
  if (!AudioCtx) return null
  if (!ctx) ctx = new AudioCtx()
  return ctx
}

// Les navigateurs bloquent le son tant que la personne n'a pas interagi avec la page.
// On "débloque" donc l'audio au premier clic, touche ou dépôt de fichier.
export function unlockAudio() {
  try {
    const c = getContext()
    if (c && c.state === 'suspended') c.resume()
  } catch {
    // Pas de son disponible : l'alerte visuelle reste active
  }
}

export function playAlertSound() {
  try {
    const c = getContext()
    if (!c) return
    if (c.state === 'suspended') c.resume()
    const t0 = c.currentTime
    // Deux bips courts
    for (const offset of [0, 0.25]) {
      const osc = c.createOscillator()
      const gain = c.createGain()
      osc.type = 'square'
      osc.frequency.value = 880
      gain.gain.setValueAtTime(0.15, t0 + offset)
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + offset + 0.18)
      osc.connect(gain)
      gain.connect(c.destination)
      osc.start(t0 + offset)
      osc.stop(t0 + offset + 0.2)
    }
  } catch {
    // Pas de son disponible : l'alerte visuelle reste active
  }
}
