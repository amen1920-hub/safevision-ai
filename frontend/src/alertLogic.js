// Logique anti-doublons des alertes webcam.
//
// Une « situation » commence à la première non-conformité et dure tant qu'il y
// en a encore au moins une toutes les RESET_MS. Pendant la situation, on garde
// pour chaque classe (NO-Hardhat, NO-Mask...) le nombre maximal vu. Une nouvelle
// alerte n'est déclenchée que si ce maximum augmente : une nouvelle classe
// manquante, ou une personne de plus en infraction.

export const RESET_MS = 4000 // sans infraction pendant ce délai : situation terminée

export function emptyEpisode() {
  return { max: {}, lastViolationAt: 0 }
}

// counts : { 'NO-Mask': 2, ... } pour l'image courante.
// Retourne { episode, newViolations } sans modifier l'épisode reçu.
export function evaluateEpisode(episode, counts, now) {
  let max = episode.max
  if (now - episode.lastViolationAt > RESET_MS) max = {}

  const nextMax = { ...max }
  let newViolations = 0
  for (const [cls, count] of Object.entries(counts)) {
    const previous = max[cls] || 0
    if (count > previous) {
      newViolations += count - previous
      nextMax[cls] = count
    }
  }

  const hasViolation = Object.keys(counts).length > 0
  return {
    episode: {
      max: nextMax,
      lastViolationAt: hasViolation ? now : episode.lastViolationAt,
    },
    newViolations,
  }
}
