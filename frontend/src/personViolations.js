// Supprime les boîtes Person quasi identiques d'une même image (doublons du modèle).
// Deux personnes réellement séparées restent comptées séparément.
export function distinctPersonDetections(detections) {
  const candidates = detections
    .filter((detection) => detection.class_name === 'Person' && detection.bbox)
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
  const distinct = []

  for (const candidate of candidates) {
    const duplicate = distinct.some(
      (person) => intersectionOverUnion(candidate.bbox, person.bbox) >= 0.55,
    )
    if (!duplicate) distinct.push(candidate)
  }
  return distinct
}

export function countDistinctPeople(detections) {
  return distinctPersonDetections(detections).length
}

// Compte les personnes distinctes auxquelles au moins une détection NO-* est associée.
// Retourne null si le modèle n'a fourni aucune boîte Person.
export function countViolatingPeople(detections) {
  const people = distinctPersonDetections(detections)
  if (people.length === 0) return null

  const missingItems = detections.filter(
    (detection) => detection.class_name.startsWith('NO-') && detection.bbox,
  )
  const violatingPeople = new Set()

  for (const item of missingItems) {
    const itemBox = item.bbox
    const itemCenter = {
      x: (itemBox.x1 + itemBox.x2) / 2,
      y: (itemBox.y1 + itemBox.y2) / 2,
    }
    const containing = people
      .map((person, index) => ({ person, index }))
      .filter(
        ({ person }) =>
          itemCenter.x >= person.bbox.x1 &&
          itemCenter.x <= person.bbox.x2 &&
          itemCenter.y >= person.bbox.y1 &&
          itemCenter.y <= person.bbox.y2,
      )
      .sort((a, b) => boxArea(a.person.bbox) - boxArea(b.person.bbox))

    if (containing.length > 0) {
      violatingPeople.add(containing[0].index)
      continue
    }

    const itemArea = boxArea(itemBox)
    const best = people
      .map((person, index) => ({
        index,
        overlap: itemArea
          ? intersectionArea(itemBox, person.bbox) / itemArea
          : 0,
      }))
      .sort((a, b) => b.overlap - a.overlap)[0]
    if (best && best.overlap >= 0.5) violatingPeople.add(best.index)
  }

  return violatingPeople.size
}

export function countDetectionsByClass(detections) {
  const counts = {}
  for (const detection of detections) {
    if (detection.class_name !== 'Person') {
      counts[detection.class_name] = (counts[detection.class_name] || 0) + 1
    }
  }
  const people = countDistinctPeople(detections)
  if (people > 0) counts.Person = people
  return counts
}

function boxArea(box) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1)
}

function intersectionArea(a, b) {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1))
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1))
  return width * height
}

function intersectionOverUnion(a, b) {
  const intersection = intersectionArea(a, b)
  const union = boxArea(a) + boxArea(b) - intersection
  return union > 0 ? intersection / union : 0
}
