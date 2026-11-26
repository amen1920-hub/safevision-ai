import {
  countDetectionsByClass,
  countDistinctPeople,
} from './personViolations.js'

export const initialDashboardStats = {
  imageAnalyses: 0,
  imageObjects: 0,
  imagePeople: 0,
  imageViolations: 0,
  imageNonCompliant: 0,
  imageByClass: {},
  webcamSeen: false,
  webcamObjectsPeak: 0,
  webcamPeoplePeak: 0,
  webcamViolationsPeak: 0,
  webcamByClassPeak: {},
}

function addCounts(previous, current) {
  const result = { ...previous }
  for (const [name, count] of Object.entries(current)) {
    result[name] = (result[name] || 0) + count
  }
  return result
}

function maxCounts(previous, current) {
  const result = { ...previous }
  for (const [name, count] of Object.entries(current)) {
    result[name] = Math.max(result[name] || 0, count)
  }
  return result
}

export function addDetectionStats(previous, detections, source, violationPeople) {
  const frameClasses = countDetectionsByClass(detections)
  const peopleCount = countDistinctPeople(detections)
  const safeViolationCount = Math.max(0, violationPeople || 0)

  if (source === 'Webcam') {
    return {
      ...previous,
      webcamSeen: true,
      webcamObjectsPeak: Math.max(previous.webcamObjectsPeak, detections.length),
      webcamPeoplePeak: Math.max(previous.webcamPeoplePeak, peopleCount),
      webcamViolationsPeak: Math.max(
        previous.webcamViolationsPeak,
        safeViolationCount,
      ),
      webcamByClassPeak: maxCounts(previous.webcamByClassPeak, frameClasses),
    }
  }

  return {
    ...previous,
    imageAnalyses: previous.imageAnalyses + 1,
    imageObjects: previous.imageObjects + detections.length,
    imagePeople: previous.imagePeople + peopleCount,
    imageViolations: previous.imageViolations + safeViolationCount,
    imageNonCompliant:
      previous.imageNonCompliant + Number(safeViolationCount > 0),
    imageByClass: addCounts(previous.imageByClass, frameClasses),
  }
}

export function combineDashboardStats(stats) {
  return {
    analyses: stats.imageAnalyses + Number(stats.webcamSeen),
    objects: stats.imageObjects + stats.webcamObjectsPeak,
    people: stats.imagePeople + stats.webcamPeoplePeak,
    violations: stats.imageViolations + stats.webcamViolationsPeak,
    nonCompliant:
      stats.imageNonCompliant + Number(stats.webcamViolationsPeak > 0),
    byClass: addCounts(stats.imageByClass, stats.webcamByClassPeak),
    imageAnalyses: stats.imageAnalyses,
    webcamSessions: Number(stats.webcamSeen),
    imageObjects: stats.imageObjects,
    webcamObjects: stats.webcamObjectsPeak,
    imagePeople: stats.imagePeople,
    webcamPeople: stats.webcamPeoplePeak,
    imageViolations: stats.imageViolations,
    webcamViolations: stats.webcamViolationsPeak,
  }
}
