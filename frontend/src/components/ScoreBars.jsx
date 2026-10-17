import { colorOf } from '../classColors'

export default function ScoreBars({ detections }) {
  // Meilleure confiance et nombre d'objets pour chaque classe
  const byClass = {}
  for (const d of detections) {
    const current = byClass[d.class_name] || { max: 0, count: 0 }
    byClass[d.class_name] = {
      max: Math.max(current.max, d.confidence),
      count: current.count + 1,
    }
  }
  const rows = Object.entries(byClass).sort((a, b) => b[1].max - a[1].max)

  if (rows.length === 0) {
    return <p className="text-slate-400">Aucun objet détecté.</p>
  }

  return (
    <div className="space-y-3">
      {rows.map(([name, { max, count }]) => (
        <div key={name}>
          <div className="mb-1 flex justify-between text-sm text-slate-200">
            <span>
              {name} <span className="text-slate-400">×{count}</span>
            </span>
            <span>{(max * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
            <div
              className="h-full rounded-full"
              style={{
                width: `${max * 100}%`,
                backgroundColor: colorOf(name),
                animation: 'grow 0.8s ease-out',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
