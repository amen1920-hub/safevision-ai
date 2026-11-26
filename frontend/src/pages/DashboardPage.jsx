import { colorOf } from '../classColors'

export default function DashboardPage({ stats }) {
  const compliance =
    stats.analyses > 0
      ? Math.round(((stats.analyses - stats.nonCompliant) / stats.analyses) * 100)
      : null

  const cards = [
    ['Analyses', stats.analyses],
    ['Objets détectés', stats.objects],
    ['Non-conformités', stats.violations],
    ['Taux de conformité', compliance === null ? '–' : `${compliance} %`],
  ]
  const rows = Object.entries(stats.byClass).sort((a, b) => b[1] - a[1])

  return (
    <main className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {cards.map(([label, value]) => (
          <div key={label} className="rounded-2xl bg-slate-800 p-5">
            <p className="text-sm text-slate-400">{label}</p>
            <p className="text-3xl font-bold">{value}</p>
          </div>
        ))}
      </div>

      <section className="space-y-3 rounded-2xl bg-slate-800 p-5">
        <h2 className="text-lg font-semibold">Détections par classe</h2>
        {rows.length === 0 ? (
          <p className="text-slate-400">
            Aucune donnée : analysez une image ou lancez la webcam.
          </p>
        ) : (
          rows.map(([name, count]) => (
            <div key={name} className="flex items-center gap-3 text-sm">
              <span
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: colorOf(name) }}
              />
              <span className="flex-1">{name}</span>
              <span className="font-bold">{count}</span>
            </div>
          ))
        )}
      </section>
      <p className="text-xs text-slate-500">
        Les compteurs portent sur la session en cours (une analyse = une image
        ou une image de la webcam).
      </p>
    </main>
  )
}
