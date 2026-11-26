import { colorOf } from '../classColors'

export default function DashboardPage({ stats }) {
  const peopleCount = stats.people || 0
  const violationCount = stats.violations || 0
  const compliance =
    peopleCount > 0
      ? Math.round(
          ((peopleCount - Math.min(peopleCount, violationCount)) / peopleCount) *
            100,
        )
      : null

  const cards = [
    ['Analyses (images + webcam)', stats.analyses],
    ['Objets détectés (cumul)', stats.objects],
    ['Personnes en infraction', violationCount],
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

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl bg-slate-800 p-5">
          <h2 className="mb-2 font-semibold">Images importées</h2>
          <p className="text-sm text-slate-300">
            {stats.imageAnalyses} analyse(s) · {stats.imageObjects} objet(s) ·{' '}
            {stats.imageViolations} personne(s) en infraction
          </p>
        </div>
        <div className="rounded-2xl bg-slate-800 p-5">
          <h2 className="mb-2 font-semibold">Webcam (pic observé)</h2>
          <p className="text-sm text-slate-300">
            {stats.webcamSessions} session · {stats.webcamPeople} personne(s) ·{' '}
            {stats.webcamObjects} objet(s) · {stats.webcamViolations}{' '}
            personne(s) en infraction
          </p>
        </div>
      </section>

      <section className="space-y-3 rounded-2xl bg-slate-800 p-5">
        <h2 className="text-lg font-semibold">Détections par classe (images + webcam)</h2>
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
        Les résultats des images s’additionnent. La webcam ajoute son pic de
        détections observé au lieu de recompter la même personne à chaque frame.
        Les deux sources restent dans le tableau de bord, quel que soit l’ordre
        d’utilisation.
      </p>
    </main>
  )
}
