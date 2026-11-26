export default function AlertHistory({ history, onClear }) {
  return (
    <section className="mx-auto max-w-6xl px-6 pb-10">
      <div className="rounded-2xl bg-slate-800 p-5">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Historique des alertes</h2>
          {history.length > 0 && (
            <button
              onClick={onClear}
              className="text-sm text-slate-400 transition hover:text-slate-200"
            >
              Effacer
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-slate-400">Aucune alerte pour le moment.</p>
        ) : (
          <ul className="space-y-2">
            {history.map((a) => (
              <li
                key={a.id}
                className="flex items-center justify-between gap-3 rounded-lg bg-slate-900 p-3 text-sm"
              >
                <span>
                  <span className="mr-2 rounded bg-red-500/20 px-2 py-0.5 text-red-300">
                    {a.source}
                  </span>
                  {a.count} {a.unit || 'non-conformité'}
                  {a.count > 1 ? 's' : ''} : {a.classes.join(', ')}
                </span>
                <time className="shrink-0 text-slate-400">
                  {a.time.toLocaleTimeString('fr-FR')}
                </time>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
