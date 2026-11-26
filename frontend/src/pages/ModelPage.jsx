import { useEffect, useState } from 'react'
import { getMetadata } from '../services/api'
import { colorOf } from '../classColors'

const formatValue = (v) =>
  typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)

export default function ModelPage() {
  const [meta, setMeta] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getMetadata().then(setMeta).catch((e) => setError(e.message))
  }, [])

  if (error) {
    return (
      <main className="mx-auto max-w-6xl p-6">
        <div className="rounded-xl bg-red-500/20 p-4 text-red-200">{error}</div>
      </main>
    )
  }
  if (!meta) {
    return <p className="p-6 text-slate-400">Chargement du modèle...</p>
  }

  const info = [
    ['Nom', meta.name],
    ['Architecture', meta.architecture],
    ['Version', meta.version],
    ['Format', meta.format],
    ['Taille d\u2019entrée', `${meta.input_size}×${meta.input_size} px`],
    ['Classes', meta.num_classes],
  ]

  return (
    <main className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-2">
      <section className="space-y-4 rounded-2xl bg-slate-800 p-5">
        <h2 className="text-lg font-semibold">Modèle</h2>
        <dl className="grid grid-cols-2 gap-3 text-sm">
          {info.map(([label, value]) => (
            <div key={label} className="rounded-lg bg-slate-900 p-3">
              <dt className="text-slate-400">{label}</dt>
              <dd className="text-lg font-bold">{value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="space-y-4 rounded-2xl bg-slate-800 p-5">
        <h2 className="text-lg font-semibold">Métriques</h2>
        <dl className="grid grid-cols-2 gap-3 text-sm">
          {Object.entries(meta.metrics).map(([label, value]) => (
            <div key={label} className="rounded-lg bg-slate-900 p-3">
              <dt className="text-slate-400">{label}</dt>
              <dd className="text-lg font-bold">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="space-y-4 rounded-2xl bg-slate-800 p-5 lg:col-span-2">
        <h2 className="text-lg font-semibold">Classes détectables</h2>
        <div className="flex flex-wrap gap-2">
          {meta.classes.map((name) => (
            <span
              key={name}
              className="rounded-full px-3 py-1 text-sm font-medium text-white"
              style={{ backgroundColor: colorOf(name) }}
            >
              {name}
            </span>
          ))}
        </div>
        <p className="text-sm text-slate-400">
          Les classes « NO-… » signalent un équipement manquant et déclenchent
          l’alerte.
        </p>
      </section>
    </main>
  )
}
