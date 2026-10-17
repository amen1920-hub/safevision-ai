import { useEffect, useState } from 'react'
import Dropzone from './components/Dropzone'
import BoundingBoxCanvas from './components/BoundingBoxCanvas'
import ScoreBars from './components/ScoreBars'
import { getHealth, predict } from './services/api'

export default function App() {
  const [health, setHealth] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [runId, setRunId] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', model_loaded: false }))
  }, [])

  const handleFile = async (file) => {
    setError(null)
    setResult(null)
    setLoading(true)
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(URL.createObjectURL(file))
    try {
      const data = await predict(file)
      setResult(data)
      setRunId((id) => id + 1)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const apiOk = health?.model_loaded === true
  const violations = result
    ? result.detections.filter((d) => d.class_name.startsWith('NO-')).length
    : 0

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
        <h1 className="text-2xl font-bold">SafeVision AI</h1>
        <span
          className={`rounded-full px-3 py-1 text-sm ${
            apiOk
              ? 'bg-emerald-500/20 text-emerald-300'
              : 'bg-red-500/20 text-red-300'
          }`}
        >
          {health === null
            ? 'Connexion...'
            : apiOk
              ? 'API connectée'
              : 'API hors ligne'}
        </span>
      </header>

      <main className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-3">
        <section className="space-y-4 lg:col-span-2">
          <Dropzone onFile={handleFile} disabled={loading} />

          {error && (
            <div className="rounded-xl bg-red-500/20 p-4 text-red-200">
              {error}
            </div>
          )}

          {result && violations > 0 && (
            <div className="rounded-xl bg-red-600 p-4 font-semibold text-white">
              ALERTE : {violations} non-conformité{violations > 1 ? 's' : ''}{' '}
              détectée{violations > 1 ? 's' : ''}
            </div>
          )}

          {imageUrl && result && (
            <BoundingBoxCanvas
              imageUrl={imageUrl}
              detections={result.detections}
            />
          )}
          {imageUrl && !result && (
            <img src={imageUrl} alt="" className="w-full rounded-xl" />
          )}
          {loading && <p className="text-slate-400">Analyse en cours...</p>}
        </section>

        <aside className="space-y-6 rounded-2xl bg-slate-800 p-5">
          <h2 className="text-lg font-semibold">Résultats</h2>
          {result ? (
            <>
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-lg bg-slate-900 p-3">
                  <dt className="text-slate-400">Objets</dt>
                  <dd className="text-xl font-bold">{result.num_detections}</dd>
                </div>
                <div className="rounded-lg bg-slate-900 p-3">
                  <dt className="text-slate-400">Latence serveur</dt>
                  <dd className="text-xl font-bold">
                    {Math.round(result.inference_time_ms)} ms
                  </dd>
                </div>
                <div className="rounded-lg bg-slate-900 p-3">
                  <dt className="text-slate-400">Latence totale</dt>
                  <dd className="text-xl font-bold">
                    {result.client_latency_ms} ms
                  </dd>
                </div>
                <div className="rounded-lg bg-slate-900 p-3">
                  <dt className="text-slate-400">Image</dt>
                  <dd className="text-xl font-bold">
                    {result.image_width}×{result.image_height}
                  </dd>
                </div>
              </dl>
              <ScoreBars key={runId} detections={result.detections} />
            </>
          ) : (
            <p className="text-slate-400">
              Envoyez une image pour lancer la détection.
            </p>
          )}
        </aside>
      </main>
    </div>
  )
}
