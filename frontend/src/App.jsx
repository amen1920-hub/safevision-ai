import { useEffect, useRef, useState } from 'react'
import { HashRouter, NavLink, Route, Routes } from 'react-router-dom'
import ModelPage from './pages/ModelPage'
import DashboardPage from './pages/DashboardPage'
import Dropzone from './components/Dropzone'
import BoundingBoxCanvas from './components/BoundingBoxCanvas'
import ScoreBars from './components/ScoreBars'
import Gallery from './components/Gallery'
import Webcam from './components/Webcam'
import AlertHistory from './components/AlertHistory'
import { useAlerts } from './hooks/useAlerts'
import { fingerprintFile } from './imageFingerprint'
import { getHealth, predict } from './services/api'

function AppShell() {
  const [health, setHealth] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [runId, setRunId] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [duplicateImage, setDuplicateImage] = useState(false)
  const analyzedImagesRef = useRef(new Map())
  const { muted, toggleMute, history, notify, clearHistory, stats } =
    useAlerts()

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', model_loaded: false }))
  }, [])

  const handleFile = async (file) => {
    setError(null)
    setDuplicateImage(false)
    setResult(null)
    setLoading(true)
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(URL.createObjectURL(file))
    try {
      const fingerprint = await fingerprintFile(file)
      const previousResult = analyzedImagesRef.current.get(fingerprint)
      if (previousResult) {
        setResult(previousResult)
        setRunId((id) => id + 1)
        setDuplicateImage(true)
        return
      }

      const data = await predict(file)
      setResult(data)
      setRunId((id) => id + 1)
      analyzedImagesRef.current.set(fingerprint, data)
      notify(data.detections, 'Image')
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
        <nav className="flex gap-2">
          {[
            ['/', 'Image'],
            ['/webcam', 'Webcam'],
            ['/dashboard', 'Tableau de bord'],
            ['/modele', 'Modèle'],
          ].map(([to, label]) => (
            <NavLink
              key={to}
              to={to}
              end
              className={({ isActive }) =>
                `rounded-lg px-4 py-1.5 text-sm font-medium transition ${
                  isActive
                    ? 'bg-emerald-500 text-white'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noreferrer"
            className="rounded-lg bg-slate-800 px-4 py-1.5 text-sm font-medium text-slate-300 transition hover:bg-slate-700"
          >
            MLflow ↗
          </a>
        </nav>
        <div className="flex items-center gap-3">
        <button
          onClick={toggleMute}
          className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-slate-300 transition hover:bg-slate-700"
        >
          {muted ? 'Son coupé' : 'Son activé'}
        </button>
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
        </div>
      </header>

      <Routes>
        <Route path="/webcam" element={<Webcam onDetections={notify} />} />
        <Route path="/dashboard" element={<DashboardPage stats={stats} />} />
        <Route path="/modele" element={<ModelPage />} />
        <Route
          path="*"
          element={
      <main className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-3">
        <section className="space-y-4 lg:col-span-2">
          <Dropzone onFile={handleFile} disabled={loading} />
          <Gallery onFile={handleFile} disabled={loading} />

          {error && (
            <div className="rounded-xl bg-red-500/20 p-4 text-red-200">
              {error}
            </div>
          )}

          {duplicateImage && (
            <div
              role="status"
              className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4 text-amber-200"
            >
              Cette image a déjà été analysée. Son résultat est réaffiché,
              mais elle n’est pas ajoutée une deuxième fois au tableau de bord.
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
          }
        />
      </Routes>

      <AlertHistory history={history} onClear={clearHistory} />
    </div>
  )
}

export default function App() {
  return (
    <HashRouter>
      <AppShell />
    </HashRouter>
  )
}
