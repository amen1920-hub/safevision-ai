import { useEffect, useRef, useState } from 'react'
import { predict } from '../services/api'
import { drawDetections } from '../drawDetections'

const JPEG_QUALITY = 0.7
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const canvasToBlob = (canvas) =>
  new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY))

function cameraErrorMessage(e) {
  if (e?.name === 'NotAllowedError') return 'Accès à la caméra refusé.'
  if (e?.name === 'NotFoundError') return 'Aucune caméra détectée.'
  return e?.message || 'Impossible de démarrer la caméra.'
}

export default function Webcam({ onDetections }) {
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const streamRef = useRef(null)
  const runningRef = useRef(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [violations, setViolations] = useState(0)

  const stop = () => {
    runningRef.current = false
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    const overlay = overlayRef.current
    overlay?.getContext('2d').clearRect(0, 0, overlay.width, overlay.height)
    setRunning(false)
    setViolations(0)
  }

  // Coupe la caméra quand on quitte l'onglet
  useEffect(() => {
    return () => {
      runningRef.current = false
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  // Boucle : capture une image, l'envoie à l'API, dessine le résultat, recommence.
  // Une seule requête à la fois, pour ne pas saturer le serveur.
  const loop = async () => {
    const capture = document.createElement('canvas')
    let fps = 0
    let lastEnd = performance.now()

    while (runningRef.current) {
      const video = videoRef.current
      if (!video || video.videoWidth === 0) {
        await sleep(50)
        continue
      }
      capture.width = video.videoWidth
      capture.height = video.videoHeight
      capture.getContext('2d').drawImage(video, 0, 0)
      const blob = await canvasToBlob(capture)

      try {
        const data = await predict(
          new File([blob], 'frame.jpg', { type: 'image/jpeg' }),
        )
        if (!runningRef.current) break

        const overlay = overlayRef.current
        overlay.width = data.image_width
        overlay.height = data.image_height
        const ctx = overlay.getContext('2d')
        ctx.clearRect(0, 0, overlay.width, overlay.height)
        drawDetections(ctx, data.detections, overlay.width)

        const now = performance.now()
        const instant = 1000 / (now - lastEnd)
        lastEnd = now
        fps = fps === 0 ? instant : fps * 0.8 + instant * 0.2

        setStats({
          fps: fps.toFixed(1),
          latency: data.client_latency_ms,
          server: Math.round(data.inference_time_ms),
          objects: data.num_detections,
        })
        setViolations(
          data.detections.filter((d) => d.class_name.startsWith('NO-')).length,
        )
        onDetections?.(data.detections, 'Webcam')
      } catch (e) {
        setError(e.message)
        stop()
        break
      }
    }
  }

  const start = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      })
      streamRef.current = stream
      const video = videoRef.current
      video.srcObject = stream
      await video.play()
      runningRef.current = true
      setRunning(true)
      loop()
    } catch (e) {
      setError(cameraErrorMessage(e))
    }
  }

  return (
    <main className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-3">
      <section className="space-y-4 lg:col-span-2">
        <div className="relative aspect-video overflow-hidden rounded-xl bg-slate-800">
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full object-contain"
          />
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 h-full w-full object-contain"
          />
          {!running && (
            <div className="absolute inset-0 flex items-center justify-center text-slate-400">
              Caméra arrêtée
            </div>
          )}
        </div>

        <button
          onClick={running ? stop : start}
          className={`rounded-lg px-5 py-2 font-medium text-white transition ${
            running
              ? 'bg-red-600 hover:bg-red-500'
              : 'bg-emerald-600 hover:bg-emerald-500'
          }`}
        >
          {running ? 'Arrêter la caméra' : 'Démarrer la caméra'}
        </button>

        {error && (
          <div className="rounded-xl bg-red-500/20 p-4 text-red-200">
            {error}
          </div>
        )}

        {running && violations > 0 && (
          <div className="rounded-xl bg-red-600 p-4 font-semibold text-white">
            ALERTE : {violations} non-conformité{violations > 1 ? 's' : ''}{' '}
            détectée{violations > 1 ? 's' : ''}
          </div>
        )}
      </section>

      <aside className="space-y-4 rounded-2xl bg-slate-800 p-5">
        <h2 className="text-lg font-semibold">Direct</h2>
        {stats ? (
          <dl className="grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-lg bg-slate-900 p-3">
              <dt className="text-slate-400">FPS</dt>
              <dd className="text-xl font-bold">{stats.fps}</dd>
            </div>
            <div className="rounded-lg bg-slate-900 p-3">
              <dt className="text-slate-400">Objets</dt>
              <dd className="text-xl font-bold">{stats.objects}</dd>
            </div>
            <div className="rounded-lg bg-slate-900 p-3">
              <dt className="text-slate-400">Latence totale</dt>
              <dd className="text-xl font-bold">{stats.latency} ms</dd>
            </div>
            <div className="rounded-lg bg-slate-900 p-3">
              <dt className="text-slate-400">Latence serveur</dt>
              <dd className="text-xl font-bold">{stats.server} ms</dd>
            </div>
          </dl>
        ) : (
          <p className="text-slate-400">
            Démarrez la caméra pour lancer la détection en direct.
          </p>
        )}
        <p className="text-xs text-slate-500">
          Une image est analysée à la fois : la fréquence dépend de la vitesse
          du serveur.
        </p>
      </aside>
    </main>
  )
}
