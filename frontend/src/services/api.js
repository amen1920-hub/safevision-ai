const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

async function request(path, options) {
  try {
    return await fetch(`${API_URL}${path}`, options)
  } catch {
    throw new Error(
      `API injoignable (${API_URL}). Vérifiez que le backend est lancé.`,
    )
  }
}

export async function getHealth() {
  const res = await request('/health')
  if (!res.ok) throw new Error(`Erreur ${res.status}`)
  return res.json()
}

export async function predict(file) {
  const form = new FormData()
  form.append('file', file)

  const start = performance.now()
  const res = await request('/predict', { method: 'POST', body: form })
  const clientLatencyMs = Math.round(performance.now() - start)

  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || `Erreur ${res.status}`)
  return { ...data, client_latency_ms: clientLatencyMs }
}
