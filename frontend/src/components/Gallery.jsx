const SAMPLES = [1, 2, 3, 4, 5, 6].map((n) => `/samples/exemple-${n}.jpg`)

export default function Gallery({ onFile, disabled }) {
  // Télécharge l'image d'exemple et la traite comme un fichier choisi par l'utilisateur
  const pick = async (url) => {
    const res = await fetch(url)
    const blob = await res.blob()
    onFile(new File([blob], url.split('/').pop(), { type: blob.type }))
  }

  return (
    <div>
      <p className="mb-2 text-sm text-slate-400">
        Ou testez avec une image d'exemple :
      </p>
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
        {SAMPLES.map((url) => (
          <button
            key={url}
            type="button"
            disabled={disabled}
            onClick={() => pick(url)}
            className="overflow-hidden rounded-lg border border-slate-700 transition hover:border-emerald-400 disabled:opacity-50"
          >
            <img
              src={url}
              alt="Exemple"
              className="aspect-square w-full object-cover"
            />
          </button>
        ))}
      </div>
    </div>
  )
}
