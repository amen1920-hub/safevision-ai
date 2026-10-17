import { useRef, useState } from 'react'

export default function Dropzone({ onFile, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFiles = (files) => {
    const file = files?.[0]
    if (file) onFile(file)
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (!disabled) handleFiles(e.dataTransfer.files)
      }}
      className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition ${
        dragging
          ? 'border-emerald-400 bg-emerald-400/10'
          : 'border-slate-600 hover:border-slate-400'
      }`}
    >
      <p className="text-lg text-slate-200">Glissez une image ici</p>
      <p className="mt-1 text-sm text-slate-400">
        ou cliquez pour choisir un fichier
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files)
          e.target.value = ''
        }}
      />
    </div>
  )
}
