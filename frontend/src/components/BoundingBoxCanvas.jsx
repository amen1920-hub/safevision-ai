import { useEffect, useRef } from 'react'
import { colorOf } from '../classColors'

export default function BoundingBoxCanvas({ imageUrl, detections }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)

      // Épaisseur et taille du texte proportionnelles à la taille de l'image
      const fontSize = Math.max(12, Math.round(canvas.width / 45))
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 250))
      ctx.font = `bold ${fontSize}px sans-serif`
      ctx.textBaseline = 'top'

      for (const d of detections) {
        const color = colorOf(d.class_name)
        const { x1, y1, x2, y2 } = d.bbox
        ctx.strokeStyle = color
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        const label = `${d.class_name} ${Math.round(d.confidence * 100)}%`
        const labelW = ctx.measureText(label).width + 8
        const labelH = fontSize + 6
        const labelY = y1 - labelH >= 0 ? y1 - labelH : y1
        ctx.fillStyle = color
        ctx.fillRect(x1, labelY, labelW, labelH)
        ctx.fillStyle = '#ffffff'
        ctx.fillText(label, x1 + 4, labelY + 3)
      }
    }
    img.src = imageUrl
  }, [imageUrl, detections])

  return <canvas ref={canvasRef} className="w-full rounded-xl bg-slate-800" />
}
