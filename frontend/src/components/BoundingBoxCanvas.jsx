import { useEffect, useRef } from 'react'
import { drawDetections } from '../drawDetections'

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
      drawDetections(ctx, detections, canvas.width)
    }
    img.src = imageUrl
  }, [imageUrl, detections])

  return <canvas ref={canvasRef} className="w-full rounded-xl bg-slate-800" />
}
