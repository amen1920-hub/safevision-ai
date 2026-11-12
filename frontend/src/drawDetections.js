import { colorOf } from './classColors'

// Dessine les boîtes et leurs étiquettes sur un canvas 2D.
// 'width' sert à adapter l'épaisseur des traits et la taille du texte.
export function drawDetections(ctx, detections, width) {
  const fontSize = Math.max(12, Math.round(width / 45))
  ctx.lineWidth = Math.max(2, Math.round(width / 250))
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
