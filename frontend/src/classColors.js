// Une couleur distincte par classe, utilisée pour les boîtes et les barres de score
export const CLASS_COLORS = {
  Person: '#3b82f6',
  Hardhat: '#22c55e',
  'NO-Hardhat': '#ef4444',
  'Safety Vest': '#14b8a6',
  'NO-Safety Vest': '#f97316',
  Mask: '#a855f7',
  'NO-Mask': '#ec4899',
  Gloves: '#eab308',
  'Safety Cone': '#06b6d4',
}

export const colorOf = (name) => CLASS_COLORS[name] || '#ffffff'
