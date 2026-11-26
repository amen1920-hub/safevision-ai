export async function fingerprintFile(file) {
  if (!globalThis.crypto?.subtle) {
    throw new Error('La vérification des images identiques nécessite un navigateur sécurisé.')
  }

  const bytes = await file.arrayBuffer()
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('')
}
