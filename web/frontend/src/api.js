export async function uploadFolders(fileList) {
  const formData = new FormData()
  for (const file of fileList) {
    formData.append('files', file, file.webkitRelativePath || file.name)
  }
  const res = await fetch('/api/upload', { method: 'POST', body: formData })
  if (!res.ok) throw new Error('upload failed')
  return res.json()
}

export async function getBatchStatus(batchId) {
  const res = await fetch(`/api/batches/${batchId}/status`)
  if (!res.ok) throw new Error('status fetch failed')
  return res.json()
}

export async function listPhotos(batchId, cameraId) {
  const res = await fetch(`/api/batches/${batchId}/cameras/${cameraId}/photos`)
  if (!res.ok) throw new Error('photo list fetch failed')
  return res.json()
}

export function photoUrl(batchId, cameraId, photo) {
  return `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}.png?t=${Date.now()}`
}

export function slotPatchUrl(batchId, cameraId, photo, slotId) {
  return `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}/slots/${slotId}/patch.png`
}

export async function editLabel(batchId, cameraId, photo, slotId, action, extra = {}) {
  const res = await fetch(
    `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}/labels/${slotId}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, ...extra }),
    },
  )
  if (!res.ok) throw new Error('label edit failed')
  return res.json()
}

export function downloadCameraUrl(batchId, cameraId) {
  return `/api/batches/${batchId}/cameras/${cameraId}/download`
}

export function downloadBatchUrl(batchId) {
  return `/api/batches/${batchId}/download`
}
