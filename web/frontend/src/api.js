function generateAdhocCameraId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return `adhoc-${crypto.randomUUID()}`
  return `adhoc-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

// every endpoint here returns {"error": "..."} on failure (see web/backend/app.py) --
// surface that real message instead of a generic "X failed" string that hides
// the actual reason (bad file type, unknown camera, etc) from the user.
async function _checkOk(res, fallbackMessage) {
  if (res.ok) return
  let detail = null
  try {
    detail = (await res.json()).error
  } catch {
    // response wasn't JSON -- fall through to the generic message
  }
  throw new Error(detail || fallbackMessage)
}

// a real folder drag-in carries videos, .DS_Store, docs, etc. -- the backend
// filters those too, but they still ship in the request body, and a couple of
// videos alone can blow past the server's upload size limit and fail the
// whole upload. Filter to photos client-side before anything is sent.
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

function onlyImageFiles(fileList) {
  return Array.from(fileList).filter((file) =>
    IMAGE_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext)))
}

// a huge folder (thousands of photos) cannot go up as one request -- it
// trips server multipart/size limits and one hiccup fails everything.
// Upload in chunks instead: one request per camera folder (or fixed-size
// groups in split mode), all sharing the batch_id the first response returns.
async function _uploadChunks(chunks, onProgress) {
  let batchId = null
  const cameras = []
  let done = 0
  for (const chunk of chunks) {
    const formData = new FormData()
    if (batchId) formData.append('batch_id', batchId)
    for (const [path, file] of chunk) formData.append('files', file, path)
    const res = await fetch('/api/upload', { method: 'POST', body: formData })
    await _checkOk(res, 'upload failed')
    const body = await res.json()
    batchId = body.batch_id
    cameras.push(...body.cameras)
    done += 1
    if (onProgress) onProgress(done, chunks.length)
  }
  return { batch_id: batchId, cameras }
}

export async function uploadFolders(fileList, { splitEachPhoto = false, onProgress } = {}) {
  const images = onlyImageFiles(fileList)
  if (images.length === 0) {
    throw new Error('사진 파일(jpg/jpeg/png)이 없습니다 -- 폴더에 사진이 들어있는지 확인해주세요')
  }
  let chunks
  if (splitEachPhoto) {
    // splitEachPhoto: the folder was just a pile of unrelated photos, not
    // one-subfolder-per-camera -- ignore the real subfolder name and give
    // every file its own camera id instead of grouping/median-stacking them.
    const entries = images.map((file) => [`${generateAdhocCameraId()}/${file.name}`, file])
    chunks = []
    for (let i = 0; i < entries.length; i += 100) chunks.push(entries.slice(i, i + 100))
  } else {
    // one request per camera folder
    const byFolder = new Map()
    for (const file of images) {
      const path = file.webkitRelativePath || file.name
      const parts = path.split('/')
      const folder = parts.length >= 2 ? parts[parts.length - 2] : '(root)'
      if (!byFolder.has(folder)) byFolder.set(folder, [])
      byFolder.get(folder).push([path, file])
    }
    chunks = [...byFolder.values()]
  }
  const body = await _uploadChunks(chunks, onProgress)
  body.skipped_files = fileList.length - images.length
  return body
}

export async function listCameras() {
  const res = await fetch('/api/cameras')
  await _checkOk(res, 'camera list fetch failed')
  return res.json()
}

export async function uploadLooseFiles(fileList, { onProgress } = {}) {
  const images = onlyImageFiles(fileList)
  if (images.length === 0) {
    throw new Error('사진 파일(jpg/jpeg/png)이 없습니다')
  }
  // each loose file can be a completely different scene/angle -- never
  // shared under one camera id, since the backend median-stacks every
  // frame under the same camera together (fine for repeat frames of one
  // fixed view, nonsense for unrelated photos).
  const entries = images.map((file) => [`${generateAdhocCameraId()}/${file.name}`, file])
  const chunks = []
  for (let i = 0; i < entries.length; i += 100) chunks.push(entries.slice(i, i + 100))
  const body = await _uploadChunks(chunks, onProgress)
  body.skipped_files = fileList.length - images.length
  return body
}

export async function getBatchStatus(batchId) {
  const res = await fetch(`/api/batches/${batchId}/status`)
  await _checkOk(res, 'status fetch failed')
  return res.json()
}

export async function listPhotos(batchId, cameraId) {
  const res = await fetch(`/api/batches/${batchId}/cameras/${cameraId}/photos`)
  await _checkOk(res, 'photo list fetch failed')
  return res.json()
}

export function photoUrl(batchId, cameraId, photo) {
  return `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}.png?t=${Date.now()}`
}

export async function addLabel(batchId, cameraId, photo, rawPolygon) {
  const res = await fetch(
    `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}/labels`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ raw_polygon: rawPolygon }),
    },
  )
  await _checkOk(res, 'label add failed')
  return res.json()
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
  await _checkOk(res, 'label edit failed')
  return res.json()
}

export function downloadCameraUrl(batchId, cameraId) {
  return `/api/batches/${batchId}/cameras/${cameraId}/download`
}

export function downloadBatchUrl(batchId) {
  return `/api/batches/${batchId}/download`
}
