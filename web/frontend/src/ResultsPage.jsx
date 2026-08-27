import { useEffect, useState } from 'react'
import { listPhotos, photoUrl, slotPatchUrl, editLabel, downloadCameraUrl, downloadBatchUrl } from './api.js'

const PATCH_SIZE = 300 // must match pipeline.DEFAULT_PATCH_SIZE in src/pipeline.py

function SlotAdjustModal({ batchId, cameraId, photo, slotId, onClose, onSaved }) {
  const [drag, setDrag] = useState(null)

  function handleMouseDown(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    setDrag({ x0: e.clientX - rect.left, y0: e.clientY - rect.top, x1: e.clientX - rect.left, y1: e.clientY - rect.top })
  }
  function handleMouseMove(e) {
    if (!drag) return
    const rect = e.currentTarget.getBoundingClientRect()
    setDrag((d) => ({ ...d, x1: e.clientX - rect.left, y1: e.clientY - rect.top }))
  }
  async function handleMouseUp() {
    if (!drag) return
    const finished = drag
    setDrag(null)
    if (Math.abs(finished.x1 - finished.x0) < 10 || Math.abs(finished.y1 - finished.y0) < 10) return

    await editLabel(batchId, cameraId, photo, slotId, 'adjust', {
      patch_box: { x0: finished.x0, y0: finished.y0, x1: finished.x1, y1: finished.y1 },
    })
    onSaved()
  }

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
      <div style={{ background: 'white', padding: 16 }}>
        <p>{slotId} 위치/크기를 드래그로 지정하세요 (사각형 하나 그리면 자동 저장)</p>
        <div
          style={{ position: 'relative', width: PATCH_SIZE, height: PATCH_SIZE, cursor: 'crosshair' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <img
            src={slotPatchUrl(batchId, cameraId, photo, slotId)}
            width={PATCH_SIZE}
            height={PATCH_SIZE}
            style={{ position: 'absolute', top: 0, left: 0 }}
            draggable={false}
            alt={`${slotId} rectified patch`}
          />
          {drag && (
            <div
              style={{
                position: 'absolute',
                border: '2px solid orange',
                left: Math.min(drag.x0, drag.x1),
                top: Math.min(drag.y0, drag.y1),
                width: Math.abs(drag.x1 - drag.x0),
                height: Math.abs(drag.y1 - drag.y0),
              }}
            />
          )}
        </div>
        <button onClick={onClose}>닫기</button>
      </div>
    </div>
  )
}

function PhotoCard({ batchId, cameraId, photo, onChanged }) {
  const [editingSlot, setEditingSlot] = useState(null)

  async function handleDelete(slotId) {
    await editLabel(batchId, cameraId, photo.photo, slotId, 'delete')
    onChanged()
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: 8, margin: 8, display: 'inline-block', verticalAlign: 'top' }}>
      <img src={photoUrl(batchId, cameraId, photo.photo)} width={240} alt={photo.photo} />
      <div>{photo.photo}</div>
      <table>
        <tbody>
          {photo.slot_ids.map((slotId) => (
            <tr key={slotId}>
              <td>{slotId}{photo.excluded_slots.includes(slotId) ? ' (삭제됨)' : ''}</td>
              <td><button onClick={() => handleDelete(slotId)}>삭제</button></td>
              <td><button onClick={() => setEditingSlot(slotId)}>수정</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      {editingSlot && (
        <SlotAdjustModal
          batchId={batchId}
          cameraId={cameraId}
          photo={photo.photo}
          slotId={editingSlot}
          onClose={() => setEditingSlot(null)}
          onSaved={() => {
            setEditingSlot(null)
            onChanged()
          }}
        />
      )}
    </div>
  )
}

export default function ResultsPage({ batchId, cameras }) {
  const [cameraId, setCameraId] = useState(cameras[0]?.camera_id)
  const [photos, setPhotos] = useState([])

  async function refresh() {
    const data = await listPhotos(batchId, cameraId)
    setPhotos(data.photos)
  }

  useEffect(() => {
    refresh()
  }, [cameraId])

  return (
    <div>
      <h2>결과</h2>
      <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
        {cameras.map((c) => (
          <option key={c.camera_id} value={c.camera_id}>{c.camera_id}</option>
        ))}
      </select>
      {' '}
      <a href={downloadCameraUrl(batchId, cameraId)}>이 카메라 다운로드</a>
      {' | '}
      <a href={downloadBatchUrl(batchId)}>전체 다운로드</a>
      <div>
        {photos.map((p) => (
          <PhotoCard key={p.photo} batchId={batchId} cameraId={cameraId} photo={p} onChanged={refresh} />
        ))}
      </div>
    </div>
  )
}
