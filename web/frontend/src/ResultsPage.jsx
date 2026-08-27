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
    <div className="modal-overlay">
      <div className="modal-box">
        <p className="eyebrow">{slotId} · ADJUST</p>
        <p className="hint">위치/크기를 드래그로 지정하세요 (사각형 하나 그리면 자동 저장)</p>
        <div
          style={{ position: 'relative', width: PATCH_SIZE, height: PATCH_SIZE, cursor: 'crosshair', border: '1px solid var(--border)', borderRadius: 4, overflow: 'hidden' }}
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
                border: '2px solid var(--accent-cyan)',
                background: 'rgba(79, 209, 219, 0.15)',
                left: Math.min(drag.x0, drag.x1),
                top: Math.min(drag.y0, drag.y1),
                width: Math.abs(drag.x1 - drag.x0),
                height: Math.abs(drag.y1 - drag.y0),
              }}
            />
          )}
        </div>
        <p style={{ marginTop: 14 }}><button className="btn" onClick={onClose}>닫기</button></p>
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
    <div className="photo-card">
      <img src={photoUrl(batchId, cameraId, photo.photo)} width={240} alt={photo.photo} />
      <div className="photo-name">{photo.photo}</div>
      <table className="slot-table">
        <tbody>
          {photo.slot_ids.map((slotId) => (
            <tr key={slotId}>
              <td className={photo.excluded_slots.includes(slotId) ? 'slot-excluded' : ''}>
                {slotId}{photo.excluded_slots.includes(slotId) ? ' · 삭제됨' : ''}
              </td>
              <td><button className="btn btn-danger" onClick={() => handleDelete(slotId)}>삭제</button></td>
              <td><button className="btn" onClick={() => setEditingSlot(slotId)}>수정</button></td>
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
      <p className="eyebrow">03 · RESULTS</p>
      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>결과 확인</h2>
        <p className="hint">카메라를 선택해서 사진별 라벨을 확인하고, 필요하면 슬롯 단위로 삭제·수정하세요.</p>
        <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
          {cameras.map((c) => (
            <option key={c.camera_id} value={c.camera_id}>{c.camera_id}</option>
          ))}
        </select>
        <div className="link-row" style={{ marginTop: 12 }}>
          <a href={downloadCameraUrl(batchId, cameraId)}>이 카메라 다운로드</a>
          {' · '}
          <a href={downloadBatchUrl(batchId)}>전체 다운로드</a>
        </div>
      </div>
      <div>
        {photos.map((p) => (
          <PhotoCard key={p.photo} batchId={batchId} cameraId={cameraId} photo={p} onChanged={refresh} />
        ))}
      </div>
    </div>
  )
}
