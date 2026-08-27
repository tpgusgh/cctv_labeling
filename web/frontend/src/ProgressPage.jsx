import { useEffect, useState } from 'react'
import { getBatchStatus } from './api.js'

export default function ProgressPage({ batchId, cameras, onAllDone }) {
  const [statuses, setStatuses] = useState(cameras.map((c) => ({ ...c, status: 'queued' })))

  useEffect(() => {
    let cancelled = false
    const interval = setInterval(async () => {
      const data = await getBatchStatus(batchId)
      if (cancelled) return
      setStatuses(data.cameras)
      const allDone = data.cameras.every((c) => c.status === 'done' || c.status === 'error')
      if (allDone) {
        clearInterval(interval)
        onAllDone()
      }
    }, 2000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [batchId, onAllDone])

  return (
    <div>
      <p className="eyebrow">02 · PROCESSING</p>
      <div className="panel">
        <h2>카메라별 처리 현황</h2>
        <p className="hint">새 카메라는 슬롯 탐지 후 라벨링, 기존 카메라는 바로 라벨링됩니다.</p>
        {statuses.map((c) => (
          <div className="status-line" key={c.camera_id}>
            <span>{c.camera_id} <span style={{ color: 'var(--text-muted)' }}>({c.photo_count}장)</span></span>
            <span className={`status-badge status-${c.status}`}>{c.status}</span>
            {c.error && <span className="error-text">{c.error}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}
