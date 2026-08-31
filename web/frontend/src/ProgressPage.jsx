import { useEffect, useState } from 'react'
import { getBatchStatus } from './api.js'

export default function ProgressPage({ batchId, cameras, onAllDone }) {
  const [statuses, setStatuses] = useState(cameras.map((c) => ({ ...c, status: 'queued' })))
  const [pollError, setPollError] = useState(null)

  useEffect(() => {
    let cancelled = false
    let timeoutId = null

    // self-rescheduling instead of setInterval: the next poll is only
    // queued after the previous one finishes (success or failure), so a
    // slow response can't overlap with the next tick and race it.
    async function poll() {
      try {
        const data = await getBatchStatus(batchId)
        if (cancelled) return
        setStatuses(data.cameras)
        setPollError(null)
        const allDone = data.cameras.every((c) => c.status === 'done' || c.status === 'error')
        if (allDone) {
          onAllDone()
          return
        }
      } catch (err) {
        if (cancelled) return
        setPollError(err.message)
      }
      if (!cancelled) timeoutId = setTimeout(poll, 2000)
    }

    timeoutId = setTimeout(poll, 2000)
    return () => {
      cancelled = true
      clearTimeout(timeoutId)
    }
  }, [batchId, onAllDone])

  return (
    <div>
      <p className="eyebrow">02 · PROCESSING</p>
      <div className="panel">
        <h2>카메라별 처리 현황</h2>
        <p className="hint">새 카메라는 슬롯 탐지 후 라벨링, 기존 카메라는 바로 라벨링됩니다.</p>
        {pollError && (
          <p className="error-text">상태 확인 중 오류: {pollError} (계속 재시도 중)</p>
        )}
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
