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
      <h2>처리 중...</h2>
      <ul>
        {statuses.map((c) => (
          <li key={c.camera_id}>
            {c.camera_id}: {c.status} ({c.photo_count}장){c.error ? ` - ${c.error}` : ''}
          </li>
        ))}
      </ul>
    </div>
  )
}
