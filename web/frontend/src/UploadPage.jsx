import { useState } from 'react'
import { uploadFolders, uploadLooseFiles } from './api.js'

function generateAdhocCameraId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return `adhoc-${crypto.randomUUID()}`
  return `adhoc-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export default function UploadPage({ onUploaded }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function handleFolderChange(e) {
    const files = e.target.files
    if (!files || files.length === 0) return
    setBusy(true)
    setError(null)
    try {
      const result = await uploadFolders(files)
      onUploaded(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleLooseChange(e) {
    const files = e.target.files
    if (!files || files.length === 0) return
    setBusy(true)
    setError(null)
    try {
      const result = await uploadLooseFiles(files, generateAdhocCameraId())
      onUploaded(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <p className="eyebrow">01 · SOURCE SELECT</p>
      <div className="panel">
        <h2>카메라 폴더 업로드</h2>
        <p className="hint">
          카메라 폴더가 들어있는 상위 폴더 하나를 선택하세요 (폴더 최대 100개 권장).
          폴더 하나 = 카메라 하나로 인식합니다. 새 카메라는 슬롯 자동 탐지부터 시작합니다.
        </p>
        <input
          type="file"
          webkitdirectory="true"
          directory="true"
          multiple
          disabled={busy}
          onChange={handleFolderChange}
        />

        <div className="divider-label">또는</div>

        <h2>낱장 사진 업로드</h2>
        <p className="hint">
          카메라 등록 여부와 상관없이 사진 1장 이상만 바로 올리면 AI가 자동으로 슬롯을 탐지해 라벨링합니다.
        </p>
        <input type="file" multiple disabled={busy} onChange={handleLooseChange} />

        {busy && <p className="hint">업로드 중...</p>}
        {error && <p className="error-text">{error}</p>}
      </div>
    </div>
  )
}
