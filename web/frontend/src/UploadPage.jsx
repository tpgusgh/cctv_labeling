import { useEffect, useState } from 'react'
import { uploadFolders, uploadLooseFiles, listCameras } from './api.js'

export default function UploadPage({ onUploaded }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState('')

  useEffect(() => {
    listCameras()
      .then((data) => {
        setCameras(data.cameras)
        if (data.cameras.length > 0) setSelectedCamera(data.cameras[0])
      })
      .catch(() => {})
  }, [])

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
    if (!files || files.length === 0 || !selectedCamera) return
    setBusy(true)
    setError(null)
    try {
      const result = await uploadLooseFiles(files, selectedCamera)
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
          이미 등록된 카메라라면 폴더 없이 사진 1장 이상만 바로 올려서 라벨링할 수 있습니다.
        </p>
        {cameras.length === 0 ? (
          <p className="hint">등록된 카메라가 아직 없습니다 — 폴더 업로드로 먼저 카메라를 등록하세요.</p>
        ) : (
          <>
            <select value={selectedCamera} onChange={(e) => setSelectedCamera(e.target.value)} disabled={busy}>
              {cameras.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
            {' '}
            <input type="file" multiple disabled={busy} onChange={handleLooseChange} />
          </>
        )}

        {busy && <p className="hint">업로드 중...</p>}
        {error && <p className="error-text">{error}</p>}
      </div>
    </div>
  )
}
