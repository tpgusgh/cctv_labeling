import { useState } from 'react'
import { uploadFolders, uploadLooseFiles } from './api.js'

export default function UploadPage({ onUploaded }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(null)
  const [splitEachPhoto, setSplitEachPhoto] = useState(false)

  const onProgress = (done, total) => setProgress({ done, total })

  async function handleFolderChange(e) {
    const files = e.target.files
    if (!files || files.length === 0) return
    setBusy(true)
    setError(null)
    setProgress(null)
    try {
      const result = await uploadFolders(files, { splitEachPhoto, onProgress })
      onUploaded(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      setProgress(null)
    }
  }

  async function handleLooseChange(e) {
    const files = e.target.files
    if (!files || files.length === 0) return
    setBusy(true)
    setError(null)
    setProgress(null)
    try {
      const result = await uploadLooseFiles(files, { onProgress })
      onUploaded(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      setProgress(null)
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
        <label style={{ display: 'block', marginBottom: 8 }}>
          <input
            type="checkbox"
            checked={splitEachPhoto}
            onChange={(e) => setSplitEachPhoto(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          하위 폴더 구분 없이 그냥 사진들을 넣었음 (사진마다 서로 다른 카메라로 따로 처리)
        </label>
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
          여러 장을 한 번에 올려도 각 사진은 서로 다른 장면으로 보고 따로따로 처리합니다 (같은 카메라의
          여러 프레임을 합쳐서 보고 싶으면 위쪽 폴더 업로드를 쓰세요).
        </p>
        <input type="file" multiple disabled={busy} onChange={handleLooseChange} />

        {busy && (
          <p className="hint">
            {progress ? `업로드 중... (${progress.done}/${progress.total})` : '업로드 준비 중...'}
          </p>
        )}
        {error && <p className="error-text">{error}</p>}
      </div>
    </div>
  )
}
