import { useState } from 'react'
import { uploadFolders } from './api.js'

export default function UploadPage({ onUploaded }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function handleChange(e) {
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

  return (
    <div>
      <p>
        카메라 폴더가 들어있는 상위 폴더 하나를 선택하세요 (폴더 최대 100개 권장).
        폴더 하나 = 카메라 하나로 인식합니다. 사진 1장짜리 폴더도 됩니다.
      </p>
      <input
        type="file"
        webkitdirectory="true"
        directory="true"
        multiple
        disabled={busy}
        onChange={handleChange}
      />
      {busy && <p>업로드 중...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  )
}
