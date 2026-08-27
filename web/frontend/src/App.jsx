import { useState } from 'react'
import UploadPage from './UploadPage.jsx'
import ProgressPage from './ProgressPage.jsx'

export default function App() {
  const [batch, setBatch] = useState(null)
  const [stage, setStage] = useState('upload')

  function handleUploaded(batchData) {
    setBatch(batchData)
    setStage('progress')
  }

  function handleAllDone() {
    setStage('results')
  }

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 16, fontFamily: 'sans-serif' }}>
      <h1>CCTV 라벨링</h1>
      {stage === 'upload' && <UploadPage onUploaded={handleUploaded} />}
      {stage === 'progress' && batch && (
        <ProgressPage batchId={batch.batch_id} cameras={batch.cameras} onAllDone={handleAllDone} />
      )}
      {stage === 'results' && batch && <p>결과 화면 (다음 작업에서 추가)</p>}
    </div>
  )
}
