import { useState } from 'react'
import UploadPage from './UploadPage.jsx'
import ProgressPage from './ProgressPage.jsx'
import ResultsPage from './ResultsPage.jsx'

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
      {stage === 'results' && batch && <ResultsPage batchId={batch.batch_id} cameras={batch.cameras} />}
    </div>
  )
}
