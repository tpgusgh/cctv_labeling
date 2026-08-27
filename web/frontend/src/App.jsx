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

  const steps = [
    { key: 'upload', label: '01 UPLOAD' },
    { key: 'progress', label: '02 PROCESS' },
    { key: 'results', label: '03 RESULTS' },
  ]

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title-row">
          <h1 className="app-title">CCTV LABELING</h1>
          <span className="app-live"><span className="app-live-dot" />LIVE</span>
        </div>
        <nav className="app-stepper">
          {steps.map((s) => (
            <span key={s.key} className={stage === s.key ? 'active' : ''}>{s.label}</span>
          ))}
        </nav>
      </header>
      {stage === 'upload' && <UploadPage onUploaded={handleUploaded} />}
      {stage === 'progress' && batch && (
        <ProgressPage batchId={batch.batch_id} cameras={batch.cameras} onAllDone={handleAllDone} />
      )}
      {stage === 'results' && batch && <ResultsPage batchId={batch.batch_id} cameras={batch.cameras} />}
    </div>
  )
}
