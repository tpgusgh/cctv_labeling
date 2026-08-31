import { useEffect, useRef, useState } from 'react'
import UploadPage from './UploadPage.jsx'
import ProgressPage from './ProgressPage.jsx'
import ResultsPage from './ResultsPage.jsx'
import { getBatchStatus } from './api.js'

// batch_id + stage live in the URL (not just React state) so a refresh, a
// back/forward navigation, or sharing the link restores the same screen
// instead of dropping back to an empty upload page.
function readUrlState() {
  const params = new URLSearchParams(window.location.search)
  return { batchId: params.get('batch'), stage: params.get('stage') }
}

function writeUrlState(batchId, stage) {
  const params = new URLSearchParams()
  if (batchId) params.set('batch', batchId)
  if (stage) params.set('stage', stage)
  const query = params.toString()
  window.history.pushState({}, '', query ? `?${query}` : window.location.pathname)
}

export default function App() {
  const [batch, setBatch] = useState(null)
  const [stage, setStage] = useState('upload')
  const [ready, setReady] = useState(false)
  // guards against a slow load losing to (and then overwriting) a faster
  // one that started later -- e.g. two quick Back-button presses firing
  // popstate twice before the first getBatchStatus() call resolves.
  const requestIdRef = useRef(0)

  async function loadFromUrl() {
    const requestId = ++requestIdRef.current
    const { batchId, stage: urlStage } = readUrlState()
    if (!batchId) {
      setBatch(null)
      setStage('upload')
      setReady(true)
      return
    }
    try {
      const data = await getBatchStatus(batchId)
      if (requestId !== requestIdRef.current) return
      setBatch({ batch_id: batchId, cameras: data.cameras })
      setStage(urlStage === 'results' ? 'results' : 'progress')
    } catch {
      // batch unknown (e.g. server restarted since) -- fall back to upload
      // instead of showing a broken progress/results page.
      if (requestId !== requestIdRef.current) return
      setBatch(null)
      setStage('upload')
    }
    if (requestId === requestIdRef.current) setReady(true)
  }

  useEffect(() => {
    loadFromUrl()
    window.addEventListener('popstate', loadFromUrl)
    return () => window.removeEventListener('popstate', loadFromUrl)
  }, [])

  function handleUploaded(batchData) {
    setBatch(batchData)
    setStage('progress')
    writeUrlState(batchData.batch_id, 'progress')
  }

  function handleAllDone() {
    setStage('results')
    writeUrlState(batch.batch_id, 'results')
  }

  function goHome() {
    setBatch(null)
    setStage('upload')
    writeUrlState(null, null)
  }

  const steps = [
    { key: 'upload', label: '01 UPLOAD' },
    { key: 'progress', label: '02 PROCESS' },
    { key: 'results', label: '03 RESULTS' },
  ]

  if (!ready) return null

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title-row">
          <h1
            className="app-title"
            role="button"
            tabIndex={0}
            onClick={goHome}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goHome() } }}
            style={{ cursor: 'pointer' }}
            title="처음으로"
          >
            CCTV LABELING
          </h1>
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
      {stage === 'results' && batch && (
        <ResultsPage batchId={batch.batch_id} cameras={batch.cameras} onGoHome={goHome} />
      )}
    </div>
  )
}
