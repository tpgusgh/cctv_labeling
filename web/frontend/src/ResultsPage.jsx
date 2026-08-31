import { useEffect, useMemo, useRef, useState } from 'react'
import { listPhotos, photoUrl, editLabel, addLabel, downloadCameraUrl, downloadBatchUrl } from './api.js'

const BASE_MAX_DISPLAY_WIDTH = 900

function boxBounds(boxRaw) {
  const xs = boxRaw.map((p) => p[0])
  const ys = boxRaw.map((p) => p[1])
  const left = Math.min(...xs)
  const top = Math.min(...ys)
  return { left, top, width: Math.max(...xs) - left, height: Math.max(...ys) - top }
}

function PhotoEditor({ batchId, cameraId, photo, onChanged }) {
  // mode: null (idle)
  //   | {type:'adjust', slotId} -- redrawing an existing slot's label position (drag)
  //   | {type:'add', points:[[x,y]...]} -- a brand-new slot, 4 clicks like the review app's missed-slot page
  const [mode, setMode] = useState(null)
  const [drag, setDrag] = useState(null)
  const [naturalSize, setNaturalSize] = useState(null)
  const [error, setError] = useState(null)
  // ×로 가린 라벨은 기본적으로 화면에서 완전히 사라짐 -- 남은 라벨과 겹쳐
  // 보이면 안 되므로. 복원하고 싶을 때만 토글로 표시.
  const [showHidden, setShowHidden] = useState(false)

  // window release outside this container (another photo card, the page
  // background, anywhere) never fired this component's own onMouseUp, so a
  // drag left `mode`/`drag` stuck forever with no way to recover short of
  // reloading. A single window-level mouseup listener, driven by refs (so
  // it always reads the current drag/mode instead of the values from
  // whenever the listener was attached), ends the drag no matter where the
  // button comes up.
  const modeRef = useRef(mode)
  const dragRef = useRef(drag)
  modeRef.current = mode
  dragRef.current = drag

  // photoUrl() appends a fresh cache-busting timestamp -- memoized on
  // `photo` itself (a new object only when refresh() actually re-fetches)
  // so it doesn't regenerate on every render. Without this, the onLoad
  // below sets state -> re-render -> new timestamp -> <img> reloads ->
  // onLoad fires again -> infinite reload loop (verified: was hammering
  // the backend with GET requests every ~100ms).
  const src = useMemo(() => photoUrl(batchId, cameraId, photo.photo), [batchId, cameraId, photo])

  const slots = useMemo(
    () => [...photo.slots].sort((a, b) => (a.confidence ?? Infinity) - (b.confidence ?? Infinity)),
    [photo.slots],
  )

  // real photos here can be much bigger than this project's usual small
  // camera frames (a 4000x3000 phone photo shown 1:1 is unusably large to
  // edit) -- scale the display down and convert every coordinate through
  // the same factor so raw_polygon sent to the backend stays in true
  // image-pixel space regardless of how small it's shown on screen. Capped
  // by the viewport too so a narrow window doesn't force horizontal overflow.
  const maxDisplayWidth = typeof window !== 'undefined'
    ? Math.min(BASE_MAX_DISPLAY_WIDTH, window.innerWidth - 48)
    : BASE_MAX_DISPLAY_WIDTH
  const scale = naturalSize ? Math.min(1, maxDisplayWidth / naturalSize.w) : 1
  const displayW = naturalSize ? naturalSize.w * scale : undefined
  const displayH = naturalSize ? naturalSize.h * scale : undefined
  const toRaw = (x, y) => [x / scale, y / scale]

  async function sendAdjust(slotId, rawX0, rawY0, rawX1, rawY1, applyAll) {
    try {
      // applyAll (shift) = 이 위치를 이 슬롯의 기본값으로 저장
      // (이 카메라의 모든 사진/배치에 적용) -- 사진마다 반복 드래그 불필요.
      await editLabel(batchId, cameraId, photo.photo, slotId, applyAll ? 'adjust_all' : 'adjust', {
        raw_polygon: [[rawX0, rawY0], [rawX1, rawY1]],
      })
      setError(null)
      onChanged()
      return true
    } catch (err) {
      setError(err.message)
      return false
    }
  }

  async function finishDrag(applyAll) {
    const finished = dragRef.current
    const activeMode = modeRef.current
    if (!finished || activeMode?.type !== 'adjust') return
    setDrag(null)
    const slotId = activeMode.slotId

    if (finished.kind === 'move') {
      // 박스 자체를 잡아 끌기: 크기 유지, 위치만 이동
      const dx = finished.x1 - finished.x0
      const dy = finished.y1 - finished.y0
      if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return // 그냥 클릭 -- 선택 유지
      const b = finished.bbox
      const [x0, y0] = toRaw(b.left + dx, b.top + dy)
      const [x1, y1] = toRaw(b.left + b.width + dx, b.top + b.height + dy)
      if (await sendAdjust(slotId, x0, y0, x1, y1, applyAll)) setMode(null)
      return
    }

    if (Math.abs(finished.x1 - finished.x0) < 6 || Math.abs(finished.y1 - finished.y0) < 6) {
      setMode(null)
      return
    }
    const [x0, y0] = toRaw(Math.min(finished.x0, finished.x1), Math.min(finished.y0, finished.y1))
    const [x1, y1] = toRaw(Math.max(finished.x0, finished.x1), Math.max(finished.y0, finished.y1))
    // keep `mode` on failure (the slot stays selected) so the user can just
    // drag again without reselecting it
    if (await sendAdjust(slotId, x0, y0, x1, y1, applyAll)) setMode(null)
  }

  useEffect(() => {
    function handleWindowMouseUp(e) {
      finishDrag(e.shiftKey)
    }
    window.addEventListener('mouseup', handleWindowMouseUp)
    return () => window.removeEventListener('mouseup', handleWindowMouseUp)
  }, [])

  function handleMouseDown(e) {
    if (mode?.type !== 'adjust') return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    // 선택된 박스 안에서 누르면 이동, 밖에서 누르면 새로 그리기
    const slot = photo.slots.find((s) => s.id === mode.slotId)
    if (slot?.box_raw) {
      const b = boxBounds(slot.box_raw)
      const disp = { left: b.left * scale, top: b.top * scale, width: b.width * scale, height: b.height * scale }
      if (px >= disp.left && px <= disp.left + disp.width && py >= disp.top && py <= disp.top + disp.height) {
        setDrag({ kind: 'move', x0: px, y0: py, x1: px, y1: py, bbox: disp })
        return
      }
    }
    setDrag({ x0: px, y0: py, x1: px, y1: py })
  }
  function handleMouseMove(e) {
    if (!drag) return
    const rect = e.currentTarget.getBoundingClientRect()
    setDrag((d) => (d ? { ...d, x1: e.clientX - rect.left, y1: e.clientY - rect.top } : d))
  }

  // 미세 이동: 선택된 슬롯을 화살표 키로 raw 픽셀 단위 nudge (shift = 8px)
  async function nudgeSelected(dxRaw, dyRaw) {
    const activeMode = modeRef.current
    if (activeMode?.type !== 'adjust') return
    const slot = photo.slots.find((s) => s.id === activeMode.slotId)
    if (!slot?.box_raw) return
    const b = boxBounds(slot.box_raw)
    await sendAdjust(activeMode.slotId, b.left + dxRaw, b.top + dyRaw,
                      b.left + b.width + dxRaw, b.top + b.height + dyRaw, false)
  }

  useEffect(() => {
    async function handleKeyDown(e) {
      const activeMode = modeRef.current
      if (activeMode?.type !== 'adjust') return
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return
      const step = e.shiftKey ? 8 : 2
      if (e.key === 'ArrowLeft') { e.preventDefault(); nudgeSelected(-step, 0) }
      else if (e.key === 'ArrowRight') { e.preventDefault(); nudgeSelected(step, 0) }
      else if (e.key === 'ArrowUp') { e.preventDefault(); nudgeSelected(0, -step) }
      else if (e.key === 'ArrowDown') { e.preventDefault(); nudgeSelected(0, step) }
      else if (e.key === 'Escape') { setMode(null); setDrag(null) }
      else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        const slotId = activeMode.slotId
        setMode(null)
        try {
          await editLabel(batchId, cameraId, photo.photo, slotId, 'delete')
          setError(null)
          onChanged()
        } catch (err) {
          setError(err.message)
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [photo])

  async function handleContainerClick(e) {
    if (mode?.type !== 'add') return
    const rect = e.currentTarget.getBoundingClientRect()
    const points = [...mode.points, [e.clientX - rect.left, e.clientY - rect.top]]
    if (points.length < 4) {
      setMode({ type: 'add', points })
      return
    }
    try {
      const rawPolygon = points.map(([x, y]) => toRaw(x, y))
      await addLabel(batchId, cameraId, photo.photo, rawPolygon)
      setMode(null)
      setError(null)
      onChanged()
    } catch (err) {
      // keep the 4 clicked points so the user doesn't have to redo them --
      // only the "전체 취소" link or a fresh 4th click retries/abandons it
      setError(err.message)
    }
  }

  function undoLastPoint() {
    setMode((m) => (m.points.length ? { ...m, points: m.points.slice(0, -1) } : m))
  }

  async function handleDelete(slotId, e) {
    e.stopPropagation()
    // shift+× = "이 슬롯 자체가 잘못됨": 카메라 config에서 제거 (모든 사진에
    // 반영), 그리고 리뷰 로그에 거부로 기록되어 다음 재탐지에서도 이 위치가
    // 자동으로 걸러짐. 일반 × = 이 사진에서만 가림 (차가 가렸을 때 등).
    const deleteAll = e.shiftKey
    if (deleteAll && !window.confirm(`${slotId} 슬롯을 모든 사진에서 삭제할까요?\n(카메라 설정에서 제거되고, 재탐지 시에도 이 위치는 다시 잡히지 않습니다)`)) {
      return
    }
    try {
      await editLabel(batchId, cameraId, photo.photo, slotId, deleteAll ? 'delete_all' : 'delete')
      setError(null)
      onChanged()
    } catch (err) {
      setError(err.message)
    }
  }

  function handleSelectSlot(slotId) {
    if (mode) return
    setError(null)
    setMode({ type: 'adjust', slotId })
  }

  const hint = mode?.type === 'adjust'
    ? <>{mode.slotId} 선택됨 -- 박스를 끌면 이동, 바깥에 드래그하면 새로 그리기, 방향키 미세이동(Shift=크게), Delete=이 사진에서 삭제, Esc=취소. Shift 누른 채 놓으면 모든 사진에 적용 (<a href="#" onClick={(e) => { e.preventDefault(); setMode(null) }}>취소</a>)</>
    : mode?.type === 'add'
      ? (
        <>
          새 슬롯 추가 중 -- 놓친 슬롯의 네 꼭짓점을 순서대로 클릭 ({mode.points.length}/4)
          {' '}(<a href="#" onClick={(e) => { e.preventDefault(); undoLastPoint() }}>마지막 점 취소</a>
          {' · '}<a href="#" onClick={(e) => { e.preventDefault(); setMode(null) }}>전체 취소</a>)
        </>
      )
      : null

  return (
    <div className="photo-card">
      <div className="photo-name">{photo.photo}</div>
      {hint && <p className="hint">{hint}</p>}
      {error && <p className="error-text">{error} (<a href="#" onClick={(e) => { e.preventDefault(); setError(null) }}>닫기</a>)</p>}
      {!mode && (
        <p style={{ marginBottom: 8 }}>
          <button className="btn" onClick={() => setMode({ type: 'add', points: [] })}>+ 새 슬롯 추가</button>
          {slots.some((s) => s.excluded) && (
            <button className="btn" style={{ marginLeft: 8 }} onClick={() => setShowHidden((v) => !v)}>
              {showHidden ? '가려진 라벨 숨기기' : `가려진 라벨 ${slots.filter((s) => s.excluded).length}개 보기`}
            </button>
          )}
        </p>
      )}
      <div
        style={{
          position: 'relative', display: 'inline-block', width: displayW, height: displayH,
          cursor: mode ? 'crosshair' : 'default',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onClick={handleContainerClick}
      >
        <img
          src={src}
          style={{ display: 'block', width: displayW, height: displayH }}
          draggable={false}
          alt={photo.photo}
          onLoad={(e) => setNaturalSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
        />
        {naturalSize && (
          <svg
            style={{ position: 'absolute', top: 0, left: 0, width: displayW, height: displayH, pointerEvents: 'none' }}
          >
            {slots.map((s) => {
              if (!s.box_raw) return null
              if (s.excluded && !showHidden) return null
              const isSelected = mode?.type === 'adjust' && mode.slotId === s.id
              return (
                <polygon
                  key={s.id}
                  points={s.box_raw.map(([x, y]) => `${x * scale},${y * scale}`).join(' ')}
                  fill="none"
                  stroke={s.excluded ? '#888' : isSelected ? '#ff4d4f' : 'var(--accent-cyan)'}
                  strokeWidth="2"
                  strokeDasharray={s.excluded ? '6 4' : undefined}
                  opacity={s.excluded ? 0.5 : 1}
                />
              )
            })}
          </svg>
        )}
        {naturalSize && slots.map((s) => {
          if (!s.box_raw) return null
          if (s.excluded && !showHidden) return null
          const b = boxBounds(s.box_raw)
          const isSelected = mode?.type === 'adjust' && mode.slotId === s.id
          return (
            <div
              key={s.id}
              role="button"
              tabIndex={s.excluded || mode ? -1 : 0}
              aria-label={`슬롯 ${s.id}, 신뢰도 ${s.confidence != null ? s.confidence.toFixed(2) : '알 수 없음'}`}
              onClick={(e) => { if (!s.excluded && !mode) { e.stopPropagation(); handleSelectSlot(s.id) } }}
              onKeyDown={(e) => {
                if ((e.key === 'Enter' || e.key === ' ') && !s.excluded && !mode) {
                  e.preventDefault()
                  handleSelectSlot(s.id)
                }
              }}
              style={{
                position: 'absolute',
                left: b.left * scale,
                top: b.top * scale,
                width: b.width * scale,
                height: b.height * scale,
                // 실제 라벨 모양은 위 SVG 폴리곤이 그림 -- 이 div는 클릭/버튼용
                // 투명 히트박스만 담당 (bbox 사각형을 테두리로 그리면 기울어진
                // 이웃 라벨들이 화면에서만 겹쳐 보이는 착시가 생김)
                cursor: mode ? (isSelected ? 'move' : 'default') : 'pointer',
                boxSizing: 'border-box',
              }}
            >
              <span
                style={{
                  position: 'absolute', top: -18, left: 0, fontSize: 11, whiteSpace: 'nowrap',
                  background: '#000', color: '#fff', padding: '0 4px', borderRadius: 2,
                }}
              >
                {s.confidence != null ? s.confidence.toFixed(2) : '-'}
              </span>
              {!s.excluded && !mode && (
                <button
                  className="btn btn-danger"
                  aria-label={`슬롯 ${s.id} 삭제`}
                  onClick={(e) => handleDelete(s.id, e)}
                  style={{ position: 'absolute', top: -12, right: -12, width: 20, height: 20, padding: 0, lineHeight: '16px', fontSize: 12 }}
                >
                  ×
                </button>
              )}
              {s.excluded && !mode && (
                <button
                  className="btn"
                  aria-label={`슬롯 ${s.id} 복원`}
                  onClick={async (e) => {
                    e.stopPropagation()
                    try {
                      await editLabel(batchId, cameraId, photo.photo, s.id, 'restore')
                      setError(null)
                      onChanged()
                    } catch (err) {
                      setError(err.message)
                    }
                  }}
                  style={{ position: 'absolute', top: -14, right: -14, height: 20, padding: '0 5px', lineHeight: '16px', fontSize: 11 }}
                >
                  복원
                </button>
              )}
            </div>
          )
        })}
        {drag && drag.kind === 'move' && (
          <div
            style={{
              position: 'absolute',
              border: '2px dashed orange',
              left: drag.bbox.left + (drag.x1 - drag.x0),
              top: drag.bbox.top + (drag.y1 - drag.y0),
              width: drag.bbox.width,
              height: drag.bbox.height,
              pointerEvents: 'none',
            }}
          />
        )}
        {drag && drag.kind !== 'move' && (
          <div
            style={{
              position: 'absolute',
              border: '2px dashed orange',
              left: Math.min(drag.x0, drag.x1),
              top: Math.min(drag.y0, drag.y1),
              width: Math.abs(drag.x1 - drag.x0),
              height: Math.abs(drag.y1 - drag.y0),
            }}
          />
        )}
        {mode?.type === 'add' && mode.points.length > 0 && (
          <svg style={{ position: 'absolute', top: 0, left: 0, width: displayW, height: displayH, pointerEvents: 'none' }}>
            <polyline points={mode.points.map((p) => p.join(',')).join(' ')} fill="none" stroke="orange" strokeWidth="2" />
            {mode.points.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="4" fill="orange" />)}
          </svg>
        )}
      </div>
    </div>
  )
}

export default function ResultsPage({ batchId, cameras, onGoHome }) {
  const [cameraId, setCameraId] = useState(cameras[0]?.camera_id)
  const [photos, setPhotos] = useState([])
  const [loadError, setLoadError] = useState(null)
  // guards against out-of-order responses: if the user switches cameras
  // again before an in-flight listPhotos() for the *previous* camera
  // resolves, that stale response must not overwrite the newer one.
  const requestIdRef = useRef(0)

  async function refresh() {
    if (!cameraId) return
    const requestId = ++requestIdRef.current
    try {
      const data = await listPhotos(batchId, cameraId)
      if (requestId !== requestIdRef.current) return
      setPhotos(data.photos)
      setLoadError(null)
    } catch (err) {
      if (requestId !== requestIdRef.current) return
      setLoadError(err.message)
    }
  }

  useEffect(() => {
    // clear immediately on camera switch -- otherwise the old camera's
    // photos briefly render paired with the new cameraId until refresh()
    // resolves, so PhotoEditor requests a photo name under the wrong
    // camera and gets a 404.
    setPhotos([])
    refresh()
  }, [cameraId])

  if (!cameraId) {
    return (
      <div>
        <p className="eyebrow">03 · RESULTS</p>
        <div className="panel"><p className="hint">카메라 정보가 없습니다.</p></div>
      </div>
    )
  }

  return (
    <div>
      <p className="eyebrow">03 · RESULTS</p>
      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>결과 확인</h2>
        <p className="hint">
          신뢰도가 낮은 순으로 정렬됩니다. 박스를 클릭해 선택하면: 박스를 끌어서 이동,
          바깥에 드래그해서 새로 그리기, 방향키로 미세 이동(Shift=크게), Delete 키로 이 사진에서 삭제.
          × = 이 사진에서만 가림 (가려진 박스의 "복원" 버튼으로 되돌리기),
          Shift+× = 잘못 잡힌 슬롯을 모든 사진에서 삭제 + 재탐지 때도 그 위치를 다시 잡지 않게 기억.
          위치 조정을 Shift 누른 채 놓으면 그 위치가 모든 사진에 적용됩니다.
        </p>
        <label>
          카메라 선택{' '}
          <select aria-label="카메라 선택" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            {cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>{c.camera_id}</option>
            ))}
          </select>
        </label>
        <div className="link-row" style={{ marginTop: 12 }}>
          <a href={downloadCameraUrl(batchId, cameraId)}>이 카메라 다운로드</a>
          {' · '}
          <a href={downloadBatchUrl(batchId)}>전체 다운로드</a>
        </div>
        <p style={{ marginTop: 12 }}>
          <button className="btn" onClick={onGoHome}>다른 사진 올리기</button>
        </p>
      </div>
      {loadError && (
        <p className="error-text">
          결과를 불러오지 못했습니다: {loadError} (<a href="#" onClick={(e) => { e.preventDefault(); refresh() }}>다시 시도</a>)
        </p>
      )}
      <div>
        {photos.map((p) => (
          <PhotoEditor key={p.photo} batchId={batchId} cameraId={cameraId} photo={p} onChanged={refresh} />
        ))}
      </div>
    </div>
  )
}
