import { useEffect, useMemo, useRef, useState } from 'react'
import { listPhotos, photoUrl, editLabel, addLabel, undoPhoto, redoPhoto, downloadCameraUrl, downloadBatchUrl } from './api.js'

const BASE_MAX_DISPLAY_WIDTH = 900

function boxBounds(boxRaw) {
  const xs = boxRaw.map((p) => p[0])
  const ys = boxRaw.map((p) => p[1])
  const left = Math.min(...xs)
  const top = Math.min(...ys)
  return { left, top, width: Math.max(...xs) - left, height: Math.max(...ys) - top }
}

function PhotoEditor({ batchId, cameraId, photo, onChanged }) {
  const [tool, setTool] = useState('select')      // 'select' | 'pen' | 'erase'
  const [selected, setSelected] = useState(null)  // slot id (select tool)
  const [drag, setDrag] = useState(null)
  const [penPoints, setPenPoints] = useState([])  // 펜: 꼭짓점 4개 클릭
  const [naturalSize, setNaturalSize] = useState(null)
  const [error, setError] = useState(null)
  // ×로 가린 라벨은 기본적으로 화면에서 완전히 사라짐 -- 남은 라벨과 겹쳐
  // 보이면 안 되므로. 복원하고 싶을 때만 토글로 표시.
  const [showHidden, setShowHidden] = useState(false)

  // window-level mouseup/keydown read state through refs so they always see
  // the current values (a drag released outside the container used to stick
  // forever otherwise).
  const toolRef = useRef(tool); toolRef.current = tool
  const selectedRef = useRef(selected); selectedRef.current = selected
  const dragRef = useRef(drag); dragRef.current = drag
  const eraseIdsRef = useRef(new Set())

  // photoUrl() appends a fresh cache-busting timestamp -- memoized on
  // `photo` itself so it doesn't regenerate every render (was causing an
  // infinite <img> reload loop).
  const src = useMemo(() => photoUrl(batchId, cameraId, photo.photo), [batchId, cameraId, photo])

  const slots = useMemo(
    () => [...photo.slots].sort((a, b) => (a.confidence ?? Infinity) - (b.confidence ?? Infinity)),
    [photo.slots],
  )

  const maxDisplayWidth = typeof window !== 'undefined'
    ? Math.min(BASE_MAX_DISPLAY_WIDTH, window.innerWidth - 48)
    : BASE_MAX_DISPLAY_WIDTH
  const scale = naturalSize ? Math.min(1, maxDisplayWidth / naturalSize.w) : 1
  const displayW = naturalSize ? naturalSize.w * scale : undefined
  const displayH = naturalSize ? naturalSize.h * scale : undefined
  const toRaw = (x, y) => [x / scale, y / scale]

  function slotById(slotId) {
    return photo.slots.find((s) => s.id === slotId)
  }

  function displayBounds(slot) {
    const b = boxBounds(slot.box_raw)
    return { left: b.left * scale, top: b.top * scale, width: b.width * scale, height: b.height * scale }
  }

  // 화면 좌표 고정 편집: 편집된 라벨 사각형을 픽셀 그대로 보냄. 평면 좌표
  // 왕복 변환은 어안 재투영 때문에 이동만 해도 크기/위치가 출렁였음 (실측).
  async function sendQuad(slotId, quadRaw, applyAll) {
    try {
      await editLabel(batchId, cameraId, photo.photo, slotId, applyAll ? 'adjust_all' : 'adjust', {
        quad_raw: quadRaw,
      })
      setError(null)
      onChanged()
      return true
    } catch (err) {
      setError(err.message)
      return false
    }
  }

  async function finishDrag(shiftHeld) {
    const finished = dragRef.current
    if (!finished) return
    setDrag(null)

    if (finished.kind === 'move') {
      const dx = finished.x1 - finished.x0
      const dy = finished.y1 - finished.y0
      if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return
      const quadRaw = finished.quad.map(([x, y]) => [x + dx / scale, y + dy / scale])
      await sendQuad(finished.slotId, quadRaw, shiftHeld)
      return
    }

    if (finished.kind === 'resize') {
      const [ax, ay] = finished.anchor
      const newW = Math.abs(finished.x1 - ax)
      const newH = Math.abs(finished.y1 - ay)
      if (newW < 6 || newH < 6) return
      const fw = newW / finished.bbox.width
      const fh = newH / finished.bbox.height
      const [axr, ayr] = toRaw(ax, ay)
      // 앵커(반대쪽 모서리) 기준으로 사각형 자체를 배율 -- 모양(기울기) 유지
      const quadRaw = finished.quad.map(([x, y]) => [axr + (x - axr) * fw, ayr + (y - ayr) * fh])
      await sendQuad(finished.slotId, quadRaw, shiftHeld)
      return
    }

    if (finished.kind === 'erase') {
      const ids = [...eraseIdsRef.current]
      eraseIdsRef.current = new Set()
      if (ids.length === 0) return
      const deleteAll = shiftHeld
      if (deleteAll && !window.confirm(`${ids.length}개 슬롯을 모든 사진에서 삭제할까요?\n(재탐지 때도 이 위치들은 다시 잡히지 않습니다)`)) {
        return
      }
      try {
        for (const id of ids) {
          await editLabel(batchId, cameraId, photo.photo, id, deleteAll ? 'delete_all' : 'delete')
        }
        setError(null)
        onChanged()
      } catch (err) {
        setError(err.message)
        onChanged()
      }
    }
  }

  useEffect(() => {
    function handleWindowMouseUp(e) { finishDrag(e.shiftKey) }
    window.addEventListener('mouseup', handleWindowMouseUp)
    return () => window.removeEventListener('mouseup', handleWindowMouseUp)
  }, [photo, scale])

  function collectErase(x, y) {
    for (const s of photo.slots) {
      if (!s.box_raw || s.excluded) continue
      const d = displayBounds(s)
      if (x >= d.left && x <= d.left + d.width && y >= d.top && y <= d.top + d.height) {
        eraseIdsRef.current.add(s.id)
      }
    }
  }

  async function handleMouseDown(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top

    if (tool === 'pen') {
      // 꼭짓점 4개 클릭 -> 슬롯 추가 (4번째 클릭에서 확정)
      const pts = [...penPoints, [px, py]]
      if (pts.length < 4) {
        setPenPoints(pts)
        return
      }
      setPenPoints([])
      try {
        await addLabel(batchId, cameraId, photo.photo, pts.map(([x, y]) => toRaw(x, y)))
        setError(null)
        onChanged()
      } catch (err) {
        setError(err.message)
      }
      return
    }
    if (tool === 'erase') {
      eraseIdsRef.current = new Set()
      collectErase(px, py)
      setDrag({ kind: 'erase', x1: px, y1: py })
      return
    }
    // select tool: drag inside the selected box = move; empty space = deselect
    if (selected) {
      const slot = slotById(selected)
      if (slot?.box_raw) {
        const d = displayBounds(slot)
        if (px >= d.left && px <= d.left + d.width && py >= d.top && py <= d.top + d.height) {
          setDrag({ kind: 'move', slotId: selected, x0: px, y0: py, x1: px, y1: py, bbox: d, quad: slot.box_raw })
          return
        }
      }
      setSelected(null)
    }
  }

  function handleMouseMove(e) {
    if (!drag) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    if (drag.kind === 'erase') {
      collectErase(px, py)
      setDrag((d) => (d ? { ...d, x1: px, y1: py } : d))
    } else {
      setDrag((d) => (d ? { ...d, x1: px, y1: py } : d))
    }
  }

  // 미세 이동: 방향키 (Shift = 크게), Delete = 이 사진에서 삭제, Esc = 선택 해제
  async function nudgeSelected(dxRaw, dyRaw) {
    const slotId = selectedRef.current
    const slot = slotId && slotById(slotId)
    if (!slot?.box_raw) return
    await sendQuad(slotId, slot.box_raw.map(([x, y]) => [x + dxRaw, y + dyRaw]), false)
  }

  useEffect(() => {
    async function handleKeyDown(e) {
      if (toolRef.current !== 'select' || !selectedRef.current) return
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return
      const step = e.shiftKey ? 8 : 2
      if (e.key === 'ArrowLeft') { e.preventDefault(); nudgeSelected(-step, 0) }
      else if (e.key === 'ArrowRight') { e.preventDefault(); nudgeSelected(step, 0) }
      else if (e.key === 'ArrowUp') { e.preventDefault(); nudgeSelected(0, -step) }
      else if (e.key === 'ArrowDown') { e.preventDefault(); nudgeSelected(0, step) }
      else if (e.key === 'Escape') { setSelected(null); setDrag(null) }
      else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        const slotId = selectedRef.current
        setSelected(null)
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

  const selectedSlot = selected ? slotById(selected) : null
  const hiddenCount = slots.filter((s) => s.excluded).length

  const toolHint = tool === 'select'
    ? '박스 클릭 = 선택 · 선택 후 끌기 = 이동, 모서리 핸들 = 크기조절 (모양은 유지되고 위치/크기만 변함), 방향키 = 미세이동(Shift=크게), Delete = 이 사진에서 삭제. Shift 누른 채 놓으면 모든 사진에 적용.'
    : tool === 'pen'
      ? `펜: 슬롯의 네 꼭짓점을 순서대로 클릭 (${penPoints.length}/4) -- 4번째 클릭에서 추가됩니다.`
      : '지우개: 지울 박스들 위를 드래그하면 이 사진에서 가려집니다. Shift 누른 채 놓으면 모든 사진에서 삭제 + 재탐지 차단.'

  async function handleHistory(fn) {
    try {
      await fn(batchId, cameraId, photo.photo)
      setError(null)
      onChanged()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="photo-card">
      <div className="photo-name">{photo.photo}</div>
      <p style={{ marginBottom: 8 }}>
        {[['select', '선택'], ['pen', '펜'], ['erase', '지우개']].map(([key, label]) => (
          <button
            key={key}
            className={tool === key ? 'btn btn-danger' : 'btn'}
            style={{ marginRight: 6 }}
            onClick={() => { setTool(key); setSelected(null); setDrag(null); setPenPoints([]) }}
          >
            {label}
          </button>
        ))}
        <button className="btn" style={{ marginRight: 6 }} title="뒤로가기 (편집 취소)"
                onClick={() => handleHistory(undoPhoto)}>↩ 뒤로</button>
        <button className="btn" style={{ marginRight: 6 }} title="앞으로가기 (다시 실행)"
                onClick={() => handleHistory(redoPhoto)}>↪ 앞으로</button>
        <a className="btn" style={{ marginRight: 6 }} href={src} download={`${photo.photo}_labeled.png`}>
          이 사진 저장
        </a>
        {tool === 'pen' && penPoints.length > 0 && (
          <button className="btn" onClick={() => setPenPoints((p) => p.slice(0, -1))}>마지막 점 취소</button>
        )}
        {hiddenCount > 0 && (
          <button className="btn" onClick={() => setShowHidden((v) => !v)}>
            {showHidden ? '가려진 라벨 숨기기' : `가려진 라벨 ${hiddenCount}개 보기`}
          </button>
        )}
      </p>
      <p className="hint">{toolHint}</p>
      {error && <p className="error-text">{error} (<a href="#" onClick={(e) => { e.preventDefault(); setError(null) }}>닫기</a>)</p>}
      <div
        style={{
          position: 'relative', display: 'inline-block', width: displayW, height: displayH,
          cursor: tool === 'pen' ? 'crosshair' : tool === 'erase' ? 'cell' : 'default',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
      >
        <img
          src={src}
          style={{ display: 'block', width: displayW, height: displayH }}
          draggable={false}
          alt={photo.photo}
          onLoad={(e) => setNaturalSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
        />
        {naturalSize && (
          <svg style={{ position: 'absolute', top: 0, left: 0, width: displayW, height: displayH, pointerEvents: 'none' }}>
            {slots.map((s) => {
              if (!s.box_raw) return null
              if (s.excluded && !showHidden) return null
              const isSelected = selected === s.id
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
            {tool === 'pen' && penPoints.length > 0 && (
              <>
                <polyline
                  points={penPoints.map((p) => p.join(',')).join(' ')}
                  fill="none" stroke="orange" strokeWidth="2"
                />
                {penPoints.map(([x, y], i) => <circle key={i} cx={x} cy={y} r="4" fill="orange" />)}
              </>
            )}
          </svg>
        )}
        {naturalSize && slots.map((s) => {
          if (!s.box_raw) return null
          if (s.excluded && !showHidden) return null
          const d = displayBounds(s)
          return (
            <div
              key={s.id}
              role="button"
              tabIndex={s.excluded || tool !== 'select' ? -1 : 0}
              aria-label={`슬롯 ${s.id}, 신뢰도 ${s.confidence != null ? s.confidence.toFixed(2) : '알 수 없음'}`}
              onClick={(e) => {
                if (tool === 'select' && !s.excluded) { e.stopPropagation(); setSelected(s.id) }
              }}
              onKeyDown={(e) => {
                if ((e.key === 'Enter' || e.key === ' ') && tool === 'select' && !s.excluded) {
                  e.preventDefault()
                  setSelected(s.id)
                }
              }}
              style={{
                position: 'absolute',
                left: d.left, top: d.top, width: d.width, height: d.height,
                cursor: tool === 'select' ? (selected === s.id ? 'move' : 'pointer') : 'inherit',
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
              {s.excluded && tool === 'select' && (
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
        {/* 크기조절 핸들: 선택된 박스의 네 모서리 */}
        {naturalSize && tool === 'select' && selectedSlot?.box_raw && !drag && (() => {
          const d = displayBounds(selectedSlot)
          const corners = [
            [d.left, d.top], [d.left + d.width, d.top],
            [d.left + d.width, d.top + d.height], [d.left, d.top + d.height],
          ]
          return corners.map(([hx, hy], i) => (
            <div
              key={i}
              onMouseDown={(e) => {
                e.stopPropagation()
                const opposite = corners[(i + 2) % 4]
                setDrag({ kind: 'resize', slotId: selectedSlot.id, anchor: opposite, x1: hx, y1: hy, bbox: d, quad: selectedSlot.box_raw })
              }}
              style={{
                position: 'absolute', left: hx - 5, top: hy - 5, width: 10, height: 10,
                background: '#ff4d4f', border: '1px solid #fff', cursor: 'nwse-resize', zIndex: 3,
              }}
            />
          ))
        })()}
        {(drag?.kind === 'move' || drag?.kind === 'resize') && (
          <div
            style={{
              position: 'absolute',
              border: '2px dashed orange',
              pointerEvents: 'none',
              ...(drag.kind === 'move'
                ? {
                    left: drag.bbox.left + (drag.x1 - drag.x0),
                    top: drag.bbox.top + (drag.y1 - drag.y0),
                    width: drag.bbox.width,
                    height: drag.bbox.height,
                  }
                : {
                    left: Math.min(drag.anchor[0], drag.x1),
                    top: Math.min(drag.anchor[1], drag.y1),
                    width: Math.abs(drag.x1 - drag.anchor[0]),
                    height: Math.abs(drag.y1 - drag.anchor[1]),
                  }),
            }}
          />
        )}
      </div>
    </div>
  )
}

export default function ResultsPage({ batchId, cameras, onGoHome }) {
  const [cameraId, setCameraId] = useState(cameras[0]?.camera_id)
  const [photos, setPhotos] = useState([])
  const [loadError, setLoadError] = useState(null)
  // guards against out-of-order responses on fast camera switching
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
          신뢰도가 낮은 순으로 정렬됩니다. 사진마다 도구를 골라 편집:
          <b> 선택</b>(이동·크기조절·미세이동) · <b>펜</b>(둘레를 그리면 슬롯 추가) ·
          <b> 지우개</b>(드래그로 삭제, Shift는 모든 사진 + 재탐지 차단).
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
