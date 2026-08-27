# React 라벨링 웹 애플리케이션 설계

## 배경

지금까지 라벨링은 전부 CLI(`generate_config.py`, `main.py`, `batch_processor.py`)로만
가능했다. README의 sub-project 7("GUI")이 원래 PySide6 데스크톱 앱으로 계획돼
있었으나, 이번 세션에 사용자가 React 기반 웹 애플리케이션으로 방향을 바꿔
요청했다 — 사진(폴더 단위, 카메라당 폴더 하나)을 드래그드롭으로 올리면
자동으로 슬롯 탐지/라벨링하고, 결과에서 라벨을 사진별로 삭제/수정하고,
다운로드까지 하는 하나의 웹사이트.

## 핵심 제약 (기존 원칙 유지)

- **슬롯 좌표 손입력 금지 원칙은 "공유 config"에 대해서만 유지된다.** 웹에서
  하는 라벨 삭제/수정은 카메라 config(`config/<camera_id>.json`)의
  `polygon_raw`를 절대 건드리지 않는다 — 특정 사진 한 장의 출력 결과에만
  적용되는 오버라이드다.
- 완전 로컬. Flask/React 둘 다 로컬 실행, 외부 서비스 호출 없음.
- 기존 `src/` 파이프라인 함수를 그대로 재사용한다 — 탐지/라벨링 로직을
  웹 백엔드에서 새로 짜지 않는다.

## 요구사항 정리 (브레인스토밍으로 확정)

1. **업로드 = 폴더 단위.** 폴더 이름 = 카메라 id. 한 번에 폴더 최대 100개까지
   (UI에서 안내하는 가이드라인, 하드 리밋 아님). 폴더 안 사진은 1장이어도 됨.
2. **카메라별 분기**: 이미 `config/<camera_id>.json`이 있으면 탐지 생략하고
   업로드된 사진 전부 바로 라벨링(1장이어도 정상 동작). 없으면 업로드된
   사진들로 `generate_config()`와 동일한 로직(median 스택+탐지)을 돌려서 새
   config를 만들고, 그 결과로 review 후보/crop도 기존과 동일하게
   `review/candidates.jsonl`에 쌓은 뒤 라벨링까지 진행. 사진이 적으면(특히
   1장) 탐지 품질이 낮을 수 있음 — 별도 처리 없이 기존 `needs_review` 표시로
   흘러가게 둔다(최소 프레임 수 강제 안 함).
3. **오래 걸리는 배치는 백그라운드 처리**, 진행상황 폴링으로 확인.
4. **라벨 삭제 = 사진 한 장 안에서 슬롯 하나만 지움**(그 사진의 나머지 슬롯
   라벨은 유지). 카메라 공유 config는 안 바뀜.
5. **라벨 수정 = 사진 한 장 안에서 슬롯 하나의 라벨 박스 위치/크기만 조정**
   (드래그). 마찬가지로 그 사진에만 적용, 공유 config 불변.
6. **삭제는 "문제 신호"로 리뷰 도구에 넘어간다** — 웹에서 삭제한다고 곧바로
   `review/labels.jsonl`(정식 학습 라벨)에 거부로 기록되지는 않는다. 대신
   `review/web_flags.jsonl`에 쌓이고, `review_server.py`가 이 플래그 걸린
   후보를 리뷰 큐에서 우선적으로 보여줘서 사람이 정식으로 승인/거부
   판단하도록 유도한다.

## 아키텍처

```
web/
  backend/          Flask 앱 (신규 의존성: Flask 하나만 추가)
  frontend/          React + Vite (신규 의존성: React 생태계, 프론트 전용)
run_web.sh          프론트 빌드 + Flask가 빌드 결과물+API 같이 서빙 (서버 1개, 포트 1개)
```

**백엔드로 Flask를 고른 이유**: 폴더 업로드(파일 다수 + 상대경로 구조 보존)
파싱과 백그라운드 작업 상태 추적을 순수 stdlib `http.server`로 손으로 짜면
(특히 멀티파트 파싱은 Python 3.13+에서 `cgi` 모듈이 아예 제거됨) 코드량도
많고 파일 업로드라는 보안 민감 영역에서 버그 위험이 큼. FastAPI까지는
과함(로컬 1인 툴이라 비동기/타입검증 이점이 크지 않음) — Flask 하나로 충분.

**백엔드는 처리 로직을 새로 짜지 않는다** — `src/generate_config.py`,
`src/pipeline.py`, `src/slot_classifier.py`, `src/review_store.py`를 그대로
import해서 호출한다.

## 데이터 흐름

1. `POST /api/upload` — multipart로 폴더(들) 업로드. 최상위 폴더명으로
   그룹핑 → `web_uploads/<batch_id>/<camera_id>/` 에 원본 저장(gitignore
   대상, ephemeral) → 카메라별로 백그라운드 작업 하나씩 생성 →
   `{batch_id, cameras: [{camera_id, job_id, photo_count}]}` 즉시 응답.
2. 백그라운드 작업(스레드풀, `concurrent.futures.ThreadPoolExecutor`):
   - config 없으면: `generate_config.generate_config(camera_id, frames_dir=업로드폴더, output_path=config/<camera_id>.json, classifier=있으면 로드)` 호출
     (기존 CLI와 동일 부수효과: `config/*.json` 갱신, `review/candidates.jsonl`+`review/crops/` 갱신)
   - 업로드된 사진 전부에 대해 `pipeline.run_auto_all()`로 라벨링 →
     `web_uploads/<batch_id>/<camera_id>/labeled/<photo>.png`에 저장
   - 작업 상태를 메모리 내 레지스트리에 갱신(큐잉/탐지중/라벨링중/완료/에러)
3. 프론트는 `GET /api/batches/<batch_id>/status`로 폴링, 완료된 카메라부터
   결과 화면에 노출.
4. 결과 화면: `GET /api/batches/<batch_id>/cameras/<camera_id>/photos`로
   사진 목록(썸네일 URL, 슬롯별 상태) 조회. 사진 클릭 시 슬롯별 삭제/수정 UI.
5. `POST /api/.../photos/<photo>/labels/<slot_id>` — `{action: "delete"}`
   또는 `{action: "adjust", box: {...}}`. 오버라이드를
   `web_uploads/<batch_id>/<camera_id>/overrides/<photo>.json`에 저장하고
   해당 사진만 재렌더링. `action=="delete"`일 때만 `review_store`에
   web flag 기록(수정은 단순 위치 보정이라 오탐 신호로 안 봄).
6. 다운로드: `GET /api/batches/<batch_id>/cameras/<camera_id>/download`(카메라
   단위 zip), `GET /api/batches/<batch_id>/download`(배치 전체 zip). 항상
   최신 오버라이드가 반영된 이미지로 압축.

## 오버라이드 저장 형식

```json
// web_uploads/<batch_id>/<camera_id>/overrides/<photo>.json
{
  "excluded_slots": ["slot-2"],
  "adjusted": {
    "slot-0": {"cx": 0.52, "cy": 0.48, "w": 0.6, "h": 0.6}
  }
}
```
`cx/cy/w/h`는 기존 `pipeline.py`의 정규화 좌표 관례(0~1, 슬롯 폴리곤
기준)를 그대로 따른다. 재렌더링 시 `excluded_slots`에 있는 슬롯은
건너뛰고, `adjusted`에 있는 슬롯은 그 값으로 라벨 위치/크기를 덮어쓴다.

## review_store.py / review_server.py 추가

- `review_store.py`: `WEB_FLAGS_PATH`, `append_web_flag(record)`,
  `load_web_flags()` — 기존 `append_decision`/`load_labels`와 동일한 얇은
  append/load 패턴. 레코드: `{id, camera_id, slot_id, photo, ts}` (`id`는
  기존 `candidate_id(camera_id, slot_polygon)`과 동일 해시 체계 재사용).
- `review_server.py`: `_next_unreviewed()`가 web flag 걸린 후보를 큐 맨
  앞으로 우선 배치. 홈 화면에 "웹에서 지적된 후보 N개" 카운트 표시. 별도
  페이지는 만들지 않음(큐 우선순위 조정 + 카운트 표시로 충분, YAGNI).

## API 요약

| 메서드/경로 | 설명 |
|---|---|
| `POST /api/upload` | 폴더(들) 업로드, 배치+카메라별 작업 생성 |
| `GET /api/batches/<batch_id>/status` | 카메라별 작업 상태 |
| `GET /api/batches/<batch_id>/cameras/<camera_id>/photos` | 사진 목록+상태 |
| `GET /api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>` | 현재(오버라이드 반영) 이미지 |
| `POST /api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/labels/<slot_id>` | 삭제/수정 |
| `GET /api/batches/<batch_id>/cameras/<camera_id>/download` | 카메라 단위 zip |
| `GET /api/batches/<batch_id>/download` | 배치 전체 zip |

## 프론트엔드 (React + Vite)

- 업로드 화면: 폴더 드래그드롭(브라우저 네이티브
  `DataTransferItem.webkitGetAsEntry` — 추가 라이브러리 불필요), 업로드 전
  폴더별 파일 개수 미리보기.
- 진행상황 화면: 배치 상태 폴링, 카메라별 진행 표시.
- 결과 화면: 카메라별 사진 썸네일 그리드 → 클릭 시 캔버스 오버레이로 슬롯별
  삭제 버튼 + 드래그 리사이즈, 카메라/전체 다운로드 버튼.

## 실행 스크립트

`run_web.sh` — `web/frontend`에서 `npm install && npm run build`(빌드
결과물 없으면), 그다음 `web/backend`의 Flask 앱을 `.venv` 파이썬으로
실행(빌드된 정적 파일 서빙 + API 같은 포트). 사용자는 이 스크립트 하나만
실행하면 됨.

## 에러 처리

- 업로드 폴더가 비었거나 이미지가 하나도 없으면 그 카메라 작업은 즉시
  `error` 상태로 표시, 나머지 카메라 작업에는 영향 없음.
- 탐지 실패(0개 슬롯)는 기존 `needs_review` 그대로 프론트에 노출.
- 존재하지 않는 batch_id/camera_id/photo/slot_id 요청은 404.

## 테스트

- Flask 백엔드: Flask test client로 업로드→작업 생성, 라벨 삭제/수정→
  오버라이드 반영, 다운로드→zip 응답 검증(실제 서버 기동 없이).
- 프론트엔드: 자동 테스트 없음(기존 `review_server.py`와 동일하게 수동
  확인으로 충분 — YAGNI, 로컬 1인 툴).
- 기존 `tests/` 스위트는 변경 없음(웹 계층은 `src/` 함수를 호출만 함).

## 범위 밖

- 배치 상태의 서버 재시작 후 영속성 — 메모리 내 레지스트리로 충분(로컬
  세션 툴, `review_server.py`와 동일 전제).
- 인증/다중 사용자 — 로컬 1인 사용 전제.
- 미탐(슬롯 자체가 안 잡힘) 수정 — 기존 원칙대로 범위 밖, `/missed` 도구가
  이미 별도로 다룸.
