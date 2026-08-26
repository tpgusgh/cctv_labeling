# 사람 승인/거부 피드백 기반 슬롯 후보 분류기

## 배경

`slot_detection.detect_slots()`는 기하/색상 필터(면적, 직사각형도, 색상)만으로
슬롯 후보를 판정한다. 실제 슬롯이 아닌 바닥 표시(경고판, 배경 구조물,
횡단보도 등)가 여전히 오탐으로 통과하는 경우가 있음 — 순수 기하/색상으로는
"이게 뭔지 이해해야 하는" 내용 판별 문제라 원천적 한계가 있다는 게 이전
세션에 충분히 확인됨(고전 CV 7종, CLIP, YOLO, LaneNet 계열 전부 시도 후
폐기 — `HANDOFF.md` 참고).

남은 현실적 선택지는 소량의 사람 피드백으로 가벼운 분류기를 학습시키는
것뿐. 이 문서는 그 설계.

## 핵심 제약 (유지)

- **슬롯 좌표를 사람이 직접 입력/수정하는 건 여전히 금지.** 이 기능은
  기존 `detect_slots()`가 이미 만든 후보 폴리곤에 대해 "이거 진짜
  슬롯 맞음/아님"만 승인·거부하게 한다. 좌표를 그리거나 옮기는 UI는
  없음.
- **완전 로컬/오프라인.** 새 웹서버도 표준 라이브러리로, 분류기도 로컬
  scikit-learn으로.
- **범위 밖(의도적으로 안 함): 미탐(false negative) 수정.** 애초에
  후보로 안 만들어진 슬롯은 리뷰 화면에 뜨지도 않고, 이 기능으로 못
  고침. 미탐이 많은 카메라는 지금처럼 `needs_review` 플래그로만 사람이
  훑어봄. (사용자 확인: "승인/거부만, 미탐은 원칙상 안 건드림")

## 목표

`detect_slots()`가 만든 후보 중 오탐(진짜 슬롯이 아닌 것)을, 사람이 승인/
거부한 소량의 예시로 학습한 분류기로 추가로 걸러낸다. 승인/거부 데이터가
쌓일수록(=리뷰 라운드를 반복할수록) 분류기가 더 정확해지는 반복 루프
("강화학습 느낌"으로 요청됨 — 실제로는 보상함수 기반 RL이 아니라, 사람
피드백으로 점점 정확해지는 반복적 지도학습 루프에 가까움).

## 컴포넌트

### 1. `src/review_store.py`
- `review/labels.jsonl` — append-only. 레코드 하나당 한 줄:
  ```json
  {"id": "<sha1(camera_id + polygon)>", "camera_id": "P1_B1_1_9",
   "image_path": "no_label/P1_B1_1_9/20260820_115033.jpg",
   "polygon": [[x,y],[x,y],[x,y],[x,y]], "crop_path": "review/crops/<id>.png",
   "confidence": 0.87, "decision": "accept", "ts": "2026-08-26T16:00:00"}
  ```
- `id`는 `(camera_id, polygon)`을 안정적으로 해시한 값 — 같은 후보를
  나중에 다시 리뷰해도 최신 결정으로 덮어쓰기(같은 id, 마지막 줄 우선).
- `append_decision(record)`, `load_labels() -> list[dict]`(id별 최신만),
  `unreviewed_ids(all_candidate_ids, labels) -> list[str]` 정도의 얇은
  함수만 제공.

### 2. `src/review_server.py`
- 표준 라이브러리 `http.server.BaseHTTPRequestHandler` 기반, 새 웹
  프레임워크 의존성 없음.
- `GET /` — 아직 리뷰 안 된 후보 crop을 하나(또는 소량 그리드) 보여주는
  최소 HTML + 승인/거부 버튼.
- `POST /decide` — `{id, decision}` 받아서 `review_store.append_decision()`
  호출.
- crop 이미지는 리뷰 시점에 `review/crops/<id>.png`로 미리 저장해둔
  것을 서빙(원본 프레임 폴더가 나중에 사라지거나 바뀌어도 리뷰/학습
  데이터가 안전하도록 — crop을 원본 경로 참조가 아니라 실체 파일로 보관).
- 실행: `python src/review_server.py --port 8765`, 브라우저에서
  `localhost:8765`.

### 3. `src/slot_classifier.py`
- `extract_features(crop_bgr, confidence) -> np.ndarray` — 손으로 뽑은
  특징: 기존 rectangularity(`confidence` 그대로 재사용), grayscale/HSV
  히스토그램 요약 통계, edge 밀도(Canny 픽셀 비율) 등. CNN 아님 — 라벨
  수십 개 수준에서도 즉시 학습되고 오버피팅 위험이 낮아야 하니까.
- `train(labels) -> sklearn Pipeline` — `StandardScaler` +
  `LogisticRegression`. 데이터 늘어나서 선형으로 부족해지면
  `RandomForestClassifier`로 교체 가능하게 함수 시그니처만 안정적으로.
- `save(model, path)` / `load(path)` — `joblib` 사용,
  `models/slot_classifier.joblib`에 저장(재생성 가능한 산출물, gitignore
  대상 — `.gitignore`에 `models/*.joblib` 패턴 추가 필요).

### 4. `slot_detection.detect_slots()` 확장
- 새 파라미터 `classifier=None` (기존 `require_studs`처럼 opt-in,
  기본값은 지금 동작 그대로 안 바뀜).
- 주어지면: 기존 면적/직사각형도/hazard-color 필터 통과한 후보에 대해
  `extract_features` → `classifier.predict()` → reject면 후보 목록에서
  제외.

### 5. `src/train_from_reviews.py`
- CLI 한 줄짜리: `review/labels.jsonl` 로드 → `slot_classifier.train()`
  → `models/slot_classifier.joblib` 저장. 이게 루프의 "재학습" 스텝.

## 루프

1. `generate_config.py` (또는 새 배치 스크립트)로 카메라들 후보 생성 —
   이때 각 후보의 crop을 `review/crops/`에 저장.
2. `review_server.py` 켜고 새로 생긴 후보만 승인/거부(이미 리뷰한 건
   `review_store`가 걸러줘서 안 뜸).
3. `train_from_reviews.py`로 재학습.
4. `detect_slots(..., classifier=load(...))`로 재탐지 → 오탐 줄어든 결과
   확인.
5. 2~4 반복. 라운드 돌수록 분류기가 이미 맞히는 후보는 리뷰 화면에 계속
   나오지 않으므로(아직 "새 후보"의 정의는 카메라+폴리곤 id 기준이라,
   같은 카메라를 다시 돌려도 폴리곤이 안 바뀌면 새로 안 뜸) 리뷰 부담이
   점점 줄어듦.

## 에러 처리

- `review/labels.jsonl`이 아직 없으면 `train_from_reviews.py`는 "라벨
  없음" 메시지 내고 종료(분류기 없이 기존 필터만 쓰는 상태 유지).
- 승인/거부 라벨이 너무 적어서(예: 한쪽 클래스가 0개) 학습이 안 되는
  경우 `slot_classifier.train()`이 명시적 예외를 던지고, 호출부
  (`train_from_reviews.py`)는 이유를 출력하고 종료 — 조용히 이상한
  모델을 저장하지 않음.
- `detect_slots()`에 `classifier`를 안 넘기면 이 기능 전체가 no-op —
  기존 파이프라인/테스트에 영향 없음.

## 테스트

- `tests/test_review_store.py` — 기록 append/로드/같은 id 덮어쓰기(최신
  우선) 라운드트립.
- `tests/test_slot_classifier.py` — 합성으로 명확히 분리되는 accept/
  reject 예시 소량(예: 20개)으로 학습시켜서 held-out 예시를 올바르게
  분류하는지 확인. 실제 카메라 이미지 의존 없음(빠르고 결정적).
- `review_server.py`는 수동 통합 테스트로 충분(HTTP 서버 자체를
  자동화하진 않음) — 브라우저로 열어서 승인/거부 눌러보고
  `labels.jsonl`에 기록되는지 확인.

## 범위 밖 (지금은 안 함)

- 미탐(false negative) 수정 UI — 원칙상 배제(위 제약 참고).
- CNN 기반 분류기 — 데이터 적을 때 손 특징 기반이 더 안정적, 나중에
  라벨이 수백 개 이상 쌓이면 재검토.
- 여러 리뷰어 동시 사용, 인증, 원격 접속 — 로컬 1인 사용 전제.
