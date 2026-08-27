# YOLOv8-seg 기반 슬롯 탐지 전환

## 배경

현재 `slot_detection.detect_slots()`는 전부 classical CV(median stack →
Canny/Hough 라인 → 연결영역 → 면적/직사각형도/색상 필터)로 동작한다.
`HANDOFF.md`에 기록된 대로, 진짜 슬롯이 아닌 바닥 표시(차선 화살표, 경고
구역, 배경 구조물)를 구분하는 건 "이게 뭔지 이해해야 하는" 내용 판별
문제라 기하/색상 필터만으로는 원천적 한계가 있다는 게 이전 세션에 여러
번(고전 CV 7종, CLIP 제로샷, YOLOP) 실측으로 확인됐다.

단, CLIP/YOLOP 실패는 **제로샷**(사전학습 그대로, 파인튜닝 없음) 시도였고,
실패 원인은 전부 동일한 도메인 미스매치(천장 어안렌즈 탑다운 시점이 학습
분포 밖)였다 — 이 프로젝트 자체 라벨로 파인튜닝하는 것과는 다른 얘기다.

원래 프로젝트 스펙 문서(사용자 지시)는 처음부터 "YOLOv8-seg를 사전학습
기반으로 전이학습"을 모델 선택으로 명시하고 있고, 지금 시점엔 이미
`review/labels.jsonl`(accept 95건) + `review/missed.jsonl`(60건) = 폴리곤
정답 155개, 23개 카메라 커버 데이터가 쌓여 있어 전이학습을 시작할 씨앗
데이터가 있다. 이 문서는 그 전환 설계.

## 핵심 제약 (유지)

- 사람이 슬롯 좌표를 카메라마다 손으로 입력하는 건 여전히 금지 — 이번
  전환도 그 원칙을 안 건드림(오히려 강화: 탐지 자체가 더 정확해지는 게
  목적).
- 강화학습 아님 — 지도학습(세그멘테이션) 기반 전이학습.
- Active Learning은 이번 범위 밖(다음 단계, 데이터 더 쌓이면).

## 목표

`detect_slots()`의 whole-image 탐지를 classical CV에서 YOLOv8-seg
파인튜닝 모델로 교체하되, 기존 반환 포맷(`[{"polygon": [[x,y]x4],
"confidence": float}]`)을 그대로 유지해서 `generate_config.py` 밖의 어떤
코드(`pipeline.py`, 웹앱, 리뷰 UI)도 안 건드린다. Hough 기반 후보 생성에
의존하지 않는 whole-image 탐지이므로, `missed.jsonl`이 기록해온 "애초에
후보로도 안 잡히던 슬롯" 문제를 구조적으로 해결할 잠재력이 있다.

## 검토한 대안과 선택 이유

1. **완전 교체(폴백 없음)** — 카메라 23개뿐인 극소 데이터셋 상태로
   안전망 없이 전면 전환은 위험. 기각.
2. **기존 accept/reject 분류기(`slot_classifier.py`)만 YOLO로 교체** —
   후보 자체는 여전히 Hough/연결영역이 제안 → missed-slot 문제 그대로
   남음. 기각.
3. **채택: whole-image YOLOv8-seg를 1차 탐지기로, 저신뢰/0건 시 기존
   `needs_review` 플래그로 폴백.** 아래 컴포넌트 참고.

## 컴포넌트

### 1. `src/yolo_slot_detector.py` (신규)

- `detect_slots(image_bgr, model, conf=0.25) -> list[{"polygon": [[x,y]×4],
  "confidence": float}]` — `slot_detection.detect_slots()`와 동일한 반환
  계약.
- ultralytics YOLOv8-seg 추론 → 마스크별 컨투어 추출 → **4점 quad 피팅은
  `slot_detection.py`에 이미 있는 로직(`approxPolyDP` 4점 성공 시 그대로,
  아니면 `cv2.boxPoints(minAreaRect)`)을 공용 헬퍼로 뽑아서 재사용** —
  중복 구현 안 함. (`perspective.plane_to_pixel_homography`가 정확히
  4점을 요구하므로 — `perspective.py:9` — quad 강제는 선택이 아니라
  필수.)
- `confidence`는 YOLO의 mask/box confidence score 그대로 사용.

### 2. `scripts/export_yolo_dataset.py` (신규)

- 입력: `review_store.load_labels()`에서 `decision == "accept"`인 것 +
  `review_store.load_missed_annotations()` 전부 → 카메라별 폴리곤 집합.
- **핵심 증폭 트릭 (고정 카메라 전제 활용):** 같은 카메라의 폴리곤
  세트는 `no_label/<camera_id>/`의 모든 원본 프레임(카메라당 수십 장,
  조명/차량 배치만 다름)에 그대로 복제 — 슬롯 좌표는 안 바뀌니까. 라벨링
  1세트 → 학습 이미지 수십 장으로 증폭.
- 출력: ultralytics YOLO-seg 포맷(이미지 + `class x1 y1 x2 y2 x3 y3 x4 y4`
  정규화 좌표 txt, 단일 클래스 `parking_slot`).
- **카메라 단위로 train/val 분리** (프레임 단위 아님) — 같은 카메라의
  다른 프레임끼리 train/val에 나뉘어 들어가면 "새 카메라 일반화 성능"을
  거짓으로 높게 측정하게 됨. `reject`(38건) 좌표는 훈련 이미지에 포함은
  되지만 별도 라벨로 안 만듦 — 그 이미지 안에서 암묵적 배경으로 이미
  처리됨(추가 로직 불필요).

### 3. `scripts/train_yolo_seg.py` (신규)

- ultralytics `yolov8n-seg.pt`(COCO 사전학습, 최초 1회 다운로드) 로드 →
  단일 클래스로 파인튜닝 → `models/yolov8_seg_slots.pt` 저장.
- GPU(CUDA/MPS) 사용 가능 전제.

### 4. `generate_config.py` 확장

- 기존 `classifier=` 파라미터 옆에 `yolo_model=None` 파라미터 추가
  (opt-in, 기본값 None = 지금 동작 그대로 — 하위호환).
- 주어지면 `slot_detection.detect_slots()` 대신
  `yolo_slot_detector.detect_slots()` 호출, 그 외 로직(리뷰 후보 저장,
  `needs_review` 판정)은 동일.

## 데이터 흐름

```
review/labels.jsonl (accept) + review/missed.jsonl
        │
        ▼
scripts/export_yolo_dataset.py  (카메라별 프레임에 폴리곤 복제)
        │
        ▼
YOLO-seg 데이터셋 폴더 (train/ val/, 카메라 단위 분리)
        │
        ▼
scripts/train_yolo_seg.py  (yolov8n-seg.pt 파인튜닝)
        │
        ▼
models/yolov8_seg_slots.pt
        │
        ▼
generate_config.py(yolo_model=...)  →  신규/미등록 카메라 config 생성
```

## 에러 처리

- YOLO 추론 결과 0건 또는 전부 저신뢰 → 기존 `needs_review=True` 그대로
  반환 (새 로직 아님, 기존 메커니즘 재사용).
- 학습 데이터 카메라 수가 너무 적어(예: val 카메라 0개) 분리 불가능한
  경우 `export_yolo_dataset.py`가 명시적 에러로 종료 — 조용히 val 없이
  진행하지 않음.
- `yolo_model` 안 넘기면 이 기능 전체 no-op — 기존 파이프라인/테스트
  영향 없음.

## 테스트

- `yolo_slot_detector.py`: 저장된 소형 테스트 체크포인트(또는 mock 모델
  객체)로 quad 피팅 유닛 테스트 — 실제 대형 가중치 의존 없이 결정적으로.
- `export_yolo_dataset.py`: `review/` 픽스처로 출력 포맷 + 카메라 단위
  분리 검증.
- 실사용 탐지 정확도 검증은 held-out 카메라(학습에 아예 안 쓴 카메라)로
  수동 확인 — 유닛테스트로 대체 불가(23개뿐인 데이터셋에서 통계적
  유의미성 자체가 제한적이라는 점 명시).

## 범위 밖 (안 함)

- `pipeline.py`/웹앱/리뷰 UI 변경 — 인터페이스 보존, 전부 그대로.
- Active Learning, 강화학습.
- 프레임별 재탐지 — config 생성 시 1회, 지금 구조 그대로 유지.
- 기존 `slot_classifier.py`(accept/reject 분류기) 제거 — 나란히 유지,
  당장 안 건드림.

## 신규 의존성

`requirements.txt`에 `ultralytics`, `torch` 추가.
