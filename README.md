# CCTV 주차공간 라벨링 자동화

저장된 CCTV(어안렌즈) 이미지에 주차공간별 라벨을 자동으로 배치하고 PNG로
저장하는 이미지 라벨링 자동화 프로그램. 전체 목표와 요구사항은 서브프로젝트로
쪼개서 진행 중.

## 진행 상태

| 서브프로젝트 | 상태 | 설명 |
|---|---|---|
| 1. 핵심 기하/렌더링 파이프라인 | ✅ 완료 | 어안렌즈 왜곡 보정 + 슬롯별 로컬 재투영 + 라벨 합성 |
| 2. 앞유리 탐지 기반 라벨 위치 자동 결정 | ✅ 완료 | 차량 앞유리(검은 영역) 탐지 → 그 위치를 포함하는 라벨 위치/크기 자동 계산, 안 보이면 스킵 |
| 3. 여러 슬롯 한번에 처리 | ✅ 완료 | 이미지 1장 넣으면 config의 모든 슬롯 자동 검사/라벨링 (`--auto-all`) |
| 4. 탐지 신뢰도 + 검수 플래그 | ✅ 완료 | blob 모양 기반 confidence score, 낮으면 `review`로 표시 |
| 5. 배치 처리기 + 검수/로그 | ✅ 완료 | 카메라 폴더 전체 일괄 처리, output/review 자동 분류 + JSON 로그 |
| 6. 슬롯 경계 자동 탐지 | ✅ 완료 | 카메라 원본 프레임 폴더만 주면 주차 슬롯 폴리곤 자체를 자동 생성 (`generate_config.py`), 사람이 좌표 입력 안 해도 됨 |
| 7. GUI (PySide6) | ⬜ 미착수 | 작업자용 데스크톱 앱 |

지금 할 수 있는 것: **카메라의 원본 프레임 폴더만 있으면 슬롯 좌표도 사람이
입력할 필요 없이** 자동으로 주차 슬롯 폴리곤을 찾아 config를 생성하고
(`generate_config.py`), 그 config로 슬롯 안 차량의 앞유리를 자동 탐지해서
라벨을 합성한 PNG를 만든다. 앞유리가 안 보이면(빈 슬롯이거나 이웃 차량에
가려짐) 자동으로 스킵. 한 이미지 안 슬롯 여러 개도 한번에 처리 가능
(`--auto-all`). **카메라 폴더 전체(이미지 수십~수백 장)를 한 번에 배치
처리해서 결과 자동 분류(성공 → output/, 검수 필요 → review/)까지 됨.**
GUI는 아직 없음.

원래 계획은 sub-project 4를 "label/no_label 분류기"(학습된 모델)로 잡았으나,
학습 데이터가 없어서(label/no_label 샘플이 실제로는 그런 용도가 아님, 앞선
브레인스토밍에서 확인됨) 대신 blob 모양 기반 휴리스틱 confidence score로
대체 — 옆으로 길쭉한(반사광/그림자일 가능성 높은) 탐지는 낮은 점수를 받아
자동으로 검수 대상이 됨.

원래 계획은 sub-project 2를 "차량 detection(YOLO)"으로 잡았으나, COCO
사전학습 YOLO가 이 카메라의 top-down 앵글에서 전혀 인식을 못해(spike로 확인)
"차량 앞유리 검은 영역 탐지"로 방향을 바꿈 — 실제 필요(라벨이 앞유리를
포함해야 함)에도 더 맞았음.

설계 배경과 판단 근거는 `docs/superpowers/specs/`, `docs/superpowers/plans/`에
있음 — 특히 이 카메라가 실측상 거의 180° FOV 어안렌즈라 슬롯별 로컬 gnomonic
재투영 방식을 쓴다는 배경은
`docs/superpowers/specs/2026-08-26-local-gnomonic-rectification-design.md`,
앞유리 탐지 기반 라벨 배치 배경은
`docs/superpowers/specs/2026-08-26-windshield-label-placement-design.md` 참고.

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 테스트 실행

```bash
.venv/bin/pytest -v
```

44개 테스트, 실제 샘플 CCTV 프레임(`no_label/`)을 사용하므로 이 저장소에
`no_label/` 폴더(gitignore됨, 원본 CCTV 데이터라 커밋 안 함)가 있어야 통과함.

## 슬롯 config 자동 생성 (사람이 좌표 입력 안 함)

카메라별 config(`config/<camera_id>.json`)는 이제 사람이 만들 필요 없다.
그 카메라의 원본 프레임이 모여 있는 폴더만 주면 자동으로 만들어진다:

```bash
.venv/bin/python src/generate_config.py \
  --camera-id P1_B1_1_9 \
  --frames-dir no_label/P1_B1_1_9 \
  --output config/P1_B1_1_9.json
```

내부 동작(`src/slot_detection.py`):

1. **median 스택** — 그 카메라의 모든 원본 프레임을 픽셀 단위 median으로
   합쳐 참조 이미지 하나를 만든다. 차량/사람/움직이는 반사광처럼 프레임마다
   달라지는 건 지워지고, 항상 그 자리에 있는 바닥 주차선만 선명하게 남는다.
2. **국소 왜곡보정 격자 + Hough** — 어안렌즈에서는 카메라 중심을 안 지나는
   실제 직선이 원본 픽셀에서 곡선으로 찍혀서 일반 Hough로는 못 잡는다.
   이미 있는 슬롯별 로컬 gnomonic 재투영(`LocalView`, 원래 라벨 렌더링용)을
   재사용해 바닥을 여러 개 겹치는 국소 패치로 나눠 각각 왜곡보정 후
   Canny+Hough로 직선을 찾고, 원본 좌표로 역변환해 모은다.
3. **연결영역 분리 + 필터** — 찾은 선들로 닫힌 영역을 분리하고, 면적/
   직사각형도(윤곽선 면적 대 최소외접사각형 면적 비)로 그럴듯한 슬롯
   크기만 남긴 뒤, 노란/주황 사선(주차금지 경고선) 색상 영역은 제외한다.
4. **인셋** — 2번 단계의 morphological closing이 경계를 실제 흰 선보다
   몇 픽셀 밖으로 부풀리므로, 폴리곤 각 꼭짓점을 중심 방향으로 살짝
   당겨서 렌더링된 라벨이 실제 흰 선 안에 들어오게 한다.

`--cx/--cy/--f/--radius`로 다른 카메라의 렌즈 파라미터도 넘길 수 있다
(기본값은 이 프로젝트 샘플 데이터의 실측값 `f≈204, radius≈320`).

**알려진 한계**: 슬롯 폴리곤 판정은 순수 기하/색상 특징(면적, 직사각형도,
색상)만 쓴다 — 바닥 위에 있는 다른 표시(차선 방향 화살표, 볼라드 경고
구역 등)는 모양이 슬롯과 비슷해 여전히 오탐될 수 있다. 이건 "이게 뭔지
이해해야 하는" 내용 판별 문제라 이 세션에서 시도한 5가지 이상의 기하/색상
조합(면적, 직사각형도, 텍스처 분산, 지면평면 재투영, BEV 재투영,
adaptive threshold)과 로컬 CLIP 제로샷 분류(도메인이 너무 특이해서 실패,
COCO 학습 YOLO가 이 카메라 각도에서 실패했던 것과 같은 이유)로도 완전히
못 없앴다. 그래서 `generate_config()`는 탐지된 슬롯 중 confidence가 낮은
게 하나라도 있거나 슬롯이 0개면 `needs_review=True`를 반환한다 — 슬롯이
없어야 정상인 차선/교차로 전용 카메라를 포함해, 생성된 config는 실제
배치 처리 전에 한 번 훑어보는 걸 권장한다(카메라당 1회, 슬롯 좌표를
찍는 게 아니라 이미 그려진 결과를 확인만 하는 수준).

## 현재 파이프라인 실행 (단일 이미지)

`config/<camera_id>.json`은 위 자동 생성 스크립트로 만든다. 예시
(`config/P1_B1_1_9.json`, 실제로 자동 생성/실행 검증된 값):

```json
{
  "camera_id": "P1_B1_1_9",
  "image_width": 640,
  "image_height": 640,
  "calibration": { "model": "equidistant_1param", "center": [320.0, 320.0], "f": 204.0, "radius": 320.0 },
  "slots": [
    { "id": "slot-0", "polygon_raw": [[238.4, 25.6], [321.3, 20.2], [318.9, 130.5], [242.1, 135.8]] }
  ],
  "label_spec": { "shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": null, "border_width": 3 }
}
```

- `calibration.f`: 이 카메라(640x640, 거의 180° FOV)는 실측상 `f≈204`가 맞음 —
  다른 카메라라면 `src/calibration.py`의 `fit()`으로 raw 이미지에서 실제 직선
  구조를 클릭해 추정 필요 (아직 CLI/GUI 노출 안 됨, 코드에서 직접 호출).
- `slots[].polygon_raw`: 위 `generate_config.py`가 자동으로 채운다.
- `label_spec`: 라벨은 채워진 도형이 아니라 **테두리만**(`cv2.polylines`) 그림 —
  `color`는 BGR 순서(OpenCV 관례), `border_width`는 선 두께(생략 시 3px).

실행:

```bash
.venv/bin/python src/main.py \
  --config config/P1_B1_1_9.json \
  --image no_label/P1_B1_1_9/20260820_115033.jpg \
  --slot-id A \
  --candidate-u 0.5 \
  --candidate-v 0.5 \
  --output output/demo.png
```

`--candidate-u`/`--candidate-v`는 슬롯 polygon 내부 정규화 좌표(0~1, 라벨
중심 위치). `output/demo.png`에 라벨이 합성된 최종 이미지가 생성됨 —
슬롯 바깥 영역은 원본과 픽셀 단위로 동일.

### 자동 모드 (`--auto`)

후보 위치를 직접 안 정해도, 슬롯 안 차량의 앞유리를 자동 탐지해서 라벨
위치/크기를 계산한다:

```bash
.venv/bin/python src/main.py \
  --config config/P1_B1_1_9.json \
  --image no_label/P1_B1_1_9/20260820_115033.jpg \
  --slot-id A \
  --auto \
  --output output/demo_auto.png
```

앞유리가 탐지 안 되면(빈 슬롯이거나 이웃 차량에 가려짐) 파일을 안 만들고
`slot 'A': no visible windshield, skipped (no output written)`만 출력함 —
에러 아니라 정상 결과.

### 여러 슬롯 한번에 (`--auto-all`)

`--slot-id` 없이 config의 모든 슬롯을 한번에 검사해서, 앞유리 보이는 슬롯은
전부 라벨링하고 한 장의 출력 이미지에 합성한다:

```bash
.venv/bin/python src/main.py \
  --config config/P1_B1_1_21.json \
  --image no_label/P1_B1_1_21/20260820_115029.jpg \
  --auto-all \
  --output output/demo_all.png
```

슬롯별 결과(`labeled (confidence=0.91)`/`skipped`/`review (confidence=0.25)`/
`error: ...`)를 슬롯마다 한 줄씩 출력함. confidence는 탐지된 앞유리 blob의
모양(길쭉하지 않고 compact할수록 높음) 기반 휴리스틱 점수 — 0.5 미만이면
`review`로 표시됨(라벨은 일단 그려지지만 검수 대상 표시). 한 슬롯 처리 중
에러(예: 슬롯 polygon이 너무 커서 로컬 patch 범위 벗어남)가 나도 다른 슬롯
처리는 안 멈춤 — 앞유리 탐지는 원본 raw 프레임에서 한 번만 수행(먼저
라벨링된 슬롯이 나중 슬롯의 탐지 결과를 오염시키지 않도록).

### 카메라 폴더 전체 배치 처리

카메라 하나의 raw 프레임이 담긴 폴더 전체를 한 번에 처리한다. 슬롯 중
하나라도 `review`나 `error` 상태면 그 이미지 전체가 `review-dir`로, 아니면
`output-dir`로 감:

```bash
.venv/bin/python src/batch_processor.py \
  --config config/P1_B1_1_9.json \
  --input-dir no_label/P1_B1_1_9 \
  --output-dir output/P1_B1_1_9 \
  --review-dir review/P1_B1_1_9 \
  --log output/P1_B1_1_9/log.json
```

`log.json`에 이미지별 상태(`success`/`review`/`error`)와 슬롯별 상세 결과가
전부 기록됨. 실제 카메라 P1_B1_1_9(57장)로 검증: 55장 성공, 2장 검수 대상,
0장 에러.

## 프로젝트 구조

```
src/
  calibration.py      어안렌즈 보정 (전역 2D 모델 + 슬롯별 로컬 gnomonic 뷰)
  slot_detection.py    median 스택 + 슬롯 폴리곤 자동 탐지 + 탐지기 앙상블 병합
  yolo_slot_detector.py YOLOv8-seg 체크포인트 로드/추론 (classical CV와 앙상블)
  generate_config.py   카메라 프레임 폴더 -> config JSON 자동 생성 CLI
                       (거부 이력 지역 자동 억제, 탐지기 합의 자동 승인 포함)
  review_store.py      승인/거부/미탐/웹플래그 JSONL 저장소 (최신 결정 우선)
  review_server.py     후보 승인/거부 리뷰 웹 UI (stdlib http.server, a/r 단축키)
  train_from_reviews.py 리뷰 라벨로 slot_classifier 재학습
  export_yolo_dataset.py 리뷰 라벨 -> YOLO-seg 학습 데이터셋 export
  train_yolo_seg.py    YOLOv8-seg 파인튜닝 CLI (MPS/CUDA 자동 선택)
  parking_slot.py  카메라별 슬롯 config 로드/저장
  perspective.py   정규화 좌표 <-> 픽셀 homography (양방향)
  renderer.py      라벨 도형 직접 그리기 (PNG 에셋 없음)
  windshield.py       차량 앞유리(검은 영역) 탐지 + confidence score
  candidate.py        탐지된 앞유리 -> 슬롯 배정 + 라벨 위치/크기 계산 (슬롯 경계 clamp)
  pipeline.py         오케스트레이션 (run() 수동 / run_auto() 슬롯1개 자동 / run_auto_all() 전체 슬롯 자동)
  main.py             CLI 진입점 (--candidate-u/-v 수동, --auto, --auto-all)
  batch_processor.py  카메라 폴더 전체 일괄 처리, output/review 분류 + JSON 로그
web/
  backend/           Flask API (업로드 -> 탐지 -> 라벨링 -> 결과/수정/다운로드)
  frontend/          React (Vite) — 업로드/진행/결과 페이지, 슬롯 추가·조정·삭제
models/            학습된 체크포인트 — 실제 운영에 쓰는 3개만 git에 커밋됨
                   (slot_classifier.joblib, yolov8_seg_slots_production.pt,
                   yolov8_pose_marking_points.pt/실험용). retrain_yolo.py가
                   만드는 후보/구버전 스냅샷(v2~v8, archive/)은 gitignore 유지.
tests/             pytest, 실제 샘플 이미지 기반
config/            카메라별 config JSON (자동 생성 + 웹/리뷰로 다듬음)
review/            리뷰 로그 (labels/candidates/missed/web_flags .jsonl + crops/)
docs/superpowers/  설계 문서(spec)와 구현 계획(plan) 이력
label/, no_label/  원본 CCTV 참고 자료 (gitignore, 커밋 안 함)
output/            배치 처리 결과 (gitignore)
```

웹앱 실행: `./run_web.sh` → `http://localhost:5050` (사진 업로드 → 자동
탐지/라벨링 → 결과 확인/수정 → zip 다운로드). 결과 페이지에서 ×는 해당
사진에서만 가림, Shift+×는 모든 사진에서 삭제 + 재탐지 억제 기록, 슬롯
추가는 승인 기록으로 남아 재학습 데이터가 됨. 위치 조정 드래그를 Shift 누른
채 놓으면 그 위치가 슬롯 기본값으로 저장되어 모든 사진/배치에 적용됨.

## 재학습 (수정 기록 100건쯤 쌓이면)

웹에서 한 수정(Shift+×, 슬롯 추가, 승인/거부)이 전부 학습 데이터로
쌓이므로, 주기적으로 이거 한 줄만 실행하면 모델이 좋아짐:

```bash
.venv/bin/python src/retrain_yolo.py
```

자동으로 데이터셋 추출 → 현재 운영 모델에서 파인튜닝(약 1시간) → 성능
자동 비교 → **더 좋을 때만** 운영 모델(`models/yolov8_seg_slots_production.pt`)
교체. 지든 이기든 모든 체크포인트는 `models/archive/`에 보관(롤백 가능).
끝나면 `./run_web.sh`로 웹앱 재시작만 하면 적용. 자세한 건
`docs/COMMANDS.md` 7-4절.

## 알려진 한계

- 슬롯 좌표는 자동 생성됨(`generate_config.py` + YOLOv8-seg/classical CV
  앙상블) — 단, 천장 구조물/녹색 바닥 도색/바닥 화살표 같은 비슬롯 표시를
  여전히 오탐할 수 있어 사람 검수(웹 결과 페이지 또는 review_server)가
  전제임. 한번 거부한 위치는 재탐지에서 자동으로 걸러짐.
- 카메라별 `f`/`radius`는 프레임 크기에서 자동 유도 (640x640 실측값을
  비례 스케일) — 렌즈가 크게 다른 카메라는 `--cx/--cy/--f/--radius` 수동
  지정 필요.
- 앞유리 탐지는 실제 프레임 1장 기준으로 튜닝한 고정 threshold — 같은 프레임에서
  차량 1대에 배경 노이즈(바닥 반사광 줄무늬, 그림자 등) blob이 18개까지 잡힘.
  길쭉한 반사광 줄무늬는 aspect-ratio 필터(2.5:1 초과 제외)로 6개 걸러짐(18→12),
  나머지는 슬롯 안에서 가장 큰 blob 선택 + confidence score 기반 review 플래그로
  방어. 조명/차종 다양한 데이터 쌓이면 적응형 threshold로 교체 필요.
- confidence score는 학습된 분류기가 아니라 blob 모양(aspect ratio) 기반
  휴리스틱 — label/no_label 학습 데이터가 없어서 실제 분류 모델은 못 만듦.
- 고정 크기 로컬 패치(`pipeline.py`의 `DEFAULT_PATCH_SIZE`/`DEFAULT_LOCAL_F`)가
  슬롯 크기 상한을 만듦 — 이 카메라 기준 raw 픽셀로 약 142x142 정도. 실제
  주차 슬롯이 이보다 크면 `DEFAULT_LOCAL_F`를 낮춰야 함.
- 웹앱이 여러 카메라 폴더 업로드를 한번에 처리함 (폴더별로 카메라 분리,
  백그라운드 잡 4개 병렬). CLI 쪽은 여전히 카메라 1개씩.
