# CCTV 주차공간 라벨링 자동화

저장된 CCTV(어안렌즈) 이미지에 주차공간별 라벨을 자동으로 배치하고 PNG로
저장하는 이미지 라벨링 자동화 프로그램. 전체 목표와 요구사항은 서브프로젝트로
쪼개서 진행 중.

## 진행 상태

| 서브프로젝트 | 상태 | 설명 |
|---|---|---|
| 1. 핵심 기하/렌더링 파이프라인 | ✅ 완료 | 어안렌즈 왜곡 보정 + 슬롯별 로컬 재투영 + 라벨 합성 |
| 2. 앞유리 탐지 기반 라벨 위치 자동 결정 | ✅ 완료 | 차량 앞유리(검은 영역) 탐지 → 그 위치를 포함하는 라벨 위치/크기 자동 계산, 안 보이면 스킵 |
| 3. 후보 생성 + 스코어링 | ⬜ 미착수 | 여러 이미지 통합 안정 위치 등 고도화 (옆 슬롯 침범 방지는 sub-project 2에서 처리됨) |
| 4. label/no_label 분류기 | ⬜ 미착수 | 좋은/나쁜 위치 판별 모델 |
| 5. 배치 처리기 + 검수/로그 | ⬜ 미착수 | 대량 이미지 일괄 처리, output/review 분류 |
| 6. GUI (PySide6) | ⬜ 미착수 | 작업자용 데스크톱 앱 |

지금 할 수 있는 것: 카메라 config + 슬롯 좌표만 있으면, **라벨 위치를 사람이
지정하지 않아도** 슬롯 안 차량의 앞유리를 자동 탐지해서 그 위치를 포함하는
라벨을 합성한 PNG를 만든다(`--auto`). 앞유리가 안 보이면(빈 슬롯이거나 이웃
차량에 가려짐) 자동으로 스킵. 수동 위치 지정(`--candidate-u`/`-v`)도 계속
가능. 여러 이미지 일괄 처리, GUI는 아직 없음 — 한 번에 이미지 1장·슬롯 1개.

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

37개 테스트, 실제 샘플 CCTV 프레임(`no_label/`)을 사용하므로 이 저장소에
`no_label/` 폴더(gitignore됨, 원본 CCTV 데이터라 커밋 안 함)가 있어야 통과함.

## 현재 파이프라인 실행 (단일 이미지)

카메라 config가 없으면 먼저 `config/<camera_id>.json` 형태로 하나 만든다.
예시(`config/P1_B1_1_9.json`, 실제로 실행 검증된 값):

```json
{
  "camera_id": "P1_B1_1_9",
  "image_width": 640,
  "image_height": 640,
  "calibration": { "model": "equidistant_1param", "center": [320.0, 320.0], "f": 204.0, "radius": 320.0 },
  "slots": [
    { "id": "A", "polygon_raw": [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]] }
  ],
  "label_spec": { "shape": "rect", "width": 0.6, "height": 0.6, "color": [235, 206, 135], "alpha": 1.0, "text": null, "border_width": 3 }
}
```

- `calibration.f`: 이 카메라(640x640, 거의 180° FOV)는 실측상 `f≈204`가 맞음 —
  다른 카메라라면 `src/calibration.py`의 `fit()`으로 raw 이미지에서 실제 직선
  구조를 클릭해 추정 필요 (아직 CLI/GUI 노출 안 됨, 코드에서 직접 호출).
- `slots[].polygon_raw`: 사람이 raw 이미지에서 슬롯 모서리 4점을 클릭한 픽셀
  좌표 그대로. 지금은 이미지 뷰어로 좌표를 직접 확인해서 손으로 넣어야 함
  (GUI 미구현).
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

## 프로젝트 구조

```
src/
  calibration.py   어안렌즈 보정 (전역 2D 모델 + 슬롯별 로컬 gnomonic 뷰)
  parking_slot.py  카메라별 슬롯 config 로드/저장
  perspective.py   정규화 좌표 <-> 픽셀 homography (양방향)
  renderer.py      라벨 도형 직접 그리기 (PNG 에셋 없음)
  windshield.py    차량 앞유리(검은 영역) 탐지
  candidate.py     탐지된 앞유리 -> 슬롯 배정 + 라벨 위치/크기 계산
  pipeline.py      전체 오케스트레이션 (수동 run() / 자동 run_auto())
  main.py          CLI 진입점 (--candidate-u/-v 수동, --auto 자동)
tests/             pytest, 실제 샘플 이미지 기반
config/            카메라별 config JSON (직접 작성)
docs/superpowers/  설계 문서(spec)와 구현 계획(plan) 이력
label/, no_label/  원본 CCTV 참고 자료 (gitignore, 커밋 안 함)
output/            CLI 실행 결과 (gitignore)
```

## 알려진 한계

- 슬롯 좌표를 사람이 직접 raw 이미지에서 읽어 넣어야 함 (GUI 없음).
- 카메라별 `f` 값도 수동 추정/입력 필요 (`fit()` 함수는 있으나 CLI 미노출).
- 앞유리 탐지는 실제 프레임 1장 기준으로 튜닝한 고정 threshold — 같은 프레임에서
  차량 1대에 배경 노이즈(바닥 반사광 줄무늬, 그림자 등) blob이 18개까지 잡힘.
  길쭉한 반사광 줄무늬는 aspect-ratio 필터(2.5:1 초과 제외)로 6개 걸러짐(18→12),
  나머지는 슬롯 안에서 가장 큰 blob을 고르는 방식으로 방어. 조명/차종 다양한
  데이터 쌓이면 적응형 threshold로 교체 필요.
- 고정 크기 로컬 패치(`pipeline.py`의 `DEFAULT_PATCH_SIZE`/`DEFAULT_LOCAL_F`)가
  슬롯 크기 상한을 만듦 — 이 카메라 기준 raw 픽셀로 약 142x142 정도. 실제
  주차 슬롯이 이보다 크면 `DEFAULT_LOCAL_F`를 낮춰야 함.
- 여러 이미지 일괄 처리, GUI 전부 없음 — 이미지·슬롯 1개씩만 처리 가능.
