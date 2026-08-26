# CCTV 주차공간 라벨링 자동화

저장된 CCTV(어안렌즈) 이미지에 주차공간별 라벨을 자동으로 배치하고 PNG로
저장하는 이미지 라벨링 자동화 프로그램. 전체 목표와 요구사항은 서브프로젝트로
쪼개서 진행 중.

## 진행 상태

| 서브프로젝트 | 상태 | 설명 |
|---|---|---|
| 1. 핵심 기하/렌더링 파이프라인 | ✅ 완료 | 어안렌즈 왜곡 보정 + 슬롯별 로컬 재투영 + 라벨 합성 |
| 2. 차량 detection 연동 | ⬜ 미착수 | YOLO 등으로 차량 인식 |
| 3. 후보 생성 + 스코어링 | ⬜ 미착수 | 라벨 위치 자동 탐색, 가림 위험 판정 |
| 4. label/no_label 분류기 | ⬜ 미착수 | 좋은/나쁜 위치 판별 모델 |
| 5. 배치 처리기 + 검수/로그 | ⬜ 미착수 | 대량 이미지 일괄 처리, output/review 분류 |
| 6. GUI (PySide6) | ⬜ 미착수 | 작업자용 데스크톱 앱 |

지금 할 수 있는 것: **카메라 config + 슬롯 좌표 + 후보 위치를 사람이 직접
지정하면, 이미지 1장에 라벨 1개를 올바르게 합성한 PNG를 만든다.** 차량 인식이나
자동 위치 선정, 대량 처리는 아직 없음.

설계 배경과 판단 근거는 `docs/superpowers/specs/`, `docs/superpowers/plans/`에
있음 — 특히 이 카메라가 실측상 거의 180° FOV 어안렌즈라 슬롯별 로컬 gnomonic
재투영 방식을 쓴다는 배경은
`docs/superpowers/specs/2026-08-26-local-gnomonic-rectification-design.md` 참고.

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 테스트 실행

```bash
.venv/bin/pytest -v
```

25개 테스트, 실제 샘플 CCTV 프레임(`no_label/`)을 사용하므로 이 저장소에
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
  "label_spec": { "shape": "rect", "width": 0.6, "height": 0.6, "color": [0, 200, 90], "alpha": 0.75, "text": null }
}
```

- `calibration.f`: 이 카메라(640x640, 거의 180° FOV)는 실측상 `f≈204`가 맞음 —
  다른 카메라라면 `src/calibration.py`의 `fit()`으로 raw 이미지에서 실제 직선
  구조를 클릭해 추정 필요 (아직 CLI/GUI 노출 안 됨, 코드에서 직접 호출).
- `slots[].polygon_raw`: 사람이 raw 이미지에서 슬롯 모서리 4점을 클릭한 픽셀
  좌표 그대로. 지금은 이미지 뷰어로 좌표를 직접 확인해서 손으로 넣어야 함
  (GUI 미구현).

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

## 프로젝트 구조

```
src/
  calibration.py   어안렌즈 보정 (전역 2D 모델 + 슬롯별 로컬 gnomonic 뷰)
  parking_slot.py  카메라별 슬롯 config 로드/저장
  perspective.py   정규화 좌표 <-> 픽셀 homography
  renderer.py      라벨 도형 직접 그리기 (PNG 에셋 없음)
  pipeline.py       전체 오케스트레이션 (raw -> 합성 -> raw 로 재배치)
  main.py          CLI 진입점
tests/             pytest, 실제 샘플 이미지 기반
config/            카메라별 config JSON (직접 작성)
docs/superpowers/  설계 문서(spec)와 구현 계획(plan) 이력
label/, no_label/  원본 CCTV 참고 자료 (gitignore, 커밋 안 함)
output/            CLI 실행 결과 (gitignore)
```

## 알려진 한계

- 슬롯 좌표를 사람이 직접 raw 이미지에서 읽어 넣어야 함 (GUI 없음).
- 카메라별 `f` 값도 수동 추정/입력 필요 (`fit()` 함수는 있으나 CLI 미노출).
- 차량 인식/가림 판정/자동 위치선정/대량 처리 전부 없음 — 이미지·슬롯·후보위치
  1개씩만 처리 가능.
