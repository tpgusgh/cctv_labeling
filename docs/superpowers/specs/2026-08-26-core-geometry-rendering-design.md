# 서브프로젝트 1: 핵심 기하/렌더링 파이프라인 — 설계

## 상태
구현 계획 단계 진행 승인됨.

## 배경

전체 시스템 목표(원본 요구사항 참고, 여기 재수록 안함): CCTV 주차공간 이미지에
가림 위험을 고려한 라벨을 배치하고, 렌즈/원근 왜곡을 반영해서 일괄 처리하는 것.
전체 시스템은 스펙 하나로 다루기엔 너무 커서 서브프로젝트로 분해, 다음 순서로
빌드:

1. **핵심 기하/렌더링 파이프라인** (본 문서)
2. 차량 detection 연동
3. 후보 생성 + 스코어링 엔진
4. label/no_label 분류기
5. 배치 처리기 + 검수/로그 파이프라인
6. GUI (PySide6)

본 서브프로젝트는 기하학적 핵심만 증명: raw CCTV 프레임 1장 + 카메라 config +
주차슬롯 polygon + 고정 라벨 위치가 주어졌을 때, 올바르게 왜곡 보정된 합성 PNG
1장을 만들어내는 것. ML 없음, 배치 없음, 후보 스코어링 없음.

### 실제 프로젝트 데이터에서 확인한 사실

요구사항 문서의 가정과 실제 샘플 데이터가 달랐음. `label/`, `no_label/`
폴더에서 확인한 실제 내용:

- 카메라는 **원형 어안렌즈(fisheye)** — 천장에 달린 기계식 주차타워 카메라이지
  일반적인 비스듬한 각도의 CCTV 아님. raw 프레임은 640x640 JPG, 원형 유효
  영역 바깥은 검은색.
- `no_label/<camera_id>/*.jpg` — 폴더 23개, 폴더 하나 = **카메라 하나**(주차슬롯
  하나 아님). 폴더명(`P1_B1_1_1` 등)은 카메라ID. 카메라 하나의 fisheye 화면
  안에 주차슬롯 여러 개가 동시에 들어감(참고 스크린샷 하나에 `parkingLocations-*`
  polygon이 8개 찍혀있는 것으로 확인). 폴더 안 파일들은 그 카메라가 시간대별로
  찍은 raw 프레임 — 이건 배치 처리 대상 데이터셋이지, good/bad 학습 예시 쌍이
  아님.
- `label/*.png` — 무관한 외부 툴에서 캡처한 스크린샷 18장. 폴리곤 도형은
  "각도가 애매한" 슬롯 인식 엣지케이스 참고자료일 뿐; 텍스트 오버레이
  (`parkingLocations-978: 0.714`)는 우리 출력 형식과 무관, 무시함.
- 라벨 그래픽 에셋(PNG 템플릿) 존재 안함, 제공 예정도 없음. 시스템이 라벨
  도형을 직접 그림(아래 Renderer 참고) — 외부 이미지 파일 합성 아님.
- 최종 출력 포맷은 PNG, 원본 spec대로(배치 처리 결과 `output/`에 저장).

## 접근 방식

**정류(rectify) → 합성 → 재왜곡.** raw fisheye 프레임을 먼저 평평한 rectified
이미지로 undistort. 원근/라벨배치 계산은 전부 그 평면 공간에서 수행(일반
사각형/homography 기하). 라벨은 거기서 합성. 그 다음 합성된 rectified
이미지를 원본 fisheye 픽셀 배치로 다시 forward-warp(재왜곡)해서 최종 출력.

두 가지 대안 대비 이 방식을 선택함:
- 라벨 4개 모서리만 homography+왜곡으로 변환(참고 외부 툴이 이렇게 하는
  것으로 보임, `label/` 스크린샷의 폴리곤이 대체로 직선 변임) — 더 간단하지만,
  실제 fisheye 매핑이면 곡선이어야 할 라벨 변이 특히 화면 주변부에서 직선으로
  나와 부정확.
- 라벨 변마다 경계점을 다수 샘플링해서 전체 왜곡모델로 각각 매핑, rectified
  전체 이미지는 만들지 않는 방식 — 정확하지만 왜곡 계산이 렌더러 호출마다
  들어가버림(이미지 단위 연산 2번이 아니라).

Rectify-first 방식이 나은 이유: 원근왜곡과 렌즈왜곡을 **별도 단계**로 처리하라는
요구사항(undistort/재왜곡 = 렌즈 단계, homography = 원근 단계)을 OpenCV
이미지 단위 연산 2개로 정확히 충족하고, 이후 모든 서브프로젝트(가림 판정,
후보 생성)가 왜곡된 fisheye 공간이 아니라 일반 사각형 공간에서 작업 가능해짐.

## 컴포넌트

전부 `src/` 아래.

### `calibration.py`
카메라별 fisheye 왜곡 모델.

- MVP 모델: 단일 파라미터 equidistant radial 모델, 중심 = 이미지 중심(원형
  fisheye crop이라 근거 있음), 반지름 = 검출된 원 경계. `ponytail: 단일
  스칼라 왜곡계수, 사람이 raw 프레임에서 실제 직선(예: 주차선) 위 점들을
  클릭하면 undistort 후 일직선이 되도록 최소제곱으로 피팅 — 체커보드 촬영
  데이터가 생기면 cv2.fisheye.calibrate() 정식 캘리브레이션으로 업그레이드.`
- `fit(raw_image, clicked_line_points) -> CalibrationModel`
- `CalibrationModel.undistort_image(raw) -> rectified`
- `CalibrationModel.undistort_points(pts) -> pts` (raw px → rectified px)
- `CalibrationModel.distort_points(pts) -> pts` (rectified px → raw px,
  최종 재왜곡 remap에 사용)
- 모델은 카메라별 JSON으로 저장(`config/<camera_id>.json`, `calibration` 키).

### `parking_slot.py`
카메라별 주차슬롯 polygon config.

- 사람이 **raw** 프레임(실제로 보는 화면) 위에서 슬롯 모서리 클릭.
  `undistort_points`로 변환해서 **rectified 픽셀 공간** 기준 polygon으로
  저장 — 이후 모든 기하 계산은 rectified 공간만 사용.
- `config/<camera_id>.json` 로드/저장(`slots: [{id, polygon_rectified}]`).

### `perspective.py`
정규화 주차면 좌표(`(0,0)`~`(1,1)`)와 슬롯의 rectified-픽셀 사변형 사이
homography.

- `plane_to_pixel(slot_polygon_rectified) -> homography`
  (`cv2.getPerspectiveTransform` 사용).
- 라벨의 정규화 공간 도형을 rectified 픽셀 좌표로 배치하는 데 사용.

### `renderer.py`
라벨을 직접 그림 — 외부 이미지 에셋 없음.

- 라벨 spec(config에서): 도형(`rect` | `rounded_rect`), 정규화 주차면 단위
  크기, 채우기 색상, 투명도, 테두리, 선택적 텍스트.
- `render_label(rectified_image, homography, candidate_point, label_spec) ->
  rectified_image_with_label` — 정규화 공간에서 라벨 모서리점 계산 →
  homography로 rectified 픽셀 매핑 → `cv2.fillPoly`/`cv2.polylines`로 그리기
  → rectified 이미지에 alpha 합성.

### 재왜곡 단계
`CalibrationModel.distort_points`로 remap 그리드 생성(`cv2.remap`, 역방향
매핑)해서 합성된 rectified 이미지를 원본 fisheye 배치로 되돌림. 출력 크기 =
입력 raw 프레임과 동일.

### `main.py` (본 서브프로젝트 진입점)
CLI 테스트 하네스: 카메라 config + raw 프레임 1장 + 후보점 1개 로드, 전체
파이프라인 실행, 출력 PNG 저장. 본 서브프로젝트 수동 검증용 스캐폴딩일 뿐 —
실제 배치 진입점은 서브프로젝트 5에서 구현.

## 데이터 흐름

```
raw.jpg (fisheye, no_label/<camera_id>/ 에서)
  -> CalibrationModel.undistort_image
rectified.png (평면, 디버그용 중간 산출물)
  -> perspective.plane_to_pixel(슬롯 polygon)
  -> renderer.render_label(후보점, 라벨 spec)
composited_rectified.png
  -> CalibrationModel.distort_points 기반 remap
final.png (raw.jpg와 크기 동일)
```

## Config 스키마 (카메라별, `config/<camera_id>.json`)

```json
{
  "camera_id": "P1_B1_1_1",
  "image_width": 640,
  "image_height": 640,
  "calibration": { "model": "equidistant_1param", "center": [320, 320], "radius": 310, "k": 0.42 },
  "slots": [
    { "id": "P1_B1_1_1-A", "polygon_rectified": [[100,300],[350,280],[380,650],[80,680]] }
  ],
  "label_spec": { "shape": "rounded_rect", "width": 0.6, "height": 0.25, "color": [30,180,90], "alpha": 0.75, "text": null }
}
```

## 에러 처리

이 단계엔 아직 검수/로그 파이프라인 없음(서브프로젝트 5에서 추가). 본
서브프로젝트에서는: calibration 없음, 슬롯 config 없음, 후보점이 rectified
이미지 범위 밖 — 이 경우 명확한 메시지와 함께 예외 발생. 삼키지 않고
기본값으로 넘기지 않음.

## 테스트

Ponytail 규칙: 비trivial 로직은 테스트 프레임워크 대신 실행 가능한 self-check
1개.

- `calibration.py`: `if __name__ == "__main__"` 아래 `demo()` — 이미지 전역
  샘플점 여러 개로 `distort_points(undistort_points(p)) ≈ p`(라운드트립
  항등성) assert.
- `renderer.py`/파이프라인: `test_pipeline.py` 1개 — `no_label/`의 실제
  샘플 프레임 1장으로 raw→final 전체 파이프라인 실행, (a) 출력 크기 =
  입력 크기 (b) 라벨의 raw-공간 bounding box 내부 픽셀이 원본과 달라짐(실제로
  뭔가 그려졌는지) (c) 슬롯 polygon 바깥 먼 픽셀은 그대로(누출 없음) assert.

## 범위 밖 (이후 서브프로젝트)

차량 detection, 가림 스코어링, 후보 생성, label/no_label 분류기, 배치 처리,
검수 분류, 로깅, GUI. 본 서브프로젝트는 단일 이미지 기하/렌더링 정확성만
증명.
