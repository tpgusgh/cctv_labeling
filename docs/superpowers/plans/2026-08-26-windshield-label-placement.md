# 앞유리 탐지 기반 라벨 위치 자동 결정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사람이 후보 위치를 직접 지정하지 않아도, 슬롯 안 차량의 앞유리(검은
영역)를 자동 탐지해서 그 위치를 포함하는 라벨 위치/크기를 계산하고 합성한다.
앞유리가 안 보이면(빈 슬롯이거나 이웃 차량에 가려짐) 그 슬롯은 라벨을 붙이지
않고 건너뛴다.

**Architecture:** 원형 바닥 마스크 + 어두운 영역 threshold로 앞유리 후보
blob을 raw 픽셀 공간에서 탐지 → blob 중심점이 슬롯 polygon 내부에 있는지로
슬롯 배정 → 배정된 blob의 bbox를 서브프로젝트 1의 `LocalView`/homography로
정규화 슬롯 평면 좌표로 변환해 라벨 후보 위치/크기 계산 → 기존
`render_label`/합성 파이프라인 그대로 재사용.

**Tech Stack:** Python 3, OpenCV, NumPy, pytest (기존 venv 재사용).

**Spec:** [docs/superpowers/specs/2026-08-26-windshield-label-placement-design.md](../specs/2026-08-26-windshield-label-placement-design.md)

## Global Constraints

- `renderer.py`는 수정하지 않는다 — 기존 `render_label` 시그니처 그대로
  사용(단, `label_spec`의 `width`/`height` 값은 호출 전에 앞유리 기반으로
  계산해 넣는다).
- "앞유리 안 보임(슬롯 배정 실패)"은 예외가 아니라 `None` 반환으로 표현한다
  — 삼키는 게 아니라 이 서브프로젝트가 의도적으로 다루는 정상 결과.
- 그 외 에러(슬롯 없음, 이미지 못 읽음, 출력 파일 쓰기 실패, 후보점 범위
  밖 등)는 기존과 동일하게 명확한 예외.
- `DARK_THRESHOLD`/`MIN_BLOB_AREA`/`MAX_BLOB_AREA`/`coverage_margin` 등은
  고정 상수로 시작한다 (ponytail: 적응형 threshold는 범위 밖, 향후 과제).
- 테스트는 무거운 프레임워크나 매트릭스가 아니라, 핵심 동작 하나씩만 pytest로
  검증하는 가벼운 self-check로 유지한다.

---

## Task 1: `perspective.py`에 역방향 매핑 추가

**Files:**
- Modify: `src/perspective.py`
- Modify: `tests/test_perspective.py`

**Interfaces:**
- Produces: `pixel_to_plane_points(homography, pixel_points) -> np.ndarray[N,2]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_perspective.py`에 추가:
```python
def test_pixel_to_plane_points_roundtrips_with_plane_points_to_pixel():
    quad = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    homography = plane_to_pixel_homography(quad)
    normalized = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.3, 0.7]]

    pixels = plane_points_to_pixel(homography, normalized)
    roundtripped = pixel_to_plane_points(homography, pixels)

    np.testing.assert_allclose(roundtripped, np.asarray(normalized), atol=1e-4)
```

(파일 상단에 `import numpy as np`가 이미 있는지 확인 — 없으면 추가.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_perspective.py -v`
Expected: FAIL — `ImportError: cannot import name 'pixel_to_plane_points'`

- [ ] **Step 3: 구현**

`src/perspective.py`에 추가:
```python
def pixel_to_plane_points(homography, pixel_points):
    pts = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    inverse_homography = np.linalg.inv(homography)
    out = cv2.perspectiveTransform(pts, inverse_homography)
    return out.reshape(-1, 2)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_perspective.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/perspective.py tests/test_perspective.py
git commit -m "feat: add pixel-to-plane inverse homography mapping"
```

---

## Task 2: `windshield.py` — 앞유리(어두운 영역) 탐지

**Files:**
- Create: `src/windshield.py`
- Create: `tests/test_windshield.py`

**Interfaces:**
- Produces: `WindshieldBlob` (dataclass: `contour`, `bbox`, `centroid`, `area`),
  `detect_windshields(raw_image, calibration) -> list[WindshieldBlob]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_windshield.py`:
```python
import numpy as np
import cv2

from calibration import CalibrationModel
from windshield import detect_windshields

CAR_SAMPLE_IMAGE = "no_label/P1_B1_1_21/20260820_115029.jpg"


def test_detect_windshields_finds_at_least_one_blob_in_real_car_frame():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    raw = cv2.imread(CAR_SAMPLE_IMAGE)
    assert raw is not None

    blobs = detect_windshields(raw, calibration)

    assert len(blobs) >= 1


def test_detect_windshields_masks_out_background_outside_floor_circle():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=200.0)
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (300, 300), (340, 340), (10, 10, 10), -1)  # inside circle
    cv2.rectangle(img, (10, 10), (50, 50), (10, 10, 10), -1)      # outside circle

    blobs = detect_windshields(img, calibration)

    centroids = [b.centroid for b in blobs]
    assert any(280 < cx < 360 and 280 < cy < 360 for cx, cy in centroids)
    assert not any(0 < cx < 60 and 0 < cy < 60 for cx, cy in centroids)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_windshield.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'windshield'`

- [ ] **Step 3: 구현**

`src/windshield.py`:
```python
from dataclasses import dataclass

import numpy as np
import cv2

DARK_THRESHOLD = 60
MIN_BLOB_AREA = 150
MAX_BLOB_AREA = 8000


@dataclass
class WindshieldBlob:
    contour: object
    bbox: tuple
    centroid: tuple
    area: float


def detect_windshields(raw_image, calibration):
    h, w = raw_image.shape[:2]
    radius = calibration.radius
    if radius is None:
        radius = min(calibration.cx, calibration.cy, w - calibration.cx, h - calibration.cy)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(calibration.cx)), int(round(calibration.cy))), int(round(radius)), 255, thickness=-1)

    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            x, y, bw, bh = cv2.boundingRect(c)
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx_b = m["m10"] / m["m00"]
            cy_b = m["m01"] / m["m00"]
            blobs.append(WindshieldBlob(contour=c, bbox=(x, y, bw, bh), centroid=(cx_b, cy_b), area=area))
    return blobs
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_windshield.py -v`
Expected: PASS. 만약 `test_detect_windshields_finds_at_least_one_blob_in_real_car_frame`가
실패하면 `DARK_THRESHOLD`/`MIN_BLOB_AREA`/`MAX_BLOB_AREA` 상수를 조정해서
다시 시도 — 실제 이미지 기준 값이므로 여기서 조정하는 것이 맞고, 조정한
값과 이유를 리포트에 기록한다(임계값을 조정하는 것과, 테스트가 검증하는
"배경은 걸러지고 실제 차량은 잡힌다"는 요구사항 자체를 느슨하게 하는 것은
다르다 — 후자는 하면 안 됨).

- [ ] **Step 5: 커밋**

```bash
git add src/windshield.py tests/test_windshield.py
git commit -m "feat: add floor-masked dark-blob windshield detector"
```

---

## Task 3: `candidate.py` — 슬롯 배정 (`point_in_polygon`, `find_slot_windshield`)

**Files:**
- Create: `src/candidate.py`
- Create: `tests/test_candidate.py`

**Interfaces:**
- Consumes: `WindshieldBlob` (Task 2)
- Produces: `point_in_polygon(point, polygon) -> bool`,
  `find_slot_windshield(slot_polygon_raw, blobs) -> WindshieldBlob | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_candidate.py`:
```python
from windshield import WindshieldBlob
from candidate import point_in_polygon, find_slot_windshield


def test_point_in_polygon_true_and_false():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]

    assert point_in_polygon((50.0, 50.0), polygon) is True
    assert point_in_polygon((150.0, 50.0), polygon) is False


def test_find_slot_windshield_picks_blob_inside_polygon_only():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    inside_blob = WindshieldBlob(contour=None, bbox=(40, 40, 10, 10), centroid=(45.0, 45.0), area=100)
    outside_blob = WindshieldBlob(contour=None, bbox=(200, 200, 10, 10), centroid=(205.0, 205.0), area=100)

    assert find_slot_windshield(polygon, [outside_blob, inside_blob]) is inside_blob
    assert find_slot_windshield(polygon, [outside_blob]) is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_candidate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'candidate'`

- [ ] **Step 3: 구현**

`src/candidate.py`:
```python
import numpy as np
import cv2


def point_in_polygon(point, polygon):
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


def find_slot_windshield(slot_polygon_raw, blobs):
    for blob in blobs:
        if point_in_polygon(blob.centroid, slot_polygon_raw):
            return blob
    return None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_candidate.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/candidate.py tests/test_candidate.py
git commit -m "feat: add point-in-polygon slot assignment for detected windshields"
```

---

## Task 4: `candidate.py` — 라벨 후보 위치/크기 계산

**Files:**
- Modify: `src/candidate.py`
- Modify: `tests/test_candidate.py`

**Interfaces:**
- Consumes: `pixel_to_plane_points` (Task 1), `LocalView.raw_to_local` (기존),
  `WindshieldBlob` (Task 2)
- Produces: `compute_label_candidate(view, homography, blob, coverage_margin=1.3)
  -> (candidate_point, width, height)`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_candidate.py`에 추가:
```python
import numpy as np

from calibration import CalibrationModel, LocalView
from perspective import plane_to_pixel_homography, pixel_to_plane_points
from candidate import compute_label_candidate


def test_compute_label_candidate_region_fully_contains_blob():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 60.0), patch_size=(300, 300), local_f=300.0)
    slot_polygon_raw = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]
    polygon_local = view.raw_to_local(slot_polygon_raw)
    homography = plane_to_pixel_homography(polygon_local)

    blob = WindshieldBlob(contour=None, bbox=(300, 40, 20, 20), centroid=(310.0, 50.0), area=400)

    candidate_point, width, height = compute_label_candidate(view, homography, blob)

    corners_raw = [[300, 40], [320, 40], [320, 60], [300, 60]]
    corners_local = view.raw_to_local(corners_raw)
    corners_plane = pixel_to_plane_points(homography, corners_local)

    u_min = candidate_point[0] - width / 2.0
    u_max = candidate_point[0] + width / 2.0
    v_min = candidate_point[1] - height / 2.0
    v_max = candidate_point[1] + height / 2.0

    assert np.all(corners_plane[:, 0] >= u_min - 1e-6)
    assert np.all(corners_plane[:, 0] <= u_max + 1e-6)
    assert np.all(corners_plane[:, 1] >= v_min - 1e-6)
    assert np.all(corners_plane[:, 1] <= v_max + 1e-6)
```

`WindshieldBlob`를 이 테스트 파일에서 쓰려면 상단에 `from windshield import
WindshieldBlob` 임포트 추가(Task 3에서 이미 추가됐다면 그대로 재사용).

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_candidate.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_label_candidate'`

- [ ] **Step 3: 구현**

`src/candidate.py`에 추가:
```python
from perspective import pixel_to_plane_points


def compute_label_candidate(view, homography, blob, coverage_margin=1.3):
    x, y, w, h = blob.bbox
    corners_raw = [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ]
    corners_local = view.raw_to_local(corners_raw)
    corners_plane = pixel_to_plane_points(homography, corners_local)

    u_min, v_min = corners_plane.min(axis=0)
    u_max, v_max = corners_plane.max(axis=0)
    u_center = (u_min + u_max) / 2.0
    v_center = (v_min + v_max) / 2.0
    width = (u_max - u_min) * coverage_margin
    height = (v_max - v_min) * coverage_margin

    return (float(u_center), float(v_center)), float(width), float(height)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_candidate.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/candidate.py tests/test_candidate.py
git commit -m "feat: compute label candidate region from a detected windshield blob"
```

---

## Task 5: `pipeline.py` — 공용 헬퍼 추출 + `run_auto()`

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `detect_windshields` (Task 2), `find_slot_windshield`,
  `compute_label_candidate` (Task 3/4)
- Produces: `run_auto(config_path, raw_image_path, slot_id, output_path) ->
  np.ndarray | None` (`None` = 앞유리 안 보여서 스킵). 기존 `run()`은 동작
  변화 없음(내부 구현만 공용 헬퍼로 리팩터링).

- [ ] **Step 1: 현재 `src/pipeline.py` 확인**

먼저 현재 파일을 읽는다. `run()`이 슬롯 조회 → 이미지 로드 →
`_polygon_centroid` → `LocalView.centered_on` → `corner_rays`/`polygon_local`
검증(범위 밖이면 `ValueError`) → `plane_to_pixel_homography` → `rectify` →
`render_label` → `unrectify_into` → `cv2.imwrite`(반환값 체크) 순서로 되어
있을 것이다. 이 Step 1은 실패하는 테스트가 아니라 확인 단계 — 만약 위 설명과
현재 파일 구조가 크게 다르면(예: 검증 로직이 없거나 함수 이름이 다르면)
구현을 진행하지 말고 NEEDS_CONTEXT로 보고한다.

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_pipeline.py`에 추가 (파일 상단에 `from pipeline import run,
run_auto`로 임포트 갱신 필요):
```python
CAR_SAMPLE_IMAGE = "no_label/P1_B1_1_21/20260820_115029.jpg"
CAR_SLOT_POLYGON_RAW = [[200.0, 260.0], [460.0, 260.0], [460.0, 500.0], [200.0, 500.0]]
EMPTY_SLOT_POLYGON_RAW = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]


def _write_auto_test_config(tmp_path, polygon_raw):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [{"id": "slot-A", "polygon_raw": polygon_raw}]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config = SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "config.json"
    config.save(str(path))
    return str(path)


def test_run_auto_labels_slot_with_visible_windshield(tmp_path):
    config_path = _write_auto_test_config(tmp_path, CAR_SLOT_POLYGON_RAW)
    output_path = tmp_path / "final.png"

    raw = cv2.imread(CAR_SAMPLE_IMAGE)
    assert raw is not None

    final = run_auto(config_path, CAR_SAMPLE_IMAGE, "slot-A", str(output_path))

    assert final is not None
    assert final.shape == raw.shape
    assert not np.array_equal(final, raw)
    assert output_path.exists()


def test_run_auto_skips_slot_with_no_visible_windshield(tmp_path):
    config_path = _write_auto_test_config(tmp_path, EMPTY_SLOT_POLYGON_RAW)
    output_path = tmp_path / "final.png"

    result = run_auto(config_path, CAR_SAMPLE_IMAGE, "slot-A", str(output_path))

    assert result is None
    assert not output_path.exists()
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_auto'`

- [ ] **Step 4: 구현**

`src/pipeline.py`를 다음 내용으로 전체 교체:
```python
import cv2
import numpy as np

from calibration import LocalView
from candidate import compute_label_candidate, find_slot_windshield
from parking_slot import SlotConfig
from perspective import plane_to_pixel_homography
from renderer import render_label
from windshield import detect_windshields

DEFAULT_PATCH_SIZE = (300, 300)
DEFAULT_LOCAL_F = 300.0


def _polygon_centroid(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    return tuple(pts.mean(axis=0))


def _find_slot(config, slot_id, config_path):
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")
    return slot


def _prepare_slot_view(config, slot, slot_id):
    center_raw = _polygon_centroid(slot["polygon_raw"])
    if not np.all(np.isfinite(center_raw)):
        raise ValueError(f"slot '{slot_id}' has a degenerate polygon_raw (empty or non-finite): {slot['polygon_raw']}")

    view = LocalView.centered_on(config.calibration, center_raw, DEFAULT_PATCH_SIZE, DEFAULT_LOCAL_F)

    corner_rays = view._local_rays(slot["polygon_raw"])
    patch_w, patch_h = DEFAULT_PATCH_SIZE
    if not np.all(corner_rays[:, 2] > 0):
        raise ValueError(f"slot '{slot_id}' has a polygon_raw corner behind the local view (camera cannot represent it)")

    polygon_local = view.raw_to_local(slot["polygon_raw"])
    if not (np.all(polygon_local[:, 0] >= 0) and np.all(polygon_local[:, 0] <= patch_w - 1)
            and np.all(polygon_local[:, 1] >= 0) and np.all(polygon_local[:, 1] <= patch_h - 1)):
        raise ValueError(f"slot '{slot_id}' polygon_raw falls outside the local patch bounds for the current DEFAULT_PATCH_SIZE/DEFAULT_LOCAL_F")

    homography = plane_to_pixel_homography(polygon_local)
    return view, homography


def run(config_path, raw_image_path, slot_id, candidate_point, output_path):
    config = SlotConfig.load(config_path)
    slot = _find_slot(config, slot_id, config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    view, homography = _prepare_slot_view(config, slot, slot_id)

    local_patch = view.rectify(raw)
    composited_local = render_label(local_patch, homography, candidate_point, config.label_spec)
    final = view.unrectify_into(composited_local, raw)

    if not cv2.imwrite(output_path, final):
        raise ValueError(f"could not write output image to {output_path}")
    return final


def run_auto(config_path, raw_image_path, slot_id, output_path):
    config = SlotConfig.load(config_path)
    slot = _find_slot(config, slot_id, config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    blobs = detect_windshields(raw, config.calibration)
    blob = find_slot_windshield(slot["polygon_raw"], blobs)
    if blob is None:
        return None

    view, homography = _prepare_slot_view(config, slot, slot_id)
    candidate_point, width, height = compute_label_candidate(view, homography, blob)

    label_spec = dict(config.label_spec)
    label_spec["width"] = width
    label_spec["height"] = height

    local_patch = view.rectify(raw)
    composited_local = render_label(local_patch, homography, candidate_point, label_spec)
    final = view.unrectify_into(composited_local, raw)

    if not cv2.imwrite(output_path, final):
        raise ValueError(f"could not write output image to {output_path}")
    return final
```

기존 `run()`을 호출하는 테스트(예: 슬롯 없음/이미지 못 읽음/후보점 범위 밖
에러 케이스)는 동작이 그대로여야 한다 — Step 5에서 전체 스위트로 확인.

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: PASS — 기존 테스트 포함 전부.

- [ ] **Step 6: 커밋**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "refactor: extract shared slot-view setup, add run_auto for windshield-based placement"
```

---

## Task 6: `main.py` — `--auto` CLI 플래그

**Files:**
- Modify: `src/main.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `run_auto` (Task 5)
- Produces: `main(argv=None)`에 `--auto` 플래그 추가 (기존 시그니처 유지)

- [ ] **Step 1: 현재 `src/main.py` 확인**

먼저 현재 파일을 읽는다. `build_parser()` + `main(argv=None)` 구조로,
`--config`/`--image`/`--slot-id`/`--candidate-u`/`--candidate-v`/`--output`
플래그가 있고 `pipeline.run(...)`을 호출할 것이다. 크게 다르면
NEEDS_CONTEXT로 보고.

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_main.py`에 추가 (상단에 `from calibration import
CalibrationModel`, `from parking_slot import SlotConfig` 이미 있으면 재사용):
```python
def test_cli_auto_flag_writes_output_for_visible_windshield(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [{"id": "slot-A", "polygon_raw": [[200.0, 260.0], [460.0, 260.0], [460.0, 500.0], [200.0, 500.0]]}]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config_path = tmp_path / "config.json"
    SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", "no_label/P1_B1_1_21/20260820_115029.jpg",
        "--slot-id", "slot-A",
        "--auto",
        "--output", str(output_path),
    ])

    assert output_path.exists()


def test_cli_auto_flag_skips_without_writing_when_no_windshield(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [{"id": "slot-A", "polygon_raw": [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]}]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config_path = tmp_path / "config.json"
    SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", "no_label/P1_B1_1_21/20260820_115029.jpg",
        "--slot-id", "slot-A",
        "--auto",
        "--output", str(output_path),
    ])

    assert not output_path.exists()
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_main.py -v`
Expected: FAIL — `error: unrecognized arguments: --auto` (argparse가 아직
`--auto`를 모름)

- [ ] **Step 4: 구현**

`src/main.py`를 다음 내용으로 전체 교체:
```python
import argparse

from pipeline import run, run_auto


def build_parser():
    parser = argparse.ArgumentParser(
        description="Composite a label onto a parking slot in a CCTV image.")
    parser.add_argument("--config", required=True, help="path to camera config JSON")
    parser.add_argument("--image", required=True, help="path to raw input frame")
    parser.add_argument("--slot-id", required=True, help="slot id from the config to place the label in")
    parser.add_argument("--candidate-u", type=float, help="candidate label center, normalized 0-1 (ignored with --auto)")
    parser.add_argument("--candidate-v", type=float, help="candidate label center, normalized 0-1 (ignored with --auto)")
    parser.add_argument("--output", required=True, help="path to write the final PNG")
    parser.add_argument("--auto", action="store_true",
                         help="auto-detect the windshield and compute label position/size instead of using --candidate-u/-v")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.auto:
        result = run_auto(args.config, args.image, args.slot_id, args.output)
        if result is None:
            print(f"slot '{args.slot_id}': no visible windshield, skipped (no output written)")
        return

    if args.candidate_u is None or args.candidate_v is None:
        raise SystemExit("--candidate-u and --candidate-v are required unless --auto is set")

    run(args.config, args.image, args.slot_id, (args.candidate_u, args.candidate_v), args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 6: 전체 테스트 스위트 확인**

Run: `.venv/bin/pytest -v`
Expected: 전부 PASS

- [ ] **Step 7: 커밋**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: add --auto CLI flag for windshield-based label placement"
```
