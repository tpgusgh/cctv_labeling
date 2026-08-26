# 핵심 기하/렌더링 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** raw fisheye CCTV 프레임 1장 + 카메라 config + 주차슬롯 polygon + 고정
후보점을 입력받아, 렌즈왜곡+원근왜곡을 반영해 라벨을 합성한 PNG 1장을 출력하는
CLI 파이프라인을 만든다.

**Architecture:** 단일 스칼라 파라미터(equidistant) fisheye 모델로 raw
프레임을 rectified(평면) 이미지로 undistort → rectified 공간에서 homography로
라벨 도형을 직접 그려 합성 → 합성 결과를 같은 왜곡 모델로 raw 좌표계에
재왜곡(redistort)해서 최종 이미지 생성. 렌즈왜곡과 원근왜곡을 별도 단계로
분리 처리.

**Tech Stack:** Python 3, OpenCV(`opencv-python`), NumPy, pytest.

**Spec:** [docs/superpowers/specs/2026-08-26-core-geometry-rendering-design.md](../specs/2026-08-26-core-geometry-rendering-design.md)

## Global Constraints

- 라벨은 외부 PNG 에셋 합성이 아니라 프로그램이 직접 도형을 그린다 (spec:
  "Renderer" 섹션).
- 원근왜곡과 렌즈왜곡은 별도 단계로 처리한다 — 하나의 변환으로 합치지 않는다.
- 최종 출력은 입력 raw 프레임과 동일한 픽셀 크기의 PNG.
- 에러(calibration 없음, 슬롯 없음, 후보점 범위 밖)는 명확한 예외로 던진다 —
  삼키거나 기본값으로 넘어가지 않는다.
- 테스트는 무거운 프레임워크나 다수의 단위테스트 매트릭스가 아니라, 각 모듈의
  핵심 동작 하나씩만 pytest로 검증하는 가벼운 self-check로 유지한다 (ponytail
  규칙, spec의 "테스트" 섹션과 동일).

---

## Task 1: 프로젝트 초기화 + CalibrationModel 좌표 변환 코어

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py`
- Create: `src/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationModel(cx: float, cy: float, f: float)`,
  `CalibrationModel.undistort_points(points: Sequence[[float,float]]) -> np.ndarray[N,2]`,
  `CalibrationModel.distort_points(points: Sequence[[float,float]]) -> np.ndarray[N,2]`

- [ ] **Step 1: 의존성 파일 작성**

`requirements.txt`:
```
opencv-python>=4.8
numpy>=1.24
pytest>=7.4
```

`conftest.py` (repo root — pytest가 `src/`를 import 경로에 넣도록):
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
```

- [ ] **Step 2: 의존성 설치**

Run: `pip install -r requirements.txt`

- [ ] **Step 3: 실패하는 테스트 작성**

`tests/test_calibration.py`:
```python
import numpy as np
from calibration import CalibrationModel


def test_undistort_distort_roundtrip():
    model = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    points = [[320.0, 320.0], [400.0, 320.0], [320.0, 450.0], [200.0, 500.0], [500.0, 150.0]]

    undistorted = model.undistort_points(points)
    roundtripped = model.distort_points(undistorted)

    np.testing.assert_allclose(roundtripped, np.asarray(points), atol=1e-6)
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibration'`

- [ ] **Step 5: CalibrationModel 구현**

`src/calibration.py`:
```python
import numpy as np
import cv2


class CalibrationModel:
    """Single-parameter equidistant fisheye <-> rectilinear mapping.

    ponytail: one scalar focal parameter fit by least-squares against
    user-clicked collinear points on a real straight line — upgrade to
    full cv2.fisheye.calibrate() + checkerboard captures if that data
    ever becomes available.
    """

    def __init__(self, cx: float, cy: float, f: float):
        self.cx = float(cx)
        self.cy = float(cy)
        self.f = float(f)

    def undistort_points(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        r_d = np.sqrt(dx ** 2 + dy ** 2)
        theta = r_d / self.f
        r_u = self.f * np.tan(theta)
        safe_r_d = np.where(r_d > 1e-9, r_d, 1.0)
        scale = np.where(r_d > 1e-9, r_u / safe_r_d, 1.0)
        return np.stack([self.cx + dx * scale, self.cy + dy * scale], axis=1)

    def distort_points(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        r_u = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan(r_u / self.f)
        r_d = self.f * theta
        safe_r_u = np.where(r_u > 1e-9, r_u, 1.0)
        scale = np.where(r_u > 1e-9, r_d / safe_r_u, 1.0)
        return np.stack([self.cx + dx * scale, self.cy + dy * scale], axis=1)

    def undistort_image(self, raw_bgr):
        h, w = raw_bgr.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        rectified_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        raw_sample_coords = self.distort_points(rectified_coords)
        map_x = raw_sample_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = raw_sample_coords[:, 1].reshape(h, w).astype(np.float32)
        return cv2.remap(raw_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)

    def redistort_image(self, rectified_bgr, output_shape):
        h, w = output_shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        raw_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        rectified_sample_coords = self.undistort_points(raw_coords)
        map_x = rectified_sample_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = rectified_sample_coords[:, 1].reshape(h, w).astype(np.float32)
        return cv2.remap(rectified_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)

    def to_dict(self):
        return {"model": "equidistant_1param", "center": [self.cx, self.cy], "f": self.f}

    @classmethod
    def from_dict(cls, d):
        cx, cy = d["center"]
        return cls(cx, cy, d["f"])
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add requirements.txt conftest.py src/calibration.py tests/test_calibration.py
git commit -m "feat: add fisheye CalibrationModel point round-trip"
```

---

## Task 2: CalibrationModel.fit() + JSON 직렬화

**Files:**
- Modify: `src/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `CalibrationModel` from Task 1
- Produces: `fit(clicked_points, cx, cy, f_min=50.0, f_max=1000.0, n_coarse=50) -> CalibrationModel`,
  `CalibrationModel.to_dict()` / `CalibrationModel.from_dict(d)` (already added in Task 1, exercised here)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_calibration.py`에 추가:
```python
import json
from calibration import fit


def test_fit_recovers_known_focal_length():
    cx, cy, f_true = 320.0, 320.0, 300.0
    true_model = CalibrationModel(cx, cy, f_true)
    rectified_line_points = [[200.0, 250.0], [260.0, 250.0], [320.0, 250.0], [380.0, 250.0], [440.0, 250.0]]
    raw_clicks = true_model.distort_points(rectified_line_points)

    fitted = fit(raw_clicks, cx=cx, cy=cy)

    assert abs(fitted.f - f_true) / f_true < 0.05


def test_to_dict_from_dict_roundtrip(tmp_path):
    model = CalibrationModel(cx=321.5, cy=318.0, f=287.3)
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(model.to_dict()))

    loaded = CalibrationModel.from_dict(json.loads(path.read_text()))

    assert loaded.cx == model.cx
    assert loaded.cy == model.cy
    assert loaded.f == model.f
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL — `ImportError: cannot import name 'fit'`

- [ ] **Step 3: fit() 구현**

`src/calibration.py`에 추가 (클래스 밖, 모듈 레벨 함수):
```python
def fit(clicked_points, cx, cy, f_min=50.0, f_max=1000.0, n_coarse=50):
    pts = np.asarray(clicked_points, dtype=np.float64).reshape(-1, 2)

    def residual(f):
        model = CalibrationModel(cx, cy, f)
        undistorted = model.undistort_points(pts)
        centered = undistorted - undistorted.mean(axis=0)
        singular_values = np.linalg.svd(centered, full_matrices=False)[1]
        return singular_values[-1]

    candidates = np.linspace(f_min, f_max, n_coarse)
    residuals = [residual(f) for f in candidates]
    best_idx = int(np.argmin(residuals))
    lo = candidates[max(best_idx - 1, 0)]
    hi = candidates[min(best_idx + 1, n_coarse - 1)]

    for _ in range(40):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if residual(m1) < residual(m2):
            hi = m2
        else:
            lo = m1

    return CalibrationModel(cx, cy, (lo + hi) / 2)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 커밋**

```bash
git add src/calibration.py tests/test_calibration.py
git commit -m "feat: fit fisheye focal parameter from clicked line points"
```

---

## Task 3: 이미지 단위 undistort/redistort + 실제 샘플 프레임 검증

**Files:**
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `CalibrationModel.undistort_image`, `CalibrationModel.redistort_image` (구현은 Task 1에서 이미 완료됨 — 이 태스크는 실제 이미지로 검증만 추가)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_calibration.py`에 추가:
```python
import cv2

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def test_undistort_redistort_image_roundtrip():
    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert raw is not None, f"sample image not found at {SAMPLE_RAW_IMAGE}"
    model = CalibrationModel(cx=320.0, cy=320.0, f=300.0)

    rectified = model.undistort_image(raw)
    assert rectified.shape == raw.shape
    assert not np.array_equal(rectified, raw)

    redistorted = model.redistort_image(rectified, output_shape=raw.shape[:2])
    assert redistorted.shape == raw.shape

    mean_abs_diff = np.mean(np.abs(redistorted.astype(np.int16) - raw.astype(np.int16)))
    assert mean_abs_diff < 15.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_calibration.py::test_undistort_redistort_image_roundtrip -v`
Expected: FAIL — `NameError: name 'cv2' is not defined` (테스트 파일에 아직 `import cv2` 없음)

- [ ] **Step 3: 통과 확인**

`undistort_image`/`redistort_image`는 Task 1에서 이미 구현됨. 테스트 파일에
`import cv2` 추가한 상태로 재실행.

Run: `pytest tests/test_calibration.py -v`
Expected: PASS (5 tests)

- [ ] **Step 4: 커밋**

```bash
git add tests/test_calibration.py
git commit -m "test: verify undistort/redistort round-trip on real sample frame"
```

---

## Task 4: 주차슬롯 Config (raw 클릭 → rectified polygon, JSON 저장/로드)

**Files:**
- Create: `src/parking_slot.py`
- Test: `tests/test_parking_slot.py`

**Interfaces:**
- Consumes: `CalibrationModel` (Task 1)
- Produces: `raw_clicks_to_slot_polygon(raw_points, calibration) -> list[[float,float]]`,
  `SlotConfig(camera_id, image_width, image_height, calibration, slots, label_spec)`,
  `SlotConfig.save(path)`, `SlotConfig.load(path) -> SlotConfig`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_parking_slot.py`:
```python
from calibration import CalibrationModel
from parking_slot import raw_clicks_to_slot_polygon, SlotConfig


def test_raw_clicks_to_slot_polygon_converts_through_calibration():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    raw_points = [[250.0, 180.0], [400.0, 190.0], [410.0, 300.0], [240.0, 290.0]]

    polygon = raw_clicks_to_slot_polygon(raw_points, calibration)

    assert len(polygon) == 4
    assert polygon != raw_points


def test_slot_config_save_load_roundtrip(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    slots = [{"id": "P1_B1_1_1-A", "polygon_rectified": [[220.0, 150.0], [380.0, 150.0], [400.0, 280.0], [200.0, 280.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.25, "color": [30, 180, 90], "alpha": 0.75, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "P1_B1_1_1.json"

    config.save(str(path))
    loaded = SlotConfig.load(str(path))

    assert loaded.camera_id == "P1_B1_1_1"
    assert loaded.image_width == 640
    assert loaded.slots == slots
    assert loaded.label_spec == label_spec
    assert loaded.calibration.f == calibration.f
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_parking_slot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parking_slot'`

- [ ] **Step 3: 구현**

`src/parking_slot.py`:
```python
import json
from pathlib import Path

from calibration import CalibrationModel


def raw_clicks_to_slot_polygon(raw_points, calibration):
    rectified = calibration.undistort_points(raw_points)
    return rectified.tolist()


class SlotConfig:
    def __init__(self, camera_id, image_width, image_height, calibration, slots, label_spec):
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.calibration = calibration
        self.slots = slots
        self.label_spec = label_spec

    def save(self, path):
        payload = {
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "calibration": self.calibration.to_dict(),
            "slots": self.slots,
            "label_spec": self.label_spec,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        calibration = CalibrationModel.from_dict(payload["calibration"])
        return cls(
            payload["camera_id"],
            payload["image_width"],
            payload["image_height"],
            calibration,
            payload["slots"],
            payload["label_spec"],
        )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_parking_slot.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/parking_slot.py tests/test_parking_slot.py
git commit -m "feat: add per-camera slot config with raw-click-to-rectified conversion"
```

---

## Task 5: Homography (정규화 주차면 좌표 <-> rectified 픽셀)

**Files:**
- Create: `src/perspective.py`
- Test: `tests/test_perspective.py`

**Interfaces:**
- Produces: `plane_to_pixel_homography(slot_polygon_rectified) -> np.ndarray[3,3]`,
  `plane_points_to_pixel(homography, normalized_points) -> np.ndarray[N,2]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_perspective.py`:
```python
import numpy as np
import pytest

from perspective import plane_to_pixel_homography, plane_points_to_pixel


def test_plane_points_map_to_expected_pixel_square():
    quad = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    homography = plane_to_pixel_homography(quad)

    normalized = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    pixels = plane_points_to_pixel(homography, normalized)

    expected = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0], [200.0, 200.0]])
    np.testing.assert_allclose(pixels, expected, atol=1e-3)


def test_plane_to_pixel_homography_rejects_non_quad():
    with pytest.raises(ValueError):
        plane_to_pixel_homography([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_perspective.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perspective'`

- [ ] **Step 3: 구현**

`src/perspective.py`:
```python
import numpy as np
import cv2

_UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


def plane_to_pixel_homography(slot_polygon_rectified):
    dst = np.asarray(slot_polygon_rectified, dtype=np.float32)
    if dst.shape != (4, 2):
        raise ValueError(f"slot polygon must have exactly 4 points, got shape {dst.shape}")
    return cv2.getPerspectiveTransform(_UNIT_SQUARE, dst)


def plane_points_to_pixel(homography, normalized_points):
    pts = np.asarray(normalized_points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, homography)
    return out.reshape(-1, 2)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_perspective.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/perspective.py tests/test_perspective.py
git commit -m "feat: add plane-to-pixel homography for slot polygons"
```

---

## Task 6: 라벨 렌더러 (직접 그리기, PNG 에셋 없음)

**Files:**
- Create: `src/renderer.py`
- Test: `tests/test_renderer.py`

**Interfaces:**
- Consumes: `plane_points_to_pixel` (Task 5)
- Produces: `render_label(rectified_image, homography, candidate_point, label_spec) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_renderer.py`:
```python
import numpy as np
import pytest

from perspective import plane_to_pixel_homography
from renderer import render_label


def _blank_canvas():
    return np.full((400, 400, 3), 255, dtype=np.uint8)


def test_render_label_fills_candidate_region_with_full_alpha():
    canvas = _blank_canvas()
    homography = plane_to_pixel_homography([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])
    label_spec = {"shape": "rect", "width": 0.4, "height": 0.4, "color": [0, 0, 255], "alpha": 1.0, "text": None}

    result = render_label(canvas, homography, (0.5, 0.5), label_spec)

    assert list(result[200, 200]) == [0, 0, 255]
    assert list(result[10, 10]) == [255, 255, 255]


def test_render_label_rejects_unsupported_shape():
    canvas = _blank_canvas()
    homography = plane_to_pixel_homography([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])
    label_spec = {"shape": "rounded_rect", "width": 0.4, "height": 0.4, "color": [0, 0, 255], "alpha": 1.0, "text": None}

    with pytest.raises(NotImplementedError):
        render_label(canvas, homography, (0.5, 0.5), label_spec)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_renderer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'renderer'`

- [ ] **Step 3: 구현**

`src/renderer.py`:
```python
import numpy as np
import cv2

from perspective import plane_points_to_pixel


def _label_corners_normalized(candidate_point, width, height):
    cu, cv_ = candidate_point
    hw, hh = width / 2.0, height / 2.0
    return [
        [cu - hw, cv_ - hh],
        [cu + hw, cv_ - hh],
        [cu + hw, cv_ + hh],
        [cu - hw, cv_ + hh],
    ]


def render_label(rectified_image, homography, candidate_point, label_spec):
    shape = label_spec.get("shape", "rect")
    if shape != "rect":
        raise NotImplementedError(f"label shape '{shape}' not implemented yet; only 'rect' is supported")

    width = label_spec["width"]
    height = label_spec["height"]
    color = tuple(int(c) for c in label_spec["color"])
    alpha = float(label_spec["alpha"])
    text = label_spec.get("text")

    corners_norm = _label_corners_normalized(candidate_point, width, height)
    corners_px = plane_points_to_pixel(homography, corners_norm)
    poly = corners_px.reshape(-1, 1, 2).astype(np.int32)

    overlay = rectified_image.copy()
    cv2.fillPoly(overlay, [poly], color)
    if text:
        centroid = corners_px.mean(axis=0).astype(int)
        cv2.putText(overlay, text, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, rectified_image, 1 - alpha, 0)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_renderer.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/renderer.py tests/test_renderer.py
git commit -m "feat: add direct-draw label renderer (no PNG asset)"
```

---

## Task 7: 전체 파이프라인 (raw -> rectified -> 합성 -> redistort)

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `SlotConfig` (Task 4), `plane_to_pixel_homography` (Task 5), `render_label` (Task 6)
- Produces: `run(config_path, raw_image_path, slot_id, candidate_point, output_path) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_pipeline.py`:
```python
import cv2
import numpy as np

from calibration import CalibrationModel
from parking_slot import SlotConfig
from pipeline import run

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def _write_test_config(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    slots = [{"id": "slot-A", "polygon_rectified": [[220.0, 150.0], [380.0, 150.0], [400.0, 280.0], [200.0, 280.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.6, "color": [0, 255, 0], "alpha": 0.8, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "P1_B1_1_1.json"
    config.save(str(path))
    return str(path), calibration, slots[0]["polygon_rectified"]


def test_pipeline_composites_label_within_slot_bbox_only(tmp_path):
    config_path, calibration, polygon_rectified = _write_test_config(tmp_path)
    output_path = str(tmp_path / "final.png")

    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert raw is not None

    final = run(config_path, SAMPLE_RAW_IMAGE, "slot-A", (0.5, 0.5), output_path)

    assert final.shape == raw.shape

    raw_space_polygon = calibration.distort_points(polygon_rectified)
    x_min, y_min = raw_space_polygon.min(axis=0).astype(int)
    x_max, y_max = raw_space_polygon.max(axis=0).astype(int)
    bbox_before = raw[y_min:y_max, x_min:x_max]
    bbox_after = final[y_min:y_max, x_min:x_max]
    assert not np.array_equal(bbox_before, bbox_after)

    far_before = raw[600:630, 10:40]
    far_after = final[600:630, 10:40]
    np.testing.assert_array_equal(far_before, far_after)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: 구현**

`src/pipeline.py`:
```python
import cv2

from parking_slot import SlotConfig
from perspective import plane_to_pixel_homography
from renderer import render_label


def run(config_path, raw_image_path, slot_id, candidate_point, output_path):
    config = SlotConfig.load(config_path)
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    rectified = config.calibration.undistort_image(raw)
    homography = plane_to_pixel_homography(slot["polygon_rectified"])
    composited_rectified = render_label(rectified, homography, candidate_point, config.label_spec)
    final = config.calibration.redistort_image(composited_rectified, output_shape=raw.shape[:2])

    cv2.imwrite(output_path, final)
    return final
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire full raw-to-final label compositing pipeline"
```

---

## Task 8: CLI 진입점

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `run` (Task 7)
- Produces: `main(argv=None)` — CLI entry point

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_main.py`:
```python
import cv2

from calibration import CalibrationModel
from parking_slot import SlotConfig
from main import main

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def test_cli_writes_output_png(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    slots = [{"id": "slot-A", "polygon_rectified": [[220.0, 150.0], [380.0, 150.0], [400.0, 280.0], [200.0, 280.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.6, "color": [0, 255, 0], "alpha": 0.8, "text": None}
    config_path = tmp_path / "P1_B1_1_1.json"
    SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", SAMPLE_RAW_IMAGE,
        "--slot-id", "slot-A",
        "--candidate-u", "0.5",
        "--candidate-v", "0.5",
        "--output", str(output_path),
    ])

    written = cv2.imread(str(output_path))
    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert written is not None
    assert written.shape == raw.shape
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: 구현**

`src/main.py`:
```python
import argparse

from pipeline import run


def build_parser():
    parser = argparse.ArgumentParser(
        description="Composite a label onto one raw CCTV frame (sub-project 1 test harness).")
    parser.add_argument("--config", required=True, help="path to camera config JSON")
    parser.add_argument("--image", required=True, help="path to raw input frame")
    parser.add_argument("--slot-id", required=True, help="slot id from the config to place the label in")
    parser.add_argument("--candidate-u", type=float, required=True, help="candidate label center, normalized 0-1")
    parser.add_argument("--candidate-v", type=float, required=True, help="candidate label center, normalized 0-1")
    parser.add_argument("--output", required=True, help="path to write the final PNG")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.config, args.image, args.slot_id, (args.candidate_u, args.candidate_v), args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: 전체 테스트 스위트 확인**

Run: `pytest -v`
Expected: 전부 PASS (calibration 5개, parking_slot 2개, perspective 2개, renderer 2개, pipeline 1개, main 1개)

- [ ] **Step 6: 커밋**

```bash
git add src/main.py tests/test_main.py
git commit -m "feat: add CLI entry point for single-image label compositing"
```
