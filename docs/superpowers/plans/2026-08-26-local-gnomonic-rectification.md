# 슬롯별 로컬 Gnomonic 재투영 수정 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 서브프로젝트 1의 "프레임 전체를 한 번에 rectify" 방식을, 실측된
거의-180° FOV 카메라에서도 성립하는 "슬롯별 로컬 gnomonic 재투영" 방식으로
교체한다.

**Architecture:** raw 픽셀 → 3D 단위 광선(equidistant) → 슬롯 중심 방향이
`(0,0,1)`이 되도록 회전 → 로컬 rectilinear 투영으로 작은 패치 생성. 라벨
합성은 기존 `perspective.py`/`renderer.py`를 그대로 재사용. 합성된 패치는
raw 이미지의 해당 bounding box에만 역변환해 합성 — 슬롯 바깥은 원본과 픽셀
단위로 동일.

**Tech Stack:** Python 3, OpenCV, NumPy, pytest (기존 서브프로젝트 1과 동일
worktree/venv 재사용).

**Spec:** [docs/superpowers/specs/2026-08-26-local-gnomonic-rectification-design.md](../specs/2026-08-26-local-gnomonic-rectification-design.md)

## Global Constraints

- 기존 `perspective.py`, `renderer.py`, `main.py`는 수정하지 않는다 (spec: 두
  모듈 모두 "어떤 rectified 공간이 주어지든" 동작하도록 설계되어 재사용 가능).
- 기존 2D `CalibrationModel.undistort_points`/`distort_points`/`undistort_image`/
  `redistort_image`/`to_dict`/`from_dict`는 삭제하지 않고 그대로 둔다 (다른
  테스트가 계속 참조).
- `SlotConfig.slots[].polygon_rectified` 키를 `polygon_raw`로 변경한다 — 값의
  의미가 "전역 rectified 좌표"에서 "raw 클릭 좌표 그대로"로 바뀌었으므로 같은
  키 이름을 재사용하지 않는다.
- 에러는 명확한 예외로 던진다 — 삼키거나 기본값으로 넘어가지 않는다.
- 테스트는 무거운 프레임워크나 매트릭스가 아니라, 핵심 동작 하나씩만 pytest로
  검증하는 가벼운 self-check로 유지한다 (ponytail 규칙).
- `patch_size`/`local_f`는 고정 상수로 시작한다 (ponytail: 슬롯 실제 각크기
  기반 동적 산정은 범위 밖, 향후 서브프로젝트에서 다룸).

---

## Task 1: 3D 광선 변환 (`pixel_to_ray`/`ray_to_pixel`)

**Files:**
- Modify: `src/calibration.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationModel.pixel_to_ray(points) -> np.ndarray[N,3]`,
  `CalibrationModel.ray_to_pixel(rays) -> np.ndarray[N,2]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_calibration.py`에 추가:
```python
def test_pixel_to_ray_ray_to_pixel_roundtrip():
    model = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    points = [[320.0, 320.0], [400.0, 320.0], [320.0, 450.0], [200.0, 500.0], [500.0, 150.0]]

    rays = model.pixel_to_ray(points)
    roundtripped = model.ray_to_pixel(rays)

    np.testing.assert_allclose(roundtripped, np.asarray(points), atol=1e-6)


def test_pixel_to_ray_at_center_is_forward_axis():
    model = CalibrationModel(cx=320.0, cy=320.0, f=204.0)

    ray = model.pixel_to_ray([[320.0, 320.0]])[0]

    np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_calibration.py -v`
Expected: FAIL — `AttributeError: 'CalibrationModel' object has no attribute 'pixel_to_ray'`

- [ ] **Step 3: 구현**

`src/calibration.py`의 `CalibrationModel` 클래스에 메서드 추가 (기존 메서드는
그대로 유지, 삭제하지 않음):
```python
    def pixel_to_ray(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = r / self.f
        phi = np.arctan2(dy, dx)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack([x, y, z], axis=1)

    def ray_to_pixel(self, rays):
        rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        rays = rays / norms
        theta = np.arccos(np.clip(rays[:, 2], -1.0, 1.0))
        phi = np.arctan2(rays[:, 1], rays[:, 0])
        r = self.f * theta
        dx = r * np.cos(phi)
        dy = r * np.sin(phi)
        return np.stack([self.cx + dx, self.cy + dy], axis=1)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_calibration.py -v`
Expected: PASS (전체 통과, 새로 추가된 2개 포함)

- [ ] **Step 5: 커밋**

```bash
git add src/calibration.py tests/test_calibration.py
git commit -m "feat: add 3D ray point mapping to CalibrationModel"
```

---

## Task 2: 회전 정렬 헬퍼 + `LocalView` 점 단위 매핑

**Files:**
- Modify: `src/calibration.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `CalibrationModel.pixel_to_ray`/`ray_to_pixel` (Task 1)
- Produces: `LocalView.centered_on(calibration, center_raw_point, patch_size, local_f) -> LocalView`,
  `LocalView.raw_to_local(raw_points) -> np.ndarray[N,2]`,
  `LocalView.local_to_raw(local_points) -> np.ndarray[N,2]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_calibration.py`에 추가:
```python
from calibration import LocalView


def test_local_view_raw_to_local_roundtrip_off_axis_center():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 20.0), patch_size=(300, 300), local_f=300.0)

    raw_points = [[320.0, 20.0], [340.0, 40.0], [300.0, 10.0], [330.0, 60.0]]
    local_points = view.raw_to_local(raw_points)
    roundtripped = view.local_to_raw(local_points)

    np.testing.assert_allclose(roundtripped, np.asarray(raw_points), atol=1e-3)


def test_local_view_center_maps_to_patch_center():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    center_raw = (320.0, 20.0)
    view = LocalView.centered_on(calibration, center_raw, patch_size=(300, 300), local_f=300.0)

    local_center = view.raw_to_local([center_raw])[0]

    np.testing.assert_allclose(local_center, [150.0, 150.0], atol=1e-6)


def test_local_view_handles_center_at_optical_axis():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 320.0), patch_size=(300, 300), local_f=300.0)

    raw_points = [[320.0, 320.0], [340.0, 330.0], [300.0, 310.0]]
    local_points = view.raw_to_local(raw_points)
    roundtripped = view.local_to_raw(local_points)

    np.testing.assert_allclose(roundtripped, np.asarray(raw_points), atol=1e-3)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_calibration.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalView'`

- [ ] **Step 3: 구현**

`src/calibration.py`에 추가 (모듈 레벨 함수 + 새 클래스):
```python
def _rotation_aligning_to_z(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = np.dot(v, z)

    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
        return 2 * np.outer(axis, axis) - np.eye(3)

    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    axis = axis / s
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + K * s + (K @ K) * (1 - c)


class LocalView:
    def __init__(self, calibration, rotation, local_f, patch_size):
        self.calibration = calibration
        self.rotation = rotation
        self.local_f = local_f
        self.patch_size = patch_size

    @classmethod
    def centered_on(cls, calibration, center_raw_point, patch_size, local_f):
        ray = calibration.pixel_to_ray([center_raw_point])[0]
        rotation = _rotation_aligning_to_z(ray)
        return cls(calibration, rotation, local_f, patch_size)

    def raw_to_local(self, raw_points):
        rays = self.calibration.pixel_to_ray(raw_points)
        local_rays = rays @ self.rotation.T
        w, h = self.patch_size
        lx = self.local_f * local_rays[:, 0] / local_rays[:, 2] + w / 2.0
        ly = self.local_f * local_rays[:, 1] / local_rays[:, 2] + h / 2.0
        return np.stack([lx, ly], axis=1)

    def local_to_raw(self, local_points):
        pts = np.asarray(local_points, dtype=np.float64).reshape(-1, 2)
        w, h = self.patch_size
        x = (pts[:, 0] - w / 2.0) / self.local_f
        y = (pts[:, 1] - h / 2.0) / self.local_f
        z = np.ones_like(x)
        local_rays = np.stack([x, y, z], axis=1)
        local_rays = local_rays / np.linalg.norm(local_rays, axis=1, keepdims=True)
        world_rays = local_rays @ self.rotation
        return self.calibration.ray_to_pixel(world_rays)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/calibration.py tests/test_calibration.py
git commit -m "feat: add LocalView gnomonic point mapping centered on arbitrary raw direction"
```

---

## Task 3: `LocalView` 이미지 단위 rectify/unrectify + 주변부 회귀 테스트

**Files:**
- Modify: `src/calibration.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `LocalView.raw_to_local`/`local_to_raw` (Task 2)
- Produces: `LocalView.rectify(raw_image) -> np.ndarray`,
  `LocalView.unrectify_into(local_patch, raw_image) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_calibration.py`에 추가:
```python
import cv2

PERIPHERAL_SAMPLE_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def test_local_view_rectify_output_shape():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    raw = cv2.imread(PERIPHERAL_SAMPLE_IMAGE)
    assert raw is not None
    view = LocalView.centered_on(calibration, (320.0, 20.0), patch_size=(300, 300), local_f=300.0)

    patch = view.rectify(raw)

    assert patch.shape == (300, 300, 3)


def test_local_view_roundtrip_preserves_peripheral_content():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    raw = cv2.imread(PERIPHERAL_SAMPLE_IMAGE)
    assert raw is not None

    center_raw = (320.0, 20.0)  # raw radius ~300px from (320,320) -- near the measured
                                  # real content boundary where the old global model
                                  # zero-filled everything
    view = LocalView.centered_on(calibration, center_raw, patch_size=(300, 300), local_f=300.0)

    patch = view.rectify(raw)
    roundtripped = view.unrectify_into(patch, raw)

    corners_local = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype=np.float64)
    corners_raw = view.local_to_raw(corners_local)
    x_min, y_min = corners_raw.min(axis=0).astype(int)
    x_max, y_max = corners_raw.max(axis=0).astype(int)
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, raw.shape[1]), min(y_max, raw.shape[0])

    bbox_raw = raw[y_min:y_max, x_min:x_max]
    bbox_roundtripped = roundtripped[y_min:y_max, x_min:x_max]
    mean_abs_diff = np.mean(np.abs(bbox_roundtripped.astype(np.int16) - bbox_raw.astype(np.int16)))
    assert mean_abs_diff < 10.0

    far_raw = raw[550:580, 550:580]
    far_roundtripped = roundtripped[550:580, 550:580]
    np.testing.assert_array_equal(far_raw, far_roundtripped)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_calibration.py -v`
Expected: FAIL — `AttributeError: 'LocalView' object has no attribute 'rectify'`

- [ ] **Step 3: 구현**

`src/calibration.py`의 `LocalView` 클래스에 메서드 추가:
```python
    def rectify(self, raw_image):
        w, h = self.patch_size
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        local_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        raw_sample_coords = self.local_to_raw(local_coords)
        map_x = raw_sample_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = raw_sample_coords[:, 1].reshape(h, w).astype(np.float32)
        return cv2.remap(raw_image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)

    def unrectify_into(self, local_patch, raw_image):
        w, h = self.patch_size
        corners_local = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        corners_raw = self.local_to_raw(corners_local)
        x_min = max(int(np.floor(corners_raw[:, 0].min())), 0)
        x_max = min(int(np.ceil(corners_raw[:, 0].max())) + 1, raw_image.shape[1])
        y_min = max(int(np.floor(corners_raw[:, 1].min())), 0)
        y_max = min(int(np.ceil(corners_raw[:, 1].max())) + 1, raw_image.shape[0])

        result = raw_image.copy()
        if x_max <= x_min or y_max <= y_min:
            return result

        grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        raw_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        local_sample_coords = self.raw_to_local(raw_coords)
        map_x = local_sample_coords[:, 0].reshape(y_max - y_min, x_max - x_min).astype(np.float32)
        map_y = local_sample_coords[:, 1].reshape(y_max - y_min, x_max - x_min).astype(np.float32)
        patch_region = cv2.remap(local_patch, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
        result[y_min:y_max, x_min:x_max] = patch_region
        return result
```

`src/calibration.py` 파일 상단에 `import cv2`가 이미 있는지 확인 (기존
`undistort_image` 등에서 이미 사용 중이므로 이미 있을 것 — 없다면 추가).

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_calibration.py -v`
Expected: PASS — 특히 `test_local_view_roundtrip_preserves_peripheral_content`가
통과하는 것이 이번 수정 전체의 핵심 증명 (기존 전역 모델이었다면 이 위치는
검게 뭉개졌을 영역).

- [ ] **Step 5: 커밋**

```bash
git add src/calibration.py tests/test_calibration.py
git commit -m "feat: add LocalView image rectify/unrectify, verify no zero-fill at fisheye periphery"
```

---

## Task 4: `parking_slot.py` 스키마 변경 (`polygon_rectified` → `polygon_raw`)

**Files:**
- Modify: `src/parking_slot.py`
- Modify: `tests/test_parking_slot.py`

**Interfaces:**
- Produces: `SlotConfig(camera_id, image_width, image_height, calibration, slots, label_spec)`
  (시그니처 동일, `slots[].polygon_raw` 의미만 변경), `SlotConfig.save(path)`,
  `SlotConfig.load(path) -> SlotConfig` (변경 없음)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_parking_slot.py`를 다음 내용으로 전체 교체:
```python
from calibration import CalibrationModel
from parking_slot import SlotConfig


def test_slot_config_save_load_roundtrip():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    slots = [{"id": "P1_B1_1_1-A", "polygon_raw": [[250.0, 180.0], [400.0, 190.0], [410.0, 300.0], [240.0, 290.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.25, "color": [30, 180, 90], "alpha": 0.75, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = "/tmp/test_slot_config_roundtrip.json"

    config.save(path)
    loaded = SlotConfig.load(path)

    assert loaded.camera_id == "P1_B1_1_1"
    assert loaded.image_width == 640
    assert loaded.slots == slots
    assert loaded.label_spec == label_spec
    assert loaded.calibration.f == calibration.f
```

(참고: 이전 버전 테스트는 `raw_clicks_to_slot_polygon`도 검증했으나, 이번
스키마 변경으로 그 함수 자체가 불필요해져 삭제되므로 이 테스트도 함께
제거한다.)

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_parking_slot.py -v`
Expected: 이미 통과할 수도 있음 — `SlotConfig` 자체는 키 이름을 검사하지
않으므로. 이 경우 `grep raw_clicks_to_slot_polygon src/parking_slot.py`로
"현재 코드에 그 함수가 아직 남아있음"을 확인하는 것으로 이 Step을 대체 —
Step 3에서 그 함수를 제거한 뒤 Step 4에서 스위트가 여전히 통과하는지로
검증한다.

- [ ] **Step 3: 구현**

`src/parking_slot.py`를 다음 내용으로 전체 교체 (기존
`raw_clicks_to_slot_polygon` 함수와 그 안의 `undistort_points` 호출 제거,
`SlotConfig` 클래스는 로직 변경 없이 유지):
```python
import json
from pathlib import Path

from calibration import CalibrationModel


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

Run: `.venv/bin/pytest tests/test_parking_slot.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/parking_slot.py tests/test_parking_slot.py
git commit -m "refactor: store slot polygons as raw clicks, drop global-rectified conversion"
```

---

## Task 5: `pipeline.py` 재작성 (로컬 뷰 기반) + end-to-end 주변부 테스트

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `LocalView` (Task 2, 3), `SlotConfig` (Task 4),
  `plane_to_pixel_homography` (기존, 변경 없음), `render_label` (기존, 변경 없음)
- Produces: `run(config_path, raw_image_path, slot_id, candidate_point, output_path)`
  (시그니처 동일, 내부 구현만 로컬 뷰 기반으로 교체)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_pipeline.py`를 다음 내용으로 전체 교체:
```python
import cv2
import numpy as np
import pytest

from calibration import CalibrationModel
from parking_slot import SlotConfig
from pipeline import run

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"

# Near-periphery slot: raw radius ~260-300px from (320,320), the region where
# the old global-rectify approach zero-filled real content (see
# docs/superpowers/specs/2026-08-26-local-gnomonic-rectification-design.md).
PERIPHERAL_POLYGON_RAW = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]


def _write_test_config(tmp_path, polygon_raw=None):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    slots = [{"id": "slot-A", "polygon_raw": polygon_raw or PERIPHERAL_POLYGON_RAW}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.6, "color": [0, 255, 0], "alpha": 0.8, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "P1_B1_1_1.json"
    config.save(str(path))
    return str(path)


def test_pipeline_composites_label_near_periphery_without_corrupting_far_pixels(tmp_path):
    config_path = _write_test_config(tmp_path)
    output_path = str(tmp_path / "final.png")

    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert raw is not None

    final = run(config_path, SAMPLE_RAW_IMAGE, "slot-A", (0.5, 0.5), output_path)

    assert final.shape == raw.shape

    pts = np.asarray(PERIPHERAL_POLYGON_RAW)
    x_min, y_min = (pts.min(axis=0) - 20).astype(int)
    x_max, y_max = (pts.max(axis=0) + 20).astype(int)
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, 640), min(y_max, 640)

    bbox_before = raw[y_min:y_max, x_min:x_max]
    bbox_after = final[y_min:y_max, x_min:x_max]
    assert not np.array_equal(bbox_before, bbox_after)

    far_before = raw[550:580, 550:580]
    far_after = final[550:580, 550:580]
    np.testing.assert_array_equal(far_before, far_after)


def test_pipeline_raises_for_unknown_slot_id(tmp_path):
    config_path = _write_test_config(tmp_path)
    with pytest.raises(ValueError):
        run(config_path, SAMPLE_RAW_IMAGE, "does-not-exist", (0.5, 0.5), str(tmp_path / "out.png"))


def test_pipeline_raises_for_unreadable_raw_image(tmp_path):
    config_path = _write_test_config(tmp_path)
    with pytest.raises(ValueError):
        run(config_path, "no_such_file.jpg", "slot-A", (0.5, 0.5), str(tmp_path / "out.png"))


def test_pipeline_raises_when_candidate_maps_outside_local_patch_bounds(tmp_path):
    config_path = _write_test_config(tmp_path)
    with pytest.raises(ValueError):
        run(config_path, SAMPLE_RAW_IMAGE, "slot-A", (5.0, 5.0), str(tmp_path / "out.png"))
```

`tests/test_main.py`에서 config 픽스처가 `polygon_rectified`를 쓰고 있다면
`polygon_raw`로, 좌표값은 위 `PERIPHERAL_POLYGON_RAW`와 동일한 값으로
교체한다. `main.py` 자체는 수정하지 않는다 — `pipeline.run()` 시그니처가
그대로이므로 CLI 계층은 영향받지 않는다.

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_main.py -v`
Expected: FAIL — `slot["polygon_raw"]` 관련 `KeyError`, 또는 기존
`undistort_image`/`redistort_image` 기반 로직이 새 스키마와 안 맞아 발생하는
에러 (구현 전이므로 실패 사유는 실제 실행 결과로 확인).

- [ ] **Step 3: 구현**

`src/pipeline.py`를 다음 내용으로 전체 교체:
```python
import cv2
import numpy as np

from calibration import LocalView
from parking_slot import SlotConfig
from perspective import plane_to_pixel_homography
from renderer import render_label

DEFAULT_PATCH_SIZE = (300, 300)
DEFAULT_LOCAL_F = 300.0


def _polygon_centroid(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    return tuple(pts.mean(axis=0))


def run(config_path, raw_image_path, slot_id, candidate_point, output_path):
    config = SlotConfig.load(config_path)
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    center_raw = _polygon_centroid(slot["polygon_raw"])
    view = LocalView.centered_on(config.calibration, center_raw, DEFAULT_PATCH_SIZE, DEFAULT_LOCAL_F)

    local_patch = view.rectify(raw)
    polygon_local = view.raw_to_local(slot["polygon_raw"])
    homography = plane_to_pixel_homography(polygon_local)
    composited_local = render_label(local_patch, homography, candidate_point, config.label_spec)
    final = view.unrectify_into(composited_local, raw)

    if not cv2.imwrite(output_path, final):
        raise ValueError(f"could not write output image to {output_path}")
    return final
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: 전체 테스트 스위트 확인**

Run: `.venv/bin/pytest -v`
Expected: 전부 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/pipeline.py tests/test_pipeline.py tests/test_main.py
git commit -m "refactor: rewrite pipeline to use per-slot LocalView instead of whole-frame rectify"
```
