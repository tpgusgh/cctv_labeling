# YOLOv8-seg Slot Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a YOLOv8-seg-based whole-image slot detector that plugs into the existing `generate_config.py` pipeline as an opt-in alternative to the current classical-CV `detect_slots()`, trained on the polygon labels already collected in `review/labels.jsonl` and `review/missed.jsonl`.

**Architecture:** New `src/yolo_slot_detector.py` module returns the exact same `[{"polygon": [[x,y]x4], "confidence": float}]` contract as `slot_detection.detect_slots()`, so `generate_config.py` can dispatch to either detector via a new `yolo_model=None` parameter without any other file (pipeline, web app, review UI) changing. A shared `fit_quad()` helper (extracted from `slot_detection.py`) guarantees both detectors always emit exactly 4 points, since `perspective.plane_to_pixel_homography` requires that downstream. Two new CLI scripts (`export_yolo_dataset.py`, `train_yolo_seg.py`) turn the existing review data into a YOLO-seg training set and fine-tune a COCO-pretrained `yolov8n-seg.pt` checkpoint on it.

**Tech Stack:** Python, OpenCV, ultralytics (YOLOv8-seg), PyTorch (GPU: CUDA/MPS available).

**Spec:** `docs/superpowers/specs/2026-08-27-yolov8-seg-slot-detection-design.md`

## Global Constraints

- No per-camera manual coordinate entry, ever — this migration only changes *how* polygons are auto-detected, never adds a way for a human to type/drag coordinates.
- Supervised learning only, no reinforcement learning.
- `generate_config.py`'s existing behavior (no `yolo_model` passed) must stay byte-for-byte identical — this is an opt-in addition, not a replacement.
- Every new detector must return exactly 4-point polygons — `perspective.plane_to_pixel_homography` (`src/perspective.py:9-10`) raises `ValueError` on anything else.
- `pipeline.py`, `web/backend/*`, and the review UI are out of scope for this plan — not touched.

---

### Task 1: Extract reusable `fit_quad()` helper from `slot_detection.py`

**Files:**
- Modify: `src/slot_detection.py:96-161` (the `detect_slots` function body)
- Test: `tests/test_slot_detection.py`

**Interfaces:**
- Produces: `fit_quad(cnt, inset_px=6.0) -> np.ndarray` shape `(4, 2)` float64 — importable as `from slot_detection import fit_quad`. Used by Task 3's `yolo_slot_detector.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_slot_detection.py`:

```python
import numpy as np

from slot_detection import fit_quad


def test_fit_quad_returns_four_points_for_a_quad_contour():
    cnt = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    poly = fit_quad(cnt, inset_px=0.0)
    assert poly.shape == (4, 2)


def test_fit_quad_returns_four_points_for_a_non_quad_contour():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cnt = np.stack([50 + 40 * np.cos(angles), 50 + 40 * np.sin(angles)], axis=1).astype(np.float32)
    poly = fit_quad(cnt, inset_px=0.0)
    assert poly.shape == (4, 2)


def test_fit_quad_shrinks_toward_centroid():
    cnt = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    poly = fit_quad(cnt, inset_px=10.0)
    centroid = poly.mean(axis=0)
    assert np.all(np.abs(centroid - [50, 50]) < 1.0)
    # every corner moved inward, so max coordinate spread shrank
    assert poly[:, 0].max() < 100 and poly[:, 1].max() < 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_slot_detection.py -k fit_quad -v`
Expected: FAIL with `ImportError: cannot import name 'fit_quad'`

- [ ] **Step 3: Add `fit_quad()` and refactor `detect_slots()` to use it**

In `src/slot_detection.py`, add this function right after `_shrink_polygon` (currently ending at line 93):

```python
def fit_quad(cnt, inset_px=6.0):
    """Reduce an arbitrary contour/polygon to a 4-point quad.

    perspective.plane_to_pixel_homography requires exactly 4 points, but a
    real contour (classical CV connected-component boundary, or a YOLO-seg
    mask polygon) rarely comes out as a clean quad. Try a 4-corner
    approximation first; if the shape doesn't reduce to 4 corners, fall back
    to its minimum-area bounding rect. Either way, shrink the result inward
    so it doesn't overshoot the true painted line (see _shrink_polygon).
    """
    cnt = np.asarray(cnt, dtype=np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        poly = approx.reshape(-1, 2)
    else:
        rect = cv2.minAreaRect(cnt)
        poly = cv2.boxPoints(rect)
    return _shrink_polygon(poly, inset_px)
```

Then in `detect_slots()` (`src/slot_detection.py:96-161`), replace this block:

```python
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < min_rectangularity:
            continue
        poly = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(rect)
        poly = _shrink_polygon(poly, inset_px)
```

with:

```python
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < min_rectangularity:
            continue
        poly = fit_quad(cnt, inset_px)
```

(Rectangularity stays computed inline here since it's a classical-CV-specific filter, not part of the generic quad-fit — `fit_quad` only owns the "make it 4 points" job.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_slot_detection.py -v`
Expected: all PASS, including the pre-existing `test_detect_slots_finds_plausible_candidates_in_real_camera_folder` (this confirms the refactor didn't change `detect_slots()` behavior).

- [ ] **Step 5: Run full test suite as a regression check**

Run: `pytest -q`
Expected: same pass count as before this change, no new failures.

- [ ] **Step 6: Commit**

```bash
git add src/slot_detection.py tests/test_slot_detection.py
git commit -m "refactor: extract fit_quad() helper from detect_slots for reuse"
```

---

### Task 2: Add ultralytics/PyTorch dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `ultralytics.YOLO` importable in the environment, used by Task 3 and Task 5.

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt`:

```
ultralytics>=8.0
```

(`ultralytics` pulls in `torch`/`torchvision` transitively — no need to pin them separately. If `torch` doesn't pick up your GPU (CUDA/MPS) automatically after this install, follow the platform-specific install command at https://pytorch.org/get-started/locally/ and re-run.)

- [ ] **Step 2: Install and verify**

Run: `pip install -r requirements.txt`
Run: `python -c "import ultralytics, torch; print('cuda:', torch.cuda.is_available()); print('mps:', torch.backends.mps.is_available())"`
Expected: no import errors, and at least one of `cuda`/`mps` prints `True` on your GPU machine.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add ultralytics/torch dependency for YOLOv8-seg detection"
```

---

### Task 3: `src/yolo_slot_detector.py` — YOLOv8-seg drop-in detector

**Files:**
- Create: `src/yolo_slot_detector.py`
- Test: `tests/test_yolo_slot_detector.py`

**Interfaces:**
- Consumes: `fit_quad(cnt, inset_px=6.0)` from `slot_detection.py` (Task 1).
- Produces: `load(model_path) -> ultralytics.YOLO`; `detect_slots(image_bgr, model, conf=0.25) -> list[{"polygon": [[x,y]x4], "confidence": float}]`. Used by Task 6's `generate_config.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_yolo_slot_detector.py`:

```python
import numpy as np
import pytest

import yolo_slot_detector


class _FakeConf:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeMasks:
    def __init__(self, polygons):
        self.xy = polygons


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _FakeConf(confs)


class _FakeResult:
    def __init__(self, polygons, confs):
        self.masks = _FakeMasks(polygons) if polygons else None
        self.boxes = _FakeBoxes(confs)


class _FakeModel:
    def __init__(self, polygons, confs):
        self._polygons = polygons
        self._confs = confs

    def predict(self, image, conf, verbose):
        return [_FakeResult(self._polygons, self._confs)]


def test_detect_slots_converts_mask_polygons_to_quads():
    polygon = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.float32)
    model = _FakeModel(polygons=[polygon], confs=[0.87])

    detections = yolo_slot_detector.detect_slots(np.zeros((100, 150, 3), dtype=np.uint8), model)

    assert len(detections) == 1
    assert len(detections[0]["polygon"]) == 4
    assert detections[0]["confidence"] == 0.87


def test_detect_slots_returns_empty_list_when_no_masks():
    model = _FakeModel(polygons=None, confs=[])

    detections = yolo_slot_detector.detect_slots(np.zeros((100, 150, 3), dtype=np.uint8), model)

    assert detections == []


def test_load_constructs_YOLO_with_given_path(monkeypatch):
    captured = {}

    class _FakeYOLO:
        def __init__(self, path):
            captured["path"] = path

    monkeypatch.setattr(yolo_slot_detector, "YOLO", _FakeYOLO)

    yolo_slot_detector.load("models/yolov8_seg_slots.pt")

    assert captured["path"] == "models/yolov8_seg_slots.pt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_yolo_slot_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'yolo_slot_detector'`

- [ ] **Step 3: Write the implementation**

Create `src/yolo_slot_detector.py`:

```python
from ultralytics import YOLO

from slot_detection import fit_quad


def load(model_path):
    return YOLO(str(model_path))


def detect_slots(image_bgr, model, conf=0.25):
    """Whole-image slot detection via a fine-tuned YOLOv8-seg model.

    Returns the same [{"polygon": [[x,y]x4], "confidence": float}] contract
    as slot_detection.detect_slots() -- callers (generate_config.py) swap
    detectors without anything downstream (pipeline.py, the web app, the
    review UI) knowing the difference.
    """
    results = model.predict(image_bgr, conf=conf, verbose=False)
    result = results[0]
    if result.masks is None:
        return []

    detections = []
    confidences = result.boxes.conf.tolist()
    for polygon_pts, confidence in zip(result.masks.xy, confidences):
        poly = fit_quad(polygon_pts, inset_px=6.0)
        detections.append({"polygon": poly.tolist(), "confidence": round(float(confidence), 3)})
    return detections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_yolo_slot_detector.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/yolo_slot_detector.py tests/test_yolo_slot_detector.py
git commit -m "feat: add YOLOv8-seg drop-in slot detector"
```

---

### Task 4: `src/export_yolo_dataset.py` — build a YOLO-seg dataset from review data

**Files:**
- Create: `src/export_yolo_dataset.py`
- Test: `tests/test_export_yolo_dataset.py`

**Interfaces:**
- Consumes: `review_store.load_labels(path)`, `review_store.load_missed_annotations(path)` (existing, `src/review_store.py:52-53,73-74`) — each record has `camera_id` (str), `polygon` (`[[x,y]x4]`), and (labels only) `decision` (`"accept"`/`"reject"`).
- Produces: `export_dataset(no_label_dir, output_dir, labels_path, missed_path, image_extensions=(".jpg", ".jpeg", ".png"), val_every=5) -> dict` with keys `train_cameras`, `val_cameras`, `train_images`, `val_images` (all counts/lists). Writes `output_dir/images/{train,val}/`, `output_dir/labels/{train,val}/`, `output_dir/dataset.yaml`. Used by Task 5 (consumes the `dataset.yaml` it writes).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_yolo_dataset.py`:

```python
import json

import cv2
import numpy as np
import pytest

import export_yolo_dataset


def _write_frame(path, size=20):
    cv2.imwrite(str(path), np.zeros((size, size, 3), dtype=np.uint8))


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_export_dataset_replicates_camera_polygons_across_all_its_frames(tmp_path):
    no_label_dir = tmp_path / "no_label"
    (no_label_dir / "camA").mkdir(parents=True)
    (no_label_dir / "camB").mkdir(parents=True)
    _write_frame(no_label_dir / "camA" / "frame1.jpg")
    _write_frame(no_label_dir / "camA" / "frame2.jpg")
    _write_frame(no_label_dir / "camB" / "frame1.jpg")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": "l1", "camera_id": "camA", "decision": "accept",
         "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"id": "l2", "camera_id": "camA", "decision": "reject",
         "polygon": [[5, 5], [15, 5], [15, 15], [5, 15]]},
    ])
    missed_path = tmp_path / "missed.jsonl"
    _write_jsonl(missed_path, [
        {"id": "m1", "camera_id": "camB", "polygon": [[2, 2], [8, 2], [8, 8], [2, 8]]},
    ])

    output_dir = tmp_path / "dataset"
    summary = export_yolo_dataset.export_dataset(
        no_label_dir=no_label_dir, output_dir=output_dir,
        labels_path=labels_path, missed_path=missed_path, val_every=2)

    # camA is index 0 of sorted(["camA", "camB"]) -> val; camB -> train
    assert summary["val_cameras"] == ["camA"]
    assert summary["train_cameras"] == ["camB"]
    assert summary["val_images"] == 2  # camA has 2 frames
    assert summary["train_images"] == 1  # camB has 1 frame

    assert (output_dir / "images" / "val" / "camA__frame1.jpg").exists()
    assert (output_dir / "images" / "val" / "camA__frame2.jpg").exists()
    assert (output_dir / "images" / "train" / "camB__frame1.jpg").exists()

    # camA's accept polygon [[0,0],[10,0],[10,10],[0,10]] on a 20x20 image
    # normalizes to (0,0) (0.5,0) (0.5,0.5) (0,0.5); the reject polygon must
    # NOT appear.
    label_text = (output_dir / "labels" / "val" / "camA__frame1.txt").read_text().strip()
    lines = label_text.splitlines()
    assert len(lines) == 1
    parts = [float(x) for x in lines[0].split()]
    assert parts[0] == 0  # class id
    assert parts[1:] == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5])

    # camB's missed-annotation polygon [[2,2],[8,2],[8,8],[2,8]] on 20x20
    label_text = (output_dir / "labels" / "train" / "camB__frame1.txt").read_text().strip()
    parts = [float(x) for x in label_text.split()]
    assert parts[1:] == pytest.approx([0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4])

    dataset_yaml = (output_dir / "dataset.yaml").read_text()
    assert "train: images/train" in dataset_yaml
    assert "val: images/val" in dataset_yaml
    assert "parking_slot" in dataset_yaml


def test_export_dataset_raises_when_fewer_than_two_labeled_cameras(tmp_path):
    no_label_dir = tmp_path / "no_label"
    (no_label_dir / "camA").mkdir(parents=True)
    _write_frame(no_label_dir / "camA" / "frame1.jpg")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": "l1", "camera_id": "camA", "decision": "accept",
         "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    ])
    missed_path = tmp_path / "missed_empty.jsonl"

    with pytest.raises(ValueError):
        export_yolo_dataset.export_dataset(
            no_label_dir=no_label_dir, output_dir=tmp_path / "dataset",
            labels_path=labels_path, missed_path=missed_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export_yolo_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_yolo_dataset'`

- [ ] **Step 4: Write the implementation**

Create `src/export_yolo_dataset.py`:

```python
import argparse
import glob
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

import review_store

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _polygons_by_camera(labels_path, missed_path):
    by_camera = defaultdict(list)
    for record in review_store.load_labels(labels_path):
        if record.get("decision") == "accept":
            by_camera[record["camera_id"]].append(record["polygon"])
    for record in review_store.load_missed_annotations(missed_path):
        by_camera[record["camera_id"]].append(record["polygon"])
    return by_camera


def _write_label_file(path, polygons, width, height):
    lines = []
    for polygon in polygons:
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in polygon)
        lines.append(f"0 {coords}")
    path.write_text("\n".join(lines) + "\n")


def export_dataset(no_label_dir, output_dir, labels_path=review_store.LABELS_PATH,
                    missed_path=review_store.MISSED_PATH, image_extensions=IMAGE_EXTENSIONS, val_every=5):
    no_label_dir = Path(no_label_dir)
    output_dir = Path(output_dir)

    polygons_by_camera = _polygons_by_camera(labels_path, missed_path)
    camera_ids = sorted(polygons_by_camera.keys())
    if len(camera_ids) < 2:
        raise ValueError(
            f"need at least 2 labeled cameras to make a train/val split, found {len(camera_ids)}")

    val_cameras = camera_ids[::val_every] or [camera_ids[-1]]
    train_cameras = [c for c in camera_ids if c not in val_cameras]

    summary = {"train_cameras": train_cameras, "val_cameras": val_cameras,
               "train_images": 0, "val_images": 0}

    for split, cameras in (("train", train_cameras), ("val", val_cameras)):
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in cameras:
            polygons = polygons_by_camera[camera_id]
            frame_paths = sorted(
                {p for ext in image_extensions for p in glob.glob(str(no_label_dir / camera_id / f"*{ext}"))})
            for frame_path in frame_paths:
                frame_path = Path(frame_path)
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                height, width = image.shape[:2]

                dest_name = f"{camera_id}__{frame_path.stem}"
                shutil.copy2(frame_path, images_dir / f"{dest_name}{frame_path.suffix}")
                _write_label_file(labels_dir / f"{dest_name}.txt", polygons, width, height)
                summary[f"{split}_images"] += 1

    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: parking_slot\n"
    )

    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export review_store's labeled polygons into a YOLO-seg training dataset.")
    parser.add_argument("--no-label-dir", default="no_label")
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", default=str(review_store.LABELS_PATH))
    parser.add_argument("--missed", default=str(review_store.MISSED_PATH))
    parser.add_argument("--val-every", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = export_dataset(args.no_label_dir, args.output, args.labels, args.missed, val_every=args.val_every)
    print(f"train: {summary['train_images']} images across {len(summary['train_cameras'])} cameras")
    print(f"val: {summary['val_images']} images across {len(summary['val_cameras'])} cameras")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_export_yolo_dataset.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/export_yolo_dataset.py tests/test_export_yolo_dataset.py
git commit -m "feat: export review_store labels into a YOLO-seg training dataset"
```

---

### Task 5: `src/train_yolo_seg.py` — fine-tune YOLOv8-seg on the exported dataset

**Files:**
- Create: `src/train_yolo_seg.py`
- Test: `tests/test_train_yolo_seg.py`

**Interfaces:**
- Consumes: `dataset.yaml` produced by Task 4's `export_dataset()`.
- Produces: `train(data_yaml, base_model="yolov8n-seg.pt", epochs=100) -> ultralytics.YOLO` (trained model object). CLI writes the checkpoint to `models/yolov8_seg_slots.pt` by default — that path is what Task 6 (via a human running `generate_config.py --yolo-model ...`) will point at.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_yolo_seg.py`:

```python
import train_yolo_seg


def test_train_calls_YOLO_train_with_expected_args(monkeypatch):
    calls = {}

    class _FakeYOLO:
        def __init__(self, base_model):
            calls["base_model"] = base_model

        def train(self, **kwargs):
            calls["train_kwargs"] = kwargs

    monkeypatch.setattr(train_yolo_seg, "YOLO", _FakeYOLO)

    train_yolo_seg.train("dataset.yaml", base_model="yolov8n-seg.pt", epochs=5)

    assert calls["base_model"] == "yolov8n-seg.pt"
    assert calls["train_kwargs"] == {"data": "dataset.yaml", "epochs": 5, "single_cls": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_yolo_seg.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_yolo_seg'`

- [ ] **Step 3: Write the implementation**

Create `src/train_yolo_seg.py`:

```python
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(data_yaml, base_model="yolov8n-seg.pt", epochs=100):
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, single_cls=True)
    return model


def _save_checkpoint(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(output_path))
    except AttributeError:
        # ponytail: fallback for ultralytics versions where YOLO has no
        # top-level .save() -- the trainer always writes its best checkpoint
        # to <save_dir>/weights/best.pt regardless.
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        shutil.copy2(best, output_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a COCO-pretrained YOLOv8-seg checkpoint on an exported parking-slot dataset.")
    parser.add_argument("--data", required=True, help="path to dataset.yaml from export_yolo_dataset.py")
    parser.add_argument("--base-model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="models/yolov8_seg_slots.pt")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = train(args.data, args.base_model, args.epochs)
    _save_checkpoint(model, args.output)
    print(f"trained -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_yolo_seg.py -v`
Expected: PASS

- [ ] **Step 5: Manual smoke test (real ultralytics, real data — not automated)**

This step needs a real checkpoint download and real training, so it isn't a pytest step. Run it once by hand before trusting this module:

```bash
python src/export_yolo_dataset.py --output /tmp/yolo_dataset
python src/train_yolo_seg.py --data /tmp/yolo_dataset/dataset.yaml --epochs 1 --output /tmp/smoke_model.pt
python -c "from ultralytics import YOLO; YOLO('/tmp/smoke_model.pt')"
```

Expected: all three commands succeed, `/tmp/smoke_model.pt` exists, and the final line loads without error. If `model.save()` raises something other than `AttributeError`, or `model.trainer.save_dir` doesn't exist on your installed ultralytics version, fix `_save_checkpoint()` to match what your installed version actually exposes (check `pip show ultralytics` version and its docs) before moving on.

- [ ] **Step 6: Commit**

```bash
git add src/train_yolo_seg.py tests/test_train_yolo_seg.py
git commit -m "feat: add YOLOv8-seg training script"
```

---

### Task 6: Wire `yolo_model` into `generate_config.py`

**Files:**
- Modify: `src/generate_config.py:1-86` (full file — imports, `generate_config()`, `build_parser()`, `main()`)
- Test: `tests/test_generate_config.py` (new file)

**Interfaces:**
- Consumes: `yolo_slot_detector.load(path)`, `yolo_slot_detector.detect_slots(median, model)` (Task 3).
- Produces: `generate_config(camera_id, frames_dir, output_path, cx=320.0, cy=320.0, f=204.0, radius=320.0, image_width=640, image_height=640, classifier=None, yolo_model=None)` — same return `(slots, needs_review)` as before; `yolo_model` is a new trailing parameter, default `None` preserves all existing behavior/callers (e.g. `web/backend/jobs.py:60-61` keeps working unchanged).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_config.py`:

```python
import json

import cv2
import numpy as np

import generate_config


def _make_frames_dir(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame1.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
    return frames_dir


def test_generate_config_uses_yolo_model_when_given(tmp_path, monkeypatch):
    frames_dir = _make_frames_dir(tmp_path)
    fake_polygon = [[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]]
    captured = {}

    def _fake_detect_slots(median_bgr, model, conf=0.25):
        captured["model"] = model
        return [{"polygon": fake_polygon, "confidence": 0.9}]

    monkeypatch.setattr(generate_config.yolo_slot_detector, "detect_slots", _fake_detect_slots)

    sentinel_model = object()
    output_path = tmp_path / "config.json"
    slots, needs_review = generate_config.generate_config(
        "cam-1", str(frames_dir), str(output_path), yolo_model=sentinel_model)

    assert captured["model"] is sentinel_model
    assert slots == [{"id": "slot-0", "polygon_raw": fake_polygon}]
    assert needs_review is False
    assert json.loads(output_path.read_text())["camera_id"] == "cam-1"


def test_generate_config_uses_classical_detection_when_no_yolo_model(tmp_path, monkeypatch):
    frames_dir = _make_frames_dir(tmp_path)
    called = {}

    def _fake_yolo_detect_slots(median_bgr, model, conf=0.25):
        called["yolo"] = True
        return []

    monkeypatch.setattr(generate_config.yolo_slot_detector, "detect_slots", _fake_yolo_detect_slots)

    output_path = tmp_path / "config.json"
    generate_config.generate_config("cam-1", str(frames_dir), str(output_path))

    assert "yolo" not in called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_config.py -v`
Expected: FAIL — `generate_config()` raises `TypeError: generate_config() got an unexpected keyword argument 'yolo_model'`

- [ ] **Step 3: Modify `generate_config.py`**

Add the import near the top of `src/generate_config.py` (alongside the existing `from slot_detection import median_stack, detect_slots` at line 12):

```python
import yolo_slot_detector
```

Replace the body of `generate_config()` (`src/generate_config.py:40-56`):

```python
def generate_config(camera_id, frames_dir, output_path, cx=320.0, cy=320.0, f=204.0, radius=320.0,
                     image_width=640, image_height=640, classifier=None):
    image_paths = sorted({p for pattern in IMAGE_EXTENSIONS for p in glob.glob(str(Path(frames_dir) / pattern))})
    if not image_paths:
        raise ValueError(f"no frames found in {frames_dir}")

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=cx, cy=cy, f=f, radius=radius)
    detections = detect_slots(median, calibration, classifier=classifier)
    _save_review_candidates(camera_id, image_paths[0], median, detections)

    slots = [{"id": f"slot-{i}", "polygon_raw": d["polygon"]} for i, d in enumerate(detections)]
    config = SlotConfig(camera_id, image_width, image_height, calibration, slots, DEFAULT_LABEL_SPEC)
    config.save(output_path)

    needs_review = len(detections) == 0 or any(d["confidence"] < REVIEW_CONFIDENCE_THRESHOLD for d in detections)
    return slots, needs_review
```

with:

```python
def generate_config(camera_id, frames_dir, output_path, cx=320.0, cy=320.0, f=204.0, radius=320.0,
                     image_width=640, image_height=640, classifier=None, yolo_model=None):
    image_paths = sorted({p for pattern in IMAGE_EXTENSIONS for p in glob.glob(str(Path(frames_dir) / pattern))})
    if not image_paths:
        raise ValueError(f"no frames found in {frames_dir}")

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=cx, cy=cy, f=f, radius=radius)
    if yolo_model is not None:
        detections = yolo_slot_detector.detect_slots(median, yolo_model)
    else:
        detections = detect_slots(median, calibration, classifier=classifier)
    _save_review_candidates(camera_id, image_paths[0], median, detections)

    slots = [{"id": f"slot-{i}", "polygon_raw": d["polygon"]} for i, d in enumerate(detections)]
    config = SlotConfig(camera_id, image_width, image_height, calibration, slots, DEFAULT_LABEL_SPEC)
    config.save(output_path)

    needs_review = len(detections) == 0 or any(d["confidence"] < REVIEW_CONFIDENCE_THRESHOLD for d in detections)
    return slots, needs_review
```

Add a `--yolo-model` CLI option. Replace `build_parser()` (`src/generate_config.py:59-72`):

```python
    parser.add_argument("--model", default=None, help="path to a trained models/slot_classifier.joblib (optional)")
    return parser
```

with:

```python
    parser.add_argument("--model", default=None, help="path to a trained models/slot_classifier.joblib (optional)")
    parser.add_argument("--yolo-model", default=None,
                         help="path to a trained models/yolov8_seg_slots.pt (optional, overrides --model)")
    return parser
```

Replace `main()` (`src/generate_config.py:75-82`):

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    classifier = slot_classifier.load(args.model) if args.model else None
    slots, needs_review = generate_config(args.camera_id, args.frames_dir, args.output, args.cx, args.cy, args.f,
                                           args.radius, args.image_width, args.image_height, classifier)
    status = "REVIEW RECOMMENDED (low confidence or no slots found)" if needs_review else "ok"
    print(f"{args.camera_id}: detected {len(slots)} slot(s) -> {args.output} [{status}]")
```

with:

```python
def main(argv=None):
    args = build_parser().parse_args(argv)
    classifier = slot_classifier.load(args.model) if args.model else None
    yolo_model = yolo_slot_detector.load(args.yolo_model) if args.yolo_model else None
    slots, needs_review = generate_config(args.camera_id, args.frames_dir, args.output, args.cx, args.cy, args.f,
                                           args.radius, args.image_width, args.image_height, classifier, yolo_model)
    status = "REVIEW RECOMMENDED (low confidence or no slots found)" if needs_review else "ok"
    print(f"{args.camera_id}: detected {len(slots)} slot(s) -> {args.output} [{status}]")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_config.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite as a regression check**

Run: `pytest -q`
Expected: same pass count as the end of Task 1 plus the new tests from Tasks 3, 4, 5, 6 — no failures.

- [ ] **Step 6: Commit**

```bash
git add src/generate_config.py tests/test_generate_config.py
git commit -m "feat: wire optional YOLOv8-seg detector into generate_config"
```

---

## Self-Review Notes

- **Spec coverage:** every spec component has a task — `yolo_slot_detector.py` (Task 3), `export_yolo_dataset.py` (Task 4), `train_yolo_seg.py` (Task 5), `generate_config.py` extension (Task 6), shared quad-fit reuse (Task 1), new dependency (Task 2). The spec's "범위 밖" items (pipeline.py, web app, review UI, RL, Active Learning, per-frame re-detection, removing `slot_classifier.py`) are deliberately untouched by every task above.
- **Type/interface consistency:** `detect_slots(image_bgr, model, conf=0.25)` signature is identical between the spec, Task 3's implementation, and Task 6's test mocks. `fit_quad(cnt, inset_px=6.0)` signature matches between Task 1's implementation and Task 3's usage. `export_dataset(...)` return dict keys (`train_cameras`, `val_cameras`, `train_images`, `val_images`) match between Task 4's implementation and its own test.
- **No placeholders:** every step has real, complete code — the one intentionally-manual step (Task 5, Step 5) is manual because the spec itself says real-checkpoint training/accuracy verification can't be a deterministic unit test, not because anything was left vague.
