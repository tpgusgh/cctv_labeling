import cv2
import numpy as np
import pytest

from calibration import CalibrationModel
from parking_slot import SlotConfig
from pipeline import run, run_auto

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


CAR_SAMPLE_IMAGE = "no_label/P1_B1_1_21/20260820_115029.jpg"
# The windshield blob detected by detect_windshields() on this image has
# bbox (283, 418, 62, 70) / centroid ~(316, 456) -- a white car with a black
# windshield roughly centered-low in frame (verified visually). This box is
# that bbox expanded by a 40px margin in each direction: large enough to
# comfortably contain the blob (with room for detection jitter) but still
# small enough that its raw polygon projects inside the 300x300 local
# rectified patch used by _prepare_slot_view (DEFAULT_PATCH_SIZE/DEFAULT_LOCAL_F).
# A larger, more "visually generous" box (e.g. 60-100px margin, or the
# original coarse estimate [[200,260],[460,260],[460,500],[200,500]]) maps
# outside the local patch bounds and makes _prepare_slot_view raise ValueError.
CAR_SLOT_POLYGON_RAW = [[243.0, 378.0], [385.0, 378.0], [385.0, 528.0], [243.0, 528.0]]
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
