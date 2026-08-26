import cv2
import numpy as np
import pytest

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

    baseline = calibration.redistort_image(
        calibration.undistort_image(raw), output_shape=raw.shape[:2]
    )
    mask = np.ones(raw.shape[:2], dtype=bool)
    mask[y_min:y_max, x_min:x_max] = False  # exclude the label's raw-space bbox
    np.testing.assert_array_equal(final[mask], baseline[mask])


def test_pipeline_raises_for_unknown_slot_id(tmp_path):
    config_path, _, _ = _write_test_config(tmp_path)
    output_path = str(tmp_path / "final.png")

    with pytest.raises(ValueError):
        run(config_path, SAMPLE_RAW_IMAGE, "slot-does-not-exist", (0.5, 0.5), output_path)


def test_pipeline_raises_for_unreadable_raw_image(tmp_path):
    config_path, _, _ = _write_test_config(tmp_path)
    output_path = str(tmp_path / "final.png")

    with pytest.raises(ValueError):
        run(config_path, str(tmp_path / "does_not_exist.jpg"), "slot-A", (0.5, 0.5), output_path)


def test_pipeline_raises_when_candidate_maps_outside_rectified_bounds(tmp_path):
    config_path, _, _ = _write_test_config(tmp_path)
    output_path = str(tmp_path / "final.png")

    with pytest.raises(ValueError):
        run(config_path, SAMPLE_RAW_IMAGE, "slot-A", (5.0, 5.0), output_path)
