import glob

import numpy as np

from calibration import CalibrationModel
from slot_detection import median_stack, detect_slots, fit_quad

CAMERA_FOLDER = "no_label/P1_B1_1_9"


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


def test_detect_slots_finds_plausible_candidates_in_real_camera_folder():
    image_paths = sorted(glob.glob(f"{CAMERA_FOLDER}/*.jpg"))
    assert len(image_paths) > 10

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    detections = detect_slots(median, calibration)

    # real camera with visible parking bays: not zero, not absurdly many
    assert 4 <= len(detections) <= 15
    for d in detections:
        assert len(d["polygon"]) == 4
        assert 0.0 <= d["confidence"] <= 1.0
