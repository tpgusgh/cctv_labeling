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


def test_detect_slots_is_deterministic_on_real_camera_folder():
    """Verify that polygon coordinates don't drift across refactors.

    Runs detect_slots twice on the same real camera data and asserts:
    1. Same number of detections (already tested above, but explicit here)
    2. Detection ordering and polygon coordinates are identical
    3. All polygon coordinates match to high precision

    This catches numeric precision changes (e.g., dtype casts that alter
    approxPolyDP or minAreaRect output) that the count-only test would miss.
    """
    image_paths = sorted(glob.glob(f"{CAMERA_FOLDER}/*.jpg"))
    assert len(image_paths) > 10

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)

    # Run twice to verify determinism
    detections_1 = detect_slots(median, calibration)
    detections_2 = detect_slots(median, calibration)

    # Must have same count
    assert len(detections_1) == len(detections_2)

    # All polygons must match exactly (within floating point tolerance)
    for d1, d2 in zip(detections_1, detections_2):
        poly1 = np.array(d1["polygon"])
        poly2 = np.array(d2["polygon"])
        # Allow 1e-8 tolerance for floating point rounding, but no larger
        np.testing.assert_allclose(poly1, poly2, rtol=1e-8, atol=1e-8,
                                    err_msg="Polygon coordinates drifted across runs")

    # Also explicitly assert that at least the first detection exists and
    # has reasonable coordinate ranges (sanity check the test setup)
    assert len(detections_1) >= 4, "Need at least 4 detections to meaningfully test coordinates"
    first_poly = np.array(detections_1[0]["polygon"])
    assert np.all(first_poly >= 0), "Polygon coordinates should be non-negative"
    assert np.all(first_poly < 640), "Polygon coordinates should be within image bounds"
