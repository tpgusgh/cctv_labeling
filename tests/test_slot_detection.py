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


def test_detect_slots_matches_golden_polygon_from_real_camera():
    """Verify that polygon coordinates match pre-recorded golden values.

    This is the critical regression test for coordinate-level precision.
    It loads a real polygon that was detected in a prior session and verifies
    that re-running detect_slots on the same camera folder produces a
    polygon within a very tight tolerance (1e-9 absolute error).

    This test would FAIL if fit_quad() forces a dtype=float32 cast,
    which causes ~7.7e-6 coordinate drift on the approxPolyDP fast path.
    It also catches any other precision regressions.

    Golden polygon from review/candidates.jsonl for camera P1_B1_1_9:
    - Pre-recorded detection from a prior successful run
    - Used as a pin to catch coordinate drift across refactors
    """
    image_paths = sorted(glob.glob(f"{CAMERA_FOLDER}/*.jpg"))
    assert len(image_paths) > 10

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    detections = detect_slots(median, calibration)

    # Golden polygon from review/candidates.jsonl for this camera
    # This polygon was detected in a prior successful run and serves as
    # a regression pin. Coordinate precision matters: dtype casts that alter
    # approxPolyDP's numeric behavior will cause drift.
    golden_polygon = np.array([
        [308.06011571545224, 138.57660916268068],
        [239.98448195576762, 130.06716081775699],
        [254.12456246347682, 16.946461658595503],
        [322.200224905487, 25.455909858017737]
    ])

    # Find a detected polygon that matches the golden value within tolerance.
    # Sort by proximity to golden centroid to find the best match.
    golden_centroid = golden_polygon.mean(axis=0)
    matched = False
    for d in detections:
        detected_poly = np.array(d["polygon"])
        detected_centroid = detected_poly.mean(axis=0)
        # Quick proximity check: centroids must be within ~10 pixels
        if np.linalg.norm(detected_centroid - golden_centroid) > 10:
            continue
        # If centroids are close, check full polygon match with tight tolerance
        try:
            np.testing.assert_allclose(detected_poly, golden_polygon, atol=1e-9, rtol=1e-9,
                                        err_msg=f"Polygon coordinates diverged from golden: {detected_poly}")
            matched = True
            break
        except AssertionError:
            # This detected polygon was close but not an exact match; try the next one
            continue

    assert matched, (
        f"No detected polygon matched golden value within tolerance. "
        f"Golden polygon centroid: {golden_centroid}. "
        f"Detected centroids: {[np.array(d['polygon']).mean(axis=0) for d in detections]}"
    )
