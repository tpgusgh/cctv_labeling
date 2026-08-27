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


def test_fit_quad_does_not_force_a_lossy_float32_cast():
    """Verify that fit_quad preserves contour dtype, not forcing float32.

    2**24 + 1 (16777217) is exactly representable in int32 but NOT in
    float32 (24-bit mantissa rounds it to 16777216 or 16777218). A forced
    float32 cast would silently corrupt it.

    fit_quad must preserve the caller's dtype instead of forcing a cast,
    so cv2.arcLength/approxPolyDP/minAreaRect see the exact int32 value.

    This test discriminates the bug: would FAIL if fit_quad() forced
    dtype=float32, and PASSES with the fix (no forced cast).
    """
    big = 2 ** 24 + 1
    cnt = np.array([[big, 0], [big + 100, 0], [big + 100, 100], [big, 100]], dtype=np.int32)
    poly = fit_quad(cnt, inset_px=0.0)

    # Check that output contains the exact value (not rounded by float32 conversion).
    # The critical check: if float32 was forced, the corners would be
    # 16777216.0 instead of 16777217. Convert to int and check for exact match.
    poly_as_int = poly[:, 0].astype(np.int32)
    assert big in poly_as_int, (
        f"fit_quad forced a lossy dtype cast: "
        f"expected {big} in corner x-coords, got {poly[:, 0]}"
    )


def test_fit_quad_accepts_float64_input_without_crashing():
    cnt = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float64)
    poly = fit_quad(cnt, inset_px=0.0)
    assert poly.shape == (4, 2)


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


