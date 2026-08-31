import glob

import numpy as np

from calibration import CalibrationModel
from slot_detection import median_stack, detect_slots, fit_quad, merge_detections

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


def test_merge_detections_keeps_non_overlapping_slots_from_both_lists():
    a = [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "confidence": 0.9}]
    b = [{"polygon": [[100, 100], [110, 100], [110, 110], [100, 110]], "confidence": 0.5}]

    merged = merge_detections(a, b)

    assert len(merged) == 2
    assert {tuple(map(tuple, d["polygon"])) for d in merged} == {
        ((0, 0), (10, 0), (10, 10), (0, 10)),
        ((100, 100), (110, 100), (110, 110), (100, 110)),
    }


def test_merge_detections_dedupes_overlapping_slots_keeping_higher_confidence():
    low_conf = {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "confidence": 0.4}
    high_conf = {"polygon": [[1, 1], [11, 1], [11, 11], [1, 11]], "confidence": 0.95}

    merged = merge_detections([low_conf], [high_conf])

    assert len(merged) == 1
    assert merged[0]["polygon"] == high_conf["polygon"]
    assert merged[0]["confidence"] == high_conf["confidence"]
    # both input lists had a candidate here -- independent agreement
    assert merged[0]["agreement_count"] == 2


def test_merge_detections_tags_agreement_count_of_one_for_a_lone_detection():
    a = [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "confidence": 0.9}]
    b = [{"polygon": [[100, 100], [110, 100], [110, 110], [100, 110]], "confidence": 0.5}]

    merged = merge_detections(a, b)

    assert all(d["agreement_count"] == 1 for d in merged)


def test_merge_detections_dedupes_same_centroid_despite_low_bbox_iou():
    # two detectors can fit very differently-shaped quads to the same
    # physical slot (verified against real production configs: the same
    # slot detected twice with centroids a few px apart but bbox IoU as low
    # as 0.3, both surviving the old IoU-only dedup) -- centroid proximity
    # must catch this even when bbox IoU stays under iou_threshold.
    wide_low_conf = {"polygon": [[0, 0], [40, 0], [40, 40], [0, 40]], "confidence": 0.4}
    tight_high_conf = {"polygon": [[10, 10], [30, 10], [30, 30], [10, 30]], "confidence": 0.95}
    from slot_detection import _bbox_iou, _polygon_bbox
    assert _bbox_iou(_polygon_bbox(wide_low_conf["polygon"]), _polygon_bbox(tight_high_conf["polygon"])) < 0.4

    merged = merge_detections([wide_low_conf], [tight_high_conf])

    assert len(merged) == 1
    assert merged[0]["polygon"] == tight_high_conf["polygon"]
    assert merged[0]["agreement_count"] == 2


def test_regularize_quad_squares_up_a_skewed_quad_and_keeps_a_clean_rect():
    from slot_detection import regularize_quad
    calibration = CalibrationModel(cx=320, cy=320, f=204, radius=320)

    # a clean axis-aligned rect near the center must come back essentially unchanged
    rect = [[300.0, 200.0], [340.0, 200.0], [340.0, 290.0], [300.0, 290.0]]
    snapped = np.array(regularize_quad(rect, calibration))
    assert snapped.shape == (4, 2)
    assert np.all(np.isfinite(snapped))
    assert np.allclose(sorted(snapped[:, 0]), sorted(np.array(rect)[:, 0]), atol=3.0)

    # a skewed parallelogram (straight slot fitted diagonally in raw space)
    # must snap to a rectangle: in the local view its corners form right angles
    skewed = [[300.0, 200.0], [340.0, 208.0], [332.0, 290.0], [292.0, 282.0]]
    out = np.array(regularize_quad(skewed, calibration))
    from calibration import LocalView
    view = LocalView.centered_on(calibration, tuple(out.mean(axis=0)), (300, 300), 300.0)
    local = np.array(view.raw_to_local(out))
    for i in range(4):
        a = local[i] - local[(i + 1) % 4]
        b = local[(i + 2) % 4] - local[(i + 1) % 4]
        cos = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos < 0.05  # right angle within ~3 degrees


def test_is_degenerate_quad_flags_bowtie_and_keeps_tilted_thin_slot():
    from slot_detection import is_degenerate_quad
    # bowtie: corners in Z order self-intersect, shoelace area collapses
    bowtie = [[0, 0], [40, 0], [0, 30], [40, 30]]
    assert is_degenerate_quad(bowtie)
    # folded/near-flat sliver
    flat = [[0, 0], [80, 2], [82, 6], [1, 4]]
    assert is_degenerate_quad(flat)
    # a real fisheye-edge slot: thin (16px) and heavily tilted, but a clean
    # convex quad -- must NOT be flagged (verified real slots like this exist)
    tilted_thin = [[100, 100], [116, 104], [96, 190], [80, 186]]
    assert not is_degenerate_quad(tilted_thin)
    # ordinary upright slot
    normal = [[0, 0], [40, 0], [40, 90], [0, 90]]
    assert not is_degenerate_quad(normal)


def test_merge_detections_handles_empty_lists():
    assert merge_detections([], []) == []
    only = [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "confidence": 0.7}]
    merged = merge_detections(only, [])
    assert len(merged) == 1
    assert merged[0]["polygon"] == only[0]["polygon"]
    assert merged[0]["agreement_count"] == 1


