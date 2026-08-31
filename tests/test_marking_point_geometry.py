import numpy as np
import pytest

from calibration import CalibrationModel, LocalView
from marking_point_geometry import derive_marking_points, reconstruct_slot_quad


def _same_quad(actual, expected, abs_tol=1e-6):
    """Two quads describe the same rectangle if every expected point has a
    matching actual point nearby -- point order/winding isn't guaranteed to
    survive the derive -> reconstruct round trip, only the shape is."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    used = set()
    for pt in expected:
        dists = np.linalg.norm(actual - pt, axis=1)
        for i in np.argsort(dists):
            if i not in used and dists[i] < abs_tol:
                used.add(i)
                break
        else:
            return False
    return True


def test_reconstruct_reproduces_a_clean_rectangle_from_its_derived_marking_points():
    polygon = [[0, 0], [10, 0], [10, 30], [0, 30]]

    p1, p2, depth = derive_marking_points(polygon)
    quad = reconstruct_slot_quad(p1, p2, depth)

    assert _same_quad(quad, polygon)


def test_reconstruct_reproduces_a_rotated_rectangle():
    # same rectangle, rotated 30 degrees and translated
    theta = np.deg2rad(30)
    rect = np.array([[0, 0], [10, 0], [10, 30], [0, 30]], dtype=np.float64)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    polygon = (rect @ rotation.T) + np.array([50, 80])

    p1, p2, depth = derive_marking_points(polygon.tolist())
    quad = reconstruct_slot_quad(p1, p2, depth)

    assert _same_quad(quad, polygon, abs_tol=1e-4)


def test_reconstruct_reproduces_a_fisheye_distorted_rectangle_near_the_edge_with_calibration():
    # a real rectangle only looks like a clean rectangle in a locally
    # rectified tangent plane -- project one through the lens model to get
    # what it would actually look like in raw pixels near the radius=320
    # edge (this project's slots sit there), where a flat pixel-space
    # rotation points the wrong way.
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    view = LocalView.centered_on(calibration, (420.0, 420.0), (400, 400), 200.0)
    local_rect = np.array([[184.9, 175.6], [213.3, 175.6], [213.3, 224.1], [184.9, 224.1]])
    polygon = view.local_to_raw(local_rect)

    p1, p2, depth = derive_marking_points(polygon.tolist(), calibration=calibration)
    quad = reconstruct_slot_quad(p1, p2, depth, calibration=calibration)

    assert _same_quad(quad, polygon, abs_tol=1.0)


def test_derive_marking_points_rejects_non_quad():
    with pytest.raises(ValueError):
        derive_marking_points([[0, 0], [10, 0], [10, 10]])


def test_reconstruct_slot_quad_rejects_coincident_points():
    with pytest.raises(ValueError):
        reconstruct_slot_quad([5, 5], [5, 5], depth=20)
