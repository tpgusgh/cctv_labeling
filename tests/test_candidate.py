import numpy as np

from windshield import WindshieldBlob
from candidate import point_in_polygon, find_slot_windshield, compute_label_candidate
from calibration import CalibrationModel, LocalView
from perspective import plane_to_pixel_homography, pixel_to_plane_points


def test_point_in_polygon_true_and_false():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]

    assert point_in_polygon((50.0, 50.0), polygon) is True
    assert point_in_polygon((150.0, 50.0), polygon) is False


def test_find_slot_windshield_picks_blob_inside_polygon_only():
    polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    inside_blob = WindshieldBlob(contour=None, bbox=(40, 40, 10, 10), centroid=(45.0, 45.0), area=100)
    outside_blob = WindshieldBlob(contour=None, bbox=(200, 200, 10, 10), centroid=(205.0, 205.0), area=100)

    assert find_slot_windshield(polygon, [outside_blob, inside_blob]) is inside_blob
    assert find_slot_windshield(polygon, [outside_blob]) is None


def test_compute_label_candidate_region_fully_contains_blob():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 60.0), patch_size=(300, 300), local_f=300.0)
    slot_polygon_raw = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]
    polygon_local = view.raw_to_local(slot_polygon_raw)
    homography = plane_to_pixel_homography(polygon_local)

    blob = WindshieldBlob(contour=None, bbox=(300, 40, 20, 20), centroid=(310.0, 50.0), area=400)

    candidate_point, width, height = compute_label_candidate(view, homography, blob)

    corners_raw = [[300, 40], [320, 40], [320, 60], [300, 60]]
    corners_local = view.raw_to_local(corners_raw)
    corners_plane = pixel_to_plane_points(homography, corners_local)

    u_min = candidate_point[0] - width / 2.0
    u_max = candidate_point[0] + width / 2.0
    v_min = candidate_point[1] - height / 2.0
    v_max = candidate_point[1] + height / 2.0

    assert np.all(corners_plane[:, 0] >= u_min - 1e-6)
    assert np.all(corners_plane[:, 0] <= u_max + 1e-6)
    assert np.all(corners_plane[:, 1] >= v_min - 1e-6)
    assert np.all(corners_plane[:, 1] <= v_max + 1e-6)


def test_compute_label_candidate_clamps_to_slot_bounds_near_edge():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 60.0), patch_size=(300, 300), local_f=300.0)
    slot_polygon_raw = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]
    polygon_local = view.raw_to_local(slot_polygon_raw)
    homography = plane_to_pixel_homography(polygon_local)

    # blob near the slot's right edge -- margin expansion pushes past u=1.0
    # without clamping (verified: unclamped u_max would be ~1.02)
    blob = WindshieldBlob(contour=None, bbox=(345, 40, 15, 20), centroid=(352.5, 50.0), area=300)

    candidate_point, width, height = compute_label_candidate(view, homography, blob)

    assert candidate_point[0] - width / 2.0 >= -1e-6
    assert candidate_point[0] + width / 2.0 <= 1.0 + 1e-6
    assert candidate_point[1] - height / 2.0 >= -1e-6
    assert candidate_point[1] + height / 2.0 <= 1.0 + 1e-6
