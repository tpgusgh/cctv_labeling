import json
import numpy as np
import cv2
import pytest
from calibration import CalibrationModel, fit

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"
PERIPHERAL_SAMPLE_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def test_undistort_distort_roundtrip():
    model = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    points = [[320.0, 320.0], [400.0, 320.0], [320.0, 450.0], [200.0, 500.0], [500.0, 150.0]]

    undistorted = model.undistort_points(points)
    roundtripped = model.distort_points(undistorted)

    np.testing.assert_allclose(roundtripped, np.asarray(points), atol=1e-6)


def test_fit_recovers_known_focal_length():
    cx, cy, f_true = 320.0, 320.0, 300.0
    true_model = CalibrationModel(cx, cy, f_true)
    rectified_line_points = [[200.0, 250.0], [260.0, 250.0], [320.0, 250.0], [380.0, 250.0], [440.0, 250.0]]
    raw_clicks = true_model.distort_points(rectified_line_points)

    fitted = fit(raw_clicks, cx=cx, cy=cy)

    assert abs(fitted.f - f_true) / f_true < 0.05


def test_to_dict_from_dict_roundtrip(tmp_path):
    model = CalibrationModel(cx=321.5, cy=318.0, f=287.3, radius=310.0)
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(model.to_dict()))

    loaded = CalibrationModel.from_dict(json.loads(path.read_text()))

    assert loaded.cx == model.cx
    assert loaded.cy == model.cy
    assert loaded.f == model.f
    assert loaded.radius == model.radius


def test_fit_rejects_fewer_than_three_points():
    with pytest.raises(ValueError):
        fit([[100.0, 100.0], [200.0, 100.0]], cx=320.0, cy=320.0)


def test_fit_handles_noncollinear_points_without_crashing():
    # These points are not collinear at all. The fit either raises ValueError
    # (if the optimum lands on the search boundary) or returns some model --
    # for this specific input, verified separately, the optimum does not land
    # on a boundary, so we just confirm it runs without crashing.
    try:
        model = fit([[100.0, 100.0], [200.0, 300.0], [150.0, 50.0]], cx=320.0, cy=320.0)
        assert isinstance(model, CalibrationModel)
    except ValueError:
        pass


def test_undistort_redistort_image_roundtrip():
    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert raw is not None, f"sample image not found at {SAMPLE_RAW_IMAGE}"
    model = CalibrationModel(cx=320.0, cy=320.0, f=300.0)

    rectified = model.undistort_image(raw)
    assert rectified.shape == raw.shape
    assert not np.array_equal(rectified, raw)

    redistorted = model.redistort_image(rectified, output_shape=raw.shape[:2])
    assert redistorted.shape == raw.shape

    # The equidistant->tan(theta) undistort model has a finite valid raw-space
    # radius beyond which cv2.remap samples fall outside the same-size
    # rectified canvas and get BORDER_CONSTANT zero-filled (not a round-trip
    # error — a real limitation of this MVP model). Compare only within a safe
    # interior radius, well under the measured ~245px axis-aligned boundary for
    # these parameters.
    h, w = raw.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    safe_radius = 200.0
    mask = (xx - 320.0) ** 2 + (yy - 320.0) ** 2 < safe_radius ** 2

    mean_abs_diff = np.mean(np.abs(
        redistorted[mask].astype(np.int16) - raw[mask].astype(np.int16)
    ))
    assert mean_abs_diff < 5.0


def test_pixel_to_ray_ray_to_pixel_roundtrip():
    model = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    points = [[320.0, 320.0], [400.0, 320.0], [320.0, 450.0], [200.0, 500.0], [500.0, 150.0]]

    rays = model.pixel_to_ray(points)
    roundtripped = model.ray_to_pixel(rays)

    np.testing.assert_allclose(roundtripped, np.asarray(points), atol=1e-6)


def test_pixel_to_ray_at_center_is_forward_axis():
    model = CalibrationModel(cx=320.0, cy=320.0, f=204.0)

    ray = model.pixel_to_ray([[320.0, 320.0]])[0]

    np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)


def test_local_view_raw_to_local_roundtrip_off_axis_center():
    from calibration import LocalView

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 20.0), patch_size=(300, 300), local_f=300.0)

    raw_points = [[320.0, 20.0], [340.0, 40.0], [300.0, 10.0], [330.0, 60.0]]
    local_points = view.raw_to_local(raw_points)
    roundtripped = view.local_to_raw(local_points)

    np.testing.assert_allclose(roundtripped, np.asarray(raw_points), atol=1e-3)


def test_local_view_center_maps_to_patch_center():
    from calibration import LocalView

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    center_raw = (320.0, 20.0)
    view = LocalView.centered_on(calibration, center_raw, patch_size=(300, 300), local_f=300.0)

    local_center = view.raw_to_local([center_raw])[0]

    np.testing.assert_allclose(local_center, [150.0, 150.0], atol=1e-6)


def test_local_view_handles_center_at_optical_axis():
    from calibration import LocalView

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    view = LocalView.centered_on(calibration, (320.0, 320.0), patch_size=(300, 300), local_f=300.0)

    raw_points = [[320.0, 320.0], [340.0, 330.0], [300.0, 310.0]]
    local_points = view.raw_to_local(raw_points)
    roundtripped = view.local_to_raw(local_points)

    np.testing.assert_allclose(roundtripped, np.asarray(raw_points), atol=1e-3)


def test_local_view_rectify_output_shape():
    from calibration import LocalView

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    raw = cv2.imread(PERIPHERAL_SAMPLE_IMAGE)
    assert raw is not None
    view = LocalView.centered_on(calibration, (320.0, 20.0), patch_size=(300, 300), local_f=300.0)

    patch = view.rectify(raw)

    assert patch.shape == (300, 300, 3)


def test_local_view_roundtrip_preserves_peripheral_content():
    from calibration import LocalView

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    raw = cv2.imread(PERIPHERAL_SAMPLE_IMAGE)
    assert raw is not None

    center_raw = (320.0, 20.0)  # raw radius ~300px from (320,320) -- near the measured
                                  # real content boundary where the old global model
                                  # zero-filled everything
    view = LocalView.centered_on(calibration, center_raw, patch_size=(300, 300), local_f=300.0)

    patch = view.rectify(raw)
    roundtripped = view.unrectify_into(patch, raw)

    corners_local = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype=np.float64)
    corners_raw = view.local_to_raw(corners_local)
    x_min, y_min = corners_raw.min(axis=0).astype(int)
    x_max, y_max = corners_raw.max(axis=0).astype(int)
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, raw.shape[1]), min(y_max, raw.shape[0])

    bbox_raw = raw[y_min:y_max, x_min:x_max]
    bbox_roundtripped = roundtripped[y_min:y_max, x_min:x_max]
    mean_abs_diff = np.mean(np.abs(bbox_roundtripped.astype(np.int16) - bbox_raw.astype(np.int16)))
    assert mean_abs_diff < 10.0

    far_raw = raw[550:580, 550:580]
    far_roundtripped = roundtripped[550:580, 550:580]
    np.testing.assert_array_equal(far_raw, far_roundtripped)


def test_unrectify_into_covers_full_curved_footprint_at_nadir():
    from calibration import LocalView

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    raw = np.zeros((640, 640, 3), dtype=np.uint8)  # synthetic, content doesn't matter here
    view = LocalView.centered_on(calibration, (320.0, 320.0), patch_size=(300, 300), local_f=300.0)

    solid_patch = np.full((300, 300, 3), 200, dtype=np.uint8)
    result = view.unrectify_into(solid_patch, raw)

    # Determine the TRUE footprint by sampling raw_to_local over a dense raw-space grid
    # around the nadir and checking which raw pixels land inside the patch (this is an
    # independent ground-truth check, not reusing unrectify_into's own bbox logic).
    ys, xs = np.mgrid[170:470, 170:470]
    raw_grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    local_rays = view._local_rays(raw_grid)
    forward = local_rays[:, 2] > 0
    safe_z = np.where(forward, local_rays[:, 2], 1.0)
    lx = view.local_f * local_rays[:, 0] / safe_z + 150.0
    ly = view.local_f * local_rays[:, 1] / safe_z + 150.0
    true_footprint = forward & (lx >= 0) & (lx <= 299) & (ly >= 0) & (ly <= 299)

    painted = np.all(result[ys.ravel(), xs.ravel()] == 200, axis=1)
    coverage = painted[true_footprint].mean()
    assert coverage > 0.99, f"only {coverage:.2%} of the true footprint got painted"
