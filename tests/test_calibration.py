import json
import numpy as np
import cv2
import pytest
from calibration import CalibrationModel, fit

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


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
