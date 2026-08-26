import numpy as np
import pytest

from perspective import plane_to_pixel_homography
from renderer import render_label


def _blank_canvas():
    return np.full((400, 400, 3), 255, dtype=np.uint8)


def test_render_label_fills_candidate_region_with_full_alpha():
    canvas = _blank_canvas()
    homography = plane_to_pixel_homography([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])
    label_spec = {"shape": "rect", "width": 0.4, "height": 0.4, "color": [0, 0, 255], "alpha": 1.0, "text": None}

    result = render_label(canvas, homography, (0.5, 0.5), label_spec)

    assert list(result[200, 200]) == [0, 0, 255]
    assert list(result[10, 10]) == [255, 255, 255]


def test_render_label_rejects_unsupported_shape():
    canvas = _blank_canvas()
    homography = plane_to_pixel_homography([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])
    label_spec = {"shape": "rounded_rect", "width": 0.4, "height": 0.4, "color": [0, 0, 255], "alpha": 1.0, "text": None}

    with pytest.raises(NotImplementedError):
        render_label(canvas, homography, (0.5, 0.5), label_spec)


def test_render_label_raises_when_candidate_outside_rectified_bounds():
    canvas = _blank_canvas()
    homography = plane_to_pixel_homography([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])
    label_spec = {"shape": "rect", "width": 0.4, "height": 0.4, "color": [0, 0, 255], "alpha": 1.0, "text": None}

    with pytest.raises(ValueError):
        render_label(canvas, homography, (5.0, 5.0), label_spec)
