import numpy as np
import pytest

from perspective import plane_to_pixel_homography, plane_points_to_pixel, pixel_to_plane_points


def test_plane_points_map_to_expected_pixel_square():
    quad = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    homography = plane_to_pixel_homography(quad)

    normalized = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    pixels = plane_points_to_pixel(homography, normalized)

    expected = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0], [200.0, 200.0]])
    np.testing.assert_allclose(pixels, expected, atol=1e-3)


def test_plane_to_pixel_homography_rejects_non_quad():
    with pytest.raises(ValueError):
        plane_to_pixel_homography([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


def test_pixel_to_plane_points_roundtrips_with_plane_points_to_pixel():
    quad = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    homography = plane_to_pixel_homography(quad)
    normalized = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.3, 0.7]]

    pixels = plane_points_to_pixel(homography, normalized)
    roundtripped = pixel_to_plane_points(homography, pixels)

    np.testing.assert_allclose(roundtripped, np.asarray(normalized), atol=1e-4)
