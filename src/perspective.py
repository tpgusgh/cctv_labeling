import numpy as np
import cv2

_UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


def plane_to_pixel_homography(slot_polygon_rectified):
    dst = np.asarray(slot_polygon_rectified, dtype=np.float32)
    if dst.shape != (4, 2):
        raise ValueError(f"slot polygon must have exactly 4 points, got shape {dst.shape}")
    return cv2.getPerspectiveTransform(_UNIT_SQUARE, dst)


def plane_points_to_pixel(homography, normalized_points):
    pts = np.asarray(normalized_points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, homography)
    return out.reshape(-1, 2)
