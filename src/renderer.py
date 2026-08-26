import numpy as np
import cv2

from perspective import plane_points_to_pixel


def _label_corners_normalized(candidate_point, width, height):
    cu, cv_ = candidate_point
    hw, hh = width / 2.0, height / 2.0
    return [
        [cu - hw, cv_ - hh],
        [cu + hw, cv_ - hh],
        [cu + hw, cv_ + hh],
        [cu - hw, cv_ + hh],
    ]


def render_label(rectified_image, homography, candidate_point, label_spec):
    shape = label_spec.get("shape", "rect")
    if shape != "rect":
        raise NotImplementedError(f"label shape '{shape}' not implemented yet; only 'rect' is supported")

    width = label_spec["width"]
    height = label_spec["height"]
    color = tuple(int(c) for c in label_spec["color"])
    alpha = float(label_spec["alpha"])
    text = label_spec.get("text")
    border_width = int(label_spec.get("border_width", 3))

    corners_norm = _label_corners_normalized(candidate_point, width, height)
    corners_px = plane_points_to_pixel(homography, corners_norm)

    h, w = rectified_image.shape[:2]
    if (corners_px[:, 0] < 0).any() or (corners_px[:, 0] >= w).any() \
            or (corners_px[:, 1] < 0).any() or (corners_px[:, 1] >= h).any():
        raise ValueError(
            f"label at candidate {tuple(candidate_point)} maps to rectified pixels "
            f"{corners_px.tolist()}, outside the {w}x{h} rectified image"
        )

    poly = corners_px.reshape(-1, 1, 2).astype(np.int32)

    overlay = rectified_image.copy()
    cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=border_width, lineType=cv2.LINE_AA)
    if text:
        centroid = corners_px.mean(axis=0).astype(int)
        cv2.putText(overlay, text, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, rectified_image, 1 - alpha, 0)
