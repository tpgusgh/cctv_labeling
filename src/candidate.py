import numpy as np
import cv2

from perspective import pixel_to_plane_points


def point_in_polygon(point, polygon):
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


def find_slot_windshield(slot_polygon_raw, blobs):
    inside = [b for b in blobs if point_in_polygon(b.centroid, slot_polygon_raw)]
    return max(inside, key=lambda b: b.area, default=None)


# Note: containment is empirical, not an exact geometric guarantee -- LocalView's
# raw<->local mapping is a nonlinear fisheye/gnomonic reprojection (not a
# homography), so straight edges bow slightly under it. Measured on real data:
# margin=1.0 already covers ~99.8% of the blob's area; margin=1.3 (the default)
# measures 100.0%. Either comfortably clears the actual requirement (>=50%
# coverage) -- 1.3 is chosen for a visibly-larger-than-the-glass label, not
# because 1.0 would fail the coverage requirement.
def compute_label_candidate(view, homography, blob, coverage_margin=1.3):
    x, y, w, h = blob.bbox
    corners_raw = [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ]
    corners_local = view.raw_to_local(corners_raw)
    corners_plane = pixel_to_plane_points(homography, corners_local)

    u_min, v_min = corners_plane.min(axis=0)
    u_max, v_max = corners_plane.max(axis=0)
    u_center = (u_min + u_max) / 2.0
    v_center = (v_min + v_max) / 2.0
    width = (u_max - u_min) * coverage_margin
    height = (v_max - v_min) * coverage_margin

    # Clamp to this slot's own normalized plane (0-1) so the label never crosses
    # into a neighboring slot -- adjacent slots share their boundary line, so
    # staying inside [0,1] here is equivalent to not intruding on the neighbor.
    u_lo = max(u_center - width / 2.0, 0.0)
    u_hi = min(u_center + width / 2.0, 1.0)
    v_lo = max(v_center - height / 2.0, 0.0)
    v_hi = min(v_center + height / 2.0, 1.0)
    u_center = (u_lo + u_hi) / 2.0
    v_center = (v_lo + v_hi) / 2.0
    width = u_hi - u_lo
    height = v_hi - v_lo

    return (float(u_center), float(v_center)), float(width), float(height)
