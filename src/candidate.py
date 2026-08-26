import numpy as np
import cv2

from perspective import pixel_to_plane_points


def point_in_polygon(point, polygon):
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


def find_slot_windshield(slot_polygon_raw, blobs):
    for blob in blobs:
        if point_in_polygon(blob.centroid, slot_polygon_raw):
            return blob
    return None


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

    return (float(u_center), float(v_center)), float(width), float(height)
