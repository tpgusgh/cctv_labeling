import numpy as np
import cv2


def point_in_polygon(point, polygon):
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


def find_slot_windshield(slot_polygon_raw, blobs):
    for blob in blobs:
        if point_in_polygon(blob.centroid, slot_polygon_raw):
            return blob
    return None
