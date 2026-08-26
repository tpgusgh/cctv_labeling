from dataclasses import dataclass

import numpy as np
import cv2

# ponytail: thresholds tuned against a single real frame
# (no_label/P1_B1_1_21/20260820_115029.jpg, one visible car). Measured false-positive
# rate on that frame: 18 dark blobs detected for 1 real windshield (shadows, floor
# marks, structural clutter) -- find_slot_windshield's largest-in-slot-polygon rule
# is what keeps this usable today. Upgrade path: per-image adaptive threshold
# (e.g. histogram-based) instead of a fixed brightness cutoff, and/or a shape/aspect
# ratio filter to reject non-windshield-shaped blobs, once more real camera data
# is available to tune against.
DARK_THRESHOLD = 60
MIN_BLOB_AREA = 150
MAX_BLOB_AREA = 8000


@dataclass
class WindshieldBlob:
    contour: object
    bbox: tuple
    centroid: tuple
    area: float


def detect_windshields(raw_image, calibration):
    h, w = raw_image.shape[:2]
    radius = calibration.radius
    if radius is None:
        radius = min(calibration.cx, calibration.cy, w - calibration.cx, h - calibration.cy)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(calibration.cx)), int(round(calibration.cy))), int(round(radius)), 255, thickness=-1)

    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            x, y, bw, bh = cv2.boundingRect(c)
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx_b = m["m10"] / m["m00"]
            cy_b = m["m01"] / m["m00"]
            blobs.append(WindshieldBlob(contour=c, bbox=(x, y, bw, bh), centroid=(cx_b, cy_b), area=area))
    return blobs
