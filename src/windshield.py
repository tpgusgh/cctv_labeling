from dataclasses import dataclass

import numpy as np
import cv2

# ponytail: area/aspect thresholds tuned against a single real frame
# (no_label/P1_B1_1_21/20260820_115029.jpg, one visible car). Measured false-positive
# rate on that frame: 18 dark blobs before the aspect-ratio filter below (shadows,
# glossy-floor light reflections, structural clutter), 12 after -- reflections in
# particular tend to form thin elongated streaks along the floor's glossy surface,
# not the roughly-compact blob shape of a real windshield (measured aspect ratio
# 1.13 for the real windshield, vs up to 10.93 for reflection/shadow streaks).
# find_slot_windshield's largest-in-slot-polygon rule handles the remainder.
FALLBACK_DARK_THRESHOLD = 60  # used only if the floor mask is empty (degenerate calibration)
# ponytail: percentile chosen (not Otsu -- see _adaptive_dark_threshold) to
# match the original hand-tuned constant almost exactly: on the reference
# frame above, the 10th percentile of floor-masked brightness is 60.0, i.e.
# the same cutoff the fixed constant used. Validated against one real frame
# per camera (23 cameras, 2026-08-28): p10 ranges 34-80, comfortably inside
# the clamp below and producing a plausible handful of dark-blob candidates
# per frame (matching the reference frame's documented 18-before/12-after
# filter counts in order of magnitude).
DARK_PERCENTILE = 10
MIN_DARK_THRESHOLD = 20
MAX_DARK_THRESHOLD = 100
MIN_BLOB_AREA = 150
MAX_BLOB_AREA = 8000
MAX_ASPECT_RATIO = 2.5

# ponytail: no labeled dataset exists to train a real label/no_label classifier
# (see docs/superpowers/specs/2026-08-26-windshield-label-placement-design.md's
# background on why the original "차량 detection" plan was abandoned -- same
# reason applies here). This is a simple heuristic confidence in [0,1] based on
# how compact the blob's shape is (real windshields measured compact, aspect
# ratio ~1.1-1.3; elongated blobs are more likely reflections/shadows/clutter
# that survived the MAX_ASPECT_RATIO filter but are still borderline). Blob
# *area* is deliberately NOT part of this score -- apparent windshield size
# varies hugely with where the car sits in this fisheye frame, so a "closer to
# the middle of the valid area range is better" assumption would be wrong.
REVIEW_CONFIDENCE_THRESHOLD = 0.5


def _adaptive_dark_threshold(gray, mask):
    """Low-percentile brightness of just the floor-masked pixels, so the
    dark/light cutoff tracks each frame's actual lighting instead of a fixed
    brightness constant tuned on one reference frame -- car color, time of
    day, and per-camera exposure all shift the floor's own brightness.

    Deliberately NOT Otsu: tried first, but real floor pixels dominate the
    masked area (a windshield is a small minority), so Otsu ends up splitting
    the floor's own brightness variation in two instead of separating
    floor-from-object -- measured on one real frame per camera, this pushed
    Otsu's threshold to 112-162 (almost always clamped to this project's
    floor-brightness range of ~115-130), which would call most of the floor
    itself 'dark'. A low percentile of the floor population directly targets
    the dark tail instead, and empirically reproduces the original tuned
    constant (see DARK_PERCENTILE above)."""
    floor_pixels = gray[mask.astype(bool)]
    if floor_pixels.size == 0:
        return FALLBACK_DARK_THRESHOLD
    threshold = float(np.percentile(floor_pixels, DARK_PERCENTILE))
    return float(np.clip(threshold, MIN_DARK_THRESHOLD, MAX_DARK_THRESHOLD))


def confidence_score(blob):
    x, y, w, h = blob.bbox
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    return max(0.0, 1.0 - (aspect_ratio - 1.0) / (MAX_ASPECT_RATIO - 1.0))


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
    dark_threshold = _adaptive_dark_threshold(gray, mask)
    _, dark = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            if aspect_ratio > MAX_ASPECT_RATIO:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx_b = m["m10"] / m["m00"]
            cy_b = m["m01"] / m["m00"]
            blobs.append(WindshieldBlob(contour=c, bbox=(x, y, bw, bh), centroid=(cx_b, cy_b), area=area))
    return blobs
