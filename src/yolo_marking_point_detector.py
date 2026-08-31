import numpy as np
from ultralytics import YOLO

from calibration import CalibrationModel
from marking_point_geometry import reconstruct_slot_quad

# ponytail: flat fallback depth prior, only used when no calibration is given
# (so no radius can be computed) -- see _radius_aware_depth_prior() below for
# the real default. Median depth derive_marking_points() measures across this
# project's ~100 accepted labels, using the same calibration, is 85.2px.
FALLBACK_DEPTH_PRIOR_PX = 85.2

# ponytail: linear fit of depth ~ a + b*radius_from_calibration_center,
# measured 2026-08-28 across all 180 real accepted+missed entrance-pair
# labels on this project's fixed 640x640 cameras (see derive_marking_points
# in marking_point_geometry.py). A single flat depth constant systematically
# under/over-estimates real depth by ~20% between near-center and near-edge
# slots (bucketed real data: median 76.3px near-center vs 92.8px near-edge)
# -- this is the "fixed depth prior wrong under fisheye radial distortion"
# gap flagged when this module was first prototyped. The linear fit only
# explains ~16% of total depth variance (R^2=0.16), so most of the remaining
# spread is presumably genuine per-slot/label noise rather than residual
# fisheye foreshortening; a further improvement would need more signal than
# radius alone (e.g. per-camera depth, or a second geometric feature).
# Clamped to the observed [34.3, 128.9]px range to avoid extrapolating wildly
# for a detection whose corners land outside where this was measured.
DEPTH_PRIOR_INTERCEPT = -9.014
DEPTH_PRIOR_SLOPE = 0.33213
DEPTH_PRIOR_MIN_PX = 34.3
DEPTH_PRIOR_MAX_PX = 128.9

# this project's fisheye calibration -- same defaults used throughout
# (export_yolo_dataset.py, generate_config.py argparse defaults).
DEFAULT_CALIBRATION = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)


def load(model_path):
    return YOLO(str(model_path))


def _radius_aware_depth_prior(p1, p2, calibration):
    midpoint = (np.asarray(p1, dtype=np.float64) + np.asarray(p2, dtype=np.float64)) / 2
    radius = float(np.linalg.norm(midpoint - np.array([calibration.cx, calibration.cy])))
    depth = DEPTH_PRIOR_INTERCEPT + DEPTH_PRIOR_SLOPE * radius
    return float(np.clip(depth, DEPTH_PRIOR_MIN_PX, DEPTH_PRIOR_MAX_PX))


def detect_slots(image_bgr, model, conf=0.25, depth_prior_px=None, calibration=DEFAULT_CALIBRATION):
    """Whole-image slot detection via a fine-tuned YOLOv8-pose marking-point
    model.

    Returns the same [{"polygon": [[x,y]x4], "confidence": float}] contract
    as slot_detection.detect_slots()/yolo_slot_detector.detect_slots() --
    callers don't need to know a slot's full quad was reconstructed from
    just its 2 entrance corners rather than detected whole. This is the
    point of the marking-point redesign: entrance corners stay visible even
    when a parked car occludes the rest of the slot, which whole-polygon
    detection can't handle.

    calibration: passed straight through to reconstruct_slot_quad() so the
    inward-direction rotation happens in a locally-rectified tangent plane
    (correct under this camera's fisheye distortion) instead of flat raw
    pixel space. Pass None to fall back to the flat calculation.

    depth_prior_px: fixed depth to use for every detection. Default None
    picks a per-detection depth from _radius_aware_depth_prior() instead
    (falls back to FALLBACK_DEPTH_PRIOR_PX when calibration is None, since
    there's then no center to measure radius from). Pass an explicit value
    to force the old flat-constant behavior.
    """
    results = model.predict(image_bgr, conf=conf, verbose=False)
    result = results[0]
    if result.keypoints is None:
        return []

    detections = []
    confidences = result.boxes.conf.tolist()
    for kpts, confidence in zip(result.keypoints.xy, confidences):
        pts = kpts.tolist() if hasattr(kpts, "tolist") else list(kpts)
        if len(pts) < 2:
            continue
        p1, p2 = pts[0], pts[1]
        if depth_prior_px is not None:
            depth = depth_prior_px
        elif calibration is not None:
            depth = _radius_aware_depth_prior(p1, p2, calibration)
        else:
            depth = FALLBACK_DEPTH_PRIOR_PX
        quad = reconstruct_slot_quad(p1, p2, depth, calibration=calibration)
        detections.append({"polygon": quad.tolist(), "confidence": round(float(confidence), 3)})
    return detections
