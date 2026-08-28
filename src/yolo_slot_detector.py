from ultralytics import YOLO

from slot_detection import fit_quad


def load(model_path):
    return YOLO(str(model_path))


def detect_slots(image_bgr, model, conf=0.25, calibration=None):
    """Whole-image slot detection via a fine-tuned YOLOv8-seg model.

    Returns the same [{"polygon": [[x,y]x4], "confidence": float}] contract
    as slot_detection.detect_slots() -- callers (generate_config.py) swap
    detectors without anything downstream (pipeline.py, the web app, the
    review UI) knowing the difference. In particular, returned polygons are
    always in raw (fisheye-distorted) pixel space, matching what the rest
    of the pipeline (perspective.py, pipeline.py) expects -- regardless of
    whether `calibration` is given.

    calibration: optional CalibrationModel. Real slot corners in this
    project's cameras sit mostly near the fisheye edge (median ~78% of the
    radius out from center) -- exactly where distortion is worst -- so
    inference is run on a globally dewarped copy of the image instead of
    the raw frame (quad-fitting on an undistorted, genuinely rectangular
    shape is far more accurate than fitting on a curved one). Detected
    polygons are then mapped back through calibration.distort_points() so
    the raw-space contract holds either way. Default None skips all of
    this and runs on image_bgr unchanged (existing behavior).
    """
    inference_image = calibration.undistort_image(image_bgr) if calibration is not None else image_bgr
    results = model.predict(inference_image, conf=conf, verbose=False)
    result = results[0]
    if result.masks is None:
        return []

    detections = []
    confidences = result.boxes.conf.tolist()
    for polygon_pts, confidence in zip(result.masks.xy, confidences):
        if len(polygon_pts) < 3:
            continue
        poly = fit_quad(polygon_pts, inset_px=6.0)
        if calibration is not None:
            poly = calibration.distort_points(poly)
        detections.append({"polygon": poly.tolist(), "confidence": round(float(confidence), 3)})
    return detections
