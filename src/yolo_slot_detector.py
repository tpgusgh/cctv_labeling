from ultralytics import YOLO

from slot_detection import fit_quad


def load(model_path):
    return YOLO(str(model_path))


def detect_slots(image_bgr, model, conf=0.25):
    """Whole-image slot detection via a fine-tuned YOLOv8-seg model.

    Returns the same [{"polygon": [[x,y]x4], "confidence": float}] contract
    as slot_detection.detect_slots() -- callers (generate_config.py) swap
    detectors without anything downstream (pipeline.py, the web app, the
    review UI) knowing the difference.
    """
    results = model.predict(image_bgr, conf=conf, verbose=False)
    result = results[0]
    if result.masks is None:
        return []

    detections = []
    confidences = result.boxes.conf.tolist()
    for polygon_pts, confidence in zip(result.masks.xy, confidences):
        if len(polygon_pts) < 3:
            continue
        poly = fit_quad(polygon_pts, inset_px=6.0)
        detections.append({"polygon": poly.tolist(), "confidence": round(float(confidence), 3)})
    return detections
