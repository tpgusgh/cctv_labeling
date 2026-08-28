import argparse
import glob
from pathlib import Path

import cv2

import review_store
import slot_classifier
from calibration import CalibrationModel
from parking_slot import SlotConfig
from slot_classifier import crop_polygon
from slot_detection import median_stack, detect_slots

DEFAULT_LABEL_SPEC = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
REVIEW_CONFIDENCE_THRESHOLD = 0.75

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
CROPS_DIR = review_store.CROPS_DIR


def _save_review_candidates(camera_id, image_path, median, detections):
    # ponytail: crops are saved as real files (not just a path reference) so
    # review/training data survives the raw frame folder later disappearing
    # or changing -- see the review-feedback-classifier design doc.
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for d in detections:
        cid = review_store.candidate_id(camera_id, d["polygon"])
        crop_path = CROPS_DIR / f"{cid}.png"
        cv2.imwrite(str(crop_path), crop_polygon(median, d["polygon"]))
        review_store.append_candidate({
            "id": cid,
            "camera_id": camera_id,
            "image_path": str(Path(image_path).resolve()),
            "polygon": d["polygon"],
            "crop_path": str(crop_path),
            "confidence": d["confidence"],
        })


def generate_config(camera_id, frames_dir, output_path, cx=320.0, cy=320.0, f=204.0, radius=320.0,
                     image_width=640, image_height=640, classifier=None, yolo_model=None):
    image_paths = sorted({p for pattern in IMAGE_EXTENSIONS for p in glob.glob(str(Path(frames_dir) / pattern))})
    if not image_paths:
        raise ValueError(f"no frames found in {frames_dir}")

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=cx, cy=cy, f=f, radius=radius)
    if yolo_model is not None:
        import yolo_slot_detector
        # ponytail: NOT passing calibration here -- global dewarp was tried
        # and abandoned (see docs/superpowers/specs -- the equidistant
        # model's tan() singularity sits right where real slot corners are,
        # verified to blow up training-label coordinates way out of bounds).
        # yolo_slot_detector.detect_slots() still accepts calibration= for
        # any future opt-in; production just doesn't pass it right now.
        detections = yolo_slot_detector.detect_slots(median, yolo_model)
    else:
        detections = detect_slots(median, calibration, classifier=classifier)
    _save_review_candidates(camera_id, image_paths[0], median, detections)

    slots = [{"id": f"slot-{i}", "polygon_raw": d["polygon"]} for i, d in enumerate(detections)]
    config = SlotConfig(camera_id, image_width, image_height, calibration, slots, DEFAULT_LABEL_SPEC)
    config.save(output_path)

    needs_review = len(detections) == 0 or any(d["confidence"] < REVIEW_CONFIDENCE_THRESHOLD for d in detections)
    return slots, needs_review


def build_parser():
    parser = argparse.ArgumentParser(
        description="Auto-generate a camera's slot config from its folder of raw frames (no manual slot entry).")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--frames-dir", required=True, help="folder of raw CCTV frames for this camera")
    parser.add_argument("--output", required=True, help="path to write the generated config JSON")
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=320.0)
    parser.add_argument("--f", type=float, default=204.0)
    parser.add_argument("--radius", type=float, default=320.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=640)
    parser.add_argument("--model", default=None, help="path to a trained models/slot_classifier.joblib (optional)")
    parser.add_argument("--yolo-model", default=None,
                         help="path to a trained models/yolov8_seg_slots.pt (optional, overrides --model)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    classifier = slot_classifier.load(args.model) if args.model else None
    yolo_model = None
    if args.yolo_model:
        import yolo_slot_detector
        yolo_model = yolo_slot_detector.load(args.yolo_model)
    slots, needs_review = generate_config(args.camera_id, args.frames_dir, args.output, args.cx, args.cy, args.f,
                                           args.radius, args.image_width, args.image_height, classifier, yolo_model)
    status = "REVIEW RECOMMENDED (low confidence or no slots found)" if needs_review else "ok"
    print(f"{args.camera_id}: detected {len(slots)} slot(s) -> {args.output} [{status}]")


if __name__ == "__main__":
    main()
