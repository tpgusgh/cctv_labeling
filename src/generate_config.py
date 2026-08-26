import argparse
import glob
from pathlib import Path

from calibration import CalibrationModel
from parking_slot import SlotConfig
from slot_detection import median_stack, detect_slots

DEFAULT_LABEL_SPEC = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
REVIEW_CONFIDENCE_THRESHOLD = 0.75

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def generate_config(camera_id, frames_dir, output_path, cx=320.0, cy=320.0, f=204.0, radius=320.0,
                     image_width=640, image_height=640):
    image_paths = sorted({p for pattern in IMAGE_EXTENSIONS for p in glob.glob(str(Path(frames_dir) / pattern))})
    if not image_paths:
        raise ValueError(f"no frames found in {frames_dir}")

    median = median_stack(image_paths)
    calibration = CalibrationModel(cx=cx, cy=cy, f=f, radius=radius)
    detections = detect_slots(median, calibration)

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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    slots, needs_review = generate_config(args.camera_id, args.frames_dir, args.output, args.cx, args.cy, args.f,
                                           args.radius, args.image_width, args.image_height)
    status = "REVIEW RECOMMENDED (low confidence or no slots found)" if needs_review else "ok"
    print(f"{args.camera_id}: detected {len(slots)} slot(s) -> {args.output} [{status}]")


if __name__ == "__main__":
    main()
