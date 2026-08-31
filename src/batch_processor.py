import argparse
import json
from pathlib import Path

from generate_config import REVIEW_CONFIDENCE_THRESHOLD
from parking_slot import SlotConfig
from pipeline import run_auto_all

# ponytail: matched by lowercased suffix, not a glob pattern -- glob is
# case-sensitive on every platform regardless of the filesystem's own case
# sensitivity, so a real "*.JPG" export would otherwise be silently skipped
# (see generate_config.py's IMAGE_EXTENSIONS for the bug this was verified
# against).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _needs_review(results, config):
    # a bare status prefix check for "review" was dead code -- run_auto_all
    # only ever emits "excluded"/"labeled"/"error: ..." per slot, it never
    # emits anything starting with "review". Per-slot confidence (already
    # computed at detection time, see generate_config.py) never actually
    # reached this per-photo pipeline, so low-confidence detections were
    # silently baked into "successful" output with no human review flag.
    if any(status.startswith("error") for status in results.values()):
        return True
    if not results:
        return True
    if config is None:
        return False
    confidence_by_slot = {s["id"]: s.get("confidence") for s in config.slots}
    return any(
        status == "labeled" and confidence_by_slot.get(slot_id) is not None
        and confidence_by_slot[slot_id] < REVIEW_CONFIDENCE_THRESHOLD
        for slot_id, status in results.items()
    )


def process_camera_folder(config_path, input_dir, output_dir, review_dir, log_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    review_dir = Path(review_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = SlotConfig.load(config_path)
    except (FileNotFoundError, OSError, ValueError):
        config = None

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    log_entries = []
    for image_path in image_paths:
        candidate_output = output_dir / f"{image_path.stem}.png"
        try:
            results = run_auto_all(config_path, str(image_path), str(candidate_output))
        except (ValueError, OSError) as e:
            log_entries.append({
                "image": image_path.name,
                "status": "error",
                "output": None,
                "error": str(e),
            })
            continue

        if _needs_review(results, config):
            final_path = review_dir / f"{image_path.stem}.png"
            candidate_output.replace(final_path)
            status = "review"
        else:
            final_path = candidate_output
            status = "success"

        log_entries.append({
            "image": image_path.name,
            "status": status,
            "output": str(final_path),
            "slots": results,
        })

    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=False)

    return log_entries


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-process every raw frame in a camera's folder with run_auto_all.")
    parser.add_argument("--config", required=True, help="path to camera config JSON")
    parser.add_argument("--input-dir", required=True, help="folder of raw CCTV frames to process")
    parser.add_argument("--output-dir", required=True, help="folder for successfully labeled frames")
    parser.add_argument("--review-dir", required=True, help="folder for frames needing manual review")
    parser.add_argument("--log", required=True, help="path to write the JSON results log")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    entries = process_camera_folder(args.config, args.input_dir, args.output_dir, args.review_dir, args.log)
    success = sum(1 for e in entries if e["status"] == "success")
    review = sum(1 for e in entries if e["status"] == "review")
    error = sum(1 for e in entries if e["status"] == "error")
    print(f"processed {len(entries)} images: {success} -> {args.output_dir}, {review} -> {args.review_dir}, {error} errored")


if __name__ == "__main__":
    main()
