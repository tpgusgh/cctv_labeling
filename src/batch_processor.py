import argparse
import json
from pathlib import Path

from pipeline import run_auto_all

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def _needs_review(results):
    return any(status.startswith("error") or status.startswith("review") for status in results.values())


def process_camera_folder(config_path, input_dir, output_dir, review_dir, log_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    review_dir = Path(review_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted({p for pattern in IMAGE_EXTENSIONS for p in input_dir.glob(pattern)})

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

        if _needs_review(results):
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
