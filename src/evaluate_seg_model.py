"""Score a YOLO-seg checkpoint against this project's ground truth.

Ground truth = the shipped camera configs (accepted slots) plus the review
log's rejects (known junk regions). For each production camera's median
image the model runs raw (no ensemble, no suppression -- we want the model's
own behavior), and every detection is scored:
  - hits an accepted region  -> good (drives config recall)
  - hits a rejected-only region -> junk
  - neither -> neutral (a new region that would go to human review)

Primary metric: config recall (how many shipped slots the model re-finds).
Secondary: junk rate (detections on rejected-only regions).
"""
import argparse
import glob
import json
from pathlib import Path

import review_store
from slot_detection import median_stack, _polygon_bbox, _same_slot

PROJECT_ROOT = review_store.PROJECT_ROOT


def score_detections(detections_by_camera, configs_by_camera, labels):
    """Pure scoring -- detections/configs/labels in, counters out."""
    accepted_by_cam = {}
    rejected_by_cam = {}
    for l in labels:
        bb = _polygon_bbox(l["polygon"])
        target = accepted_by_cam if l["decision"] == "accept" else rejected_by_cam
        target.setdefault(l["camera_id"], []).append(bb)

    total = {"detections": 0, "on_accept": 0, "on_reject_only": 0, "neutral": 0,
             "config_slots": 0, "config_recall_hits": 0}
    for cam, detections in detections_by_camera.items():
        config_bboxes = [_polygon_bbox(s["polygon_raw"]) for s in configs_by_camera.get(cam, [])]
        det_bboxes = [_polygon_bbox(d["polygon"]) for d in detections]
        accepted = accepted_by_cam.get(cam, [])
        rejected = rejected_by_cam.get(cam, [])

        total["detections"] += len(det_bboxes)
        total["config_slots"] += len(config_bboxes)
        for db in det_bboxes:
            if any(_same_slot(db, ab, 0.4) for ab in accepted):
                total["on_accept"] += 1
            elif any(_same_slot(db, rb, 0.4) for rb in rejected):
                total["on_reject_only"] += 1
            else:
                total["neutral"] += 1
        for cb in config_bboxes:
            if any(_same_slot(cb, db, 0.4) for db in det_bboxes):
                total["config_recall_hits"] += 1

    total["recall"] = (total["config_recall_hits"] / total["config_slots"]
                        if total["config_slots"] else 0.0)
    total["junk_rate"] = (total["on_reject_only"] / total["detections"]
                           if total["detections"] else 0.0)
    return total


def evaluate(model_path, camera_glob="P1_B1_1_*", no_label_dir=None):
    import yolo_slot_detector
    no_label_dir = Path(no_label_dir) if no_label_dir else PROJECT_ROOT / "no_label"
    model = yolo_slot_detector.load(model_path)
    labels = review_store.load_labels()

    detections_by_camera = {}
    configs_by_camera = {}
    for cfg_path in sorted(glob.glob(str(PROJECT_ROOT / "config" / f"{camera_glob}.json"))):
        cfg = json.loads(Path(cfg_path).read_text())
        cam = cfg["camera_id"]
        frames = sorted(glob.glob(str(no_label_dir / cam / "*.jpg")))
        if not frames:
            continue
        median = median_stack(frames)
        detections_by_camera[cam] = yolo_slot_detector.detect_slots(median, model)
        configs_by_camera[cam] = cfg["slots"]
    return score_detections(detections_by_camera, configs_by_camera, labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="checkpoint path(s) to evaluate")
    args = parser.parse_args(argv)
    for path in args.models:
        r = evaluate(path)
        print(f"{path}:")
        print(f"  detections={r['detections']}  on_accept={r['on_accept']}  "
              f"on_reject_only={r['on_reject_only']} ({r['junk_rate']:.1%})  neutral={r['neutral']}")
        print(f"  config recall: {r['config_recall_hits']}/{r['config_slots']} ({r['recall']:.1%})")


if __name__ == "__main__":
    main()
