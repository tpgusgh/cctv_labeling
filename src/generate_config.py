import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2

import review_store
import slot_classifier
from calibration import CalibrationModel
from parking_slot import SlotConfig
from slot_classifier import crop_polygon
from slot_detection import (median_stack, detect_slots, merge_detections, is_degenerate_quad,
                             regularize_quad, _polygon_bbox, _same_slot)

DEFAULT_LABEL_SPEC = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
REVIEW_CONFIDENCE_THRESHOLD = 0.75

# ponytail: matched by lowercased suffix (not a glob pattern) so a real
# phone/camera export named "*.JPG" isn't silently skipped -- glob.glob is
# case-sensitive on every platform regardless of the filesystem's own case
# sensitivity (verified: a real upload named "....JPG" made this raise "no
# frames found" against a directory that visibly contained it).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CROPS_DIR = review_store.CROPS_DIR


def _save_review_candidates(camera_id, image_path, median, detections):
    # ponytail: crops are saved as real files (not just a path reference) so
    # review/training data survives the raw frame folder later disappearing
    # or changing -- see the review-feedback-classifier design doc.
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for d in detections:
        cid = review_store.candidate_id(camera_id, d["polygon"])
        crop_path = CROPS_DIR / f"{cid}.png"
        cv2.imwrite(str(crop_path), crop_polygon(median, d["polygon"]))
        record = {
            "id": cid,
            "camera_id": camera_id,
            "image_path": str(Path(image_path).resolve()),
            "polygon": d["polygon"],
            "crop_path": str(crop_path),
            "confidence": d["confidence"],
        }
        review_store.append_candidate(record)
        records.append(record)
    return records


def _auto_accept_agreed_candidates(candidate_records, detections, path=review_store.LABELS_PATH):
    # detections and candidate_records are the same length, built in the
    # same order (see _save_review_candidates) -- an "agreement_count" >= 2
    # means classical CV and the trained model independently landed on this
    # same region, a real cross-method quality signal. Auto-accepting those
    # skips human review for exactly the candidates two different,
    # differently-erroring methods already agree on, cutting down review
    # volume without needing more labeled data (see slot_detection.merge_detections).
    for record, d in zip(candidate_records, detections):
        if d.get("agreement_count", 1) < 2:
            continue
        accepted = dict(record)
        accepted["decision"] = "accept"
        accepted["ts"] = datetime.now(timezone.utc).isoformat()
        review_store.append_decision(accepted, path)


def _suppress_rejected_regions(camera_id, detections, labels_path=None):
    """Drop detections landing on a region a human already rejected for this
    camera -- verified need: the trained model (v4) happily re-detects the
    exact ceiling/green-floor regions a reviewer removed, because a rejected
    label only disappears from YOLO's positives, it never becomes a negative.
    Without this, re-running detection on a camera resurrects every cleaned-up
    false positive straight back into its config.

    A region is suppressed only when it matches a rejected polygon AND no
    accepted polygon -- a duplicate-of-a-real-slot rejection leaves an accept
    on the same region, and that region must stay detectable."""
    labels = review_store.load_labels(labels_path if labels_path is not None else review_store.LABELS_PATH)
    rejected = [l["polygon"] for l in labels if l["camera_id"] == camera_id and l["decision"] == "reject"]
    if not rejected:
        return detections
    accepted = [l["polygon"] for l in labels if l["camera_id"] == camera_id and l["decision"] == "accept"]
    rejected_bboxes = [_polygon_bbox(p) for p in rejected]
    accepted_bboxes = [_polygon_bbox(p) for p in accepted]

    kept = []
    for d in detections:
        bbox = _polygon_bbox(d["polygon"])
        hits_reject = any(_same_slot(bbox, rb, iou_threshold=0.4) for rb in rejected_bboxes)
        hits_accept = any(_same_slot(bbox, ab, iou_threshold=0.4) for ab in accepted_bboxes)
        if hits_reject and not hits_accept:
            continue
        kept.append(d)
    return kept


def generate_config(camera_id, frames_dir, output_path, cx=None, cy=None, f=None, radius=None,
                     image_width=None, image_height=None, classifier=None, yolo_model=None,
                     auto_accept_agreement=False):
    """cx/cy/f/radius/image_width/image_height default to None: auto-detected
    from the camera's own frames instead of hardcoded for this project's
    original 640x640 camera -- supports fisheye cameras at other
    resolutions/crops too. cx/cy default to the frame's own center; f/radius
    (fit for this project's cameras at 640x640) scale proportionally with
    frame size, assuming the same lens just captured at a different
    resolution. Pass explicit values to override any of this."""
    frames_dir = Path(frames_dir)
    image_paths = sorted(
        str(p) for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if frames_dir.is_dir() else []
    if not image_paths:
        raise ValueError(f"no frames found in {frames_dir}")

    median = median_stack(image_paths)
    height, width = median.shape[:2]
    image_width = width if image_width is None else image_width
    image_height = height if image_height is None else image_height
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    scale = min(width, height) / 640.0
    f = 204.0 * scale if f is None else f
    radius = 320.0 * scale if radius is None else radius
    calibration = CalibrationModel(cx=cx, cy=cy, f=f, radius=radius)
    if yolo_model is not None:
        import yolo_slot_detector
        # ponytail: NOT passing calibration here -- global dewarp was tried
        # and abandoned (see docs/superpowers/specs -- the equidistant
        # model's tan() singularity sits right where real slot corners are,
        # verified to blow up training-label coordinates way out of bounds).
        # yolo_slot_detector.detect_slots() still accepts calibration= for
        # any future opt-in; production just doesn't pass it right now.
        yolo_detections = yolo_slot_detector.detect_slots(median, yolo_model)
        # ensemble with the classical detector too -- the two catch
        # different slots (verified: classical CV found a clean angled slot
        # the trained model missed, and vice versa elsewhere), and merging
        # raises overall recall for free since every candidate still goes
        # through human review regardless of which detector found it. This
        # is the cheap alternative to "collect more labels and retrain" when
        # more labeled data isn't readily available.
        classical_detections = detect_slots(median, calibration, classifier=classifier)
        detections = merge_detections(yolo_detections, classical_detections)
        # row-adjacency rescue: a sub-threshold model detection sitting right
        # next to a confirmed slot (slots come in contiguous rows) is almost
        # always a real slot the model undersold -- measured 5 real slots
        # rescued / 0 junk across the 23 production cameras. Isolated weak
        # detections stay dropped. See src/row_inference.py.
        import row_inference
        low_conf = yolo_slot_detector.detect_slots(median, yolo_model, conf=0.05)
        weak = [d for d in low_conf if d["confidence"] < 0.25]
        weak = [d for d in weak
                if not any(_same_slot(_polygon_bbox(d["polygon"]), _polygon_bbox(kept["polygon"]), 0.4)
                            for kept in detections)]
        rescued = row_inference.rescue_row_adjacent(weak, detections, (calibration.cx, calibration.cy))
        if rescued:
            detections = merge_detections(detections, rescued)
    else:
        detections = detect_slots(median, calibration, classifier=classifier)
    # the camera hangs over the driving area and every bay sits on the outer
    # ring -- a detection whose center is close to the optical center is a
    # floor marking (turn arrows, crosswalk) by construction, never a slot.
    def _too_central(d):
        xs = [p[0] for p in d["polygon"]]
        ys = [p[1] for p in d["polygon"]]
        cx_d, cy_d = sum(xs) / len(xs), sum(ys) / len(ys)
        dist = ((cx_d - calibration.cx) ** 2 + (cy_d - calibration.cy) ** 2) ** 0.5
        return dist < 0.42 * radius
    detections = [d for d in detections if not _too_central(d)]
    # snap every detected quad to a physically plausible slot shape in its
    # own distortion-free local view (fixes fisheye-skewed 'diagonal' quads
    # on straight slots, and re-extends edge slots whose masks lost the far
    # end of the bay and came out flat).
    detections = [dict(d, polygon=regularize_quad(d["polygon"], calibration)) for d in detections]
    # bowtie/folded/extreme-sliver quads can't be real slots -- drop before
    # they reach the config, the review queue, or (via accepts) training data.
    detections = [d for d in detections if not is_degenerate_quad(d["polygon"])]
    detections = _suppress_rejected_regions(camera_id, detections)
    candidate_records = _save_review_candidates(camera_id, image_paths[0], median, detections)
    if auto_accept_agreement and candidate_records:
        _auto_accept_agreed_candidates(candidate_records, detections)

    slots = [{"id": f"slot-{i}", "polygon_raw": d["polygon"], "confidence": d["confidence"]}
              for i, d in enumerate(detections)]
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
    parser.add_argument("--cx", type=float, default=None, help="default: frame's own horizontal center")
    parser.add_argument("--cy", type=float, default=None, help="default: frame's own vertical center")
    parser.add_argument("--f", type=float, default=None, help="default: 204 scaled to the frame's actual size")
    parser.add_argument("--radius", type=float, default=None, help="default: 320 scaled to the frame's actual size")
    parser.add_argument("--image-width", type=int, default=None, help="default: frame's own width")
    parser.add_argument("--image-height", type=int, default=None, help="default: frame's own height")
    parser.add_argument("--model", default=None, help="path to a trained models/slot_classifier.joblib (optional)")
    parser.add_argument("--yolo-model", default=None,
                         help="path to a trained models/yolov8_seg_slots.pt (optional, overrides --model)")
    parser.add_argument("--auto-accept-agreement", action="store_true",
                         help="auto-accept candidates the classical detector and --yolo-model both independently "
                              "found (agreement_count >= 2), skipping human review for just those -- cuts review "
                              "volume without more labeled data. Only applies when --yolo-model is given.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    classifier = slot_classifier.load(args.model) if args.model else None
    yolo_model = None
    if args.yolo_model:
        import yolo_slot_detector
        yolo_model = yolo_slot_detector.load(args.yolo_model)
    slots, needs_review = generate_config(args.camera_id, args.frames_dir, args.output, args.cx, args.cy, args.f,
                                           args.radius, args.image_width, args.image_height, classifier, yolo_model,
                                           args.auto_accept_agreement)
    status = "REVIEW RECOMMENDED (low confidence or no slots found)" if needs_review else "ok"
    print(f"{args.camera_id}: detected {len(slots)} slot(s) -> {args.output} [{status}]")


if __name__ == "__main__":
    main()
