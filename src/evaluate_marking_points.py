import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import review_store
import yolo_marking_point_detector
from export_marking_points import _polygons_by_camera


def _default_val_cameras(labels_path, missed_path, val_every):
    """Same camera-level split export_marking_points.export_dataset() used to
    build the training set, so evaluation runs against the cameras that were
    actually held out of training rather than ones the model already saw."""
    camera_ids = sorted(_polygons_by_camera(labels_path, missed_path).keys())
    return camera_ids[::val_every] or camera_ids[-1:]


def _polygon_mask(polygon, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _iou(poly_a, poly_b, shape):
    mask_a = _polygon_mask(poly_a, shape)
    mask_b = _polygon_mask(poly_b, shape)
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union else 0.0


def evaluate(model_path, no_label_dir, val_cameras, labels_path=review_store.LABELS_PATH,
             missed_path=review_store.MISSED_PATH, calibration=yolo_marking_point_detector.DEFAULT_CALIBRATION,
             conf=0.25, iou_match_threshold=0.3, frames_per_camera=1):
    """Detection-quality check for the marking-point model: for each held-out
    camera, run detect_slots() on up to frames_per_camera real frames and
    match its reconstructed quads against that camera's real accepted+missed
    polygons by IoU (greedy best-match, matched independently per frame --
    every frame from a fixed camera shares the same ground-truth slots, see
    export_marking_points.py, but detection quality itself still varies frame
    to frame with lighting/occlusion, so frames_per_camera > 1 reduces
    single-frame sampling noise in the precision/recall estimate)."""
    model = yolo_marking_point_detector.load(model_path)
    no_label_dir = Path(no_label_dir)
    ground_truth_by_camera = _polygons_by_camera(labels_path, missed_path)

    ious = []
    true_positives = false_positives = false_negatives = 0
    cameras_with_frames = 0
    frames_evaluated = 0

    for camera_id in val_cameras:
        ground_truth = ground_truth_by_camera.get(camera_id, [])
        camera_dir = no_label_dir / camera_id
        frame_paths = sorted(camera_dir.glob("*.jpg"))[:frames_per_camera] if camera_dir.is_dir() else []
        if not frame_paths:
            continue
        cameras_with_frames += 1

        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            frames_evaluated += 1
            shape = image.shape[:2]

            detections = yolo_marking_point_detector.detect_slots(image, model, conf=conf, calibration=calibration)
            matched_gt = set()
            for det in detections:
                best_iou, best_idx = 0.0, None
                for idx, gt_poly in enumerate(ground_truth):
                    if idx in matched_gt:
                        continue
                    iou = _iou(det["polygon"], gt_poly, shape)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx
                if best_iou >= iou_match_threshold:
                    matched_gt.add(best_idx)
                    true_positives += 1
                    ious.append(best_iou)
                else:
                    false_positives += 1
            false_negatives += len(ground_truth) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    return {
        "cameras_requested": len(val_cameras),
        "cameras_with_frames": cameras_with_frames,
        "frames_evaluated": frames_evaluated,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "mean_iou_of_matches": round(float(np.mean(ious)), 3) if ious else 0.0,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8-pose marking-point model against held-out real polygon labels.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--no-label-dir", default="no_label")
    parser.add_argument("--val-cameras", nargs="+", default=None,
                         help="default: recompute the same held-out split export_marking_points.py used")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--labels", default=str(review_store.LABELS_PATH))
    parser.add_argument("--missed", default=str(review_store.MISSED_PATH))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-match-threshold", type=float, default=0.3)
    parser.add_argument("--frames-per-camera", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    val_cameras = args.val_cameras or _default_val_cameras(args.labels, args.missed, args.val_every)
    result = evaluate(args.model, args.no_label_dir, val_cameras, args.labels, args.missed,
                       conf=args.conf, iou_match_threshold=args.iou_match_threshold,
                       frames_per_camera=args.frames_per_camera)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
