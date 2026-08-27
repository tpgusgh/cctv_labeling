import argparse
import glob
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

import review_store

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _polygons_by_camera(labels_path, missed_path):
    by_camera = defaultdict(list)
    for record in review_store.load_labels(labels_path):
        if record.get("decision") == "accept":
            by_camera[record["camera_id"]].append(record["polygon"])
    for record in review_store.load_missed_annotations(missed_path):
        by_camera[record["camera_id"]].append(record["polygon"])
    return by_camera


def _write_label_file(path, polygons, width, height):
    lines = []
    for polygon in polygons:
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in polygon)
        lines.append(f"0 {coords}")
    path.write_text("\n".join(lines) + "\n")


def export_dataset(no_label_dir, output_dir, labels_path=review_store.LABELS_PATH,
                    missed_path=review_store.MISSED_PATH, image_extensions=IMAGE_EXTENSIONS, val_every=5):
    no_label_dir = Path(no_label_dir)
    output_dir = Path(output_dir)

    polygons_by_camera = _polygons_by_camera(labels_path, missed_path)
    camera_ids = sorted(polygons_by_camera.keys())
    if len(camera_ids) < 2:
        raise ValueError(
            f"need at least 2 labeled cameras to make a train/val split, found {len(camera_ids)}")

    val_cameras = camera_ids[::val_every] or [camera_ids[-1]]
    train_cameras = [c for c in camera_ids if c not in val_cameras]

    summary = {"train_cameras": train_cameras, "val_cameras": val_cameras,
               "train_images": 0, "val_images": 0}

    for split, cameras in (("train", train_cameras), ("val", val_cameras)):
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in cameras:
            polygons = polygons_by_camera[camera_id]
            frame_paths = sorted(
                {p for ext in image_extensions for p in glob.glob(str(no_label_dir / camera_id / f"*{ext}"))})
            for frame_path in frame_paths:
                frame_path = Path(frame_path)
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue
                height, width = image.shape[:2]

                dest_name = f"{camera_id}__{frame_path.stem}"
                shutil.copy2(frame_path, images_dir / f"{dest_name}{frame_path.suffix}")
                _write_label_file(labels_dir / f"{dest_name}.txt", polygons, width, height)
                summary[f"{split}_images"] += 1

    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: parking_slot\n"
    )

    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export review_store's labeled polygons into a YOLO-seg training dataset.")
    parser.add_argument("--no-label-dir", default="no_label")
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", default=str(review_store.LABELS_PATH))
    parser.add_argument("--missed", default=str(review_store.MISSED_PATH))
    parser.add_argument("--val-every", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = export_dataset(args.no_label_dir, args.output, args.labels, args.missed, val_every=args.val_every)
    print(f"train: {summary['train_images']} images across {len(summary['train_cameras'])} cameras")
    print(f"val: {summary['val_images']} images across {len(summary['val_cameras'])} cameras")


if __name__ == "__main__":
    main()
