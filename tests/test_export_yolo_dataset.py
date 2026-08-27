import json

import cv2
import numpy as np
import pytest

import export_yolo_dataset


def _write_frame(path, size=20):
    cv2.imwrite(str(path), np.zeros((size, size, 3), dtype=np.uint8))


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_export_dataset_replicates_camera_polygons_across_all_its_frames(tmp_path):
    no_label_dir = tmp_path / "no_label"
    (no_label_dir / "camA").mkdir(parents=True)
    (no_label_dir / "camB").mkdir(parents=True)
    _write_frame(no_label_dir / "camA" / "frame1.jpg")
    _write_frame(no_label_dir / "camA" / "frame2.jpg")
    _write_frame(no_label_dir / "camB" / "frame1.jpg")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": "l1", "camera_id": "camA", "decision": "accept",
         "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"id": "l2", "camera_id": "camA", "decision": "reject",
         "polygon": [[5, 5], [15, 5], [15, 15], [5, 15]]},
    ])
    missed_path = tmp_path / "missed.jsonl"
    _write_jsonl(missed_path, [
        {"id": "m1", "camera_id": "camB", "polygon": [[2, 2], [8, 2], [8, 8], [2, 8]]},
    ])

    output_dir = tmp_path / "dataset"
    summary = export_yolo_dataset.export_dataset(
        no_label_dir=no_label_dir, output_dir=output_dir,
        labels_path=labels_path, missed_path=missed_path, val_every=2)

    # camA is index 0 of sorted(["camA", "camB"]) -> val; camB -> train
    assert summary["val_cameras"] == ["camA"]
    assert summary["train_cameras"] == ["camB"]
    assert summary["val_images"] == 2  # camA has 2 frames
    assert summary["train_images"] == 1  # camB has 1 frame

    assert (output_dir / "images" / "val" / "camA__frame1.jpg").exists()
    assert (output_dir / "images" / "val" / "camA__frame2.jpg").exists()
    assert (output_dir / "images" / "train" / "camB__frame1.jpg").exists()

    # camA's accept polygon [[0,0],[10,0],[10,10],[0,10]] on a 20x20 image
    # normalizes to (0,0) (0.5,0) (0.5,0.5) (0,0.5); the reject polygon must
    # NOT appear.
    label_text = (output_dir / "labels" / "val" / "camA__frame1.txt").read_text().strip()
    lines = label_text.splitlines()
    assert len(lines) == 1
    parts = [float(x) for x in lines[0].split()]
    assert parts[0] == 0  # class id
    assert parts[1:] == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5])

    # camB's missed-annotation polygon [[2,2],[8,2],[8,8],[2,8]] on 20x20
    label_text = (output_dir / "labels" / "train" / "camB__frame1.txt").read_text().strip()
    parts = [float(x) for x in label_text.split()]
    assert parts[1:] == pytest.approx([0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4])

    dataset_yaml = (output_dir / "dataset.yaml").read_text()
    assert "train: images/train" in dataset_yaml
    assert "val: images/val" in dataset_yaml
    assert "parking_slot" in dataset_yaml


def test_export_dataset_raises_when_fewer_than_two_labeled_cameras(tmp_path):
    no_label_dir = tmp_path / "no_label"
    (no_label_dir / "camA").mkdir(parents=True)
    _write_frame(no_label_dir / "camA" / "frame1.jpg")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": "l1", "camera_id": "camA", "decision": "accept",
         "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    ])
    missed_path = tmp_path / "missed_empty.jsonl"

    with pytest.raises(ValueError):
        export_yolo_dataset.export_dataset(
            no_label_dir=no_label_dir, output_dir=tmp_path / "dataset",
            labels_path=labels_path, missed_path=missed_path)
