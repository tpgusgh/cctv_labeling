import json

import cv2
import numpy as np
import pytest

import export_marking_points


def _write_frame(path, size=20):
    cv2.imwrite(str(path), np.zeros((size, size, 3), dtype=np.uint8))


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_export_dataset_writes_pose_labels_derived_from_polygons(tmp_path):
    no_label_dir = tmp_path / "no_label"
    (no_label_dir / "camA").mkdir(parents=True)
    (no_label_dir / "camB").mkdir(parents=True)
    _write_frame(no_label_dir / "camA" / "frame1.jpg")
    _write_frame(no_label_dir / "camB" / "frame1.jpg")

    # a 10x30 rect on a 20x20 image: entrance edge (0,0)-(10,0), depth 30
    polygon = [[0, 0], [10, 0], [10, 30], [0, 30]]
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": "l1", "camera_id": "camA", "decision": "accept", "polygon": polygon},
        {"id": "l2", "camera_id": "camB", "decision": "accept", "polygon": polygon},
    ])
    missed_path = tmp_path / "missed_empty.jsonl"

    output_dir = tmp_path / "dataset"
    summary = export_marking_points.export_dataset(
        no_label_dir=no_label_dir, output_dir=output_dir,
        labels_path=labels_path, missed_path=missed_path, val_every=2)

    assert summary["val_cameras"] == ["camA"]
    assert summary["train_cameras"] == ["camB"]

    label_text = (output_dir / "labels" / "val" / "camA__frame1.txt").read_text().strip()
    parts = [float(x) for x in label_text.split()]
    assert parts[0] == 0  # class id
    # bbox: min/max of all 4 corners over a 20x20 image
    assert parts[1:5] == pytest.approx([0.25, 0.75, 0.5, 1.5])
    # keypoints: entrance corners (10,0) and (0,0) normalized (derive_marking_points
    # swaps to its fixed clockwise-inward convention), visibility=2
    assert parts[5:] == pytest.approx([0.5, 0.0, 2.0, 0.0, 0.0, 2.0])

    dataset_yaml = (output_dir / "dataset.yaml").read_text()
    assert "kpt_shape: [2, 3]" in dataset_yaml
    assert "parking_slot" in dataset_yaml


def test_export_dataset_raises_when_fewer_than_two_labeled_cameras(tmp_path):
    no_label_dir = tmp_path / "no_label"
    (no_label_dir / "camA").mkdir(parents=True)
    _write_frame(no_label_dir / "camA" / "frame1.jpg")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": "l1", "camera_id": "camA", "decision": "accept",
         "polygon": [[0, 0], [10, 0], [10, 30], [0, 30]]},
    ])
    missed_path = tmp_path / "missed_empty.jsonl"

    with pytest.raises(ValueError):
        export_marking_points.export_dataset(
            no_label_dir=no_label_dir, output_dir=tmp_path / "dataset",
            labels_path=labels_path, missed_path=missed_path)
