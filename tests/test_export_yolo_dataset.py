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
    assert summary["val_images"] == 3  # camA: 2 frames + 1 median composite
    assert summary["train_images"] == 2  # camB: 1 frame + 1 median composite

    assert (output_dir / "images" / "val" / "camA__frame1.jpg").exists()
    assert (output_dir / "images" / "val" / "camA__frame2.jpg").exists()
    assert (output_dir / "images" / "train" / "camB__frame1.jpg").exists()

    # median-stack composite is included too, since that's what
    # generate_config.py actually runs inference on -- not just raw frames.
    assert (output_dir / "images" / "val" / "camA__median.jpg").exists()
    assert (output_dir / "images" / "train" / "camB__median.jpg").exists()
    median_label = (output_dir / "labels" / "val" / "camA__median.txt").read_text().strip()
    parts = [float(x) for x in median_label.split()]
    assert parts[1:] == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5])

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


def test_export_dataset_rerun_does_not_leak_camera_into_both_splits(tmp_path):
    # 3 cameras, sorted: camA, camB, camC.
    no_label_dir = tmp_path / "no_label"
    for cam in ("camA", "camB", "camC"):
        (no_label_dir / cam).mkdir(parents=True)
        _write_frame(no_label_dir / cam / "frame1.jpg")

    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(labels_path, [
        {"id": f"l-{cam}", "camera_id": cam, "decision": "accept",
         "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
        for cam in ("camA", "camB", "camC")
    ])
    missed_path = tmp_path / "missed_empty.jsonl"

    output_dir = tmp_path / "dataset"

    # val_every=3 -> val_cameras = camA only (index 0), rest train.
    export_yolo_dataset.export_dataset(
        no_label_dir=no_label_dir, output_dir=output_dir,
        labels_path=labels_path, missed_path=missed_path, val_every=3)
    assert (output_dir / "images" / "train" / "camB__frame1.jpg").exists()

    # val_every=2 -> val_cameras = camA, camC (indices 0, 2); camB moves... stays train.
    # Use val_every=1 so every camera is now val, moving camB out of train.
    export_yolo_dataset.export_dataset(
        no_label_dir=no_label_dir, output_dir=output_dir,
        labels_path=labels_path, missed_path=missed_path, val_every=1)

    assert (output_dir / "images" / "val" / "camB__frame1.jpg").exists()
    assert not (output_dir / "images" / "train" / "camB__frame1.jpg").exists()
    assert not (output_dir / "labels" / "train" / "camB__frame1.txt").exists()


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
