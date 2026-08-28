import json

import cv2
import numpy as np

import generate_config
import yolo_slot_detector


def _make_frames_dir(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame1.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
    return frames_dir


def test_generate_config_uses_yolo_model_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_config, "_save_review_candidates", lambda *a, **k: None)
    frames_dir = _make_frames_dir(tmp_path)
    fake_polygon = [[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]]
    captured = {}

    def _fake_detect_slots(median_bgr, model, conf=0.25, calibration=None):
        captured["model"] = model
        captured["calibration"] = calibration
        return [{"polygon": fake_polygon, "confidence": 0.9}]

    # generate_config() imports yolo_slot_detector lazily; patch the shared
    # module object (sys.modules-cached) rather than a generate_config attribute.
    monkeypatch.setattr(yolo_slot_detector, "detect_slots", _fake_detect_slots)

    sentinel_model = object()
    output_path = tmp_path / "config.json"
    slots, needs_review = generate_config.generate_config(
        "cam-1", str(frames_dir), str(output_path), yolo_model=sentinel_model)

    assert captured["model"] is sentinel_model
    # generate_config() always builds a calibration -- the yolo path must
    # get it too, so it can dewarp before inference (see yolo_slot_detector).
    assert captured["calibration"] is not None
    assert slots == [{"id": "slot-0", "polygon_raw": fake_polygon}]
    assert needs_review is False
    assert json.loads(output_path.read_text())["camera_id"] == "cam-1"


def test_generate_config_uses_classical_detection_when_no_yolo_model(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_config, "_save_review_candidates", lambda *a, **k: None)
    frames_dir = _make_frames_dir(tmp_path)
    called = {}

    def _fake_yolo_detect_slots(median_bgr, model, conf=0.25, calibration=None):
        called["yolo"] = True
        return []

    monkeypatch.setattr(yolo_slot_detector, "detect_slots", _fake_yolo_detect_slots)

    output_path = tmp_path / "config.json"
    generate_config.generate_config("cam-1", str(frames_dir), str(output_path))

    assert "yolo" not in called
