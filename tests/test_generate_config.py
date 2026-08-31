import json

import cv2
import numpy as np
import pytest

import review_store

import generate_config
import yolo_slot_detector


def _make_frames_dir(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame1.jpg"), np.zeros((40, 40, 3), dtype=np.uint8))
    return frames_dir


def test_generate_config_uses_yolo_model_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_config, "_save_review_candidates", lambda *a, **k: None)
    # this test asserts exact polygon pass-through (yolo wiring), not the
    # local-view geometry snap -- neutralize regularize_quad for it
    monkeypatch.setattr(generate_config, "regularize_quad", lambda polygon, calibration: polygon)
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
    # global dewarp was tried and abandoned (see project memory /
    # docs/superpowers specs) -- generate_config() must NOT pass calibration
    # to the yolo path right now, production stays raw-image inference.
    assert captured["calibration"] is None
    assert slots == [{"id": "slot-0", "polygon_raw": fake_polygon, "confidence": 0.9}]
    assert needs_review is False
    assert json.loads(output_path.read_text())["camera_id"] == "cam-1"


def test_generate_config_auto_derives_calibration_from_actual_frame_size(tmp_path, monkeypatch):
    # a camera whose real frames aren't this project's usual 640x640 must
    # not get the hardcoded cx=cy=320/f=204/radius=320 defaults regardless
    # -- verified against a real 4000x3000 upload landing every slot in the
    # wrong place because the calibration model was centered on (320, 320)
    # of a frame ~6x that size.
    monkeypatch.setattr(generate_config, "_save_review_candidates", lambda *a, **k: None)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame1.jpg"), np.zeros((300, 400, 3), dtype=np.uint8))
    monkeypatch.setattr(generate_config, "detect_slots", lambda *a, **k: [])

    output_path = tmp_path / "config.json"
    generate_config.generate_config("cam-1", str(frames_dir), str(output_path))

    saved = json.loads(output_path.read_text())
    assert saved["image_width"] == 400
    assert saved["image_height"] == 300
    assert saved["calibration"]["center"] == [200.0, 150.0]
    scale = 300 / 640.0
    assert saved["calibration"]["f"] == pytest.approx(204.0 * scale)
    assert saved["calibration"]["radius"] == pytest.approx(320.0 * scale)


def test_generate_config_finds_uppercase_extension_frames(tmp_path, monkeypatch):
    # a real phone/camera export named "*.JPG" must not be silently skipped
    # -- glob.glob is case-sensitive on every platform regardless of the
    # filesystem's own case sensitivity (this broke a real upload).
    monkeypatch.setattr(generate_config, "_save_review_candidates", lambda *a, **k: None)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2.imwrite(str(frames_dir / "frame1.JPG"), np.zeros((40, 40, 3), dtype=np.uint8))

    def _fake_detect_slots(median_bgr, calibration, classifier=None):
        return []

    monkeypatch.setattr(generate_config, "detect_slots", _fake_detect_slots)

    output_path = tmp_path / "config.json"
    slots, needs_review = generate_config.generate_config("cam-1", str(frames_dir), str(output_path))

    assert slots == []
    assert needs_review is True


def test_auto_accept_agreed_candidates_only_accepts_agreement_count_2_plus(tmp_path):
    # explicit path override, not monkeypatching review_store.LABELS_PATH --
    # append_decision(record, path=LABELS_PATH) binds that default at
    # definition time, so patching the module constant afterward wouldn't
    # actually redirect it (same class of bug just fixed in storage.py's
    # CONFIG_DIR). Writing to the real project's review/labels.jsonl by
    # accident is exactly the mistake to avoid here.
    fake_labels_path = tmp_path / "labels.jsonl"
    agreed_polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    lone_polygon = [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]
    candidate_records = [
        {"id": "cid-agreed", "camera_id": "cam-1", "image_path": "x.jpg",
         "polygon": agreed_polygon, "crop_path": "x.png", "confidence": 0.8},
        {"id": "cid-lone", "camera_id": "cam-1", "image_path": "x.jpg",
         "polygon": lone_polygon, "crop_path": "y.png", "confidence": 0.6},
    ]
    detections = [
        {"polygon": agreed_polygon, "confidence": 0.8, "agreement_count": 2},
        {"polygon": lone_polygon, "confidence": 0.6, "agreement_count": 1},
    ]

    generate_config._auto_accept_agreed_candidates(candidate_records, detections, path=fake_labels_path)

    labels = review_store.load_labels(fake_labels_path)
    assert len(labels) == 1
    assert labels[0]["id"] == "cid-agreed"
    assert labels[0]["decision"] == "accept"


def test_suppress_rejected_regions_drops_rejected_only_region(tmp_path):
    # verified against v4: the trained model re-detects the exact
    # ceiling/green-floor regions a human already rejected (rejection only
    # removes a YOLO positive, it adds no negative signal) -- so
    # generate_config must suppress those regions itself from the review log.
    labels_path = tmp_path / "labels.jsonl"
    junk_polygon = [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]]
    review_store.append_decision(
        {"id": "cid-junk", "camera_id": "cam-1", "polygon": junk_polygon, "decision": "reject"}, labels_path)

    detections = [
        {"polygon": [[12.0, 11.0], [41.0, 11.0], [41.0, 41.0], [12.0, 41.0]], "confidence": 0.9},
        {"polygon": [[200.0, 200.0], [240.0, 200.0], [240.0, 240.0], [200.0, 240.0]], "confidence": 0.8},
    ]
    kept = generate_config._suppress_rejected_regions("cam-1", detections, labels_path=labels_path)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.8


def test_suppress_rejected_regions_keeps_region_that_also_has_an_accept(tmp_path):
    # a duplicate-of-a-real-slot rejection leaves an accept on the same
    # region -- that region is a real slot and must stay detectable.
    labels_path = tmp_path / "labels.jsonl"
    region = [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]]
    review_store.append_decision(
        {"id": "cid-dup", "camera_id": "cam-1", "polygon": region, "decision": "reject"}, labels_path)
    review_store.append_decision(
        {"id": "cid-real", "camera_id": "cam-1", "polygon": region, "decision": "accept"}, labels_path)

    detections = [{"polygon": region, "confidence": 0.9}]
    kept = generate_config._suppress_rejected_regions("cam-1", detections, labels_path=labels_path)
    assert kept == detections


def test_suppress_rejected_regions_ignores_other_cameras_rejects(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    region = [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]]
    review_store.append_decision(
        {"id": "cid-other", "camera_id": "cam-OTHER", "polygon": region, "decision": "reject"}, labels_path)

    detections = [{"polygon": region, "confidence": 0.9}]
    kept = generate_config._suppress_rejected_regions("cam-1", detections, labels_path=labels_path)
    assert kept == detections


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
