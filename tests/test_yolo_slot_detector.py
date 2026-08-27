import numpy as np
import pytest

import yolo_slot_detector


class _FakeConf:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeMasks:
    def __init__(self, polygons):
        self.xy = polygons


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _FakeConf(confs)


class _FakeResult:
    def __init__(self, polygons, confs):
        self.masks = _FakeMasks(polygons) if polygons else None
        self.boxes = _FakeBoxes(confs)


class _FakeModel:
    def __init__(self, polygons, confs):
        self._polygons = polygons
        self._confs = confs

    def predict(self, image, conf, verbose):
        return [_FakeResult(self._polygons, self._confs)]


def test_detect_slots_converts_mask_polygons_to_quads():
    polygon = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.float32)
    model = _FakeModel(polygons=[polygon], confs=[0.87])

    detections = yolo_slot_detector.detect_slots(np.zeros((100, 150, 3), dtype=np.uint8), model)

    assert len(detections) == 1
    assert len(detections[0]["polygon"]) == 4
    assert detections[0]["confidence"] == 0.87


def test_detect_slots_skips_degenerate_zero_point_polygon():
    degenerate = np.zeros((0, 2), dtype=np.float32)
    valid = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.float32)
    model = _FakeModel(polygons=[degenerate, valid], confs=[0.5, 0.87])

    detections = yolo_slot_detector.detect_slots(np.zeros((100, 150, 3), dtype=np.uint8), model)

    assert len(detections) == 1
    assert detections[0]["confidence"] == 0.87


def test_detect_slots_returns_empty_list_when_no_masks():
    model = _FakeModel(polygons=None, confs=[])

    detections = yolo_slot_detector.detect_slots(np.zeros((100, 150, 3), dtype=np.uint8), model)

    assert detections == []


def test_load_constructs_YOLO_with_given_path(monkeypatch):
    captured = {}

    class _FakeYOLO:
        def __init__(self, path):
            captured["path"] = path

    monkeypatch.setattr(yolo_slot_detector, "YOLO", _FakeYOLO)

    yolo_slot_detector.load("models/yolov8_seg_slots.pt")

    assert captured["path"] == "models/yolov8_seg_slots.pt"
