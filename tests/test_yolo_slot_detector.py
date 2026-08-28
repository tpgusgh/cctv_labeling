import numpy as np
import pytest

import yolo_slot_detector
from calibration import CalibrationModel
from slot_detection import fit_quad


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


def test_detect_slots_dewarps_image_and_redistorts_polygon_when_calibration_given():
    captured = {}

    class _CapturingModel:
        def __init__(self, polygon, conf):
            self._polygon = polygon
            self._conf = conf

        def predict(self, image, conf, verbose):
            captured["image"] = image
            return [_FakeResult([self._polygon], [self._conf])]

    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    raw_image = np.zeros((640, 640, 3), dtype=np.uint8)
    # a clean rectangle in the dewarped/rectified image the model "sees"
    rectified_polygon = np.array([[420, 420], [520, 420], [520, 520], [420, 520]], dtype=np.float32)
    model = _CapturingModel(rectified_polygon, 0.9)

    detections = yolo_slot_detector.detect_slots(raw_image, model, calibration=calibration)

    # the model must be run on the dewarped image, not the raw fisheye one
    expected_dewarped = calibration.undistort_image(raw_image)
    assert np.array_equal(captured["image"], expected_dewarped)

    # the returned polygon must be mapped back into raw pixel space, since
    # everything downstream (perspective.py, pipeline.py) expects raw coords
    expected_quad = calibration.distort_points(fit_quad(rectified_polygon, inset_px=6.0))
    assert np.array(detections[0]["polygon"]) == pytest.approx(expected_quad, abs=1e-6)


def test_load_constructs_YOLO_with_given_path(monkeypatch):
    captured = {}

    class _FakeYOLO:
        def __init__(self, path):
            captured["path"] = path

    monkeypatch.setattr(yolo_slot_detector, "YOLO", _FakeYOLO)

    yolo_slot_detector.load("models/yolov8_seg_slots.pt")

    assert captured["path"] == "models/yolov8_seg_slots.pt"
