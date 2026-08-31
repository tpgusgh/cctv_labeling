import numpy as np
import pytest

import yolo_marking_point_detector
from marking_point_geometry import reconstruct_slot_quad


class _FakeConf:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeKeypoints:
    def __init__(self, xy):
        self.xy = xy


class _FakeBoxes:
    def __init__(self, confs):
        self.conf = _FakeConf(confs)


class _FakeResult:
    def __init__(self, keypoints, confs):
        self.keypoints = _FakeKeypoints(keypoints) if keypoints is not None else None
        self.boxes = _FakeBoxes(confs)


class _FakeModel:
    def __init__(self, keypoints, confs):
        self._keypoints = keypoints
        self._confs = confs

    def predict(self, image, conf, verbose):
        return [_FakeResult(self._keypoints, self._confs)]


def test_detect_slots_reconstructs_quad_from_marking_point_pair():
    pair = np.array([[10.0, 10.0], [40.0, 10.0]])
    model = _FakeModel(keypoints=[pair], confs=[0.9])

    detections = yolo_marking_point_detector.detect_slots(
        np.zeros((100, 150, 3), dtype=np.uint8), model, depth_prior_px=20.0, calibration=None)

    assert len(detections) == 1
    assert detections[0]["confidence"] == 0.9
    expected = reconstruct_slot_quad(pair[0], pair[1], 20.0)
    assert np.array_equal(detections[0]["polygon"], expected)


def test_detect_slots_uses_calibration_aware_reconstruction_by_default():
    pair = np.array([[300.0, 300.0], [340.0, 300.0]])
    model = _FakeModel(keypoints=[pair], confs=[0.8])

    detections = yolo_marking_point_detector.detect_slots(
        np.zeros((640, 640, 3), dtype=np.uint8), model, depth_prior_px=20.0)

    expected = reconstruct_slot_quad(
        pair[0], pair[1], 20.0, calibration=yolo_marking_point_detector.DEFAULT_CALIBRATION)
    assert np.array(detections[0]["polygon"]) == pytest.approx(expected)


def test_detect_slots_uses_radius_aware_depth_when_not_given_explicitly():
    near_center_pair = np.array([[300.0, 320.0], [340.0, 320.0]])
    near_edge_pair = np.array([[300.0, 620.0], [340.0, 620.0]])
    model = _FakeModel(keypoints=[near_center_pair, near_edge_pair], confs=[0.9, 0.9])

    detections = yolo_marking_point_detector.detect_slots(
        np.zeros((640, 640, 3), dtype=np.uint8), model)

    center_depth = yolo_marking_point_detector._radius_aware_depth_prior(
        near_center_pair[0], near_center_pair[1], yolo_marking_point_detector.DEFAULT_CALIBRATION)
    edge_depth = yolo_marking_point_detector._radius_aware_depth_prior(
        near_edge_pair[0], near_edge_pair[1], yolo_marking_point_detector.DEFAULT_CALIBRATION)

    assert edge_depth > center_depth
    expected_center = reconstruct_slot_quad(
        near_center_pair[0], near_center_pair[1], center_depth,
        calibration=yolo_marking_point_detector.DEFAULT_CALIBRATION)
    expected_edge = reconstruct_slot_quad(
        near_edge_pair[0], near_edge_pair[1], edge_depth,
        calibration=yolo_marking_point_detector.DEFAULT_CALIBRATION)
    assert np.array(detections[0]["polygon"]) == pytest.approx(expected_center)
    assert np.array(detections[1]["polygon"]) == pytest.approx(expected_edge)


def test_detect_slots_returns_empty_list_when_no_keypoints():
    model = _FakeModel(keypoints=None, confs=[])

    detections = yolo_marking_point_detector.detect_slots(np.zeros((100, 150, 3), dtype=np.uint8), model)

    assert detections == []


def test_load_constructs_YOLO_with_given_path(monkeypatch):
    captured = {}

    class _FakeYOLO:
        def __init__(self, path):
            captured["path"] = path

    monkeypatch.setattr(yolo_marking_point_detector, "YOLO", _FakeYOLO)

    yolo_marking_point_detector.load("models/yolov8_pose_marking_points.pt")

    assert captured["path"] == "models/yolov8_pose_marking_points.pt"
