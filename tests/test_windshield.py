import numpy as np
import cv2

from calibration import CalibrationModel
from windshield import detect_windshields, confidence_score, WindshieldBlob

CAR_SAMPLE_IMAGE = "no_label/P1_B1_1_21/20260820_115029.jpg"


def test_detect_windshields_finds_at_least_one_blob_in_real_car_frame():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    raw = cv2.imread(CAR_SAMPLE_IMAGE)
    assert raw is not None

    blobs = detect_windshields(raw, calibration)

    assert len(blobs) >= 1


def test_detect_windshields_masks_out_background_outside_floor_circle():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=200.0)
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (300, 300), (340, 340), (10, 10, 10), -1)  # inside circle
    cv2.rectangle(img, (10, 10), (50, 50), (10, 10, 10), -1)      # outside circle

    blobs = detect_windshields(img, calibration)

    centroids = [b.centroid for b in blobs]
    assert any(280 < cx < 360 and 280 < cy < 360 for cx, cy in centroids)
    assert not any(0 < cx < 60 and 0 < cy < 60 for cx, cy in centroids)


def test_detect_windshields_rejects_elongated_reflection_streaks():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    # compact, roughly square blob -- a plausible windshield shape
    cv2.rectangle(img, (200, 200), (240, 235), (10, 10, 10), -1)
    # thin elongated streak -- a plausible glossy-floor light-reflection artifact
    cv2.rectangle(img, (300, 300), (310, 450), (10, 10, 10), -1)

    blobs = detect_windshields(img, calibration)

    centroids = [b.centroid for b in blobs]
    assert any(200 < cx < 240 and 200 < cy < 235 for cx, cy in centroids)
    assert not any(300 < cx < 310 and 300 < cy < 450 for cx, cy in centroids)


def test_confidence_score_favors_compact_shapes_over_elongated_ones():
    compact = WindshieldBlob(contour=None, bbox=(0, 0, 62, 70), centroid=(31, 35), area=1624)  # aspect ~1.13
    elongated = WindshieldBlob(contour=None, bbox=(0, 0, 17, 36), centroid=(8, 18), area=379)   # aspect ~2.12

    assert confidence_score(compact) > confidence_score(elongated)
    assert 0.0 <= confidence_score(elongated) <= 1.0
    assert 0.0 <= confidence_score(compact) <= 1.0
