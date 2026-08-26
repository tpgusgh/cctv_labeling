import cv2

from calibration import CalibrationModel
from parking_slot import SlotConfig
from main import main

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def test_cli_writes_output_png(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    slots = [{"id": "slot-A", "polygon_rectified": [[220.0, 150.0], [380.0, 150.0], [400.0, 280.0], [200.0, 280.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.6, "color": [0, 255, 0], "alpha": 0.8, "text": None}
    config_path = tmp_path / "P1_B1_1_1.json"
    SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", SAMPLE_RAW_IMAGE,
        "--slot-id", "slot-A",
        "--candidate-u", "0.5",
        "--candidate-v", "0.5",
        "--output", str(output_path),
    ])

    written = cv2.imread(str(output_path))
    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert written is not None
    assert written.shape == raw.shape
