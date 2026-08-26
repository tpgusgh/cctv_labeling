import cv2

from calibration import CalibrationModel
from parking_slot import SlotConfig
from main import main

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"


def test_cli_writes_output_png(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    slots = [{"id": "slot-A", "polygon_raw": [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]}]
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


def test_cli_auto_flag_writes_output_for_visible_windshield(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [{"id": "slot-A", "polygon_raw": [[243.0, 378.0], [385.0, 378.0], [385.0, 528.0], [243.0, 528.0]]}]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config_path = tmp_path / "config.json"
    SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", "no_label/P1_B1_1_21/20260820_115029.jpg",
        "--slot-id", "slot-A",
        "--auto",
        "--output", str(output_path),
    ])

    assert output_path.exists()


def test_cli_auto_flag_labels_even_an_empty_slot(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [{"id": "slot-A", "polygon_raw": [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]}]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config_path = tmp_path / "config.json"
    SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", "no_label/P1_B1_1_21/20260820_115029.jpg",
        "--slot-id", "slot-A",
        "--auto",
        "--output", str(output_path),
    ])

    assert output_path.exists()


def test_cli_auto_all_flag_labels_multiple_slots_without_slot_id(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [
        {"id": "car-slot", "polygon_raw": [[243.0, 378.0], [385.0, 378.0], [385.0, 528.0], [243.0, 528.0]]},
        {"id": "empty-slot", "polygon_raw": [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]},
    ]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config_path = tmp_path / "config.json"
    SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec).save(str(config_path))
    output_path = tmp_path / "final.png"

    main([
        "--config", str(config_path),
        "--image", "no_label/P1_B1_1_21/20260820_115029.jpg",
        "--auto-all",
        "--output", str(output_path),
    ])

    assert output_path.exists()
