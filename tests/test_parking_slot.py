from calibration import CalibrationModel
from parking_slot import SlotConfig


def test_slot_config_save_load_roundtrip():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    slots = [{"id": "P1_B1_1_1-A", "polygon_raw": [[250.0, 180.0], [400.0, 190.0], [410.0, 300.0], [240.0, 290.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.25, "color": [30, 180, 90], "alpha": 0.75, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = "/tmp/test_slot_config_roundtrip.json"

    config.save(path)
    loaded = SlotConfig.load(path)

    assert loaded.camera_id == "P1_B1_1_1"
    assert loaded.image_width == 640
    assert loaded.slots == slots
    assert loaded.label_spec == label_spec
    assert loaded.calibration.f == calibration.f
