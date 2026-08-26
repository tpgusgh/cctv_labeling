from calibration import CalibrationModel
from parking_slot import raw_clicks_to_slot_polygon, SlotConfig


def test_raw_clicks_to_slot_polygon_converts_through_calibration():
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    raw_points = [[250.0, 180.0], [400.0, 190.0], [410.0, 300.0], [240.0, 290.0]]

    polygon = raw_clicks_to_slot_polygon(raw_points, calibration)

    assert len(polygon) == 4
    assert polygon != raw_points


def test_slot_config_save_load_roundtrip(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=300.0)
    slots = [{"id": "P1_B1_1_1-A", "polygon_rectified": [[220.0, 150.0], [380.0, 150.0], [400.0, 280.0], [200.0, 280.0]]}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.25, "color": [30, 180, 90], "alpha": 0.75, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "P1_B1_1_1.json"

    config.save(str(path))
    loaded = SlotConfig.load(str(path))

    assert loaded.camera_id == "P1_B1_1_1"
    assert loaded.image_width == 640
    assert loaded.slots == slots
    assert loaded.label_spec == label_spec
    assert loaded.calibration.f == calibration.f
