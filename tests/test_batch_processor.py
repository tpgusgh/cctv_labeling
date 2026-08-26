import json

from calibration import CalibrationModel
from parking_slot import SlotConfig
from batch_processor import process_camera_folder

CAMERA_FOLDER = "no_label/P1_B1_1_21"


def _write_batch_test_config(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [
        {"id": "car-slot", "polygon_raw": [[243.0, 378.0], [385.0, 378.0], [385.0, 528.0], [243.0, 528.0]]},
        {"id": "elongated-slot", "polygon_raw": [[513.0, 366.0], [610.0, 366.0], [610.0, 482.0], [513.0, 482.0]]},
    ]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config = SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec)
    config_path = tmp_path / "config.json"
    config.save(str(config_path))
    return str(config_path)


def test_process_camera_folder_labels_every_frame_from_fixed_slot_geometry(tmp_path):
    config_path = _write_batch_test_config(tmp_path)
    output_dir = tmp_path / "output"
    review_dir = tmp_path / "review"
    log_path = tmp_path / "log.json"

    entries = process_camera_folder(config_path, CAMERA_FOLDER, str(output_dir), str(review_dir), str(log_path))

    # label placement is a fixed rule from each slot's own polygon, not
    # content-dependent detection -- with valid slot geometry, every frame in
    # the folder succeeds regardless of what's actually in the frame.
    assert len(entries) == 4
    by_image = {e["image"]: e for e in entries}
    for image_name in ("20260820_115029.jpg", "20260820_030003.jpg", "20260820_030144.jpg", "20260820_030324.jpg"):
        assert by_image[image_name]["status"] == "success"
        assert (output_dir / by_image[image_name]["image"].replace(".jpg", ".png")).exists()

    assert list(review_dir.iterdir()) == []

    with open(log_path) as f:
        logged = json.load(f)
    assert logged == entries


def test_process_camera_folder_records_error_without_stopping_other_images(tmp_path):
    output_dir = tmp_path / "output"
    review_dir = tmp_path / "review"
    log_path = tmp_path / "log.json"

    entries = process_camera_folder(
        "no_such_config.json", CAMERA_FOLDER, str(output_dir), str(review_dir), str(log_path)
    )

    assert len(entries) == 4
    assert all(e["status"] == "error" for e in entries)
    assert all(e["output"] is None for e in entries)
