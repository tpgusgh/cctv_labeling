import cv2

from parking_slot import SlotConfig
from perspective import plane_to_pixel_homography
from renderer import render_label


def run(config_path, raw_image_path, slot_id, candidate_point, output_path):
    config = SlotConfig.load(config_path)
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    rectified = config.calibration.undistort_image(raw)
    homography = plane_to_pixel_homography(slot["polygon_rectified"])
    composited_rectified = render_label(rectified, homography, candidate_point, config.label_spec)
    final = config.calibration.redistort_image(composited_rectified, output_shape=raw.shape[:2])

    cv2.imwrite(output_path, final)
    return final
