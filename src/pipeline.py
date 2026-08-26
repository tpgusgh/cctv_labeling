import cv2
import numpy as np

from calibration import LocalView
from parking_slot import SlotConfig
from perspective import plane_to_pixel_homography
from renderer import render_label

DEFAULT_PATCH_SIZE = (300, 300)
DEFAULT_LOCAL_F = 300.0


def _polygon_centroid(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    return tuple(pts.mean(axis=0))


def run(config_path, raw_image_path, slot_id, candidate_point, output_path):
    config = SlotConfig.load(config_path)
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    center_raw = _polygon_centroid(slot["polygon_raw"])
    view = LocalView.centered_on(config.calibration, center_raw, DEFAULT_PATCH_SIZE, DEFAULT_LOCAL_F)

    local_patch = view.rectify(raw)
    polygon_local = view.raw_to_local(slot["polygon_raw"])
    homography = plane_to_pixel_homography(polygon_local)
    composited_local = render_label(local_patch, homography, candidate_point, config.label_spec)
    final = view.unrectify_into(composited_local, raw)

    if not cv2.imwrite(output_path, final):
        raise ValueError(f"could not write output image to {output_path}")
    return final
