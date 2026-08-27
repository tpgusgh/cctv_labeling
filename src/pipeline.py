import cv2
import numpy as np

from calibration import LocalView
from parking_slot import SlotConfig
from perspective import plane_to_pixel_homography
from renderer import render_label

DEFAULT_PATCH_SIZE = (300, 300)
DEFAULT_LOCAL_F = 300.0

# ponytail: label position/size is a fixed rule, not a real per-image
# windshield detection -- "차가 있다고 상상하고 그리라는거야", "차 없을때 주차장
# 차 선에 배치해야지" (imagine a car parked there; with no car, place it from
# the slot's own lines). Centered + a bit over half the slot in each
# dimension stays inside the slot's boundary lines and covers a plausible
# windshield zone regardless of which way a car would face.
FIXED_CANDIDATE_POINT = (0.5, 0.5)
FIXED_LABEL_WIDTH = 0.68
FIXED_LABEL_HEIGHT = 0.68


def _polygon_centroid(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    return tuple(pts.mean(axis=0))


def _find_slot(config, slot_id, config_path):
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")
    return slot


def _prepare_slot_view(config, slot, slot_id):
    center_raw = _polygon_centroid(slot["polygon_raw"])
    if not np.all(np.isfinite(center_raw)):
        raise ValueError(f"slot '{slot_id}' has a degenerate polygon_raw (empty or non-finite): {slot['polygon_raw']}")

    view = LocalView.centered_on(config.calibration, center_raw, DEFAULT_PATCH_SIZE, DEFAULT_LOCAL_F)

    corner_rays = view._local_rays(slot["polygon_raw"])
    patch_w, patch_h = DEFAULT_PATCH_SIZE
    if not np.all(corner_rays[:, 2] > 0):
        raise ValueError(f"slot '{slot_id}' has a polygon_raw corner behind the local view (camera cannot represent it)")

    polygon_local = view.raw_to_local(slot["polygon_raw"])
    if not (np.all(polygon_local[:, 0] >= 0) and np.all(polygon_local[:, 0] <= patch_w - 1)
            and np.all(polygon_local[:, 1] >= 0) and np.all(polygon_local[:, 1] <= patch_h - 1)):
        raise ValueError(f"slot '{slot_id}' polygon_raw falls outside the local patch bounds for the current DEFAULT_PATCH_SIZE/DEFAULT_LOCAL_F")

    homography = plane_to_pixel_homography(polygon_local)
    return view, homography


def run(config_path, raw_image_path, slot_id, candidate_point, output_path):
    config = SlotConfig.load(config_path)
    slot = _find_slot(config, slot_id, config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    view, homography = _prepare_slot_view(config, slot, slot_id)

    local_patch = view.rectify(raw)
    composited_local = render_label(local_patch, homography, candidate_point, config.label_spec)
    final = view.unrectify_into(composited_local, raw)

    if not cv2.imwrite(output_path, final):
        raise ValueError(f"could not write output image to {output_path}")
    return final


def run_auto_all(config_path, raw_image_path, output_path, excluded_slots=None, adjusted_slots=None):
    excluded_slots = set(excluded_slots or ())
    adjusted_slots = adjusted_slots or {}

    config = SlotConfig.load(config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    result_image = raw
    results = {}
    for slot in config.slots:
        slot_id = slot["id"]
        if slot_id in excluded_slots:
            results[slot_id] = "excluded"
            continue
        try:
            view, homography = _prepare_slot_view(config, slot, slot_id)
            label_spec = dict(config.label_spec)
            box = adjusted_slots.get(slot_id)
            if box:
                candidate_point = (box["cx"], box["cy"])
                label_spec["width"] = box["w"]
                label_spec["height"] = box["h"]
            else:
                candidate_point = FIXED_CANDIDATE_POINT
                label_spec.setdefault("width", FIXED_LABEL_WIDTH)
                label_spec.setdefault("height", FIXED_LABEL_HEIGHT)
            local_patch = view.rectify(result_image)
            composited_local = render_label(local_patch, homography, candidate_point, label_spec)
            result_image = view.unrectify_into(composited_local, result_image)
            results[slot_id] = "labeled"
        except (ValueError, cv2.error) as e:
            results[slot_id] = f"error: {e}"

    if not cv2.imwrite(output_path, result_image):
        raise ValueError(f"could not write output image to {output_path}")
    return results


def run_auto(config_path, raw_image_path, slot_id, output_path):
    config = SlotConfig.load(config_path)
    slot = _find_slot(config, slot_id, config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    view, homography = _prepare_slot_view(config, slot, slot_id)

    label_spec = dict(config.label_spec)
    label_spec.setdefault("width", FIXED_LABEL_WIDTH)
    label_spec.setdefault("height", FIXED_LABEL_HEIGHT)

    local_patch = view.rectify(raw)
    composited_local = render_label(local_patch, homography, FIXED_CANDIDATE_POINT, label_spec)
    final = view.unrectify_into(composited_local, raw)

    if not cv2.imwrite(output_path, final):
        raise ValueError(f"could not write output image to {output_path}")
    return final
