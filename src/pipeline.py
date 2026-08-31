import cv2
import numpy as np

from calibration import LocalView
from parking_slot import SlotConfig
from perspective import plane_points_to_pixel, plane_to_pixel_homography
from renderer import render_label

DEFAULT_PATCH_SIZE = (300, 300)
DEFAULT_LOCAL_F = 300.0

# ponytail: label position/size is a fixed rule, not a real per-image
# windshield detection -- "차가 있다고 상상하고 그리라는거야", "차 없을때 주차장
# 차 선에 배치해야지" (imagine a car parked there; with no car, place it from
# the slot's own lines). Centered, wide-short rect (a real windshield's
# proportions) so it reads as "recognizing a windshield" rather than a plain
# square -- sized to most of the slot's own footprint (not just a small
# windshield-only patch) so it reads clearly at a glance; too small looked
# like a stray sticker rather than "a car is parked here". The slot's own
# LocalView already tilts this correctly with
# radius from the fisheye center when unrectified back into the raw frame
# (see _prepare_slot_view), this just makes that tilt visually legible.
FIXED_CANDIDATE_POINT = (0.5, 0.5)
# slot-proportioned portrait rect (u = entrance width, v = depth): a parking
# slot is a long rectangle, and the label should read as one -- the previous
# wide-short (0.85, 0.6) rendered as a squarish box the user explicitly
# flagged ("주차장 슬롯은 직사각형인데... 정사각형이 아니라").
FIXED_LABEL_WIDTH = 0.62
FIXED_LABEL_HEIGHT = 0.88


def _polygon_centroid(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    return tuple(pts.mean(axis=0))


def _find_slot(config, slot_id, config_path):
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        raise ValueError(f"slot id '{slot_id}' not found in {config_path}")
    return slot


def _canonicalize_quad_start(polygon_local):
    """Rotate the quad's corner order so edge0->edge1 is always the shorter
    pair of opposite edges (entrance/width), edge1->edge2 the longer pair
    (depth) -- fit_quad() (slot_detection.py, yolo_slot_detector.py) gives
    no guarantee about which corner comes first relative to the slot's real
    width/depth axes. With a square label (old FIXED_LABEL_WIDTH ==
    FIXED_LABEL_HEIGHT) that ambiguity was invisible; with the 2:1
    windshield-shaped label it rendered rotated 90 degrees for whichever
    slots happened to start on the depth edge instead of the width edge."""
    lengths = [np.linalg.norm(polygon_local[(i + 1) % 4] - polygon_local[i]) for i in range(4)]
    start = int(np.argmin(lengths))
    return np.roll(polygon_local, -start, axis=0)


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

    polygon_local = _canonicalize_quad_start(polygon_local)
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


def _label_box_plane(config, adjusted_slots, slot_id):
    """A slot's current label placement in the slot's own normalized plane
    coordinates (cx, cy, w, h). Precedence: per-photo adjusted override >
    the slot's own saved default (config slot's "label_box", set by the web
    UI's shift-adjust = "use this placement on every photo") > the camera's
    label_spec width/height > the fixed default."""
    box = (adjusted_slots or {}).get(slot_id)
    if box:
        return box["cx"], box["cy"], box["w"], box["h"]
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    slot_box = (slot or {}).get("label_box")
    if slot_box:
        return slot_box["cx"], slot_box["cy"], slot_box["w"], slot_box["h"]
    cx, cy = FIXED_CANDIDATE_POINT
    w = config.label_spec.get("width", FIXED_LABEL_WIDTH)
    h = config.label_spec.get("height", FIXED_LABEL_HEIGHT)
    return cx, cy, w, h


def _raw_quad_override(config, adjusted_slots, slot_id):
    """A user-pinned raw-pixel label quad, if one exists (per-photo override
    first, then the slot's saved default). Interactive move/resize edits are
    stored as raw quads: round-tripping every edit through the slot plane
    made the on-screen box drift and change size (fisheye reprojection), so
    an edited label stays EXACTLY where the user put it, in pixels."""
    box = (adjusted_slots or {}).get(slot_id)
    if box and box.get("quad_raw"):
        return box["quad_raw"]
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    slot_box = (slot or {}).get("label_box")
    if slot_box and slot_box.get("quad_raw"):
        return slot_box["quad_raw"]
    return None


def label_box_raw_pixels(config, slot, slot_id, adjusted_slots=None):
    """A slot's current label box, as a raw-image-pixel quad -- lets the
    results web UI draw/edit label placement directly on the original
    uploaded photo instead of a separate rectified-patch popup."""
    quad = _raw_quad_override(config, adjusted_slots, slot_id)
    if quad is not None:
        return quad
    view, homography = _prepare_slot_view(config, slot, slot_id)
    cx, cy, w, h = _label_box_plane(config, adjusted_slots, slot_id)
    corners_norm = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
    corners_local = plane_points_to_pixel(homography, corners_norm)
    return view.local_to_raw(corners_local).tolist()


def _draw_raw_quad_label(image, quad, label_spec):
    """Draw a label quad directly in raw-image pixels (same style as
    renderer.render_label, minus the local-view round trip)."""
    color = tuple(int(c) for c in label_spec["color"])
    alpha = float(label_spec.get("alpha", 1.0))
    border_width = int(label_spec.get("border_width", 3))
    poly = np.asarray(quad, dtype=np.float64).reshape(-1, 1, 2).astype(np.int32)
    overlay = image.copy()
    cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=border_width, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def _quads_overlap(quad_a, quad_b):
    a = np.asarray(quad_a, dtype=np.float32)
    b = np.asarray(quad_b, dtype=np.float32)
    inter_area, _ = cv2.intersectConvexConvex(a, b)
    return inter_area > 1e-6


def _overlapping_label_slots(config, slots, adjusted_slots):
    """Slot ids whose rendered label quad would overlap (or sit inside) a
    higher-confidence slot's label quad -- overlapping labels are never
    acceptable in output ("라벨이 겹치는건 무조건 빼줘, 라벨 안에 있는거도
    안돼"), so the lower-confidence one is skipped entirely."""
    quads = []
    for slot in slots:
        try:
            quad = label_box_raw_pixels(config, slot, slot["id"], adjusted_slots)
        except (ValueError, cv2.error):
            continue  # unrenderable slots error out in the main loop anyway
        quads.append((slot.get("confidence") if slot.get("confidence") is not None else 2.0,
                      slot["id"], quad))
    # user-drawn slots have no confidence -- treat as strongest (2.0 > any model conf)
    quads.sort(key=lambda t: t[0], reverse=True)
    kept, dropped = [], set()
    for conf, slot_id, quad in quads:
        if any(_quads_overlap(quad, kq) for _, _, kq in kept):
            dropped.add(slot_id)
            continue
        kept.append((conf, slot_id, quad))
    return dropped


def run_auto_all(config_path, raw_image_path, output_path, excluded_slots=None, adjusted_slots=None):
    excluded_slots = set(excluded_slots or ())
    adjusted_slots = adjusted_slots or {}

    config = SlotConfig.load(config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    active_slots = [s for s in config.slots if s["id"] not in excluded_slots]
    overlap_dropped = _overlapping_label_slots(config, active_slots, adjusted_slots)

    result_image = raw
    results = {}
    for slot in config.slots:
        slot_id = slot["id"]
        if slot_id in excluded_slots:
            results[slot_id] = "excluded"
            continue
        if slot_id in overlap_dropped:
            results[slot_id] = "skipped-overlap"
            continue
        try:
            pinned_quad = _raw_quad_override(config, adjusted_slots, slot_id)
            if pinned_quad is not None:
                # user-pinned raw quad: draw exactly where it was placed
                result_image = _draw_raw_quad_label(result_image, pinned_quad, config.label_spec)
                results[slot_id] = "labeled"
                continue
            view, homography = _prepare_slot_view(config, slot, slot_id)
            label_spec = dict(config.label_spec)
            cx, cy, w, h = _label_box_plane(config, adjusted_slots, slot_id)
            candidate_point = (cx, cy)
            label_spec["width"] = w
            label_spec["height"] = h
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
