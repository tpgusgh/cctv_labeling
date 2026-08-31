import cv2
import numpy as np
import pytest

from types import SimpleNamespace

from calibration import CalibrationModel
from parking_slot import SlotConfig
import pipeline as pipeline_module
from pipeline import _canonicalize_quad_start, run, run_auto, run_auto_all


def test_overlapping_label_slots_drops_lower_confidence_overlap():
    # two slots whose label quads overlap: lower-confidence one must be
    # dropped ("라벨이 겹치는건 무조건 빼줘") -- containment counts too.
    quads = {
        "slot-a": [[0, 0], [40, 0], [40, 90], [0, 90]],
        "slot-b": [[10, 10], [30, 10], [30, 60], [10, 60]],   # inside slot-a
        "slot-c": [[200, 200], [240, 200], [240, 290], [200, 290]],  # far away
    }
    config = SimpleNamespace(
        slots=[{"id": "slot-a", "confidence": 0.9},
               {"id": "slot-b", "confidence": 0.5},
               {"id": "slot-c", "confidence": 0.4}],
        label_spec={},
    )
    import pipeline as pm
    orig = pm.label_box_raw_pixels
    pm.label_box_raw_pixels = lambda cfg, slot, sid, adj: quads[sid]
    try:
        dropped = pm._overlapping_label_slots(config, config.slots, {})
    finally:
        pm.label_box_raw_pixels = orig
    assert dropped == {"slot-b"}


def test_label_box_plane_precedence_slot_default_between_override_and_fixed():
    config = SimpleNamespace(
        slots=[{"id": "slot-0", "label_box": {"cx": 0.4, "cy": 0.45, "w": 0.5, "h": 0.3}}],
        label_spec={},
    )
    # per-photo override wins over the slot default
    assert pipeline_module._label_box_plane(
        config, {"slot-0": {"cx": 0.1, "cy": 0.2, "w": 0.3, "h": 0.4}}, "slot-0") == (0.1, 0.2, 0.3, 0.4)
    # no override -> the slot's own saved default (web shift-adjust)
    assert pipeline_module._label_box_plane(config, {}, "slot-0") == (0.4, 0.45, 0.5, 0.3)
    # a slot without a saved default falls back to the fixed constants
    config.slots[0].pop("label_box")
    cx, cy, w, h = pipeline_module._label_box_plane(config, {}, "slot-0")
    assert (cx, cy) == pipeline_module.FIXED_CANDIDATE_POINT
    assert (w, h) == (pipeline_module.FIXED_LABEL_WIDTH, pipeline_module.FIXED_LABEL_HEIGHT)

SAMPLE_RAW_IMAGE = "no_label/P1_B1_1_1/20260820_030004.jpg"

# Near-periphery slot: raw radius ~260-300px from (320,320), the region where
# the old global-rectify approach zero-filled real content (see
# docs/superpowers/specs/2026-08-26-local-gnomonic-rectification-design.md).
PERIPHERAL_POLYGON_RAW = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]


def _write_test_config(tmp_path, polygon_raw=None):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0)
    slots = [{"id": "slot-A", "polygon_raw": polygon_raw or PERIPHERAL_POLYGON_RAW}]
    label_spec = {"shape": "rect", "width": 0.6, "height": 0.6, "color": [0, 255, 0], "alpha": 0.8, "text": None}
    config = SlotConfig("P1_B1_1_1", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "P1_B1_1_1.json"
    config.save(str(path))
    return str(path)


def test_canonicalize_quad_start_rotates_to_shortest_edge_first():
    # a wide-short rectangle whose corner order happens to start on the
    # long (depth) edge instead of the short (width/entrance) edge --
    # fit_quad() gives no guarantee either way.
    quad = np.array([[0.0, 0.0], [0.0, 100.0], [40.0, 100.0], [40.0, 0.0]])
    canonical = _canonicalize_quad_start(quad)
    assert np.linalg.norm(canonical[1] - canonical[0]) < np.linalg.norm(canonical[2] - canonical[1])


def test_canonicalize_quad_start_is_a_noop_when_already_shortest_first():
    quad = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 100.0], [0.0, 100.0]])
    canonical = _canonicalize_quad_start(quad)
    np.testing.assert_array_equal(canonical, quad)


def test_pipeline_composites_label_near_periphery_without_corrupting_far_pixels(tmp_path):
    config_path = _write_test_config(tmp_path)
    output_path = str(tmp_path / "final.png")

    raw = cv2.imread(SAMPLE_RAW_IMAGE)
    assert raw is not None

    final = run(config_path, SAMPLE_RAW_IMAGE, "slot-A", (0.5, 0.5), output_path)

    assert final.shape == raw.shape

    pts = np.asarray(PERIPHERAL_POLYGON_RAW)
    x_min, y_min = (pts.min(axis=0) - 20).astype(int)
    x_max, y_max = (pts.max(axis=0) + 20).astype(int)
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, 640), min(y_max, 640)

    bbox_before = raw[y_min:y_max, x_min:x_max]
    bbox_after = final[y_min:y_max, x_min:x_max]
    assert not np.array_equal(bbox_before, bbox_after)

    far_before = raw[550:580, 550:580]
    far_after = final[550:580, 550:580]
    np.testing.assert_array_equal(far_before, far_after)


def test_pipeline_raises_for_unknown_slot_id(tmp_path):
    config_path = _write_test_config(tmp_path)
    with pytest.raises(ValueError):
        run(config_path, SAMPLE_RAW_IMAGE, "does-not-exist", (0.5, 0.5), str(tmp_path / "out.png"))


def test_pipeline_raises_for_unreadable_raw_image(tmp_path):
    config_path = _write_test_config(tmp_path)
    with pytest.raises(ValueError):
        run(config_path, "no_such_file.jpg", "slot-A", (0.5, 0.5), str(tmp_path / "out.png"))


def test_pipeline_raises_when_candidate_maps_outside_local_patch_bounds(tmp_path):
    config_path = _write_test_config(tmp_path)
    with pytest.raises(ValueError):
        run(config_path, SAMPLE_RAW_IMAGE, "slot-A", (5.0, 5.0), str(tmp_path / "out.png"))


CAR_SAMPLE_IMAGE = "no_label/P1_B1_1_21/20260820_115029.jpg"
# The windshield blob detected by detect_windshields() on this image has
# bbox (283, 418, 62, 70) / centroid ~(316, 456) -- a white car with a black
# windshield roughly centered-low in frame (verified visually). This box is
# that bbox expanded by a 40px margin in each direction: large enough to
# comfortably contain the blob (with room for detection jitter) but still
# small enough that its raw polygon projects inside the 300x300 local
# rectified patch used by _prepare_slot_view (DEFAULT_PATCH_SIZE/DEFAULT_LOCAL_F).
# A larger, more "visually generous" box (e.g. 60-100px margin, or the
# original coarse estimate [[200,260],[460,260],[460,500],[200,500]]) maps
# outside the local patch bounds and makes _prepare_slot_view raise ValueError.
CAR_SLOT_POLYGON_RAW = [[243.0, 378.0], [385.0, 378.0], [385.0, 528.0], [243.0, 528.0]]
EMPTY_SLOT_POLYGON_RAW = [[280.0, 20.0], [360.0, 20.0], [360.0, 100.0], [280.0, 100.0]]


def _write_auto_test_config(tmp_path, polygon_raw):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [{"id": "slot-A", "polygon_raw": polygon_raw}]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config = SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "config.json"
    config.save(str(path))
    return str(path)


def test_run_auto_labels_slot_with_car(tmp_path):
    config_path = _write_auto_test_config(tmp_path, CAR_SLOT_POLYGON_RAW)
    output_path = tmp_path / "final.png"

    raw = cv2.imread(CAR_SAMPLE_IMAGE)
    assert raw is not None

    final = run_auto(config_path, CAR_SAMPLE_IMAGE, "slot-A", str(output_path))

    assert final is not None
    assert final.shape == raw.shape
    assert not np.array_equal(final, raw)
    assert output_path.exists()


def test_run_auto_labels_empty_slot_too(tmp_path):
    # label placement is a fixed rule driven by the slot's own geometry, not
    # real car/windshield detection -- an empty slot still gets labeled.
    config_path = _write_auto_test_config(tmp_path, EMPTY_SLOT_POLYGON_RAW)
    output_path = tmp_path / "final.png"

    result = run_auto(config_path, CAR_SAMPLE_IMAGE, "slot-A", str(output_path))

    assert result is not None
    assert output_path.exists()


def _write_multi_slot_config(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [
        {"id": "car-slot", "polygon_raw": CAR_SLOT_POLYGON_RAW},
        {"id": "empty-slot", "polygon_raw": EMPTY_SLOT_POLYGON_RAW},
    ]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config = SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec)
    path = tmp_path / "config.json"
    config.save(str(path))
    return str(path)


def test_run_auto_all_labels_every_slot(tmp_path):
    config_path = _write_multi_slot_config(tmp_path)
    output_path = tmp_path / "final.png"

    raw = cv2.imread(CAR_SAMPLE_IMAGE)
    assert raw is not None

    results = run_auto_all(config_path, CAR_SAMPLE_IMAGE, str(output_path))

    assert results["car-slot"] == "labeled"
    assert results["empty-slot"] == "labeled"
    assert output_path.exists()

    final = cv2.imread(str(output_path))
    assert final.shape == raw.shape
    assert not np.array_equal(final, raw)


def test_run_auto_all_records_error_without_aborting_other_slots(tmp_path):
    calibration = CalibrationModel(cx=320.0, cy=320.0, f=204.0, radius=320.0)
    slots = [
        # too large to fit the fixed local patch, so _prepare_slot_view
        # raises ValueError for this slot specifically (see CAR_SLOT_POLYGON_RAW
        # comment above for the original measurement).
        {"id": "bad-slot", "polygon_raw": [[200.0, 260.0], [460.0, 260.0], [460.0, 500.0], [200.0, 500.0]]},
        {"id": "car-slot", "polygon_raw": CAR_SLOT_POLYGON_RAW},
    ]
    label_spec = {"shape": "rect", "color": [235, 206, 135], "alpha": 1.0, "text": None, "border_width": 3}
    config = SlotConfig("P1_B1_1_21", 640, 640, calibration, slots, label_spec)
    config_path = tmp_path / "config.json"
    config.save(str(config_path))
    output_path = tmp_path / "final.png"

    results = run_auto_all(str(config_path), CAR_SAMPLE_IMAGE, str(output_path))

    assert results["car-slot"].startswith("labeled")
    assert results["bad-slot"].startswith("error:")
    assert output_path.exists()


def test_run_auto_all_excludes_requested_slot(tmp_path):
    config_path = _write_test_config(tmp_path)
    output_path = str(tmp_path / "excluded.png")

    results = run_auto_all(config_path, SAMPLE_RAW_IMAGE, output_path, excluded_slots={"slot-A"})

    assert results["slot-A"] == "excluded"


def test_run_auto_all_applies_adjusted_slot_box(tmp_path):
    config_path = _write_test_config(tmp_path)
    output_path = str(tmp_path / "adjusted.png")

    results = run_auto_all(
        config_path, SAMPLE_RAW_IMAGE, output_path,
        adjusted_slots={"slot-A": {"cx": 0.3, "cy": 0.3, "w": 0.4, "h": 0.4}},
    )

    assert results["slot-A"] == "labeled"


def test_run_auto_all_defaults_keep_existing_behavior(tmp_path):
    config_path = _write_test_config(tmp_path)
    output_path_a = str(tmp_path / "a.png")
    output_path_b = str(tmp_path / "b.png")

    results_no_override = run_auto_all(config_path, SAMPLE_RAW_IMAGE, output_path_a)
    results_empty_override = run_auto_all(
        config_path, SAMPLE_RAW_IMAGE, output_path_b, excluded_slots=set(), adjusted_slots={})

    assert results_no_override == results_empty_override
