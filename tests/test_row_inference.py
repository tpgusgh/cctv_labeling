from row_inference import rescue_row_adjacent


def _quad(cx, cy, w=20, h=50):
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


CENTER = (320, 320)


def test_rescues_weak_detection_next_to_confirmed_slot():
    # strong slot and weak detection side by side on the same ring (top row)
    strong = [{"polygon": _quad(300, 80), "confidence": 0.9}]
    weak = [{"polygon": _quad(325, 80), "confidence": 0.1}]
    rescued = rescue_row_adjacent(weak, strong, CENTER)
    assert len(rescued) == 1
    assert rescued[0]["source"] == "row-rescue"
    assert rescued[0]["confidence"] == 0.1


def test_drops_isolated_weak_detection():
    strong = [{"polygon": _quad(300, 80), "confidence": 0.9}]
    # same ring radius but far away in angle -- not a row neighbor
    weak = [{"polygon": _quad(560, 320), "confidence": 0.1}]
    assert rescue_row_adjacent(weak, strong, CENTER) == []


def test_drops_weak_detection_at_different_radius():
    strong = [{"polygon": _quad(300, 80), "confidence": 0.9}]
    # same angle but much closer to the center (driving area)
    weak = [{"polygon": _quad(305, 240), "confidence": 0.1}]
    assert rescue_row_adjacent(weak, strong, CENTER) == []


def test_no_strong_slots_means_no_rescue():
    weak = [{"polygon": _quad(325, 80), "confidence": 0.1}]
    assert rescue_row_adjacent(weak, [], CENTER) == []
