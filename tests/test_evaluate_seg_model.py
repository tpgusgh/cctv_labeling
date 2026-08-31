from evaluate_seg_model import score_detections


def test_score_detections_counts_recall_junk_and_neutral():
    slot_polygon = [[0.0, 0.0], [40.0, 0.0], [40.0, 90.0], [0.0, 90.0]]
    junk_polygon = [[200.0, 200.0], [240.0, 200.0], [240.0, 290.0], [200.0, 290.0]]
    new_polygon = [[400.0, 400.0], [440.0, 400.0], [440.0, 490.0], [400.0, 490.0]]

    labels = [
        {"camera_id": "cam-1", "polygon": slot_polygon, "decision": "accept"},
        {"camera_id": "cam-1", "polygon": junk_polygon, "decision": "reject"},
    ]
    configs = {"cam-1": [{"polygon_raw": slot_polygon}]}
    detections = {"cam-1": [
        {"polygon": [[1.0, 1.0], [41.0, 1.0], [41.0, 91.0], [1.0, 91.0]]},  # hits the slot
        {"polygon": junk_polygon},                                             # hits the junk region
        {"polygon": new_polygon},                                              # brand new region
    ]}

    r = score_detections(detections, configs, labels)
    assert r["on_accept"] == 1
    assert r["on_reject_only"] == 1
    assert r["neutral"] == 1
    assert r["config_recall_hits"] == 1 and r["config_slots"] == 1
    assert r["recall"] == 1.0
    assert abs(r["junk_rate"] - 1 / 3) < 1e-9


def test_score_detections_missed_config_slot_lowers_recall():
    slot_polygon = [[0.0, 0.0], [40.0, 0.0], [40.0, 90.0], [0.0, 90.0]]
    configs = {"cam-1": [{"polygon_raw": slot_polygon}]}
    r = score_detections({"cam-1": []}, configs, [])
    assert r["recall"] == 0.0
    assert r["config_slots"] == 1
