import review_store


def test_candidate_id_stable_for_same_camera_and_polygon():
    poly = [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]
    assert review_store.candidate_id("cam-1", poly) == review_store.candidate_id("cam-1", poly)
    assert review_store.candidate_id("cam-1", poly) != review_store.candidate_id("cam-2", poly)


def test_append_and_load_labels_round_trip(tmp_path):
    path = tmp_path / "labels.jsonl"
    record = {"id": "abc", "camera_id": "cam-1", "decision": "accept", "confidence": 0.9}
    review_store.append_decision(record, path)

    labels = review_store.load_labels(path)
    assert labels == [record]


def test_load_labels_missing_file_returns_empty_list(tmp_path):
    assert review_store.load_labels(tmp_path / "nope.jsonl") == []


def test_append_decision_keeps_latest_for_same_id(tmp_path):
    path = tmp_path / "labels.jsonl"
    review_store.append_decision({"id": "abc", "decision": "reject"}, path)
    review_store.append_decision({"id": "abc", "decision": "accept"}, path)

    labels = review_store.load_labels(path)
    assert len(labels) == 1
    assert labels[0]["decision"] == "accept"


def test_unreviewed_ids_excludes_ids_with_a_label():
    labels = [{"id": "a", "decision": "accept"}]
    result = review_store.unreviewed_ids(["a", "b", "c"], labels)
    assert result == ["b", "c"]


def test_remove_decision_makes_candidate_unreviewed_again(tmp_path):
    path = tmp_path / "labels.jsonl"
    review_store.append_decision({"id": "a", "decision": "reject"}, path)
    review_store.append_decision({"id": "b", "decision": "accept"}, path)

    review_store.remove_decision("a", path)

    labels = review_store.load_labels(path)
    assert [label["id"] for label in labels] == ["b"]


def test_remove_decision_on_missing_file_is_a_noop(tmp_path):
    review_store.remove_decision("a", tmp_path / "nope.jsonl")


def test_missed_annotation_round_trip_and_removal(tmp_path):
    path = tmp_path / "missed.jsonl"
    record = {"id": "m1", "camera_id": "cam-1", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    review_store.append_missed_annotation(record, path)

    assert review_store.load_missed_annotations(path) == [record]

    review_store.remove_missed_annotation("m1", path)
    assert review_store.load_missed_annotations(path) == []
