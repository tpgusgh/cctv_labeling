import overrides
import storage


def test_load_override_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    result = overrides.load_override("batch1", "cam1", "photo1")
    assert result == {"excluded_slots": [], "adjusted": {}, "undo": [], "redo": []}


def test_exclude_slot_then_adjust_slot_removes_from_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    overrides.exclude_slot("batch1", "cam1", "photo1", "slot-0")
    result = overrides.load_override("batch1", "cam1", "photo1")
    assert result["excluded_slots"] == ["slot-0"]

    overrides.adjust_slot("batch1", "cam1", "photo1", "slot-0", {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6})
    result = overrides.load_override("batch1", "cam1", "photo1")
    assert result["excluded_slots"] == []
    assert result["adjusted"] == {"slot-0": {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6}}
