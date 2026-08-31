import json

import storage


def _override_path(batch_id, camera_id, photo_stem):
    d = storage.overrides_dir(batch_id, camera_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{photo_stem}.json"


MAX_HISTORY = 50


def _empty():
    return {"excluded_slots": [], "adjusted": {}, "undo": [], "redo": []}


def load_override(batch_id, camera_id, photo_stem):
    path = _override_path(batch_id, camera_id, photo_stem)
    if not path.exists():
        return _empty()
    data = json.loads(path.read_text())
    for key, default in _empty().items():
        data.setdefault(key, default)
    return data


def save_override(batch_id, camera_id, photo_stem, override):
    path = _override_path(batch_id, camera_id, photo_stem)
    path.write_text(json.dumps(override))


def _core(override):
    return {"excluded_slots": list(override["excluded_slots"]),
            "adjusted": dict(override["adjusted"])}


def _mutate(batch_id, camera_id, photo_stem, fn):
    """Apply an edit with undo history: the pre-edit state is pushed onto the
    undo stack and the redo stack is cleared (a new edit forks history)."""
    override = load_override(batch_id, camera_id, photo_stem)
    override["undo"] = (override["undo"] + [_core(override)])[-MAX_HISTORY:]
    override["redo"] = []
    fn(override)
    save_override(batch_id, camera_id, photo_stem, override)
    return override


def exclude_slot(batch_id, camera_id, photo_stem, slot_id):
    def fn(override):
        if slot_id not in override["excluded_slots"]:
            override["excluded_slots"].append(slot_id)
        override["adjusted"].pop(slot_id, None)
    return _mutate(batch_id, camera_id, photo_stem, fn)


def adjust_slot(batch_id, camera_id, photo_stem, slot_id, box):
    def fn(override):
        if slot_id in override["excluded_slots"]:
            override["excluded_slots"].remove(slot_id)
        override["adjusted"][slot_id] = box
    return _mutate(batch_id, camera_id, photo_stem, fn)


def restore_slot(batch_id, camera_id, photo_stem, slot_id):
    """Drop every per-photo override for this slot -- un-hides an excluded
    label and reverts a per-photo position adjustment back to the slot's
    default placement."""
    def fn(override):
        if slot_id in override["excluded_slots"]:
            override["excluded_slots"].remove(slot_id)
        override["adjusted"].pop(slot_id, None)
    return _mutate(batch_id, camera_id, photo_stem, fn)


def undo(batch_id, camera_id, photo_stem):
    override = load_override(batch_id, camera_id, photo_stem)
    if not override["undo"]:
        return None
    override["redo"] = (override["redo"] + [_core(override)])[-MAX_HISTORY:]
    prev = override["undo"].pop()
    override["excluded_slots"] = prev["excluded_slots"]
    override["adjusted"] = prev["adjusted"]
    save_override(batch_id, camera_id, photo_stem, override)
    return override


def redo(batch_id, camera_id, photo_stem):
    override = load_override(batch_id, camera_id, photo_stem)
    if not override["redo"]:
        return None
    override["undo"] = (override["undo"] + [_core(override)])[-MAX_HISTORY:]
    nxt = override["redo"].pop()
    override["excluded_slots"] = nxt["excluded_slots"]
    override["adjusted"] = nxt["adjusted"]
    save_override(batch_id, camera_id, photo_stem, override)
    return override
