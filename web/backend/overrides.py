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
    override["undo"] = (override["undo"] + [{"kind": "override", "state": _core(override)}])[-MAX_HISTORY:]
    override["redo"] = []
    fn(override)
    save_override(batch_id, camera_id, photo_stem, override)
    return override


def push_add(batch_id, camera_id, photo_stem, slot_id, polygon):
    """Record a pen-drawn slot addition in this photo's history so undo can
    remove it again (the slot itself lives in the camera config -- the app
    layer applies the config change when this entry is undone/redone)."""
    override = load_override(batch_id, camera_id, photo_stem)
    override["undo"] = (override["undo"] + [{"kind": "add_slot", "slot_id": slot_id,
                                              "polygon": polygon}])[-MAX_HISTORY:]
    override["redo"] = []
    save_override(batch_id, camera_id, photo_stem, override)


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
    """Pop one history entry. Returns the popped entry (so the app layer can
    apply config-level parts like removing a pen-added slot), or None if
    there is nothing to undo."""
    override = load_override(batch_id, camera_id, photo_stem)
    if not override["undo"]:
        return None
    entry = override["undo"].pop()
    if entry["kind"] == "override":
        override["redo"] = (override["redo"] + [{"kind": "override", "state": _core(override)}])[-MAX_HISTORY:]
        override["excluded_slots"] = entry["state"]["excluded_slots"]
        override["adjusted"] = entry["state"]["adjusted"]
    else:  # add_slot: the config change is the app layer's job
        override["redo"] = (override["redo"] + [entry])[-MAX_HISTORY:]
    save_override(batch_id, camera_id, photo_stem, override)
    return entry


def redo(batch_id, camera_id, photo_stem):
    override = load_override(batch_id, camera_id, photo_stem)
    if not override["redo"]:
        return None
    entry = override["redo"].pop()
    if entry["kind"] == "override":
        override["undo"] = (override["undo"] + [{"kind": "override", "state": _core(override)}])[-MAX_HISTORY:]
        override["excluded_slots"] = entry["state"]["excluded_slots"]
        override["adjusted"] = entry["state"]["adjusted"]
    else:
        override["undo"] = (override["undo"] + [entry])[-MAX_HISTORY:]
    save_override(batch_id, camera_id, photo_stem, override)
    return entry
