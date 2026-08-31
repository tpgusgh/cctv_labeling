import json

import storage


def _override_path(batch_id, camera_id, photo_stem):
    d = storage.overrides_dir(batch_id, camera_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{photo_stem}.json"


def load_override(batch_id, camera_id, photo_stem):
    path = _override_path(batch_id, camera_id, photo_stem)
    if not path.exists():
        return {"excluded_slots": [], "adjusted": {}}
    return json.loads(path.read_text())


def save_override(batch_id, camera_id, photo_stem, override):
    path = _override_path(batch_id, camera_id, photo_stem)
    path.write_text(json.dumps(override))


def exclude_slot(batch_id, camera_id, photo_stem, slot_id):
    override = load_override(batch_id, camera_id, photo_stem)
    if slot_id not in override["excluded_slots"]:
        override["excluded_slots"].append(slot_id)
    override["adjusted"].pop(slot_id, None)
    save_override(batch_id, camera_id, photo_stem, override)
    return override


def adjust_slot(batch_id, camera_id, photo_stem, slot_id, box):
    override = load_override(batch_id, camera_id, photo_stem)
    if slot_id in override["excluded_slots"]:
        override["excluded_slots"].remove(slot_id)
    override["adjusted"][slot_id] = box
    save_override(batch_id, camera_id, photo_stem, override)
    return override


def restore_slot(batch_id, camera_id, photo_stem, slot_id):
    """Drop every per-photo override for this slot -- un-hides an excluded
    label and reverts a per-photo position adjustment back to the slot's
    default placement."""
    override = load_override(batch_id, camera_id, photo_stem)
    if slot_id in override["excluded_slots"]:
        override["excluded_slots"].remove(slot_id)
    override["adjusted"].pop(slot_id, None)
    save_override(batch_id, camera_id, photo_stem, override)
    return override
