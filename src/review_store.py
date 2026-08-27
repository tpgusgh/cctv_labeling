import hashlib
import json
from pathlib import Path

# ponytail: anchored to the repo root (not cwd) so review_server.py/
# generate_config.py find the same files regardless of which directory
# they're launched from -- cwd-relative defaults broke when run from src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_ROOT / "review"
LABELS_PATH = REVIEW_DIR / "labels.jsonl"
CANDIDATES_PATH = REVIEW_DIR / "candidates.jsonl"
CROPS_DIR = REVIEW_DIR / "crops"
# ponytail: human-drawn "a slot was missed here" boxes -- training data only
# for a future region-proposal model, never written into a camera's config
# directly (that would be per-camera manual coordinate entry, which stays
# banned -- see docs/superpowers/specs/2026-08-26-review-feedback-classifier-design.md).
MISSED_PATH = REVIEW_DIR / "missed.jsonl"


def candidate_id(camera_id, polygon):
    payload = json.dumps([camera_id, polygon], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _append(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_latest_by_id(path):
    path = Path(path)
    if not path.exists():
        return []
    latest = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            latest[record["id"]] = record
    return list(latest.values())


def append_decision(record, path=LABELS_PATH):
    _append(record, path)


def load_labels(path=LABELS_PATH):
    return _load_latest_by_id(path)


def append_candidate(record, path=CANDIDATES_PATH):
    _append(record, path)


def load_candidates(path=CANDIDATES_PATH):
    return _load_latest_by_id(path)


def unreviewed_ids(all_candidate_ids, labels):
    reviewed = {label["id"] for label in labels}
    return [cid for cid in all_candidate_ids if cid not in reviewed]


def append_missed_annotation(record, path=MISSED_PATH):
    _append(record, path)


def load_missed_annotations(path=MISSED_PATH):
    return _load_latest_by_id(path)


def remove_missed_annotation(candidate_id_, path=MISSED_PATH):
    remove_decision(candidate_id_, path)


def remove_decision(candidate_id_, path=LABELS_PATH):
    """Undo a review decision: drop every line for this id so it goes back
    to unreviewed (reappears in the review queue instead of the history)."""
    path = Path(path)
    if not path.exists():
        return
    kept = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["id"] != candidate_id_:
                kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""))
