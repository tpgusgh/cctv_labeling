# Web Labeling App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local React+Flask website: drag/select a folder (or many camera folders at once, up to ~100), auto-detect/label every photo per camera in the background, let the user delete or nudge individual slot labels on individual photos, and download the results — wrapped in one `run_web.sh` launcher.

**Architecture:** Flask backend (`web/backend/`) that imports the existing `src/` pipeline functions directly (no duplicated detection/labeling logic) and runs per-camera jobs on a thread pool; a React+Vite frontend (`web/frontend/`) that uploads via a native folder `<input>`, polls job status, and edits/downloads results. Per-photo label edits are stored as small JSON override files next to the uploaded photos — the shared `config/<camera_id>.json` slot geometry is never touched.

**Tech Stack:** Flask (new dependency), Python stdlib `concurrent.futures`/`threading`, React 18 + Vite (new dependency, frontend-only), native browser `<input webkitdirectory>` for folder selection (no upload library).

**Spec:** `docs/superpowers/specs/2026-08-27-web-labeling-app-design.md`

## Global Constraints

- Never write to `config/<camera_id>.json` from the web layer except via the existing `generate_config.generate_config()` call for a brand-new camera — per-photo edits go into override sidecar files only.
- `web/backend/` and `web/frontend/` add exactly two new runtime dependencies: Flask (backend) and the React/Vite toolchain (frontend, dev-time + built static assets only). No other new dependency.
- All web backend modules use flat imports (`import review_store`, `import pipeline`, ...) matching this repo's existing convention (no `src.` or `web.` package prefixes) — achieved by adding `web/backend` to `sys.path` alongside the existing `src` entry.
- Folder uploads are grouped by immediate parent directory name (the last path segment before the filename), not the first segment — this makes "select one folder" and "select a parent folder containing many camera subfolders" both work without special-casing.
- Background job state is in-memory only (no persistence across server restarts) — matches `review_server.py`'s existing local/ephemeral precedent.
- Frontend gets no automated tests (matches `review_server.py` precedent — manual verification only); backend gets full pytest coverage via Flask's test client.

---

## Task 1: `pipeline.run_auto_all` — per-photo label overrides

**Files:**
- Modify: `src/pipeline.py:76-93` (the `run_auto_all` function)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `run_auto_all(config_path, raw_image_path, output_path, excluded_slots=None, adjusted_slots=None) -> dict[str, str]` — `excluded_slots` is an iterable of slot ids to skip entirely (marked `"excluded"` in the result dict); `adjusted_slots` is `dict[str, dict]` mapping slot id to `{"cx": float, "cy": float, "w": float, "h": float}` (same normalized 0-1 convention as the existing `FIXED_CANDIDATE_POINT`/`FIXED_LABEL_WIDTH`/`FIXED_LABEL_HEIGHT`). Both default to `None` (empty), so existing callers (`main.py`, `batch_processor.py`) are unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py` (uses the same `_write_test_config`/`SAMPLE_RAW_IMAGE` helpers already in that file):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -k "excludes_requested_slot or applies_adjusted_slot_box or defaults_keep_existing" -v`
Expected: FAIL with `TypeError: run_auto_all() got an unexpected keyword argument 'excluded_slots'`

- [ ] **Step 3: Implement**

Replace the body of `run_auto_all` in `src/pipeline.py`:

```python
def run_auto_all(config_path, raw_image_path, output_path, excluded_slots=None, adjusted_slots=None):
    excluded_slots = set(excluded_slots or ())
    adjusted_slots = adjusted_slots or {}

    config = SlotConfig.load(config_path)

    raw = cv2.imread(raw_image_path)
    if raw is None:
        raise ValueError(f"could not read image at {raw_image_path}")

    result_image = raw
    results = {}
    for slot in config.slots:
        slot_id = slot["id"]
        if slot_id in excluded_slots:
            results[slot_id] = "excluded"
            continue
        try:
            view, homography = _prepare_slot_view(config, slot, slot_id)
            label_spec = dict(config.label_spec)
            box = adjusted_slots.get(slot_id)
            if box:
                candidate_point = (box["cx"], box["cy"])
                label_spec["width"] = box["w"]
                label_spec["height"] = box["h"]
            else:
                candidate_point = FIXED_CANDIDATE_POINT
                label_spec.setdefault("width", FIXED_LABEL_WIDTH)
                label_spec.setdefault("height", FIXED_LABEL_HEIGHT)
            local_patch = view.rectify(result_image)
            composited_local = render_label(local_patch, homography, candidate_point, label_spec)
            result_image = view.unrectify_into(composited_local, result_image)
            results[slot_id] = "labeled"
        except (ValueError, cv2.error) as e:
            results[slot_id] = f"error: {e}"

    if not cv2.imwrite(output_path, result_image):
        raise ValueError(f"could not write output image to {output_path}")
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: all PASS (including the pre-existing tests in this file — this change must not break them)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (56+ tests)

- [ ] **Step 6: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: support per-slot label exclude/adjust overrides in run_auto_all"
```

---

## Task 2: `review_store` web flags + `review_server` priority queue

**Files:**
- Modify: `src/review_store.py`
- Modify: `src/review_server.py`
- Test: `tests/test_review_store.py`

**Interfaces:**
- Consumes: existing `review_store._append`/`_load_latest_by_id` helpers, `review_store.candidate_id`.
- Produces: `review_store.WEB_FLAGS_PATH`, `review_store.append_web_flag(record, path=WEB_FLAGS_PATH)`, `review_store.load_web_flags(path=WEB_FLAGS_PATH)` — the web backend (Task 6) will call `append_web_flag` when a photo's label is deleted.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_review_store.py`:

```python
def test_web_flag_round_trip(tmp_path):
    path = tmp_path / "web_flags.jsonl"
    record = {"id": "abc", "camera_id": "cam-1", "slot_id": "slot-0", "photo": "20260101_1.jpg"}
    review_store.append_web_flag(record, path)

    assert review_store.load_web_flags(path) == [record]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_review_store.py -k web_flag -v`
Expected: FAIL with `AttributeError: module 'review_store' has no attribute 'append_web_flag'`

- [ ] **Step 3: Implement in `src/review_store.py`**

Add near the other path constants:

```python
WEB_FLAGS_PATH = REVIEW_DIR / "web_flags.jsonl"
```

Add near `append_candidate`/`load_candidates`:

```python
def append_web_flag(record, path=WEB_FLAGS_PATH):
    _append(record, path)


def load_web_flags(path=WEB_FLAGS_PATH):
    return _load_latest_by_id(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_review_store.py -v`
Expected: all PASS

- [ ] **Step 5: Wire flagged candidates into the review queue priority**

In `src/review_server.py`, replace `_next_unreviewed`:

```python
def _next_unreviewed(candidates, labels):
    by_id = {c["id"]: c for c in candidates}
    ids = review_store.unreviewed_ids(list(by_id.keys()), labels)
    if not ids:
        return None
    flagged_ids = {f["id"] for f in review_store.load_web_flags()}
    flagged_first = [i for i in ids if i in flagged_ids] + [i for i in ids if i not in flagged_ids]
    return by_id[flagged_first[0]]
```

In `_render_page`, show a flagged count. Change the home page body-building line:

```python
def _render_page(candidate):
    flagged_count = len(review_store.load_web_flags())
    flag_note = f"<p>웹에서 지적된 후보: {flagged_count}개 (우선 표시됨)</p>" if flagged_count else ""
    body = "<h1>남은 후보 없음 (전부 리뷰 완료)</h1>" if candidate is None else f"""
{flag_note}
<h1>후보 리뷰</h1>
<p>카메라: {candidate['camera_id']} | 신뢰도: {candidate['confidence']}</p>
<img src="/crops/{candidate['id']}.png" style="max-width:480px;border:1px solid #333">
<p>
<button onclick="decide('accept')">승인 (진짜 슬롯)</button>
<button onclick="decide('reject')">거부 (슬롯 아님)</button>
</p>
<script>
function decide(decision) {{
  fetch('/decide', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id: '{candidate["id"]}', decision: decision}})
  }}).then(() => location.reload());
}}
</script>"""
    return f"<html><body>{body}<p><a href=\"/history\">리뷰 기록 보기</a> | <a href=\"/missed\">미탐 슬롯 표시</a></p></body></html>"
```

- [ ] **Step 6: Manual check**

Run `.venv/bin/python src/review_server.py --port 8899 &`, then:
```bash
curl -s -X POST http://localhost:8899/undo -H "Content-Type: application/json" -d '{"id":"nonexistent"}' >/dev/null  # sanity the server is up
kill %1
```
(Full manual verification of the flagged-priority behavior happens in Task 11's end-to-end check, once the web backend can actually write a flag.)

- [ ] **Step 7: Run the full suite and commit**

```bash
.venv/bin/pytest -q
git add src/review_store.py src/review_server.py tests/test_review_store.py
git commit -m "feat: add web_flags store and surface flagged candidates first in review queue"
```

---

## Task 3: Backend storage + overrides modules

**Files:**
- Create: `web/backend/storage.py`
- Create: `web/backend/overrides.py`
- Modify: `conftest.py` (repo root)
- Test: `web/backend/tests/test_storage.py`
- Test: `web/backend/tests/test_overrides.py`

**Interfaces:**
- Produces: `storage.PROJECT_ROOT`, `storage.WEB_UPLOADS_DIR`, `storage.batch_dir(batch_id)`, `storage.camera_upload_dir(batch_id, camera_id)`, `storage.labeled_dir(batch_id, camera_id)`, `storage.overrides_dir(batch_id, camera_id)`, `storage.save_uploaded_file(batch_id, camera_id, filename, file_storage)`.
- Produces: `overrides.load_override(batch_id, camera_id, photo_stem) -> {"excluded_slots": [...], "adjusted": {...}}`, `overrides.exclude_slot(batch_id, camera_id, photo_stem, slot_id)`, `overrides.adjust_slot(batch_id, camera_id, photo_stem, slot_id, box)`.
- Consumes: nothing from `src/` (pure path/JSON helpers) — Task 4 (`jobs.py`) is the first module here to import `src/` modules.

- [ ] **Step 1: Update `conftest.py` so `web/backend` modules are importable flatly**

Read current content first, then change:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "web" / "backend"))
```

- [ ] **Step 2: Write the failing tests**

Create `web/backend/tests/test_storage.py`:

```python
from pathlib import Path

import storage


def test_camera_upload_dir_nests_under_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    result = storage.camera_upload_dir("batch1", "P1_B1_1_9")
    assert result == tmp_path / "batch1" / "P1_B1_1_9" / "original"


def test_save_uploaded_file_writes_to_camera_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    class FakeFileStorage:
        def save(self, dest):
            Path(dest).write_bytes(b"fake-jpg-bytes")

    dest = storage.save_uploaded_file("batch1", "P1_B1_1_9", "photo.jpg", FakeFileStorage())

    assert dest.read_bytes() == b"fake-jpg-bytes"
    assert dest == tmp_path / "batch1" / "P1_B1_1_9" / "original" / "photo.jpg"
```

Create `web/backend/tests/test_overrides.py`:

```python
import overrides
import storage


def test_load_override_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    result = overrides.load_override("batch1", "cam1", "photo1")
    assert result == {"excluded_slots": [], "adjusted": {}}


def test_exclude_slot_then_adjust_slot_removes_from_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    overrides.exclude_slot("batch1", "cam1", "photo1", "slot-0")
    result = overrides.load_override("batch1", "cam1", "photo1")
    assert result["excluded_slots"] == ["slot-0"]

    overrides.adjust_slot("batch1", "cam1", "photo1", "slot-0", {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6})
    result = overrides.load_override("batch1", "cam1", "photo1")
    assert result["excluded_slots"] == []
    assert result["adjusted"] == {"slot-0": {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6}}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest web/backend/tests/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'storage'`

- [ ] **Step 4: Implement `web/backend/storage.py`**

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_UPLOADS_DIR = PROJECT_ROOT / "web_uploads"


def batch_dir(batch_id):
    return WEB_UPLOADS_DIR / batch_id


def camera_upload_dir(batch_id, camera_id):
    return batch_dir(batch_id) / camera_id / "original"


def labeled_dir(batch_id, camera_id):
    return batch_dir(batch_id) / camera_id / "labeled"


def overrides_dir(batch_id, camera_id):
    return batch_dir(batch_id) / camera_id / "overrides"


def save_uploaded_file(batch_id, camera_id, filename, file_storage):
    target_dir = camera_upload_dir(batch_id, camera_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    file_storage.save(str(dest))
    return dest
```

- [ ] **Step 5: Implement `web/backend/overrides.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest web/backend/tests/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add conftest.py web/backend/storage.py web/backend/overrides.py web/backend/tests/test_storage.py web/backend/tests/test_overrides.py
git commit -m "feat: add web backend storage path helpers and per-photo label overrides"
```

---

## Task 4: Backend background jobs

**Files:**
- Create: `web/backend/jobs.py`
- Test: `web/backend/tests/test_jobs.py`

**Interfaces:**
- Consumes: `storage.PROJECT_ROOT`, `storage.labeled_dir`; `generate_config.generate_config`; `pipeline.run_auto_all`; `slot_classifier.load`.
- Produces: `jobs.submit_camera_job(batch_id, camera_id, upload_dir, photo_count) -> job_id` (submits work to a background thread and returns immediately), `jobs.get_batch_status(batch_id) -> list[dict]`, `jobs.wait_for_job(job_id, timeout=10)` (test-only helper — polls until status is `done`/`error`).

- [ ] **Step 1: Write the failing test**

Create `web/backend/tests/test_jobs.py` — reuses the real `P1_B1_1_9` sample camera (already has a committed `config/P1_B1_1_9.json`, so this exercises the "camera already known, label only" path without needing to run full slot detection in a test):

```python
import shutil
from pathlib import Path

import jobs
import storage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_CAMERA = "P1_B1_1_9"
SAMPLE_PHOTO = REPO_ROOT / "no_label" / SAMPLE_CAMERA / "20260820_030004.jpg"


def test_submit_camera_job_labels_photos_for_known_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    upload_dir = storage.camera_upload_dir("batch1", SAMPLE_CAMERA)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    job_id = jobs.submit_camera_job("batch1", SAMPLE_CAMERA, upload_dir, photo_count=1)
    status = jobs.wait_for_job(job_id, timeout=30)

    assert status["status"] == "done"
    labeled_file = storage.labeled_dir("batch1", SAMPLE_CAMERA) / f"{SAMPLE_PHOTO.stem}.png"
    assert labeled_file.is_file()


def test_get_batch_status_returns_all_cameras_in_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    upload_dir = storage.camera_upload_dir("batch2", SAMPLE_CAMERA)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    job_id = jobs.submit_camera_job("batch2", SAMPLE_CAMERA, upload_dir, photo_count=1)
    jobs.wait_for_job(job_id, timeout=30)

    statuses = jobs.get_batch_status("batch2")
    assert len(statuses) == 1
    assert statuses[0]["camera_id"] == SAMPLE_CAMERA
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest web/backend/tests/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jobs'`

- [ ] **Step 3: Implement `web/backend/jobs.py`**

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import generate_config
import pipeline
import slot_classifier
import storage

_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LOCK = threading.Lock()
_JOBS = {}

MODEL_PATH = storage.PROJECT_ROOT / "models" / "slot_classifier.joblib"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _set_status(job_id, **fields):
    with _LOCK:
        _JOBS[job_id].update(fields)


def get_batch_status(batch_id):
    with _LOCK:
        return [dict(j) for j in _JOBS.values() if j["batch_id"] == batch_id]


def wait_for_job(job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _LOCK:
            job = dict(_JOBS[job_id])
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def submit_camera_job(batch_id, camera_id, upload_dir, photo_count):
    job_id = f"{batch_id}:{camera_id}"
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "batch_id": batch_id,
            "camera_id": camera_id,
            "status": "queued",
            "photo_count": photo_count,
            "error": None,
        }
    _EXECUTOR.submit(_process_camera, job_id, batch_id, camera_id, upload_dir)
    return job_id


def _process_camera(job_id, batch_id, camera_id, upload_dir):
    try:
        config_path = storage.PROJECT_ROOT / "config" / f"{camera_id}.json"
        if not config_path.exists():
            _set_status(job_id, status="detecting")
            classifier = slot_classifier.load(MODEL_PATH) if MODEL_PATH.exists() else None
            generate_config.generate_config(
                camera_id, str(upload_dir), str(config_path), classifier=classifier)

        _set_status(job_id, status="labeling")
        labeled_dir = storage.labeled_dir(batch_id, camera_id)
        labeled_dir.mkdir(parents=True, exist_ok=True)
        for photo_path in sorted(upload_dir.iterdir()):
            if photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            output_path = labeled_dir / f"{photo_path.stem}.png"
            pipeline.run_auto_all(str(config_path), str(photo_path), str(output_path))

        _set_status(job_id, status="done")
    except Exception as e:
        _set_status(job_id, status="error", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest web/backend/tests/test_jobs.py -v`
Expected: both PASS (each takes a few seconds — real median-stack/label work on one real photo)

- [ ] **Step 5: Commit**

```bash
git add web/backend/jobs.py web/backend/tests/test_jobs.py
git commit -m "feat: add background per-camera job processing for web uploads"
```

---

## Task 5: Flask app — upload + status endpoints

**Files:**
- Create: `web/backend/app.py`
- Modify: `requirements.txt` (add `Flask>=3.0`)
- Test: `web/backend/tests/test_app.py`

**Interfaces:**
- Produces: `app` (Flask instance), `POST /api/upload`, `GET /api/batches/<batch_id>/status`.
- Consumes: `jobs.submit_camera_job`, `jobs.get_batch_status`, `storage.save_uploaded_file`.

- [ ] **Step 1: Add Flask to requirements.txt**

```
Flask>=3.0
```

Install: `.venv/bin/pip install Flask`

- [ ] **Step 2: Write the failing test**

Create `web/backend/tests/test_app.py`:

```python
import io

import app as flask_app_module


def _client():
    flask_app_module.app.testing = True
    return flask_app_module.app.test_client()


def test_upload_rejects_when_no_files():
    client = _client()
    response = client.post("/api/upload", data={})
    assert response.status_code == 400


def test_upload_groups_files_by_parent_folder(monkeypatch, tmp_path):
    import storage
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    saved = []
    monkeypatch.setattr(
        "jobs.submit_camera_job",
        lambda batch_id, camera_id, upload_dir, photo_count: saved.append((camera_id, photo_count)) or f"{batch_id}:{camera_id}",
    )

    client = _client()
    data = {
        "files": [
            (io.BytesIO(b"fake-jpg-1"), "P1_B1_1_9/a.jpg"),
            (io.BytesIO(b"fake-jpg-2"), "P1_B1_1_9/b.jpg"),
            (io.BytesIO(b"fake-jpg-3"), "P1_B1_1_1/c.jpg"),
        ]
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    body = response.get_json()
    camera_ids = {c["camera_id"] for c in body["cameras"]}
    assert camera_ids == {"P1_B1_1_9", "P1_B1_1_1"}
    assert set(saved) == {("P1_B1_1_9", 2), ("P1_B1_1_1", 1)}


def test_upload_rejects_path_traversal(tmp_path, monkeypatch):
    import storage
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    client = _client()
    data = {"files": [(io.BytesIO(b"x"), "../../etc/passwd")]}
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 400


def test_batch_status_unknown_batch_returns_404():
    client = _client()
    response = client.get("/api/batches/does-not-exist/status")
    assert response.status_code == 404
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest web/backend/tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'` (or similar import error)

- [ ] **Step 4: Implement `web/backend/app.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from flask import Flask, jsonify, request

import jobs
import storage

app = Flask(__name__)


def _grouped_upload_files():
    grouped = {}
    for file in request.files.getlist("files"):
        raw_name = file.filename or ""
        parts = raw_name.split("/")
        if not raw_name or raw_name.startswith("/") or ".." in parts or len(parts) < 2:
            continue
        camera_id, filename = parts[-2], parts[-1]
        if not camera_id or not filename:
            continue
        grouped.setdefault(camera_id, []).append((filename, file))
    return grouped


@app.post("/api/upload")
def upload():
    grouped = _grouped_upload_files()
    if not grouped:
        return jsonify({"error": "no valid files (expected a folder upload)"}), 400

    import uuid
    batch_id = uuid.uuid4().hex
    cameras = []
    for camera_id, entries in grouped.items():
        for filename, file in entries:
            storage.save_uploaded_file(batch_id, camera_id, filename, file)
        upload_dir = storage.camera_upload_dir(batch_id, camera_id)
        job_id = jobs.submit_camera_job(batch_id, camera_id, upload_dir, len(entries))
        cameras.append({"camera_id": camera_id, "job_id": job_id, "photo_count": len(entries)})

    return jsonify({"batch_id": batch_id, "cameras": cameras})


@app.get("/api/batches/<batch_id>/status")
def batch_status(batch_id):
    statuses = jobs.get_batch_status(batch_id)
    if not statuses:
        return jsonify({"error": "unknown batch_id"}), 404
    return jsonify({"cameras": statuses})


if __name__ == "__main__":
    app.run(port=5000)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest web/backend/tests/test_app.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add web/backend/app.py requirements.txt web/backend/tests/test_app.py
git commit -m "feat: add Flask upload and batch status endpoints"
```

---

## Task 6: Flask app — photos, image serving, label edit endpoints

**Files:**
- Modify: `web/backend/app.py`
- Test: `web/backend/tests/test_app.py`

**Interfaces:**
- Produces: `GET /api/batches/<batch_id>/cameras/<camera_id>/photos`, `GET /api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>.png`, `GET /api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/slots/<slot_id>/patch.png`, `POST /api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/labels/<slot_id>`.
- Consumes: `overrides.load_override`/`exclude_slot`/`adjust_slot`; `pipeline.run_auto_all`, `pipeline._prepare_slot_view`; `parking_slot.SlotConfig`; `review_store.append_web_flag`, `review_store.candidate_id`.

- [ ] **Step 1: Write the failing tests**

Uses the real `P1_B1_1_9` config + sample photo directly (same fixture approach as Task 4), so no fake camera setup is needed. Add to `web/backend/tests/test_app.py`:

```python
import shutil
from pathlib import Path

import storage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_CAMERA = "P1_B1_1_9"
SAMPLE_PHOTO = REPO_ROOT / "no_label" / SAMPLE_CAMERA / "20260820_030004.jpg"


def _seed_labeled_photo(tmp_path, monkeypatch, batch_id="batchX"):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    upload_dir = storage.camera_upload_dir(batch_id, SAMPLE_CAMERA)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    import jobs
    job_id = jobs.submit_camera_job(batch_id, SAMPLE_CAMERA, upload_dir, photo_count=1)
    jobs.wait_for_job(job_id, timeout=30)
    return batch_id, SAMPLE_PHOTO.stem


def test_list_photos_returns_seeded_photo(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.get(f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos")

    assert response.status_code == 200
    photos = response.get_json()["photos"]
    assert any(p["photo"] == photo_stem for p in photos)


def test_get_photo_png_returns_image_bytes(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.get(f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}.png")

    assert response.status_code == 200
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_delete_label_excludes_slot_and_flags_review(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/slot-0",
        json={"action": "delete"},
    )

    assert response.status_code == 200
    import overrides
    assert overrides.load_override(batch_id, SAMPLE_CAMERA, photo_stem)["excluded_slots"] == ["slot-0"]

    import review_store
    flags = review_store.load_web_flags()
    assert any(f["camera_id"] == SAMPLE_CAMERA and f["slot_id"] == "slot-0" for f in flags)


def test_adjust_label_updates_override(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/slot-0",
        json={"action": "adjust", "box": {"cx": 0.4, "cy": 0.4, "w": 0.5, "h": 0.5}},
    )

    assert response.status_code == 200
    import overrides
    override = overrides.load_override(batch_id, SAMPLE_CAMERA, photo_stem)
    assert override["adjusted"]["slot-0"] == {"cx": 0.4, "cy": 0.4, "w": 0.5, "h": 0.5}


def test_get_slot_patch_returns_image_bytes(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.get(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/slots/slot-0/patch.png")

    assert response.status_code == 200
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"
```

Note: `review_store.load_web_flags()` reads the real `review/web_flags.jsonl` — this test appends a real row there. That's consistent with how the rest of the suite already treats `review/` (see `HANDOFF.md`); no cleanup fixture is introduced here to keep this task focused, but note it for the end-to-end task (Task 11) which does a manual pass over `review/`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest web/backend/tests/test_app.py -v`
Expected: FAIL (404s) on the new routes

- [ ] **Step 3: Implement — add to `web/backend/app.py`**

Add imports at the top (alongside existing ones):

```python
import cv2
from flask import send_file, Response

import overrides
import pipeline
import review_store
from parking_slot import SlotConfig
from datetime import datetime, timezone
```

Add helper + routes:

```python
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _find_raw_photo(batch_id, camera_id, photo):
    upload_dir = storage.camera_upload_dir(batch_id, camera_id)
    for ext in IMAGE_EXTENSIONS:
        candidate = upload_dir / f"{photo}{ext}"
        if candidate.exists():
            return candidate
    return None


@app.get("/api/batches/<batch_id>/cameras/<camera_id>/photos")
def list_photos(batch_id, camera_id):
    labeled_dir = storage.labeled_dir(batch_id, camera_id)
    if not labeled_dir.exists():
        return jsonify({"photos": []})
    photos = []
    for p in sorted(labeled_dir.glob("*.png")):
        override = overrides.load_override(batch_id, camera_id, p.stem)
        photos.append({
            "photo": p.stem,
            "excluded_slots": override["excluded_slots"],
            "adjusted": override["adjusted"],
        })
    return jsonify({"photos": photos})


@app.get("/api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>.png")
def get_photo(batch_id, camera_id, photo):
    path = storage.labeled_dir(batch_id, camera_id) / f"{photo}.png"
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(path), mimetype="image/png")


@app.get("/api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/slots/<slot_id>/patch.png")
def get_slot_patch(batch_id, camera_id, photo, slot_id):
    config_path = storage.PROJECT_ROOT / "config" / f"{camera_id}.json"
    if not config_path.exists():
        return jsonify({"error": "camera not found"}), 404
    config = SlotConfig.load(str(config_path))
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        return jsonify({"error": "slot not found"}), 404

    raw_path = _find_raw_photo(batch_id, camera_id, photo)
    if raw_path is None:
        return jsonify({"error": "photo not found"}), 404

    raw = cv2.imread(str(raw_path))
    view, _ = pipeline._prepare_slot_view(config, slot, slot_id)
    patch = view.rectify(raw)
    ok, encoded = cv2.imencode(".png", patch)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(encoded.tobytes(), mimetype="image/png")


def _rerender_photo(batch_id, camera_id, photo):
    config_path = storage.PROJECT_ROOT / "config" / f"{camera_id}.json"
    raw_path = _find_raw_photo(batch_id, camera_id, photo)
    override = overrides.load_override(batch_id, camera_id, photo)
    output_path = storage.labeled_dir(batch_id, camera_id) / f"{photo}.png"
    pipeline.run_auto_all(
        str(config_path), str(raw_path), str(output_path),
        excluded_slots=set(override["excluded_slots"]),
        adjusted_slots=override["adjusted"],
    )


def _flag_web_reject(camera_id, slot_id):
    config_path = storage.PROJECT_ROOT / "config" / f"{camera_id}.json"
    config = SlotConfig.load(str(config_path))
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        return
    record = {
        "id": review_store.candidate_id(camera_id, slot["polygon_raw"]),
        "camera_id": camera_id,
        "slot_id": slot_id,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    review_store.append_web_flag(record)


@app.post("/api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/labels/<slot_id>")
def edit_label(batch_id, camera_id, photo, slot_id):
    payload = request.get_json(force=True)
    action = payload.get("action")

    if action == "delete":
        overrides.exclude_slot(batch_id, camera_id, photo, slot_id)
        _flag_web_reject(camera_id, slot_id)
    elif action == "adjust":
        overrides.adjust_slot(batch_id, camera_id, photo, slot_id, payload["box"])
    else:
        return jsonify({"error": "action must be 'delete' or 'adjust'"}), 400

    _rerender_photo(batch_id, camera_id, photo)
    return jsonify({"ok": True})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest web/backend/tests/test_app.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add web/backend/app.py web/backend/tests/test_app.py
git commit -m "feat: add photo listing, image serving, and per-slot label edit endpoints"
```

---

## Task 7: Flask app — downloads + static frontend serving + gitignore

**Files:**
- Modify: `web/backend/app.py`
- Modify: `.gitignore`
- Test: `web/backend/tests/test_app.py`

**Interfaces:**
- Produces: `GET /api/batches/<batch_id>/cameras/<camera_id>/download`, `GET /api/batches/<batch_id>/download`, catch-all `GET /<path>` serving the built frontend.

- [ ] **Step 1: Write the failing test**

Add to `web/backend/tests/test_app.py`:

```python
import zipfile
import io as io_module


def test_download_camera_returns_zip_with_labeled_photo(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.get(f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/download")

    assert response.status_code == 200
    zf = zipfile.ZipFile(io_module.BytesIO(response.data))
    assert f"{photo_stem}.png" in zf.namelist()


def test_download_batch_returns_zip_with_camera_subfolder(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.get(f"/api/batches/{batch_id}/download")

    assert response.status_code == 200
    zf = zipfile.ZipFile(io_module.BytesIO(response.data))
    assert f"{SAMPLE_CAMERA}/{photo_stem}.png" in zf.namelist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest web/backend/tests/test_app.py -k download -v`
Expected: FAIL with 404s

- [ ] **Step 3: Implement — add to `web/backend/app.py`**

```python
import io
import zipfile

from flask import send_from_directory

FRONTEND_DIST = storage.PROJECT_ROOT / "web" / "frontend" / "dist"


@app.get("/api/batches/<batch_id>/cameras/<camera_id>/download")
def download_camera(batch_id, camera_id):
    labeled_dir = storage.labeled_dir(batch_id, camera_id)
    if not labeled_dir.exists():
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in labeled_dir.glob("*.png"):
            zf.write(p, arcname=p.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{camera_id}.zip")


@app.get("/api/batches/<batch_id>/download")
def download_batch(batch_id):
    batch_root = storage.batch_dir(batch_id)
    if not batch_root.exists():
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for camera_dir in batch_root.iterdir():
            labeled_dir = camera_dir / "labeled"
            if not labeled_dir.exists():
                continue
            for p in labeled_dir.glob("*.png"):
                zf.write(p, arcname=f"{camera_dir.name}/{p.name}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{batch_id}.zip")


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def serve_frontend(path):
    if path and (FRONTEND_DIST / path).is_file():
        return send_from_directory(str(FRONTEND_DIST), path)
    return send_from_directory(str(FRONTEND_DIST), "index.html")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest web/backend/tests/test_app.py -v`
Expected: all PASS (the catch-all route will 404/500 until Task 9 builds the frontend — that's fine, not exercised by these tests)

- [ ] **Step 5: Update `.gitignore`**

Add:

```
# web app runtime data
/web_uploads/
/web/frontend/node_modules/
/web/frontend/dist/
```

- [ ] **Step 6: Run the full backend suite and commit**

```bash
.venv/bin/pytest -q
git add web/backend/app.py .gitignore web/backend/tests/test_app.py
git commit -m "feat: add zip download endpoints and static frontend serving"
```

---

## Task 8: Frontend scaffold (Vite + React) + API client

**Files:**
- Create: `web/frontend/package.json`
- Create: `web/frontend/vite.config.js`
- Create: `web/frontend/index.html`
- Create: `web/frontend/src/main.jsx`
- Create: `web/frontend/src/api.js`
- Create: `web/frontend/src/App.jsx` (placeholder shell — real pages come in Tasks 9-10)

**Interfaces:**
- Produces: `uploadFolders(fileList)`, `getBatchStatus(batchId)`, `listPhotos(batchId, cameraId)`, `photoUrl(batchId, cameraId, photo)`, `editLabel(batchId, cameraId, photo, slotId, action, box)`, `downloadCameraUrl(batchId, cameraId)`, `downloadBatchUrl(batchId)` — all exported from `src/api.js`, consumed by Tasks 9-10.

- [ ] **Step 1: `web/frontend/package.json`**

```json
{
  "name": "cctv-labeling-web-frontend",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.1",
    "vite": "^5.4.0"
  }
}
```

- [ ] **Step 2: `web/frontend/vite.config.js`**

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:5000' } },
})
```

- [ ] **Step 3: `web/frontend/index.html`**

```html
<!doctype html>
<html lang="ko">
  <head>
    <meta charset="UTF-8" />
    <title>CCTV 라벨링</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 4: `web/frontend/src/main.jsx`**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 5: `web/frontend/src/api.js`**

```js
export async function uploadFolders(fileList) {
  const formData = new FormData()
  for (const file of fileList) {
    formData.append('files', file, file.webkitRelativePath || file.name)
  }
  const res = await fetch('/api/upload', { method: 'POST', body: formData })
  if (!res.ok) throw new Error('upload failed')
  return res.json()
}

export async function getBatchStatus(batchId) {
  const res = await fetch(`/api/batches/${batchId}/status`)
  if (!res.ok) throw new Error('status fetch failed')
  return res.json()
}

export async function listPhotos(batchId, cameraId) {
  const res = await fetch(`/api/batches/${batchId}/cameras/${cameraId}/photos`)
  if (!res.ok) throw new Error('photo list fetch failed')
  return res.json()
}

export function photoUrl(batchId, cameraId, photo) {
  return `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}.png?t=${Date.now()}`
}

export function slotPatchUrl(batchId, cameraId, photo, slotId) {
  return `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}/slots/${slotId}/patch.png`
}

export async function editLabel(batchId, cameraId, photo, slotId, action, box) {
  const res = await fetch(
    `/api/batches/${batchId}/cameras/${cameraId}/photos/${photo}/labels/${slotId}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(box ? { action, box } : { action }),
    },
  )
  if (!res.ok) throw new Error('label edit failed')
  return res.json()
}

export function downloadCameraUrl(batchId, cameraId) {
  return `/api/batches/${batchId}/cameras/${cameraId}/download`
}

export function downloadBatchUrl(batchId) {
  return `/api/batches/${batchId}/download`
}
```

- [ ] **Step 6: `web/frontend/src/App.jsx` (placeholder shell)**

```jsx
export default function App() {
  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 16, fontFamily: 'sans-serif' }}>
      <h1>CCTV 라벨링</h1>
      <p>loading...</p>
    </div>
  )
}
```

- [ ] **Step 7: Install and verify the dev server boots**

```bash
cd web/frontend && npm install && npm run dev -- --port 5173 &
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5173/
kill %1
cd -
```

Expected: `200`

- [ ] **Step 8: Commit**

```bash
git add web/frontend/package.json web/frontend/package-lock.json web/frontend/vite.config.js web/frontend/index.html web/frontend/src/main.jsx web/frontend/src/api.js web/frontend/src/App.jsx
git commit -m "feat: scaffold Vite+React frontend with API client"
```

---

## Task 9: Frontend — upload + progress pages

**Files:**
- Create: `web/frontend/src/UploadPage.jsx`
- Create: `web/frontend/src/ProgressPage.jsx`
- Modify: `web/frontend/src/App.jsx`

**Interfaces:**
- Consumes: `uploadFolders`, `getBatchStatus` from `./api.js`.
- Produces: `<UploadPage onUploaded={(batchData) => void}>`, `<ProgressPage batchId cameras onAllDone={() => void}>` — both consumed by `App.jsx`'s stage state machine, and `ResultsPage` (Task 10) is the third stage.

- [ ] **Step 1: `web/frontend/src/UploadPage.jsx`**

```jsx
import { useState } from 'react'
import { uploadFolders } from './api.js'

export default function UploadPage({ onUploaded }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function handleChange(e) {
    const files = e.target.files
    if (!files || files.length === 0) return
    setBusy(true)
    setError(null)
    try {
      const result = await uploadFolders(files)
      onUploaded(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <p>
        카메라 폴더가 들어있는 상위 폴더 하나를 선택하세요 (폴더 최대 100개 권장).
        폴더 하나 = 카메라 하나로 인식합니다. 사진 1장짜리 폴더도 됩니다.
      </p>
      <input
        type="file"
        webkitdirectory="true"
        directory="true"
        multiple
        disabled={busy}
        onChange={handleChange}
      />
      {busy && <p>업로드 중...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  )
}
```

- [ ] **Step 2: `web/frontend/src/ProgressPage.jsx`**

```jsx
import { useEffect, useState } from 'react'
import { getBatchStatus } from './api.js'

export default function ProgressPage({ batchId, cameras, onAllDone }) {
  const [statuses, setStatuses] = useState(cameras.map((c) => ({ ...c, status: 'queued' })))

  useEffect(() => {
    let cancelled = false
    const interval = setInterval(async () => {
      const data = await getBatchStatus(batchId)
      if (cancelled) return
      setStatuses(data.cameras)
      const allDone = data.cameras.every((c) => c.status === 'done' || c.status === 'error')
      if (allDone) {
        clearInterval(interval)
        onAllDone()
      }
    }, 2000)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [batchId, onAllDone])

  return (
    <div>
      <h2>처리 중...</h2>
      <ul>
        {statuses.map((c) => (
          <li key={c.camera_id}>
            {c.camera_id}: {c.status} ({c.photo_count}장){c.error ? ` - ${c.error}` : ''}
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 3: Wire into `web/frontend/src/App.jsx`**

```jsx
import { useState } from 'react'
import UploadPage from './UploadPage.jsx'
import ProgressPage from './ProgressPage.jsx'

export default function App() {
  const [batch, setBatch] = useState(null)
  const [stage, setStage] = useState('upload')

  function handleUploaded(batchData) {
    setBatch(batchData)
    setStage('progress')
  }

  function handleAllDone() {
    setStage('results')
  }

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 16, fontFamily: 'sans-serif' }}>
      <h1>CCTV 라벨링</h1>
      {stage === 'upload' && <UploadPage onUploaded={handleUploaded} />}
      {stage === 'progress' && batch && (
        <ProgressPage batchId={batch.batch_id} cameras={batch.cameras} onAllDone={handleAllDone} />
      )}
      {stage === 'results' && batch && <p>결과 화면 (다음 작업에서 추가)</p>}
    </div>
  )
}
```

- [ ] **Step 4: Manual verification**

```bash
cd web/frontend && npm run dev -- --port 5173 &
```
Open `http://localhost:5173` in a browser (backend must also be running from Task 7 — `cd ../.. && .venv/bin/python web/backend/app.py &`). Select a folder via the file input, confirm the progress list appears and updates.

- [ ] **Step 5: Commit**

```bash
git add web/frontend/src/UploadPage.jsx web/frontend/src/ProgressPage.jsx web/frontend/src/App.jsx
git commit -m "feat: add upload and progress pages to web frontend"
```

---

## Task 10: Frontend — results page (grid, delete/adjust, download)

**Files:**
- Create: `web/frontend/src/ResultsPage.jsx`
- Modify: `web/frontend/src/App.jsx`

**Interfaces:**
- Consumes: `listPhotos`, `photoUrl`, `slotPatchUrl`, `editLabel`, `downloadCameraUrl`, `downloadBatchUrl` from `./api.js`.
- Produces: `<ResultsPage batchId cameras>` (terminal stage — final task in the frontend's stage state machine).

- [ ] **Step 1: `web/frontend/src/ResultsPage.jsx`**

```jsx
import { useEffect, useState } from 'react'
import { listPhotos, photoUrl, slotPatchUrl, editLabel, downloadCameraUrl, downloadBatchUrl } from './api.js'

const PATCH_SIZE = 300 // must match pipeline.DEFAULT_PATCH_SIZE in src/pipeline.py

function SlotAdjustModal({ batchId, cameraId, photo, slotId, onClose, onSaved }) {
  const [drag, setDrag] = useState(null)

  function handleMouseDown(e) {
    const rect = e.currentTarget.getBoundingClientRect()
    setDrag({ x0: e.clientX - rect.left, y0: e.clientY - rect.top, x1: e.clientX - rect.left, y1: e.clientY - rect.top })
  }
  function handleMouseMove(e) {
    if (!drag) return
    const rect = e.currentTarget.getBoundingClientRect()
    setDrag((d) => ({ ...d, x1: e.clientX - rect.left, y1: e.clientY - rect.top }))
  }
  async function handleMouseUp() {
    if (!drag) return
    const cx = (drag.x0 + drag.x1) / 2 / PATCH_SIZE
    const cy = (drag.y0 + drag.y1) / 2 / PATCH_SIZE
    const w = Math.abs(drag.x1 - drag.x0) / PATCH_SIZE
    const h = Math.abs(drag.y1 - drag.y0) / PATCH_SIZE
    const finished = drag
    setDrag(null)
    if (Math.abs(finished.x1 - finished.x0) < 10 || Math.abs(finished.y1 - finished.y0) < 10) return
    await editLabel(batchId, cameraId, photo, slotId, 'adjust', { cx, cy, w, h })
    onSaved()
  }

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
      <div style={{ background: 'white', padding: 16 }}>
        <p>{slotId} 위치/크기를 드래그로 지정하세요 (사각형 하나 그리면 자동 저장)</p>
        <div
          style={{ position: 'relative', width: PATCH_SIZE, height: PATCH_SIZE, cursor: 'crosshair' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <img
            src={slotPatchUrl(batchId, cameraId, photo, slotId)}
            width={PATCH_SIZE}
            height={PATCH_SIZE}
            style={{ position: 'absolute', top: 0, left: 0 }}
            draggable={false}
            alt={`${slotId} rectified patch`}
          />
          {drag && (
            <div
              style={{
                position: 'absolute',
                border: '2px solid orange',
                left: Math.min(drag.x0, drag.x1),
                top: Math.min(drag.y0, drag.y1),
                width: Math.abs(drag.x1 - drag.x0),
                height: Math.abs(drag.y1 - drag.y0),
              }}
            />
          )}
        </div>
        <button onClick={onClose}>닫기</button>
      </div>
    </div>
  )
}

function PhotoCard({ batchId, cameraId, photo, onChanged }) {
  const [editingSlot, setEditingSlot] = useState(null)

  async function handleDelete(slotId) {
    if (!slotId) return
    await editLabel(batchId, cameraId, photo.photo, slotId, 'delete')
    onChanged()
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: 8, margin: 8, display: 'inline-block', verticalAlign: 'top' }}>
      <img src={photoUrl(batchId, cameraId, photo.photo)} width={240} alt={photo.photo} />
      <div>{photo.photo}</div>
      <div>
        슬롯 삭제:
        <input
          placeholder="slot-id + Enter"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleDelete(e.target.value)
              e.target.value = ''
            }
          }}
        />
      </div>
      <div>
        슬롯 수정:
        <input
          placeholder="slot-id + Enter"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && e.target.value) {
              setEditingSlot(e.target.value)
              e.target.value = ''
            }
          }}
        />
      </div>
      {photo.excluded_slots.length > 0 && <div>삭제됨: {photo.excluded_slots.join(', ')}</div>}
      {editingSlot && (
        <SlotAdjustModal
          batchId={batchId}
          cameraId={cameraId}
          photo={photo.photo}
          slotId={editingSlot}
          onClose={() => setEditingSlot(null)}
          onSaved={() => {
            setEditingSlot(null)
            onChanged()
          }}
        />
      )}
    </div>
  )
}

export default function ResultsPage({ batchId, cameras }) {
  const [cameraId, setCameraId] = useState(cameras[0]?.camera_id)
  const [photos, setPhotos] = useState([])

  async function refresh() {
    const data = await listPhotos(batchId, cameraId)
    setPhotos(data.photos)
  }

  useEffect(() => {
    refresh()
  }, [cameraId])

  return (
    <div>
      <h2>결과</h2>
      <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
        {cameras.map((c) => (
          <option key={c.camera_id} value={c.camera_id}>{c.camera_id}</option>
        ))}
      </select>
      {' '}
      <a href={downloadCameraUrl(batchId, cameraId)}>이 카메라 다운로드</a>
      {' | '}
      <a href={downloadBatchUrl(batchId)}>전체 다운로드</a>
      <div>
        {photos.map((p) => (
          <PhotoCard key={p.photo} batchId={batchId} cameraId={cameraId} photo={p} onChanged={refresh} />
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Wire into `web/frontend/src/App.jsx`**

Replace the `results` stage placeholder line:

```jsx
import ResultsPage from './ResultsPage.jsx'
// ...
{stage === 'results' && batch && <ResultsPage batchId={batch.batch_id} cameras={batch.cameras} />}
```

- [ ] **Step 3: Manual verification**

With both servers running (backend + `npm run dev`), upload a small camera folder from `no_label/P1_B1_1_9/`, wait for progress to finish, confirm the results grid shows labeled photos, delete one slot on one photo (confirm the label disappears and `review/web_flags.jsonl` gets a new row), adjust one slot (confirm the label moves), and download both a per-camera zip and the full batch zip.

- [ ] **Step 4: Commit**

```bash
git add web/frontend/src/ResultsPage.jsx web/frontend/src/App.jsx
git commit -m "feat: add results page with per-slot delete/adjust and download"
```

---

## Task 11: `run_web.sh` + full end-to-end verification

**Files:**
- Create: `run_web.sh`

**Interfaces:**
- Produces: a single executable script, no other code depends on it.

- [ ] **Step 1: Create `run_web.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "web/frontend/dist" ]; then
  echo "building frontend..."
  (cd web/frontend && npm install && npm run build)
fi

echo "starting server on http://localhost:5000"
.venv/bin/python web/backend/app.py
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x run_web.sh
```

- [ ] **Step 3: Run the full backend pytest suite**

```bash
.venv/bin/pytest -q
```
Expected: all tests pass (existing `tests/` suite plus `web/backend/tests/`)

- [ ] **Step 4: End-to-end manual verification**

```bash
./run_web.sh &
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/
kill %1
```
Expected: `200`. Then open `http://localhost:5000` in a browser directly (single port, no separate dev server this time), upload a real camera folder from `no_label/`, and confirm the whole upload → progress → results → delete/adjust → download flow works end to end exactly as in Task 10's manual check, but through the single production-style launcher.

- [ ] **Step 5: Commit**

```bash
git add run_web.sh
git commit -m "feat: add run_web.sh single-command launcher"
```

---

## Self-Review Notes

- **Spec coverage:** folder-per-camera upload (Task 5), known-vs-new camera branching + single-photo support (Task 4, reuses `generate_config`/`pipeline` unchanged), background processing + polling (Tasks 4-5, 9), per-photo per-slot delete/adjust without touching `config/*.json` (Tasks 1, 3, 6), web_flags → review queue linkage (Task 2, wired in Task 6), zip downloads (Task 7), single launcher (Task 11) — all covered.
- **Out of scope reminders carried from the spec:** no auth/multi-user, no persistence of job state across restarts, no automated frontend tests, no changes to the `/missed` false-negative tool.
