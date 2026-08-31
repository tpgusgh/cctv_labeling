import io
import io as io_module
import shutil
import zipfile
from pathlib import Path

import pytest

import app as flask_app_module
import storage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_CAMERA = "P1_B1_1_9"
SAMPLE_PHOTO = REPO_ROOT / "no_label" / SAMPLE_CAMERA / "20260820_030004.jpg"


def _client():
    flask_app_module.app.testing = True
    return flask_app_module.app.test_client()


def _seed_labeled_photo(tmp_path, monkeypatch, batch_id="batchX"):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    upload_dir = storage.camera_upload_dir(batch_id, SAMPLE_CAMERA)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    import jobs
    job_id = jobs.submit_camera_job(batch_id, SAMPLE_CAMERA, upload_dir, photo_count=1)
    jobs.wait_for_job(job_id, timeout=30)
    return batch_id, SAMPLE_PHOTO.stem


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


def test_upload_reuses_batch_id_across_chunked_requests(monkeypatch, tmp_path):
    # a huge folder is uploaded as one request per camera folder -- later
    # requests pass the batch_id from the first response so every camera
    # lands in the same batch.
    import storage
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(
        "jobs.submit_camera_job",
        lambda batch_id, camera_id, upload_dir, photo_count: f"{batch_id}:{camera_id}",
    )
    client = _client()

    r1 = client.post("/api/upload", data={"files": [(io.BytesIO(b"x"), "camA/a.jpg")]},
                      content_type="multipart/form-data")
    assert r1.status_code == 200
    batch_id = r1.get_json()["batch_id"]

    r2 = client.post("/api/upload",
                      data={"batch_id": batch_id, "files": [(io.BytesIO(b"y"), "camB/b.jpg")]},
                      content_type="multipart/form-data")
    assert r2.status_code == 200
    assert r2.get_json()["batch_id"] == batch_id
    assert (tmp_path / batch_id / "camA" / "original" / "a.jpg").exists()
    assert (tmp_path / batch_id / "camB" / "original" / "b.jpg").exists()


def test_download_batch_rejects_path_traversal(tmp_path, monkeypatch):
    # security review repro: GET /api/batches/../download resolved
    # batch_dir("..") to storage.PROJECT_ROOT's parent and happily zipped
    # whatever it found there, before storage.safe_component() was added.
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    client = _client()

    response = client.get("/api/batches/../download")

    assert response.status_code == 400
    assert response.get_json()["error"]


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

    import functools
    import review_store
    flags_path = tmp_path / "web_flags.jsonl"
    monkeypatch.setattr(
        review_store, "append_web_flag",
        functools.partial(review_store.append_web_flag, path=flags_path),
    )

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/slot-0",
        json={"action": "delete"},
    )

    assert response.status_code == 200
    import overrides
    assert overrides.load_override(batch_id, SAMPLE_CAMERA, photo_stem)["excluded_slots"] == ["slot-0"]

    flags = review_store.load_web_flags(flags_path)
    assert any(f["camera_id"] == SAMPLE_CAMERA and f["slot_id"] == "slot-0" and f["photo"] == photo_stem for f in flags)


def test_adjust_label_updates_override(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    # Build a raw_polygon (raw ORIGINAL-photo pixel corners) that should
    # round-trip back to pipeline's known-good fixed label box (cx=cy=0.5,
    # w=h=0.68), proving the backend's raw-pixel->local->plane conversion
    # uses the same view/homography pipeline.run_auto_all uses to interpret
    # adjusted_slots -- editing now happens directly on the original photo,
    # not a separate rectified-patch popup.
    import pipeline
    import perspective
    from parking_slot import SlotConfig
    import storage as storage_module

    config_path = storage_module.PROJECT_ROOT / "config" / f"{SAMPLE_CAMERA}.json"
    config = SlotConfig.load(str(config_path))
    slot = next(s for s in config.slots if s["id"] == "slot-0")
    view, homography = pipeline._prepare_slot_view(config, slot, "slot-0")

    cx, cy, w, h = 0.5, 0.5, 0.68, 0.68
    local_corners = perspective.plane_points_to_pixel(homography, [
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
    ])
    raw_corners = view.local_to_raw(local_corners)
    (x0, y0), (x1, y1) = raw_corners

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/slot-0",
        json={
            "action": "adjust",
            "raw_polygon": [[float(x0), float(y0)], [float(x1), float(y1)]],
        },
    )

    assert response.status_code == 200
    import overrides
    override = overrides.load_override(batch_id, SAMPLE_CAMERA, photo_stem)
    box = override["adjusted"]["slot-0"]
    assert box["cx"] == pytest.approx(0.5, abs=0.01)
    assert box["cy"] == pytest.approx(0.5, abs=0.01)
    assert box["w"] == pytest.approx(0.68, abs=0.01)
    assert box["h"] == pytest.approx(0.68, abs=0.01)


def _isolate_project_root(tmp_path, monkeypatch):
    # add_label/delete_all write straight to storage.PROJECT_ROOT / "config" /
    # "<camera>.json" (the camera's shared, cross-photo config). Redirect
    # PROJECT_ROOT to an isolated copy so tests can't touch the real
    # committed config.
    import shutil
    import storage as storage_module
    fake_root = tmp_path / "fake_project_root"
    (fake_root / "config").mkdir(parents=True)
    shutil.copy(
        storage_module.PROJECT_ROOT / "config" / f"{SAMPLE_CAMERA}.json",
        fake_root / "config" / f"{SAMPLE_CAMERA}.json",
    )
    monkeypatch.setattr(storage_module, "PROJECT_ROOT", fake_root)
    return fake_root


def _isolate_review_store(tmp_path, monkeypatch):
    # web label edits now feed the review log (accept/reject decisions +
    # crops) -- redirect both paths so tests never write the real
    # review/labels.jsonl (that exact pollution bug already happened once
    # with a frozen CONFIG_DIR; review_store attributes are looked up at
    # call time so monkeypatching works).
    import review_store
    labels_path = tmp_path / "review_labels.jsonl"
    crops_dir = tmp_path / "review_crops"
    monkeypatch.setattr(review_store, "LABELS_PATH", labels_path)
    monkeypatch.setattr(review_store, "CROPS_DIR", crops_dir)
    return labels_path


def test_add_label_appends_new_slot_to_camera_config(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()
    fake_root = _isolate_project_root(tmp_path, monkeypatch)
    labels_path = _isolate_review_store(tmp_path, monkeypatch)

    raw_polygon = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]
    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels",
        json={"raw_polygon": raw_polygon},
    )

    assert response.status_code == 200
    new_slot_id = response.get_json()["slot_id"]

    from parking_slot import SlotConfig
    config = SlotConfig.load(str(fake_root / "config" / f"{SAMPLE_CAMERA}.json"))
    added = next(s for s in config.slots if s["id"] == new_slot_id)
    assert added["polygon_raw"] == raw_polygon

    # a human-drawn slot is an accept: protects the region from
    # rejected-region suppression and becomes a training positive.
    import review_store
    labels = review_store.load_labels(labels_path)
    assert len(labels) == 1
    assert labels[0]["decision"] == "accept"
    assert labels[0]["polygon"] == raw_polygon
    assert labels[0]["camera_id"] == SAMPLE_CAMERA


def test_delete_all_removes_slot_from_config_and_logs_reject(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()
    fake_root = _isolate_project_root(tmp_path, monkeypatch)
    labels_path = _isolate_review_store(tmp_path, monkeypatch)

    from parking_slot import SlotConfig
    config_path = fake_root / "config" / f"{SAMPLE_CAMERA}.json"
    victim = SlotConfig.load(str(config_path)).slots[0]

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/{victim['id']}",
        json={"action": "delete_all"},
    )

    assert response.status_code == 200
    config = SlotConfig.load(str(config_path))
    assert all(s["id"] != victim["id"] for s in config.slots)

    import review_store
    labels = review_store.load_labels(labels_path)
    assert len(labels) == 1
    assert labels[0]["decision"] == "reject"
    assert labels[0]["polygon"] == victim["polygon_raw"]


def test_adjust_all_saves_default_label_box_in_config(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()
    fake_root = _isolate_project_root(tmp_path, monkeypatch)
    _isolate_review_store(tmp_path, monkeypatch)

    import pipeline
    import perspective
    from parking_slot import SlotConfig
    config_path = fake_root / "config" / f"{SAMPLE_CAMERA}.json"
    config = SlotConfig.load(str(config_path))
    slot = config.slots[0]
    view, homography = pipeline._prepare_slot_view(config, slot, slot["id"])

    # a drag whose plane-space round-trip is a known box (cx=cy=0.5, w=h=0.4)
    cx, cy, w, h = 0.5, 0.5, 0.4, 0.4
    local_corners = perspective.plane_points_to_pixel(homography, [
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
    ])
    raw_corners = view.local_to_raw(local_corners)
    (x0, y0), (x1, y1) = raw_corners

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/{slot['id']}",
        json={"action": "adjust_all", "raw_polygon": [[float(x0), float(y0)], [float(x1), float(y1)]]},
    )

    assert response.status_code == 200
    saved = SlotConfig.load(str(config_path))
    box = next(s for s in saved.slots if s["id"] == slot["id"])["label_box"]
    assert box["cx"] == pytest.approx(0.5, abs=0.01)
    assert box["cy"] == pytest.approx(0.5, abs=0.01)
    assert box["w"] == pytest.approx(0.4, abs=0.01)
    assert box["h"] == pytest.approx(0.4, abs=0.01)


def test_restore_clears_per_photo_overrides(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()
    _isolate_review_store(tmp_path, monkeypatch)

    import overrides
    from parking_slot import SlotConfig
    import storage as storage_module
    config = SlotConfig.load(str(storage_module.config_path(SAMPLE_CAMERA)))
    slot_id = config.slots[0]["id"]

    # hide the slot on this photo, then restore it
    r = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/{slot_id}",
        json={"action": "delete"})
    assert r.status_code == 200
    assert slot_id in overrides.load_override(batch_id, SAMPLE_CAMERA, photo_stem)["excluded_slots"]

    r = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/{slot_id}",
        json={"action": "restore"})
    assert r.status_code == 200
    override = overrides.load_override(batch_id, SAMPLE_CAMERA, photo_stem)
    assert slot_id not in override["excluded_slots"]
    assert slot_id not in override["adjusted"]


def test_delete_all_unknown_slot_returns_404(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()
    _isolate_project_root(tmp_path, monkeypatch)
    _isolate_review_store(tmp_path, monkeypatch)

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels/no-such-slot",
        json={"action": "delete_all"},
    )
    assert response.status_code == 404


def test_add_label_fixes_z_order_clicked_corners(tmp_path, monkeypatch):
    # clicking 4 corners in a Z pattern (top-left, top-right, bottom-left,
    # bottom-right) must not save a self-intersecting bowtie quad.
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()
    fake_root = _isolate_project_root(tmp_path, monkeypatch)
    _isolate_review_store(tmp_path, monkeypatch)

    z_order = [[10.0, 10.0], [50.0, 10.0], [10.0, 50.0], [50.0, 50.0]]
    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels",
        json={"raw_polygon": z_order},
    )

    assert response.status_code == 200
    new_slot_id = response.get_json()["slot_id"]
    from parking_slot import SlotConfig
    config = SlotConfig.load(str(fake_root / "config" / f"{SAMPLE_CAMERA}.json"))
    saved = next(s for s in config.slots if s["id"] == new_slot_id)["polygon_raw"]
    # angle-sorted around the centroid: consecutive corners are adjacent,
    # never diagonal -- every consecutive pair shares exactly one coordinate
    # for this axis-aligned square.
    for i in range(4):
        a, b = saved[i], saved[(i + 1) % 4]
        assert (a[0] == b[0]) != (a[1] == b[1])


def test_add_label_rejects_non_quad_polygon(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/labels",
        json={"raw_polygon": [[10.0, 10.0], [50.0, 10.0]]},
    )

    assert response.status_code == 400


def test_edit_label_unknown_camera_returns_json_404(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    client = _client()

    response = client.post(
        "/api/batches/batchX/cameras/no-such-camera/photos/somephoto/labels/slot-0",
        json={"action": "delete"},
    )

    assert response.status_code == 404
    assert response.get_json()["error"]


def test_edit_label_missing_raw_photo_returns_json_404(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.post(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/no-such-photo/labels/slot-0",
        json={"action": "delete"},
    )

    assert response.status_code == 404
    assert response.get_json()["error"]


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
