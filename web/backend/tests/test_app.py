import io
import shutil
from pathlib import Path

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


def test_get_slot_patch_returns_image_bytes(tmp_path, monkeypatch):
    batch_id, photo_stem = _seed_labeled_photo(tmp_path, monkeypatch)
    client = _client()

    response = client.get(
        f"/api/batches/{batch_id}/cameras/{SAMPLE_CAMERA}/photos/{photo_stem}/slots/slot-0/patch.png")

    assert response.status_code == 200
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"
