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
