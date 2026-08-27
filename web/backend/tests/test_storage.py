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
