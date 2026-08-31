from pathlib import Path

import pytest

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


@pytest.mark.parametrize("unsafe", ["..", ".", "a/../../etc", "a/b", "a\\b", "", None])
def test_safe_component_rejects_traversal_and_separators(unsafe):
    with pytest.raises(ValueError):
        storage.safe_component(unsafe)


@pytest.mark.parametrize("fn_name,args", [
    ("batch_dir", ("..",)),
    ("camera_upload_dir", ("batch1", "..")),
    ("labeled_dir", ("batch1", "../../etc")),
    ("overrides_dir", ("..", "cam")),
    ("config_path", ("../../etc/passwd",)),
])
def test_path_helpers_reject_traversal_components(tmp_path, monkeypatch, fn_name, args):
    # security review finding: batch_id/camera_id reach these functions
    # straight from the URL with no validation, and Flask's <string:...>
    # route converter accepts a literal ".." segment -- verified reachable
    # (e.g. GET /api/batches/../download) before storage.safe_component()
    # was added to every one of these path builders.
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError):
        getattr(storage, fn_name)(*args)
