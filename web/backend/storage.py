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
