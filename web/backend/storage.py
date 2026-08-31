import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_UPLOADS_DIR = PROJECT_ROOT / "web_uploads"

# batch_id/camera_id/photo/filename all come straight from the URL or a
# multipart filename -- Flask's <string:...> route converter accepts any
# character except "/", so a literal ".." segment is a valid route value
# (Werkzeug does not collapse it before routing). Every function below that
# turns one of these into a path component must go through this guard, or
# a request like GET /api/batches/../download walks outside WEB_UPLOADS_DIR
# entirely (verified reachable before this fix).
_SAFE_COMPONENT = re.compile(r"^[^/\\\x00]+$")


def safe_component(name):
    if not name or name in (".", "..") or not _SAFE_COMPONENT.match(name):
        raise ValueError(f"unsafe path component: {name!r}")
    return name


def batch_dir(batch_id):
    return WEB_UPLOADS_DIR / safe_component(batch_id)


def camera_upload_dir(batch_id, camera_id):
    return batch_dir(batch_id) / safe_component(camera_id) / "original"


def labeled_dir(batch_id, camera_id):
    return batch_dir(batch_id) / safe_component(camera_id) / "labeled"


def overrides_dir(batch_id, camera_id):
    return batch_dir(batch_id) / safe_component(camera_id) / "overrides"


def config_path(camera_id):
    # PROJECT_ROOT read fresh here (not a module-load-time-derived constant)
    # so a test's monkeypatch.setattr(storage, "PROJECT_ROOT", ...) actually
    # redirects this, same as WEB_UPLOADS_DIR-based paths above already do.
    return PROJECT_ROOT / "config" / f"{safe_component(camera_id)}.json"


def batches_with_camera(camera_id):
    """Every batch id that contains uploads for this camera -- a config-level
    slot edit (add/delete_all) must refresh labeled outputs in all of them,
    not just the batch the user happened to be viewing."""
    camera_id = safe_component(camera_id)
    if not WEB_UPLOADS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in WEB_UPLOADS_DIR.iterdir()
        if d.is_dir() and (d / camera_id / "original").is_dir())


def save_uploaded_file(batch_id, camera_id, filename, file_storage):
    target_dir = camera_upload_dir(batch_id, camera_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / safe_component(filename)
    file_storage.save(str(dest))
    return dest
