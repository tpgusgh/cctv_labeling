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
