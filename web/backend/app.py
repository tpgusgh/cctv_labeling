import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import io
import zipfile
import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory, Response

import jobs
import overrides
import pipeline
import review_store
import storage
from parking_slot import SlotConfig
from datetime import datetime, timezone

app = Flask(__name__)

FRONTEND_DIST = storage.PROJECT_ROOT / "web" / "frontend" / "dist"


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
    config_path = storage.PROJECT_ROOT / "config" / f"{camera_id}.json"
    if not config_path.exists():
        return jsonify({"error": "camera not found"}), 404
    if _find_raw_photo(batch_id, camera_id, photo) is None:
        return jsonify({"error": "photo not found"}), 404

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


if __name__ == "__main__":
    app.run(port=5050)
