import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import io
import zipfile
from flask import Flask, jsonify, request, send_file, send_from_directory

import jobs
import overrides
import perspective
import pipeline
import review_store
import storage
from parking_slot import SlotConfig
from datetime import datetime, timezone

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB per upload request
# Werkzeug caps multipart parts at 1000 by default -- a real folder upload is
# thousands of photos in one request (verified: 2816-photo folder returned 413
# "upload failed" purely from the part count). The frontend now also chunks
# uploads per camera folder, but one camera alone can exceed 1000 frames.
app.config["MAX_FORM_PARTS"] = 20000
try:
    app.request_class.max_form_parts = 20000  # older Flask/Werkzeug fallback
except Exception:
    pass

FRONTEND_DIST = storage.PROJECT_ROOT / "web" / "frontend" / "dist"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@app.errorhandler(ValueError)
def _handle_unsafe_path(e):
    return jsonify({"error": str(e)}), 400


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
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        grouped.setdefault(camera_id, []).append((filename, file))
    return grouped


@app.get("/api/cameras")
def list_cameras():
    config_dir = storage.PROJECT_ROOT / "config"
    if not config_dir.exists():
        return jsonify({"cameras": []})
    camera_ids = sorted(p.stem for p in config_dir.glob("*.json"))
    return jsonify({"cameras": camera_ids})


@app.post("/api/upload")
def upload():
    grouped = _grouped_upload_files()
    if not grouped:
        return jsonify({"error": "no valid files (expected a folder upload)"}), 400

    import uuid
    # the frontend chunks big folder uploads into one request per camera --
    # every request after the first passes the batch_id from the first
    # response so all cameras land in the same batch.
    batch_id = request.form.get("batch_id") or uuid.uuid4().hex
    storage.safe_component(batch_id)
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


def _find_raw_photo(batch_id, camera_id, photo):
    storage.safe_component(photo)
    upload_dir = storage.camera_upload_dir(batch_id, camera_id)
    if not upload_dir.is_dir():
        return None
    # matched by lowercased suffix, not upload_dir / f"{photo}{ext}" -- that
    # relied on the filesystem's own case-folding (fine on macOS/APFS, not
    # guaranteed elsewhere) to find a real "*.JPG" upload.
    for candidate in upload_dir.iterdir():
        if candidate.stem == photo and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


@app.get("/api/batches/<batch_id>/cameras/<camera_id>/photos")
def list_photos(batch_id, camera_id):
    labeled_dir = storage.labeled_dir(batch_id, camera_id)
    if not labeled_dir.exists():
        return jsonify({"photos": []})

    config_path = storage.config_path(camera_id)
    config = SlotConfig.load(str(config_path)) if config_path.exists() else None

    photos = []
    for p in sorted(labeled_dir.glob("*.png")):
        override = overrides.load_override(batch_id, camera_id, p.stem)
        slots = []
        if config is not None:
            # mirror run_auto_all's overlap guard: a slot whose label quad the
            # rendered PNG dropped for overlapping a stronger slot must not
            # come back as an interactive box either ("겹치는건 무조건 빼").
            active = [s for s in config.slots if s["id"] not in override["excluded_slots"]]
            overlap_dropped = pipeline._overlapping_label_slots(config, active, override["adjusted"])
            for slot in config.slots:
                slot_id = slot["id"]
                if slot_id in overlap_dropped:
                    continue
                try:
                    box_raw = pipeline.label_box_raw_pixels(config, slot, slot_id, override["adjusted"])
                except ValueError:
                    box_raw = None
                slots.append({
                    "id": slot_id,
                    "confidence": slot.get("confidence"),
                    "box_raw": box_raw,
                    "excluded": slot_id in override["excluded_slots"],
                })
        photos.append({"photo": p.stem, "slots": slots})
    return jsonify({"photos": photos})


@app.get("/api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>.png")
def get_photo(batch_id, camera_id, photo):
    path = storage.labeled_dir(batch_id, camera_id) / f"{storage.safe_component(photo)}.png"
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(path), mimetype="image/png")


def _rerender_photo(batch_id, camera_id, photo):
    config_path = storage.config_path(camera_id)
    raw_path = _find_raw_photo(batch_id, camera_id, photo)
    override = overrides.load_override(batch_id, camera_id, photo)
    output_path = storage.labeled_dir(batch_id, camera_id) / f"{photo}.png"
    pipeline.run_auto_all(
        str(config_path), str(raw_path), str(output_path),
        excluded_slots=set(override["excluded_slots"]),
        adjusted_slots=override["adjusted"],
    )


def _rerender_all_photos(batch_id, camera_id):
    upload_dir = storage.camera_upload_dir(batch_id, camera_id)
    for raw_path in sorted(upload_dir.iterdir()):
        if raw_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        _rerender_photo(batch_id, camera_id, raw_path.stem)


def _rerender_camera_everywhere(current_batch_id, camera_id):
    """A config-level slot edit changes every batch of this camera, not just
    the one being viewed. The current batch rerenders synchronously (the
    user is looking at it right now); other batches refresh in the
    background so the request doesn't block on hundreds of old photos."""
    _rerender_all_photos(current_batch_id, camera_id)
    for other_batch in storage.batches_with_camera(camera_id):
        if other_batch == current_batch_id:
            continue
        jobs._EXECUTOR.submit(_rerender_all_photos, other_batch, camera_id)


def _validate_raw_polygon(raw_polygon, expected_len):
    if not isinstance(raw_polygon, list) or len(raw_polygon) != expected_len:
        raise ValueError(f"raw_polygon must be a list of exactly {expected_len} points")
    for pt in raw_polygon:
        if not (isinstance(pt, list) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            raise ValueError("raw_polygon points must be [x, y] numeric pairs")


@app.post("/api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/labels")
def add_label(batch_id, camera_id, photo):
    # a brand-new slot the detector missed entirely, drawn directly on the
    # original photo -- unlike delete/adjust (per-photo overrides on an
    # already-existing slot), this is a genuinely new physical parking
    # space, so it goes into the camera's shared config (every photo of
    # this camera gets it), not a per-photo override.
    config_path = storage.config_path(camera_id)
    if not config_path.exists():
        return jsonify({"error": "camera not found"}), 404
    if _find_raw_photo(batch_id, camera_id, photo) is None:
        return jsonify({"error": "photo not found"}), 404

    payload = request.get_json(force=True)
    raw_polygon = payload.get("raw_polygon")
    _validate_raw_polygon(raw_polygon, expected_len=4)
    # corners can be clicked in any order (a Z-pattern is a natural mistake)
    # -- unsorted they'd save a self-intersecting bowtie quad that renders a
    # twisted label on every photo. Angle-sort around the centroid to get a
    # simple quad regardless of click order.
    cx = sum(p[0] for p in raw_polygon) / 4
    cy = sum(p[1] for p in raw_polygon) / 4
    import math
    raw_polygon = sorted(raw_polygon, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    import uuid
    config = SlotConfig.load(str(config_path))
    new_slot_id = f"web-{uuid.uuid4().hex[:8]}"
    config.slots.append({"id": new_slot_id, "polygon_raw": raw_polygon})
    config.save(str(config_path))

    try:
        _rerender_camera_everywhere(batch_id, camera_id)
    except Exception:
        # config.json is shared across every batch/photo of this camera --
        # don't leave a slot that broke rendering permanently saved there.
        config.slots.pop()
        config.save(str(config_path))
        raise
    # a human explicitly drawing a slot is the strongest possible accept
    # signal: it protects the region from rejected-region suppression and
    # becomes a training positive for the next retrain.
    _log_review_decision(camera_id, raw_polygon, "accept", 1.0,
                          _find_raw_photo(batch_id, camera_id, photo))
    return jsonify({"ok": True, "slot_id": new_slot_id})


def _flag_web_reject(camera_id, slot_id, photo):
    config_path = storage.config_path(camera_id)
    config = SlotConfig.load(str(config_path))
    slot = next((s for s in config.slots if s["id"] == slot_id), None)
    if slot is None:
        return
    record = {
        "id": review_store.candidate_id(camera_id, slot["polygon_raw"]),
        "camera_id": camera_id,
        "slot_id": slot_id,
        "photo": photo,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    review_store.append_web_flag(record)


def _log_review_decision(camera_id, polygon, decision, confidence, source_photo_path):
    """Feed a web edit back into the review log -- an accept protects the
    region from generate_config's rejected-region suppression and becomes a
    training positive; a reject makes re-detection of that region suppressed
    on the camera's next generation instead of resurrecting into its config."""
    cid = review_store.candidate_id(camera_id, polygon)
    crop_path = review_store.CROPS_DIR / f"{cid}.png"
    if source_photo_path is not None and not crop_path.exists():
        import cv2
        import numpy as np
        img = cv2.imread(str(source_photo_path))
        if img is not None:
            review_store.CROPS_DIR.mkdir(parents=True, exist_ok=True)
            pts = np.array([[int(x), int(y)] for x, y in polygon])
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            pad = 8
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
            if x1 > x0 and y1 > y0:
                cv2.imwrite(str(crop_path), img[y0:y1, x0:x1])
    record = {
        "id": cid,
        "camera_id": camera_id,
        "image_path": str(source_photo_path) if source_photo_path else "",
        "polygon": polygon,
        "crop_path": str(crop_path),
        "confidence": confidence,
        "decision": decision,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    review_store.append_decision(record, review_store.LABELS_PATH)


@app.post("/api/batches/<batch_id>/cameras/<camera_id>/photos/<photo>/labels/<slot_id>")
def edit_label(batch_id, camera_id, photo, slot_id):
    config_path = storage.config_path(camera_id)
    if not config_path.exists():
        return jsonify({"error": "camera not found"}), 404
    if _find_raw_photo(batch_id, camera_id, photo) is None:
        return jsonify({"error": "photo not found"}), 404

    payload = request.get_json(force=True)
    action = payload.get("action")

    if action == "delete":
        overrides.exclude_slot(batch_id, camera_id, photo, slot_id)
        _flag_web_reject(camera_id, slot_id, photo)
    elif action == "restore":
        # undo for per-photo edits: un-hide an excluded label / revert a
        # per-photo adjustment back to the slot's default placement
        overrides.restore_slot(batch_id, camera_id, photo, slot_id)
    elif action == "delete_all":
        # not a per-photo occlusion but "this slot is wrong, period" -- remove
        # it from the camera's shared config (every batch/photo of this
        # camera) and log a reject so the region stays suppressed when this
        # camera's detection is ever re-run.
        config = SlotConfig.load(str(config_path))
        slot = next((s for s in config.slots if s["id"] == slot_id), None)
        if slot is None:
            return jsonify({"error": "slot not found"}), 404
        config.slots = [s for s in config.slots if s["id"] != slot_id]
        config.save(str(config_path))
        _log_review_decision(camera_id, slot["polygon_raw"], "reject",
                              slot.get("confidence", 0.0),
                              _find_raw_photo(batch_id, camera_id, photo))
        _rerender_camera_everywhere(batch_id, camera_id)
        return jsonify({"ok": True})
    elif action in ("adjust", "adjust_all"):
        config = SlotConfig.load(str(config_path))
        slot = next((s for s in config.slots if s["id"] == slot_id), None)
        if slot is None:
            return jsonify({"error": "slot not found"}), 404
        view, homography = pipeline._prepare_slot_view(config, slot, slot_id)
        # raw_polygon: the 2 opposite corners the user dragged out directly on
        # the original uploaded photo (raw pixel space) -- convert through the
        # same local-view + homography pipeline.run_auto_all uses, just
        # starting one step earlier (raw pixels instead of the old rectified
        # patch image's own pixel space).
        raw_polygon = payload.get("raw_polygon")
        _validate_raw_polygon(raw_polygon, expected_len=2)
        local_pts = view.raw_to_local(raw_polygon)
        plane_pts = perspective.pixel_to_plane_points(homography, local_pts)
        (px0, py0), (px1, py1) = plane_pts
        box = {
            "cx": float((px0 + px1) / 2),
            "cy": float((py0 + py1) / 2),
            "w": float(abs(px1 - px0)),
            "h": float(abs(py1 - py0)),
        }
        if action == "adjust_all":
            # save as this slot's default placement in the shared config --
            # every photo (every batch) of this camera renders with it,
            # instead of repeating the same drag photo by photo.
            slot["label_box"] = box
            config.save(str(config_path))
            _rerender_camera_everywhere(batch_id, camera_id)
            return jsonify({"ok": True})
        overrides.adjust_slot(batch_id, camera_id, photo, slot_id, box)
    else:
        return jsonify({"error": "action must be 'delete', 'delete_all', 'restore', 'adjust' or 'adjust_all'"}), 400

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
    # threaded=True: without it the dev server handles one request at a
    # time, so a single "add/adjust slot" request (which synchronously
    # reruns the whole render pipeline) freezes status polling for every
    # other in-flight batch. For a real deployment, run behind a proper
    # WSGI server instead (e.g. gunicorn -w 1 --threads 4) -- and if you
    # ever go to multiple worker *processes*, note jobs.py's in-memory
    # _JOBS dict isn't shared across them, so pin -w 1 or move job state
    # out of process first.
    app.run(port=5050, threaded=True)
