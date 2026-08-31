import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

import generate_config
import pipeline
import slot_classifier
import storage
from parking_slot import SlotConfig

_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LOCK = threading.Lock()
_JOBS = {}

MODEL_PATH = storage.PROJECT_ROOT / "models" / "slot_classifier.joblib"
# stable name maintained by src/retrain_yolo.py (promote-if-better) -- the
# web app no longer needs a code change when the model improves.
YOLO_MODEL_PATH = storage.PROJECT_ROOT / "models" / "yolov8_seg_slots_production.pt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _set_status(job_id, **fields):
    with _LOCK:
        _JOBS[job_id].update(fields)


def get_batch_status(batch_id):
    with _LOCK:
        return [dict(j) for j in _JOBS.values() if j["batch_id"] == batch_id]


def wait_for_job(job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _LOCK:
            job = dict(_JOBS[job_id])
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def submit_camera_job(batch_id, camera_id, upload_dir, photo_count):
    job_id = f"{batch_id}:{camera_id}"
    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "batch_id": batch_id,
            "camera_id": camera_id,
            "status": "queued",
            "photo_count": photo_count,
            "error": None,
        }
    _EXECUTOR.submit(_process_camera, job_id, batch_id, camera_id, upload_dir)
    return job_id


def _needs_fresh_detection(config_path, upload_dir):
    """An existing config is only valid for frames of the same resolution it
    was generated from -- config_path is keyed purely by the upload-folder
    name, so two different real cameras (or the same camera re-exported at
    a different resolution) sharing that name would otherwise silently reuse
    slot polygons/calibration fit for a different image entirely, producing
    exactly the "detected but in the wrong place" symptom a reviewer would
    blame on the CV/YOLO model rather than this config mixup."""
    if not config_path.exists():
        return True
    first_frame = next(
        (p for p in sorted(upload_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS), None)
    if first_frame is None:
        return False
    actual = cv2.imread(str(first_frame))
    if actual is None:
        return False
    height, width = actual.shape[:2]
    existing = SlotConfig.load(str(config_path))
    return (width, height) != (existing.image_width, existing.image_height)


def _process_camera(job_id, batch_id, camera_id, upload_dir):
    try:
        config_path = storage.config_path(camera_id)
        if _needs_fresh_detection(config_path, upload_dir):
            _set_status(job_id, status="detecting")
            classifier = slot_classifier.load(MODEL_PATH) if MODEL_PATH.exists() else None
            yolo_model = None
            if YOLO_MODEL_PATH.exists():
                # Import lazily -- ultralytics/torch only get pulled in when
                # a trained checkpoint actually exists, so the web app stays
                # light until someone deliberately drops one in.
                import yolo_slot_detector
                yolo_model = yolo_slot_detector.load(YOLO_MODEL_PATH)
            generate_config.generate_config(
                camera_id, str(upload_dir), str(config_path), classifier=classifier, yolo_model=yolo_model,
                auto_accept_agreement=True)

        _set_status(job_id, status="labeling")
        labeled_dir = storage.labeled_dir(batch_id, camera_id)
        labeled_dir.mkdir(parents=True, exist_ok=True)
        labeled_any = False
        skipped = []
        for photo_path in sorted(upload_dir.iterdir()):
            if photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            output_path = labeled_dir / f"{photo_path.stem}.png"
            try:
                pipeline.run_auto_all(str(config_path), str(photo_path), str(output_path))
                labeled_any = True
            except ValueError:
                # one corrupt/unreadable photo (or a non-image renamed .jpg)
                # must not fail the whole camera -- a real folder drag-in is
                # exactly where such files show up. Skip it and keep going.
                skipped.append(photo_path.name)

        if not labeled_any:
            raise ValueError(f"no readable photos ({len(skipped)} unreadable file(s) skipped)")
        _set_status(job_id, status="done",
                     error=f"{len(skipped)}장 손상/판독불가로 건너뜀: {', '.join(skipped[:5])}" if skipped else None)
    except Exception as e:
        _set_status(job_id, status="error", error=str(e))
