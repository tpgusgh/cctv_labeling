import threading
import time
from concurrent.futures import ThreadPoolExecutor

import generate_config
import pipeline
import slot_classifier
import storage

_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LOCK = threading.Lock()
_JOBS = {}

MODEL_PATH = storage.PROJECT_ROOT / "models" / "slot_classifier.joblib"
YOLO_MODEL_PATH = storage.PROJECT_ROOT / "models" / "yolov8_seg_slots.pt"
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


def _process_camera(job_id, batch_id, camera_id, upload_dir):
    try:
        config_path = storage.PROJECT_ROOT / "config" / f"{camera_id}.json"
        if not config_path.exists():
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
                camera_id, str(upload_dir), str(config_path), classifier=classifier, yolo_model=yolo_model)

        _set_status(job_id, status="labeling")
        labeled_dir = storage.labeled_dir(batch_id, camera_id)
        labeled_dir.mkdir(parents=True, exist_ok=True)
        for photo_path in sorted(upload_dir.iterdir()):
            if photo_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            output_path = labeled_dir / f"{photo_path.stem}.png"
            pipeline.run_auto_all(str(config_path), str(photo_path), str(output_path))

        _set_status(job_id, status="done")
    except Exception as e:
        _set_status(job_id, status="error", error=str(e))
