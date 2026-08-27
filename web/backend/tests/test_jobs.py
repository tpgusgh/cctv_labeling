import shutil
from pathlib import Path

import generate_config
import jobs
import storage
import yolo_slot_detector

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SAMPLE_CAMERA = "P1_B1_1_9"
SAMPLE_PHOTO = REPO_ROOT / "no_label" / SAMPLE_CAMERA / "20260820_030004.jpg"


def test_submit_camera_job_labels_photos_for_known_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)

    upload_dir = storage.camera_upload_dir("batch1", SAMPLE_CAMERA)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    job_id = jobs.submit_camera_job("batch1", SAMPLE_CAMERA, upload_dir, photo_count=1)
    status = jobs.wait_for_job(job_id, timeout=30)

    assert status["status"] == "done"
    labeled_file = storage.labeled_dir("batch1", SAMPLE_CAMERA) / f"{SAMPLE_PHOTO.stem}.png"
    assert labeled_file.is_file()


def test_submit_camera_job_uses_yolo_model_when_checkpoint_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path / "web_uploads")
    # config_path in jobs._process_camera is built from storage.PROJECT_ROOT
    # at call time -- redirect it so this test never writes into the real
    # repo's config/ or review/ directories.
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(generate_config, "_save_review_candidates", lambda *a, **k: None)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)

    fake_checkpoint = tmp_path / "fake_yolov8_seg_slots.pt"
    fake_checkpoint.write_text("fake checkpoint bytes")
    monkeypatch.setattr(jobs, "YOLO_MODEL_PATH", fake_checkpoint)

    fake_model = object()
    captured = {}
    monkeypatch.setattr(yolo_slot_detector, "load", lambda path: fake_model)

    def _fake_detect_slots(median_bgr, model, conf=0.25):
        captured["model"] = model
        return [{"polygon": [[1.0, 1.0], [10.0, 1.0], [10.0, 10.0], [1.0, 10.0]], "confidence": 0.9}]

    monkeypatch.setattr(yolo_slot_detector, "detect_slots", _fake_detect_slots)

    camera_id = "no-existing-config-cam"
    upload_dir = storage.camera_upload_dir("batchY", camera_id)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    job_id = jobs.submit_camera_job("batchY", camera_id, upload_dir, photo_count=1)
    status = jobs.wait_for_job(job_id, timeout=30)

    assert status["status"] == "done", status
    assert captured.get("model") is fake_model


def test_get_batch_status_returns_all_cameras_in_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "WEB_UPLOADS_DIR", tmp_path)
    upload_dir = storage.camera_upload_dir("batch2", SAMPLE_CAMERA)
    upload_dir.mkdir(parents=True)
    shutil.copy(SAMPLE_PHOTO, upload_dir / SAMPLE_PHOTO.name)

    job_id = jobs.submit_camera_job("batch2", SAMPLE_CAMERA, upload_dir, photo_count=1)
    jobs.wait_for_job(job_id, timeout=30)

    statuses = jobs.get_batch_status("batch2")
    assert len(statuses) == 1
    assert statuses[0]["camera_id"] == SAMPLE_CAMERA
