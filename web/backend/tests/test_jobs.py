import shutil
from pathlib import Path

import jobs
import storage

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
