from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

from Backend.app.main import app

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "fetalytics-dev-clinical-key-2026"}


def test_security_rejects_invalid_api_key() -> None:
    res = client.get("/api/v1/datasets/", headers={"X-API-Key": "invalid-unauthorized-key"})
    assert res.status_code == 401


def test_datasets_and_models_endpoints() -> None:
    res_ds = client.get("/api/v1/datasets/", headers=AUTH_HEADERS)
    assert res_ds.status_code == 200
    datasets = res_ds.json()
    assert isinstance(datasets, list)

    res_mv = client.get("/api/v1/models/versions", headers=AUTH_HEADERS)
    assert res_mv.status_code == 200
    assert isinstance(res_mv.json(), list)

    res_tr = client.get("/api/v1/models/training-runs", headers=AUTH_HEADERS)
    assert res_tr.status_code == 200
    assert isinstance(res_tr.json(), list)


def test_upload_validation_and_end_to_end_image_analysis_job(tmp_path: Path) -> None:
    # 1. Reject unsupported file extension (.exe)
    bad_res = client.post(
        "/api/v1/uploads/initiate",
        headers=AUTH_HEADERS,
        data={"patient_anonymized_id": "PAT-SEC-TEST"},
        files={"file": ("malware.exe", b"MZ_bad", "application/octet-stream")},
    )
    assert bad_res.status_code == 400

    # 2. Use a real extracted frame from IUGC2024 if available, else synthetic ultrasound image
    real_frame = Path("AI/datasets/extracted/frames/test_10_80_frame_80.png")
    if real_frame.exists():
        img_bytes = real_frame.read_bytes()
        fname = real_frame.name
    else:
        arr = np.full((256, 256, 3), 80, dtype=np.uint8)
        cv2.ellipse(arr, (128, 128), (60, 45), 10, 0, 360, (220, 220, 220), -1)
        ok, enc = cv2.imencode(".png", arr)
        assert ok
        img_bytes = enc.tobytes()
        fname = "sample_ultrasound.png"

    res_job = client.post(
        "/api/v1/jobs/analyze",
        headers=AUTH_HEADERS,
        data={
            "patient_anonymized_id": "PAT-JOB-TEST-01",
            "pixel_spacing_mm": "0.22",
            "run_sync": "true",
        },
        files={"file": (fname, img_bytes, "image/png")},
    )
    assert res_job.status_code == 202, res_job.text
    job_data = res_job.json()
    job_id = job_data["id"]
    assert job_data["status"] == "completed"
    assert job_data["is_calibrated"] is True

    # 3. Verify job detail, measurements, report, and media streaming endpoints
    res_detail = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH_HEADERS)
    assert res_detail.status_code == 200
    assert res_detail.json()["status"] == "completed"

    res_meas = client.get(f"/api/v1/jobs/{job_id}/measurements", headers=AUTH_HEADERS)
    assert res_meas.status_code == 200
    meas_list = res_meas.json()
    assert isinstance(meas_list, list)

    res_rep = client.get(f"/api/v1/jobs/{job_id}/report", headers=AUTH_HEADERS)
    assert res_rep.status_code == 200
    rep_json = res_rep.json()
    assert "DECISION-SUPPORT ONLY" in rep_json["disclaimer"]
    assert rep_json["source_file_sha256"] == job_data["source_sha256"]

    res_media = client.get(f"/api/v1/jobs/{job_id}/media/annotated", headers=AUTH_HEADERS)
    assert res_media.status_code == 200
    assert len(res_media.content) > 100

    # 4. Test retention cleanup dry-run and job deletion
    res_ret = client.post(
        "/api/v1/retention/cleanup",
        headers=AUTH_HEADERS,
        json={"dry_run": True},
    )
    assert res_ret.status_code == 200
    assert res_ret.json()["dry_run"] is True

    res_del = client.delete(f"/api/v1/jobs/{job_id}", headers=AUTH_HEADERS)
    assert res_del.status_code == 200
    assert res_del.json()["deleted_job_uuid"] == job_data["uuid"]
