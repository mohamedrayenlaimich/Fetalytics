import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from Backend.app.main import app

client = TestClient(app)

def test_api_workflow():
    print("=" * 60)
    print("Testing Fetalytics REST API Endpoints Workflow")
    print("=" * 60)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[1/6] Health Check: OK ->", res.json())

    # 2. Create Patient
    import time
    test_patient_id = f"PAT-TEST-{int(time.time())}"
    res = client.post("/api/v1/patients/", json={
        "anonymized_id": test_patient_id,
        "gestational_age_weeks": 22.5,
        "maternal_age": 29,
        "clinical_notes": "Routine mid-trimester fetal morphology scan."
    })
    assert res.status_code == 201, f"Create patient failed: {res.text}"
    patient_data = res.json()
    patient_pk = patient_data["id"]
    print(f"[2/6] Create Patient: OK -> ID {patient_pk} ({patient_data['anonymized_id']})")

    # 3. Upload a mock ultrasound scan
    mock_file_content = b"RIFF....WAVEfmt ....data mock ultrasound bytes"
    res = client.post(
        "/api/v1/scans/upload",
        data={
            "patient_id": patient_pk,
            "title": "Mid-Trimester Fetal Head Scan",
            "file_type": "video",
            "pixel_spacing_mm": 0.15
        },
        files={"file": ("fetal_head_scan.mp4", mock_file_content, "video/mp4")}
    )
    assert res.status_code == 201, f"Upload scan failed: {res.text}"
    scan_data = res.json()
    scan_pk = scan_data["id"]
    print(f"[3/6] Upload Scan: OK -> Scan ID {scan_pk} (Filename: {scan_data['filename']})")

    # 4. Start AI Analysis Session
    res = client.post("/api/v1/analysis/sessions", json={
        "scan_id": scan_pk,
        "model_version": "YOLO-Fetal-v1.0"
    })
    assert res.status_code == 201, f"Start session failed: {res.text}"
    session_data = res.json()
    session_pk = session_data["id"]
    print(f"[4/6] Start AI Analysis Session: OK -> Session ID {session_pk}")

    # 5. Record Biometric Measurement (BPD & HC)
    res = client.post("/api/v1/measurements/", json={
        "session_id": session_pk,
        "measurement_type": "BPD",
        "value_mm": 54.2,
        "estimated_ga_weeks": 22.4,
        "growth_percentile": 51.0,
        "confidence_score": 0.94
    })
    assert res.status_code == 201, f"Record measurement failed: {res.text}"
    meas_data = res.json()
    meas_pk = meas_data["id"]
    print(f"[5/6] Record Biometric Measurement (BPD): OK -> {meas_data['value_mm']} mm (Confidence: {meas_data['confidence_score']})")

    # Clinician adjusts measurement
    res = client.put(f"/api/v1/measurements/{meas_pk}", json={
        "verification_status": "verified",
        "adjusted_value_mm": 54.5,
        "clinician_notes": "Caliper placed accurately on outer-to-inner skull bone."
    })
    assert res.status_code == 200, f"Verify measurement failed: {res.text}"
    print(f"      Clinician Verification: OK -> Status '{res.json()['verification_status']}', Value: {res.json()['adjusted_value_mm']} mm")

    # 6. Generate Clinical Diagnostic Report
    res = client.post("/api/v1/reports/", json={
        "scan_id": scan_pk,
        "session_id": session_pk,
        "report_title": "Routine 2nd Trimester Biometry Report",
        "estimated_fetal_weight_grams": 510.0,
        "composite_ga_weeks": 22.4,
        "clinical_findings": "Fetal head contours and ventricles within normal limits. Normal biometric growth.",
        "status": "finalized"
    })
    assert res.status_code == 201, f"Create report failed: {res.text}"
    report_data = res.json()
    print(f"[6/6] Clinical Report: OK -> Report ID {report_data['id']} (Status: {report_data['status']})")

    print("\n" + "=" * 60)
    print("ALL REST API ENDPOINTS VALIDATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_api_workflow()
