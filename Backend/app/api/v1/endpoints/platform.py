import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from AI.analysis.pipelines import UltrasoundInferenceEngine
from AI.datasets.ingestion import DatasetIngestionPipeline
from Backend.app.core.config import BASE_DIR, settings
from Backend.app.core.database import SessionLocal, get_db
from Backend.app.core.security import validate_upload_file, verify_clinical_auth
from Backend.app.models import (
    AIAnalysisSession,
    AnalysisJob,
    AnalysisReport,
    BiometricMeasurement,
    CLINICAL_DISCLAIMER_TEXT,
    ClinicalReport,
    Dataset,
    Examination,
    FetalStructureDetection,
    Measurement,
    ModelVersion,
    Patient,
    TrainingRun,
    UltrasoundScan,
    UltrasoundVideo,
)
from Backend.app.services.b2_storage import B2StorageService, StorageError, compute_file_sha256
from Backend.app.services.retention import delete_job_and_artifacts, run_retention_cleanup

router = APIRouter(tags=["Cloud Storage, Datasets, Jobs & MLOps"])

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DatasetIngestRequest(BaseModel):
    dataset_name: str = "IUGC2024-Fetal-Ultrasound-V3"
    dataset_version: str = "3.0.0"
    b2_bucket: Optional[str] = None
    b2_source_key: str = "datasets/DatasetV3.zip"
    extracted_dir: Optional[str] = None
    max_videos: Optional[int] = Field(default=10, ge=1, le=500)
    dry_run: bool = False


class RetentionCleanupRequest(BaseModel):
    upload_retention_days: Optional[int] = Field(default=None, ge=1)
    artifact_retention_days: Optional[int] = Field(default=None, ge=1)
    dry_run: bool = True


def _execute_analysis_job(job_id: int) -> None:
    """
    Background worker function that runs the image or video computer-vision pipeline
    and records all traceable measurements, detections, and reports in PostgreSQL.
    """
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return

        job.status = "processing"
        job.started_at = datetime.utcnow()
        job.progress_percent = 5.0
        db.commit()

        input_path = Path(job.input_local_path or "")
        if not input_path.exists():
            raise FileNotFoundError(f"Input media file not found: {input_path}")

        # Resolve active model version checkpoint if available
        active_mv = (
            db.query(ModelVersion)
            .filter(ModelVersion.is_active == True)  # noqa: E712
            .order_by(ModelVersion.id.desc())
            .first()
        )
        ckpt_path = (
            Path(active_mv.local_artifact_path)
            if (active_mv and active_mv.local_artifact_path and Path(active_mv.local_artifact_path).exists())
            else None
        )
        engine = UltrasoundInferenceEngine(checkpoint_path=ckpt_path)

        if active_mv:
            job.model_version_id = active_mv.id

        if job.job_type == "image_analysis":
            out = engine.analyze_image(
                image_path=input_path,
                pixel_spacing_mm=job.pixel_spacing_mm,
            )
            job.progress_percent = 100.0
            job.output_local_path = out.annotated_image_path
            job.processing_config = out.preprocessing_config
            job.quality_warnings = out.quality_warnings
            job.processing_time_ms = out.processing_time_ms
            job.result_summary = {
                "head_detected": out.head_detected,
                "ps_detected": out.ps_detected,
                "standard_plane_probability": out.standard_plane_probability,
                "is_standard_plane": out.is_standard_plane,
                "overall_confidence": out.overall_confidence,
                "quality_score": out.quality_assessment.get("quality_score"),
            }

            meas_summary: Dict[str, Any] = {}
            for m in out.measurements:
                meas_rec = Measurement(
                    analysis_job_id=job.id,
                    session_id=job.session_id,
                    model_version_tag=out.model_version_tag,
                    measurement_type=m["measurement_type"],
                    value=m["value"],
                    unit=m["unit"],
                    value_mm=m["value_mm"],
                    value_px=m["value_px"],
                    pixel_spacing_mm=m["pixel_spacing_mm"],
                    is_calibrated=m["is_calibrated"],
                    confidence_score=m["confidence_score"],
                    uncertainty_px=m["uncertainty_px"],
                    uncertainty_mm=m["uncertainty_mm"],
                    estimated_ga_weeks=m.get("estimated_ga_weeks"),
                    growth_percentile=m.get("growth_percentile"),
                    method=m.get("method", "contour_ellipse_fit"),
                    quality_warnings=m.get("quality_warnings"),
                )
                db.add(meas_rec)
                meas_summary[f"{m['measurement_type']}_{m['unit']}"] = m["value"]

                if job.session_id and m["value_mm"] is not None:
                    db.add(
                        BiometricMeasurement(
                            session_id=job.session_id,
                            measurement_type=m["measurement_type"],
                            value_mm=m["value_mm"],
                            estimated_ga_weeks=m.get("estimated_ga_weeks"),
                            growth_percentile=m.get("growth_percentile"),
                            confidence_score=m["confidence_score"],
                        )
                    )

            rep = AnalysisReport(
                analysis_job_id=job.id,
                examination_id=job.examination_id,
                scan_id=job.scan_id,
                model_version_tag=out.model_version_tag,
                source_file_sha256=out.source_sha256,
                processing_config=out.preprocessing_config,
                overall_confidence=out.overall_confidence,
                quality_score=out.quality_assessment.get("quality_score"),
                quality_warnings=out.quality_warnings,
                measurements_summary=meas_summary,
                annotated_media_path=out.annotated_image_path,
                report_json=out.to_dict(),
            )
            db.add(rep)

        else:
            def _update_prog(pct: float) -> None:
                job.progress_percent = round(min(95.0, max(5.0, pct)), 1)
                db.commit()

            frame_step = int((job.processing_config or {}).get("frame_step", 4))
            out_v = engine.analyze_video(
                video_path=input_path,
                pixel_spacing_mm=job.pixel_spacing_mm,
                frame_step=frame_step,
                progress_callback=_update_prog,
            )
            job.progress_percent = 100.0
            job.output_local_path = out_v.annotated_video_path
            job.processing_config = out_v.preprocessing_config
            job.quality_warnings = out_v.quality_warnings
            job.processing_time_ms = out_v.processing_time_ms
            job.result_summary = {
                "frames_evaluated": out_v.frames_evaluated,
                "standard_plane_frames_count": out_v.standard_plane_frames_count,
                "keyframe_index": out_v.keyframe_index,
                "keyframe_timestamp_ms": out_v.keyframe_timestamp_ms,
                "overall_confidence": out_v.overall_confidence,
                "quality_score": out_v.quality_score,
                "tracking_metrics": out_v.tracking_metrics,
                "annotated_keyframe_path": out_v.annotated_keyframe_path,
            }

            meas_summary_v: Dict[str, Any] = {}
            for m in out_v.aggregated_measurements:
                meas_rec = Measurement(
                    analysis_job_id=job.id,
                    session_id=job.session_id,
                    model_version_tag=out_v.model_version_tag,
                    measurement_type=m["measurement_type"],
                    value=m["value"],
                    unit=m["unit"],
                    value_mm=m["value_mm"],
                    value_px=m["value_px"],
                    pixel_spacing_mm=m["pixel_spacing_mm"],
                    is_calibrated=m["is_calibrated"],
                    confidence_score=m["confidence_score"],
                    uncertainty_px=m["uncertainty_px"],
                    uncertainty_mm=m["uncertainty_mm"],
                    method=m.get("method", "temporal_iqr_median_aggregation"),
                    quality_warnings=out_v.quality_warnings,
                )
                db.add(meas_rec)
                meas_summary_v[f"{m['measurement_type']}_{m['unit']}"] = m["value"]

            rep = AnalysisReport(
                analysis_job_id=job.id,
                examination_id=job.examination_id,
                scan_id=job.scan_id,
                model_version_tag=out_v.model_version_tag,
                source_file_sha256=out_v.source_sha256,
                processing_config=out_v.preprocessing_config,
                overall_confidence=out_v.overall_confidence,
                quality_score=out_v.quality_score,
                quality_warnings=out_v.quality_warnings,
                measurements_summary=meas_summary_v,
                annotated_media_path=out_v.annotated_video_path,
                report_json=out_v.to_dict(),
            )
            db.add(rep)

        job.status = "completed"
        job.completed_at = datetime.utcnow()

        if job.scan_id:
            scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == job.scan_id).first()
            if scan:
                scan.status = "completed"

        if job.session_id:
            sess = db.query(AIAnalysisSession).filter(AIAnalysisSession.id == job.session_id).first()
            if sess:
                sess.status = "completed"
                sess.completed_at = datetime.utcnow()
                sess.processing_time_ms = job.processing_time_ms
                sess.annotated_video_path = job.output_local_path

        db.commit()

    except Exception as exc:
        db.rollback()
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = f"{type(exc).__name__}: {str(exc)}"
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


# ==============================================================================
# 1. Dataset Registration & Ingestion Endpoints
# ==============================================================================

@router.get("/datasets/b2-objects", dependencies=[Depends(verify_clinical_auth)])
def list_b2_dataset_objects(
    prefix: str = Query("", description="Optional object key prefix in Backblaze B2"),
) -> Dict[str, Any]:
    """List authorized dataset objects in the private Backblaze B2 bucket."""
    storage = B2StorageService()
    try:
        objects = storage.list_objects(prefix=prefix, max_keys=200)
        return {
            "bucket": storage.resolve_bucket(),
            "count": len(objects),
            "objects": [
                {
                    "key": o.key,
                    "size_bytes": o.size_bytes,
                    "etag": o.etag,
                    "last_modified": o.last_modified.isoformat() if o.last_modified else None,
                }
                for o in objects
            ],
        }
    except StorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/datasets/", dependencies=[Depends(verify_clinical_auth)])
def list_registered_datasets(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List datasets registered in Neon PostgreSQL."""
    datasets = db.query(Dataset).order_by(Dataset.id.desc()).all()
    return [
        {
            "id": d.id,
            "uuid": d.uuid,
            "name": d.name,
            "version": d.version,
            "status": d.status,
            "b2_bucket": d.b2_bucket,
            "b2_prefix": d.b2_prefix,
            "total_files": d.total_files,
            "supported_tasks": d.supported_tasks,
            "split_summary": d.split_summary,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in datasets
    ]


@router.get("/datasets/{dataset_id}/inventory", dependencies=[Depends(verify_clinical_auth)])
def get_dataset_inventory(dataset_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get detailed dataset inventory, subject split counts, and quality report."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {
        "id": ds.id,
        "uuid": ds.uuid,
        "name": ds.name,
        "version": ds.version,
        "status": ds.status,
        "total_files": ds.total_files,
        "supported_tasks": ds.supported_tasks,
        "split_summary": ds.split_summary,
        "inventory_report": ds.inventory_report,
    }


@router.post("/datasets/ingest", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_clinical_auth)])
def ingest_dataset_endpoint(
    payload: DatasetIngestRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Run dataset ingestion and validation pipeline (supports dry_run)."""
    extract_dir = (
        Path(payload.extracted_dir)
        if payload.extracted_dir
        else (settings.TEMP_STORAGE_DIR / "DatasetV3" / "extracted")
    )
    if not extract_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Extracted directory '{extract_dir}' not found. Run CLI extraction first.",
        )

    pipeline = DatasetIngestionPipeline(db=db)
    report = pipeline.ingest_extracted_directory(
        extracted_root=extract_dir,
        dataset_name=payload.dataset_name,
        dataset_version=payload.dataset_version,
        b2_bucket=payload.b2_bucket,
        b2_source_key=payload.b2_source_key,
        dry_run=payload.dry_run,
        max_videos=payload.max_videos,
    )
    return asdict(report)


# ==============================================================================
# 2. Upload Initiation & Upload Status Endpoints
# ==============================================================================

@router.post("/uploads/initiate", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_clinical_auth)])
async def initiate_upload(
    patient_anonymized_id: str = Form("PAT-CLINICAL-001"),
    title: Optional[str] = Form(None),
    pixel_spacing_mm: Optional[float] = Form(None),
    sync_to_b2: bool = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Validate and save an uploaded ultrasound image or video, compute SHA-256 checksum,
    optionally back up to private Backblaze B2 storage, and register scan metadata in PostgreSQL.
    """
    safe_name = validate_upload_file(file)
    ext = Path(safe_name).suffix.lower()
    media_type = "video" if ext in {".mp4", ".avi", ".mov"} else "image"

    patient = db.query(Patient).filter(Patient.anonymized_id == patient_anonymized_id).first()
    if not patient:
        patient = Patient(anonymized_id=patient_anonymized_id)
        db.add(patient)
        db.flush()

    dest_filename = f"upload_{patient.id}_{int(time.time() * 1000)}_{safe_name}"
    dest_path = UPLOAD_DIR / dest_filename

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    written = 0
    with dest_path.open("wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out_f.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Upload exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
                )
            out_f.write(chunk)

    if written == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    sha256_hex = compute_file_sha256(dest_path)

    width, height, total_frames, fps, duration = None, None, None, None, None
    if media_type == "video":
        cap = cv2.VideoCapture(str(dest_path))
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = round(total_frames / fps, 2) if (fps and total_frames) else None
        cap.release()
    else:
        img = cv2.imread(str(dest_path))
        if img is not None:
            height, width = img.shape[:2]
            total_frames = 1

    b2_key = None
    if sync_to_b2 and settings.b2_is_configured:
        try:
            storage = B2StorageService()
            b2_key = f"uploads/{sha256_hex[:12]}_{safe_name}"
            storage.upload_file(dest_path, key=b2_key)
        except Exception:
            b2_key = None

    scan = UltrasoundScan(
        patient_id=patient.id,
        title=title or safe_name,
        filename=safe_name,
        file_path=str(dest_path),
        file_type=media_type,
        mime_type=file.content_type,
        file_size_bytes=written,
        total_frames=total_frames,
        fps=fps,
        duration_seconds=duration,
        resolution_width=width,
        resolution_height=height,
        pixel_spacing_mm=pixel_spacing_mm,
        status="uploaded",
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return {
        "scan_id": scan.id,
        "patient_id": patient.id,
        "patient_anonymized_id": patient.anonymized_id,
        "filename": scan.filename,
        "file_type": scan.file_type,
        "file_size_bytes": scan.file_size_bytes,
        "sha256_checksum": sha256_hex,
        "b2_object_key": b2_key,
        "resolution": {"width": width, "height": height},
        "total_frames": total_frames,
        "fps": fps,
        "pixel_spacing_mm": pixel_spacing_mm,
        "status": scan.status,
        "created_at": scan.created_at.isoformat(),
    }


@router.get("/uploads/{scan_id}/status", dependencies=[Depends(verify_clinical_auth)])
def get_upload_status(scan_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Check upload status and file verification details for a scan."""
    scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    local_exists = Path(scan.file_path).exists() if scan.file_path else False
    return {
        "scan_id": scan.id,
        "filename": scan.filename,
        "file_type": scan.file_type,
        "file_size_bytes": scan.file_size_bytes,
        "status": scan.status,
        "local_verified": local_exists,
        "pixel_spacing_mm": scan.pixel_spacing_mm,
        "error_message": scan.error_message,
        "updated_at": scan.updated_at.isoformat() if scan.updated_at else None,
    }


# ==============================================================================
# 3. Image & Video Analysis Jobs, Progress, Measurements, Reports & Media Stream
# ==============================================================================

@router.post("/jobs/analyze", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_clinical_auth)])
async def create_analysis_job(
    background_tasks: BackgroundTasks,
    scan_id: Optional[int] = Form(None),
    patient_anonymized_id: str = Form("PAT-CLINICAL-001"),
    pixel_spacing_mm: Optional[float] = Form(None),
    frame_step: int = Form(4),
    run_sync: bool = Form(False),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Create and dispatch an image or video AI analysis job.
    Runs asynchronously via BackgroundTasks by default so API requests never block,
    or synchronously when run_sync=True is passed.
    """
    target_scan: Optional[UltrasoundScan] = None
    if scan_id is not None:
        target_scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == scan_id).first()
        if not target_scan:
            raise HTTPException(status_code=404, detail=f"Scan ID {scan_id} not found.")
    elif file is not None:
        up_info = await initiate_upload(
            patient_anonymized_id=patient_anonymized_id,
            title=file.filename,
            pixel_spacing_mm=pixel_spacing_mm,
            sync_to_b2=False,
            file=file,
            db=db,
        )
        target_scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == up_info["scan_id"]).first()
    else:
        raise HTTPException(status_code=400, detail="Provide either scan_id or an uploaded file.")

    assert target_scan is not None
    eff_spacing = pixel_spacing_mm if pixel_spacing_mm is not None else target_scan.pixel_spacing_mm
    media_path = Path(target_scan.file_path)
    if not media_path.exists():
        raise HTTPException(status_code=400, detail="Source media file does not exist on server.")

    sha256_hex = compute_file_sha256(media_path)
    job_type = "video_analysis" if target_scan.file_type == "video" else "image_analysis"

    # Create linked Examination and AIAnalysisSession for full backward compatibility
    exam = Examination(
        patient_id=target_scan.patient_id,
        pixel_spacing_mm=eff_spacing,
        status="analyzing",
    )
    db.add(exam)
    db.flush()

    session = AIAnalysisSession(
        scan_id=target_scan.id,
        model_version="FetalHeadUNet-v1.0",
        status="queued",
    )
    db.add(session)
    db.flush()

    job = AnalysisJob(
        examination_id=exam.id,
        scan_id=target_scan.id,
        session_id=session.id,
        job_type=job_type,
        status="queued",
        progress_percent=0.0,
        source_filename=target_scan.filename,
        source_sha256=sha256_hex,
        pixel_spacing_mm=eff_spacing,
        input_local_path=str(media_path),
        processing_config={"frame_step": frame_step},
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    if run_sync:
        _execute_analysis_job(job.id)
        db.refresh(job)
    else:
        background_tasks.add_task(_execute_analysis_job, job.id)

    return _serialize_job(job)


def _serialize_job(job: AnalysisJob) -> Dict[str, Any]:
    return {
        "id": job.id,
        "uuid": job.uuid,
        "scan_id": job.scan_id,
        "examination_id": job.examination_id,
        "session_id": job.session_id,
        "job_type": job.job_type,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "source_filename": job.source_filename,
        "source_sha256": job.source_sha256,
        "pixel_spacing_mm": job.pixel_spacing_mm,
        "is_calibrated": bool(job.pixel_spacing_mm and job.pixel_spacing_mm > 0),
        "processing_config": job.processing_config,
        "quality_warnings": job.quality_warnings or [],
        "result_summary": job.result_summary,
        "error_message": job.error_message,
        "processing_time_ms": job.processing_time_ms,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "disclaimer": CLINICAL_DISCLAIMER_TEXT,
    }


@router.get("/jobs/", dependencies=[Depends(verify_clinical_auth)])
def list_analysis_jobs(
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List recent image and video analysis jobs."""
    jobs = db.query(AnalysisJob).order_by(AnalysisJob.id.desc()).limit(limit).all()
    return [_serialize_job(j) for j in jobs]


@router.get("/jobs/{job_id}", dependencies=[Depends(verify_clinical_auth)])
def get_analysis_job(job_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get progress, status, measurements, and warnings for a specific analysis job."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    data = _serialize_job(job)
    data["measurements"] = [
        {
            "id": m.id,
            "uuid": m.uuid,
            "measurement_type": m.measurement_type,
            "value": m.value,
            "unit": m.unit,
            "value_mm": m.value_mm,
            "value_px": m.value_px,
            "pixel_spacing_mm": m.pixel_spacing_mm,
            "is_calibrated": m.is_calibrated,
            "confidence_score": m.confidence_score,
            "uncertainty_px": m.uncertainty_px,
            "uncertainty_mm": m.uncertainty_mm,
            "estimated_ga_weeks": m.estimated_ga_weeks,
            "method": m.method,
            "model_version_tag": m.model_version_tag,
        }
        for m in job.measurements
    ]
    return data


@router.get("/jobs/{job_id}/measurements", dependencies=[Depends(verify_clinical_auth)])
def get_job_measurements(job_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieve all measurements generated by an analysis job."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    return [
        {
            "id": m.id,
            "uuid": m.uuid,
            "measurement_type": m.measurement_type,
            "value": m.value,
            "unit": m.unit,
            "value_mm": m.value_mm,
            "value_px": m.value_px,
            "pixel_spacing_mm": m.pixel_spacing_mm,
            "is_calibrated": m.is_calibrated,
            "confidence_score": m.confidence_score,
            "uncertainty_px": m.uncertainty_px,
            "uncertainty_mm": m.uncertainty_mm,
            "estimated_ga_weeks": m.estimated_ga_weeks,
            "method": m.method,
            "model_version_tag": m.model_version_tag,
            "quality_warnings": m.quality_warnings or [],
        }
        for m in job.measurements
    ]


@router.get("/jobs/{job_id}/report", dependencies=[Depends(verify_clinical_auth)])
def get_job_report(job_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve the structured clinical decision-support report for an analysis job."""
    rep = db.query(AnalysisReport).filter(AnalysisReport.analysis_job_id == job_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Analysis report not found for this job.")
    return {
        "id": rep.id,
        "uuid": rep.uuid,
        "analysis_job_id": rep.analysis_job_id,
        "report_title": rep.report_title,
        "disclaimer": rep.disclaimer,
        "model_version_tag": rep.model_version_tag,
        "source_file_sha256": rep.source_file_sha256,
        "processing_config": rep.processing_config,
        "overall_confidence": rep.overall_confidence,
        "quality_score": rep.quality_score,
        "quality_warnings": rep.quality_warnings or [],
        "measurements_summary": rep.measurements_summary,
        "report_json": rep.report_json,
        "created_at": rep.created_at.isoformat() if rep.created_at else None,
    }


@router.get("/jobs/{job_id}/media/{media_kind}", dependencies=[Depends(verify_clinical_auth)])
def stream_job_media(
    job_id: int,
    media_kind: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Serve original input media, annotated output image/video, or keyframe preview
    through an authenticated backend endpoint without exposing storage keys.
    media_kind: 'original' | 'annotated' | 'keyframe'
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    target_path: Optional[Path] = None
    if media_kind == "original" and job.input_local_path:
        target_path = Path(job.input_local_path)
    elif media_kind == "annotated" and job.output_local_path:
        target_path = Path(job.output_local_path)
    elif media_kind == "keyframe":
        kf_str = (job.result_summary or {}).get("annotated_keyframe_path") or job.output_local_path
        if kf_str:
            target_path = Path(kf_str)

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Requested media '{media_kind}' is not available.")

    return FileResponse(path=str(target_path), filename=target_path.name)


@router.delete("/jobs/{job_id}", dependencies=[Depends(verify_clinical_auth)])
def delete_analysis_job(job_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Delete an analysis job and purge its associated local and B2 artifacts."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    storage = B2StorageService() if settings.b2_is_configured else None
    return delete_job_and_artifacts(db=db, job=job, storage=storage)


# ==============================================================================
# 4. Model Registry, Version & Evaluation Summary Endpoints
# ==============================================================================

@router.get("/models/versions", dependencies=[Depends(verify_clinical_auth)])
def list_model_versions(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List registered model versions, evaluation metrics, confidence intervals, and limitations."""
    versions = db.query(ModelVersion).order_by(ModelVersion.id.desc()).all()
    return [
        {
            "id": mv.id,
            "uuid": mv.uuid,
            "model_name": mv.model_name,
            "version_tag": mv.version_tag,
            "task_name": mv.task_name,
            "architecture": mv.architecture,
            "dataset_version": mv.dataset_version,
            "b2_bucket": mv.b2_bucket,
            "b2_artifact_key": mv.b2_artifact_key,
            "artifact_sha256": mv.artifact_sha256,
            "artifact_size_bytes": mv.artifact_size_bytes,
            "preprocessing_config": mv.preprocessing_config,
            "evaluation_summary": mv.evaluation_summary,
            "confidence_intervals": mv.confidence_intervals,
            "limitations_notes": mv.limitations_notes,
            "is_active": mv.is_active,
            "created_at": mv.created_at.isoformat() if mv.created_at else None,
        }
        for mv in versions
    ]


@router.get("/models/training-runs", dependencies=[Depends(verify_clinical_auth)])
def list_training_runs(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """List reproducible training runs, hyperparameters, loss curves, and split manifests."""
    runs = db.query(TrainingRun).order_by(TrainingRun.id.desc()).all()
    return [
        {
            "id": r.id,
            "uuid": r.uuid,
            "dataset_version": r.dataset_version,
            "task_name": r.task_name,
            "architecture": r.architecture,
            "random_seed": r.random_seed,
            "status": r.status,
            "epochs_completed": r.epochs_completed,
            "best_val_metric": r.best_val_metric,
            "hyperparameters": r.hyperparameters,
            "preprocessing_config": r.preprocessing_config,
            "split_manifest": r.split_manifest,
            "training_history": r.training_history,
            "evaluation_metrics": r.evaluation_metrics,
            "subgroup_metrics": r.subgroup_metrics,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


# ==============================================================================
# 5. Retention & Privacy Cleanup Endpoint
# ==============================================================================

@router.post("/retention/cleanup", dependencies=[Depends(verify_clinical_auth)])
def trigger_retention_cleanup(
    payload: RetentionCleanupRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Run retention policy inspection or cleanup for expired uploads and artifacts."""
    return run_retention_cleanup(
        db=db,
        upload_retention_days=payload.upload_retention_days,
        artifact_retention_days=payload.artifact_retention_days,
        dry_run=payload.dry_run,
    )
