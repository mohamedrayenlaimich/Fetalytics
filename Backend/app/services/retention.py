from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from Backend.app.core.config import BASE_DIR, settings
from Backend.app.models import AnalysisJob, AnalysisReport, UltrasoundScan, UltrasoundVideo
from Backend.app.services.b2_storage import B2StorageService


def run_retention_cleanup(
    db: Session,
    upload_retention_days: Optional[int] = None,
    artifact_retention_days: Optional[int] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Identify and optionally delete expired uploaded scans and derived analysis artifacts
    according to configured retention policies.
    """
    up_days = upload_retention_days if upload_retention_days is not None else settings.RETENTION_DAYS_UPLOADS
    art_days = artifact_retention_days if artifact_retention_days is not None else settings.RETENTION_DAYS_ARTIFACTS

    now = datetime.utcnow()
    upload_cutoff = now - timedelta(days=up_days)
    artifact_cutoff = now - timedelta(days=art_days)

    expired_scans = db.query(UltrasoundScan).filter(UltrasoundScan.created_at < upload_cutoff).all()
    expired_jobs = db.query(AnalysisJob).filter(AnalysisJob.created_at < artifact_cutoff).all()

    deleted_local_files: List[str] = []
    deleted_db_records = {"scans": 0, "jobs": 0}

    if not dry_run:
        for job in expired_jobs:
            for p_str in (job.input_local_path, job.output_local_path):
                if p_str:
                    p = Path(p_str)
                    if p.exists() and p.is_file():
                        p.unlink(missing_ok=True)
                        deleted_local_files.append(str(p))
            db.delete(job)
            deleted_db_records["jobs"] += 1

        for scan in expired_scans:
            if scan.file_path:
                p = Path(scan.file_path)
                if p.exists() and p.is_file():
                    p.unlink(missing_ok=True)
                    deleted_local_files.append(str(p))
            db.delete(scan)
            deleted_db_records["scans"] += 1

        db.commit()

    return {
        "dry_run": dry_run,
        "upload_retention_days": up_days,
        "artifact_retention_days": art_days,
        "expired_scans_found": len(expired_scans),
        "expired_jobs_found": len(expired_jobs),
        "deleted_local_files": deleted_local_files,
        "deleted_db_records": deleted_db_records,
    }


def delete_job_and_artifacts(
    db: Session,
    job: AnalysisJob,
    storage: Optional[B2StorageService] = None,
) -> Dict[str, Any]:
    """
    Delete an AnalysisJob along with its derived local files and private B2 objects.
    """
    removed_files: List[str] = []
    removed_b2_keys: List[str] = []

    local_paths = [job.output_local_path]
    for rep in job.reports:
        if rep.annotated_media_path:
            local_paths.append(rep.annotated_media_path)

    for p_str in local_paths:
        if p_str:
            p = Path(p_str)
            if p.exists() and p.is_file():
                p.unlink(missing_ok=True)
                removed_files.append(str(p))

    if storage and settings.b2_is_configured:
        for b2_key in (job.input_b2_key, job.output_b2_key):
            if b2_key and "#" not in b2_key:
                try:
                    storage.delete_object(b2_key)
                    removed_b2_keys.append(b2_key)
                except Exception:
                    pass

    job_uuid = job.uuid
    db.delete(job)
    db.commit()
    return {
        "deleted_job_uuid": job_uuid,
        "removed_local_files": removed_files,
        "removed_b2_keys": removed_b2_keys,
    }
