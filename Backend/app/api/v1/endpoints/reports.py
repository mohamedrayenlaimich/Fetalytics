from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from Backend.app.core.database import get_db
from Backend.app.models.report import ClinicalReport
from Backend.app.models.scan import UltrasoundScan
from Backend.app.schemas.report import (
    ClinicalReportCreate,
    ClinicalReportUpdate,
    ClinicalReportResponse,
)

router = APIRouter(prefix="/reports", tags=["Clinical Reports"])

@router.post("/", response_model=ClinicalReportResponse, status_code=status.HTTP_201_CREATED)
def create_clinical_report(payload: ClinicalReportCreate, db: Session = Depends(get_db)):
    """Create a new clinical diagnostic report for a scan."""
    scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == payload.scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ultrasound scan not found.")

    report = ClinicalReport(**payload.model_dump())
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

@router.get("/", response_model=List[ClinicalReportResponse])
def list_reports(
    scan_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List clinical reports with optional filters."""
    query = db.query(ClinicalReport)
    if scan_id is not None:
        query = query.filter(ClinicalReport.scan_id == scan_id)
    if status_filter:
        query = query.filter(ClinicalReport.status == status_filter)
    return query.offset(skip).limit(limit).all()

@router.get("/{report_id}", response_model=ClinicalReportResponse)
def get_report(report_id: int, db: Session = Depends(get_db)):
    """Retrieve detailed clinical report."""
    report = db.query(ClinicalReport).filter(ClinicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")
    return report

@router.put("/{report_id}", response_model=ClinicalReportResponse)
def update_report(report_id: int, payload: ClinicalReportUpdate, db: Session = Depends(get_db)):
    """Update report findings, recommendations, or finalize the report."""
    report = db.query(ClinicalReport).filter(ClinicalReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    update_data = payload.model_dump(exclude_unset=True)
    if update_data.get("status") == "finalized" and not report.finalized_at:
        report.finalized_at = datetime.utcnow()

    for k, v in update_data.items():
        setattr(report, k, v)

    db.commit()
    db.refresh(report)
    return report
