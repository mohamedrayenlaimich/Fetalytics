from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field

class ClinicalReportBase(BaseModel):
    report_title: str = "Fetal Biometry & Anatomy Report"
    estimated_fetal_weight_grams: Optional[float] = None
    composite_ga_weeks: Optional[float] = None
    summary_metrics: Optional[Any] = None
    clinical_findings: Optional[str] = None
    recommendations: Optional[str] = None
    status: str = "draft"

class ClinicalReportCreate(ClinicalReportBase):
    scan_id: int
    session_id: int
    clinician_id: Optional[int] = None

class ClinicalReportUpdate(BaseModel):
    report_title: Optional[str] = None
    estimated_fetal_weight_grams: Optional[float] = None
    composite_ga_weeks: Optional[float] = None
    clinical_findings: Optional[str] = None
    recommendations: Optional[str] = None
    status: Optional[str] = Field(None, description="'draft', 'finalized', 'archived'")
    pdf_report_path: Optional[str] = None

class ClinicalReportResponse(ClinicalReportBase):
    id: int
    scan_id: int
    session_id: int
    clinician_id: Optional[int] = None
    pdf_report_path: Optional[str] = None
    finalized_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
