from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class PatientBase(BaseModel):
    anonymized_id: str = Field(..., description="Unique anonymized patient identifier, e.g. PAT-2026-001")
    gestational_age_weeks: Optional[float] = Field(None, ge=1.0, le=45.0, description="Baseline gestational age in weeks")
    maternal_age: Optional[int] = Field(None, ge=10, le=70, description="Maternal age")
    clinical_notes: Optional[str] = Field(None, description="Clinical history or notes")

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    gestational_age_weeks: Optional[float] = Field(None, ge=1.0, le=45.0)
    maternal_age: Optional[int] = Field(None, ge=10, le=70)
    clinical_notes: Optional[str] = None

class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
