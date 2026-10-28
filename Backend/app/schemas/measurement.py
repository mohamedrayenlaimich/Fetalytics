from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class BiometricMeasurementBase(BaseModel):
    measurement_type: str = Field(..., description="Biometric type: 'BPD', 'HC', 'AC', 'FL', 'OFD'")
    value_mm: float = Field(..., description="Biometric measurement value in millimeters")
    estimated_ga_weeks: Optional[float] = Field(None, description="Gestational age estimate from growth curves")
    growth_percentile: Optional[float] = Field(None, ge=0.0, le=100.0, description="Growth percentile (e.g., 50.0%)")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI detection confidence")

class BiometricMeasurementCreate(BiometricMeasurementBase):
    session_id: int = Field(..., description="Target AI analysis session ID")
    detection_id: Optional[int] = Field(None, description="Associated structure detection ID")

class BiometricMeasurementUpdate(BaseModel):
    verification_status: Optional[str] = Field(None, description="'ai_generated', 'verified', 'adjusted'")
    adjusted_value_mm: Optional[float] = Field(None, description="Clinician corrected value in mm")
    clinician_notes: Optional[str] = None
    verified_by_user_id: Optional[int] = None

class BiometricMeasurementResponse(BiometricMeasurementBase):
    id: int
    session_id: int
    detection_id: Optional[int] = None
    verified_by_user_id: Optional[int] = None
    verification_status: str
    adjusted_value_mm: Optional[float] = None
    clinician_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
