from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class UltrasoundScanBase(BaseModel):
    title: Optional[str] = Field(None, description="Descriptive title or study name")
    file_type: str = Field("video", description="'video' or 'image'")
    mime_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    pixel_spacing_mm: Optional[float] = Field(None, description="Scale calibration in mm per pixel")

class UltrasoundScanCreate(UltrasoundScanBase):
    patient_id: int = Field(..., description="Target patient ID")
    filename: str = Field(..., description="Uploaded original filename")
    file_path: str = Field(..., description="Server storage path for the file")

class UltrasoundScanUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = Field(None, description="'uploaded', 'processing', 'completed', 'failed'")
    pixel_spacing_mm: Optional[float] = None
    error_message: Optional[str] = None

class UltrasoundScanResponse(UltrasoundScanBase):
    id: int
    patient_id: int
    uploaded_by_user_id: Optional[int] = None
    filename: str
    file_path: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
