from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field

class FetalStructureDetectionBase(BaseModel):
    frame_number: int = 0
    timestamp_ms: Optional[float] = None
    structure_type: str = Field(..., description="e.g. 'fetal_head', 'fetal_abdomen', 'fetal_femur'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: Optional[Any] = None
    contour_points: Optional[Any] = None
    segmentation_mask_path: Optional[str] = None
    track_id: Optional[int] = None
    is_keyframe: bool = False

class FetalStructureDetectionCreate(FetalStructureDetectionBase):
    session_id: int

class FetalStructureDetectionResponse(FetalStructureDetectionBase):
    id: int
    session_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AIAnalysisSessionBase(BaseModel):
    model_version: str = "YOLO-Fetal-v1.0"
    summary_notes: Optional[str] = None

class AIAnalysisSessionCreate(AIAnalysisSessionBase):
    scan_id: int = Field(..., description="Target ultrasound scan ID")

class AIAnalysisSessionUpdate(BaseModel):
    status: Optional[str] = Field(None, description="'queued', 'processing', 'completed', 'failed'")
    processing_time_ms: Optional[float] = None
    quality_score: Optional[float] = None
    head_detected: Optional[bool] = None
    abdomen_detected: Optional[bool] = None
    femur_detected: Optional[bool] = None
    annotated_video_path: Optional[str] = None
    summary_notes: Optional[str] = None

class AIAnalysisSessionResponse(AIAnalysisSessionBase):
    id: int
    scan_id: int
    status: str
    processing_time_ms: Optional[float] = None
    quality_score: Optional[float] = None
    head_detected: bool
    abdomen_detected: bool
    femur_detected: bool
    annotated_video_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
