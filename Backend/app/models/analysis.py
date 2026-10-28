import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base

class AIAnalysisSession(Base):
    """
    An execution run of the AI inference pipeline on an ultrasound scan.
    Tracks detection, segmentation, tracking, and measurement progress.
    """
    __tablename__ = "ai_analysis_sessions"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("ultrasound_scans.id", ondelete="CASCADE"), nullable=False, index=True)

    model_version = Column(String(100), default="YOLO-Fetal-v1.0")
    status = Column(String(50), default="queued")  # queued, processing, completed, failed
    processing_time_ms = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)  # Image/video ultrasound quality metric (0-100)

    # Detected anatomical landmarks flags
    head_detected = Column(Boolean, default=False)
    abdomen_detected = Column(Boolean, default=False)
    femur_detected = Column(Boolean, default=False)
    
    annotated_video_path = Column(String(512), nullable=True)
    summary_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    scan = relationship("UltrasoundScan", back_populates="analysis_sessions")
    detections = relationship("FetalStructureDetection", back_populates="session", cascade="all, delete-orphan")
    measurements = relationship("BiometricMeasurement", back_populates="session", cascade="all, delete-orphan")
    clinical_reports = relationship("ClinicalReport", back_populates="session")


class FetalStructureDetection(Base):
    """
    Frame-by-frame detection, segmentation, and tracking of anatomical structures
    (Fetal Head, Abdomen, Femur, etc.).
    """
    __tablename__ = "fetal_structure_detections"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("ai_analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    frame_number = Column(Integer, default=0, index=True)
    timestamp_ms = Column(Float, nullable=True)
    structure_type = Column(String(50), nullable=False)  # fetal_head, fetal_abdomen, fetal_femur
    confidence = Column(Float, nullable=False)  # Detection confidence (0.0 - 1.0)

    # Spatial annotations
    bounding_box = Column(JSON, nullable=True)  # {"x": 120, "y": 80, "w": 200, "h": 180}
    contour_points = Column(JSON, nullable=True)  # [[x1, y1], [x2, y2], ...]
    segmentation_mask_path = Column(String(512), nullable=True)

    track_id = Column(Integer, nullable=True)  # Video temporal tracking identifier
    is_keyframe = Column(Boolean, default=False)  # Optimal standard plane for measurement
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("AIAnalysisSession", back_populates="detections")
    measurements = relationship("BiometricMeasurement", back_populates="detection")
