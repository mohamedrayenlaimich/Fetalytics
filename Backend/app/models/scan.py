import datetime
from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base

class UltrasoundScan(Base):
    """Represents an uploaded ultrasound video or image scan."""
    __tablename__ = "ultrasound_scans"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    title = Column(String(255), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), default="video")  # "video" or "image"
    mime_type = Column(String(100), nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)

    # Video/Image metadata
    total_frames = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    resolution_width = Column(Integer, nullable=True)
    resolution_height = Column(Integer, nullable=True)
    pixel_spacing_mm = Column(Float, nullable=True)  # Physical scale calibration (mm per pixel)

    # Processing pipeline status
    status = Column(String(50), default="uploaded")  # uploaded, queued, processing, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="scans")
    uploaded_by_user = relationship("User", back_populates="uploaded_scans")
    analysis_sessions = relationship("AIAnalysisSession", back_populates="scan", cascade="all, delete-orphan")
    clinical_reports = relationship("ClinicalReport", back_populates="scan", cascade="all, delete-orphan")
