import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base

class BiometricMeasurement(Base):
    """
    Extracted fetal biometric measurements:
    - BPD (Biparietal Diameter)
    - HC (Head Circumference)
    - AC (Abdominal Circumference)
    - FL (Femur Length)
    - OFD (Occipitofrontal Diameter)
    Includes clinical verification workflow and growth percentiles.
    """
    __tablename__ = "biometric_measurements"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("ai_analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    detection_id = Column(Integer, ForeignKey("fetal_structure_detections.id", ondelete="SET NULL"), nullable=True)
    verified_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    measurement_type = Column(String(50), nullable=False)  # BPD, HC, AC, FL, OFD
    value_mm = Column(Float, nullable=False)  # Measurement value in millimeters
    estimated_ga_weeks = Column(Float, nullable=True)  # Estimated gestational age (Hadlock / WHO)
    growth_percentile = Column(Float, nullable=True)  # 1st - 99th percentile against reference curve
    confidence_score = Column(Float, nullable=True)  # AI confidence (0.0 - 1.0)

    # Clinician review & adjustment workflow
    verification_status = Column(String(50), default="ai_generated")  # ai_generated, verified, adjusted
    adjusted_value_mm = Column(Float, nullable=True)
    clinician_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    session = relationship("AIAnalysisSession", back_populates="measurements")
    detection = relationship("FetalStructureDetection", back_populates="measurements")
    verified_by_user = relationship("User", back_populates="verified_measurements")
