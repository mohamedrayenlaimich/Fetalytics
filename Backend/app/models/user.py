import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base

class User(Base):
    """Healthcare professionals, sonographers, radiologists, and researchers."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    role = Column(String(50), default="sonographer")  # sonographer, obstetrician, radiologist, admin
    hospital_department = Column(String(150), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    uploaded_scans = relationship("UltrasoundScan", back_populates="uploaded_by_user")
    verified_measurements = relationship("BiometricMeasurement", back_populates="verified_by_user")
    clinical_reports = relationship("ClinicalReport", back_populates="clinician")
