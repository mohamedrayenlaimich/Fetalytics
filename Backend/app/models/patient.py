import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base

class Patient(Base):
    """
    Patient record adhering to medical imaging privacy guidelines.
    Uses anonymized patient identifiers without direct personal identifiers.
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    anonymized_id = Column(String(100), unique=True, index=True, nullable=False)
    gestational_age_weeks = Column(Float, nullable=True)  # Gestational age in weeks (e.g. 21.4)
    maternal_age = Column(Integer, nullable=True)
    clinical_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    scans = relationship("UltrasoundScan", back_populates="patient", cascade="all, delete-orphan")
