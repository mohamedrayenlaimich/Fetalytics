import datetime
import uuid
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base


def _gen_uuid() -> str:
    return str(uuid.uuid4())


class Patient(Base):
    """
    Patient / pseudonymous subject record adhering to medical imaging privacy guidelines.
    Uses anonymized subject identifiers without direct personal identifiers.
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False, default=_gen_uuid)
    anonymized_id = Column(String(100), unique=True, index=True, nullable=False)
    gestational_age_weeks = Column(Float, nullable=True)  # Gestational age in weeks (e.g. 21.4)
    maternal_age = Column(Integer, nullable=True)
    clinical_notes = Column(Text, nullable=True)
    dataset_source = Column(String(100), nullable=True)
    subject_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    scans = relationship("UltrasoundScan", back_populates="patient", cascade="all, delete-orphan")
    examinations = relationship("Examination", back_populates="patient", cascade="all, delete-orphan")
    dataset_files = relationship("DatasetFile", back_populates="subject")


# Alias for pseudonymous subject terminology
Subject = Patient
