import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from Backend.app.core.database import Base

class ClinicalReport(Base):
    """
    Final clinical ultrasound analysis report.
    Consolidates AI measurements, Estimated Fetal Weight (EFW),
    and physician interpretation notes.
    """
    __tablename__ = "clinical_reports"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("ultrasound_scans.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("ai_analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    clinician_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    report_title = Column(String(255), default="Fetal Biometry & Anatomy Report")
    
    # Composite calculations
    estimated_fetal_weight_grams = Column(Float, nullable=True)  # Hadlock formula EFW (g)
    composite_ga_weeks = Column(Float, nullable=True)  # Overall composite gestational age
    summary_metrics = Column(JSON, nullable=True)  # Snapshot of all verified measurements

    # Clinical findings
    clinical_findings = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    pdf_report_path = Column(String(512), nullable=True)

    status = Column(String(50), default="draft")  # draft, finalized, approved, archived
    finalized_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    scan = relationship("UltrasoundScan", back_populates="clinical_reports")
    session = relationship("AIAnalysisSession", back_populates="clinical_reports")
    clinician = relationship("User", back_populates="clinical_reports")
