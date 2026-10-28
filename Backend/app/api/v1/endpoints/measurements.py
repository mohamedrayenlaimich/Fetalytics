from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from Backend.app.core.database import get_db
from Backend.app.models.measurement import BiometricMeasurement
from Backend.app.models.analysis import AIAnalysisSession
from Backend.app.schemas.measurement import (
    BiometricMeasurementCreate,
    BiometricMeasurementUpdate,
    BiometricMeasurementResponse,
)

router = APIRouter(prefix="/measurements", tags=["Biometric Measurements"])

@router.post("/", response_model=BiometricMeasurementResponse, status_code=status.HTTP_201_CREATED)
def record_measurement(payload: BiometricMeasurementCreate, db: Session = Depends(get_db)):
    """Add a new biometric measurement (BPD, HC, AC, FL, OFD)."""
    session = db.query(AIAnalysisSession).filter(AIAnalysisSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AI Analysis Session not found.")
    
    measurement = BiometricMeasurement(**payload.model_dump())
    db.add(measurement)
    db.commit()
    db.refresh(measurement)
    return measurement

@router.get("/", response_model=List[BiometricMeasurementResponse])
def list_measurements(
    session_id: Optional[int] = None,
    measurement_type: Optional[str] = None,
    verification_status: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Query biometric measurements with optional filters."""
    query = db.query(BiometricMeasurement)
    if session_id is not None:
        query = query.filter(BiometricMeasurement.session_id == session_id)
    if measurement_type:
        query = query.filter(BiometricMeasurement.measurement_type == measurement_type.upper())
    if verification_status:
        query = query.filter(BiometricMeasurement.verification_status == verification_status)
    return query.offset(skip).limit(limit).all()

@router.get("/{measurement_id}", response_model=BiometricMeasurementResponse)
def get_measurement(measurement_id: int, db: Session = Depends(get_db)):
    """Get single biometric measurement details."""
    measurement = db.query(BiometricMeasurement).filter(BiometricMeasurement.id == measurement_id).first()
    if not measurement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Measurement not found.")
    return measurement

@router.put("/{measurement_id}", response_model=BiometricMeasurementResponse)
def verify_or_adjust_measurement(
    measurement_id: int,
    payload: BiometricMeasurementUpdate,
    db: Session = Depends(get_db)
):
    """
    Clinician verification endpoint:
    Allows sonographer or obstetrician to verify or adjust AI measurement values and leave clinical notes.
    """
    measurement = db.query(BiometricMeasurement).filter(BiometricMeasurement.id == measurement_id).first()
    if not measurement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Measurement not found.")

    update_data = payload.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(measurement, field, val)
        
    db.commit()
    db.refresh(measurement)
    return measurement

@router.delete("/{measurement_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_measurement(measurement_id: int, db: Session = Depends(get_db)):
    """Delete a biometric measurement."""
    measurement = db.query(BiometricMeasurement).filter(BiometricMeasurement.id == measurement_id).first()
    if not measurement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Measurement not found.")
    db.delete(measurement)
    db.commit()
    return None
