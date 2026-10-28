from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from Backend.app.core.database import get_db
from Backend.app.models.analysis import AIAnalysisSession, FetalStructureDetection
from Backend.app.models.scan import UltrasoundScan
from Backend.app.schemas.analysis import (
    AIAnalysisSessionCreate,
    AIAnalysisSessionUpdate,
    AIAnalysisSessionResponse,
    FetalStructureDetectionCreate,
    FetalStructureDetectionResponse,
)

router = APIRouter(prefix="/analysis", tags=["AI Analysis & Inference"])

@router.post("/sessions", response_model=AIAnalysisSessionResponse, status_code=status.HTTP_201_CREATED)
def start_analysis_session(payload: AIAnalysisSessionCreate, db: Session = Depends(get_db)):
    """Initialize an AI analysis pipeline run for a scan."""
    scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == payload.scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")
    
    session = AIAnalysisSession(**payload.model_dump())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/sessions/{session_id}", response_model=AIAnalysisSessionResponse)
def get_analysis_session(session_id: int, db: Session = Depends(get_db)):
    """Check status and metrics of an AI analysis run."""
    session = db.query(AIAnalysisSession).filter(AIAnalysisSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis session not found.")
    return session

@router.put("/sessions/{session_id}", response_model=AIAnalysisSessionResponse)
def update_analysis_session(session_id: int, payload: AIAnalysisSessionUpdate, db: Session = Depends(get_db)):
    """Update progress, status, or results of an AI analysis run."""
    session = db.query(AIAnalysisSession).filter(AIAnalysisSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis session not found.")
    
    update_data = payload.model_dump(exclude_unset=True)
    if update_data.get("status") == "completed" and not session.completed_at:
        session.completed_at = datetime.utcnow()
        
    for k, v in update_data.items():
        setattr(session, k, v)
        
    db.commit()
    db.refresh(session)
    return session

@router.post("/detections", response_model=FetalStructureDetectionResponse, status_code=status.HTTP_201_CREATED)
def record_detection(payload: FetalStructureDetectionCreate, db: Session = Depends(get_db)):
    """Record an anatomical structure detection on a video frame or image."""
    session = db.query(AIAnalysisSession).filter(AIAnalysisSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis session not found.")
        
    detection = FetalStructureDetection(**payload.model_dump())
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection

@router.get("/sessions/{session_id}/detections", response_model=List[FetalStructureDetectionResponse])
def list_session_detections(
    session_id: int,
    frame_number: Optional[int] = None,
    structure_type: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Retrieve detections with bounding boxes & contours for a given session."""
    query = db.query(FetalStructureDetection).filter(FetalStructureDetection.session_id == session_id)
    if frame_number is not None:
        query = query.filter(FetalStructureDetection.frame_number == frame_number)
    if structure_type:
        query = query.filter(FetalStructureDetection.structure_type == structure_type)
    return query.offset(skip).limit(limit).all()
