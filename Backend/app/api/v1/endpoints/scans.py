import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Query
from sqlalchemy.orm import Session
from Backend.app.core.database import get_db
from Backend.app.models.scan import UltrasoundScan
from Backend.app.models.patient import Patient
from Backend.app.schemas.scan import (
    UltrasoundScanCreate,
    UltrasoundScanUpdate,
    UltrasoundScanResponse,
)

router = APIRouter(prefix="/scans", tags=["Ultrasound Scans"])

# Upload directory: project_root/uploads/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload", response_model=UltrasoundScanResponse, status_code=status.HTTP_201_CREATED)
async def upload_scan_file(
    patient_id: int = Form(...),
    title: Optional[str] = Form(None),
    file_type: str = Form("video"),  # "video" or "image"
    pixel_spacing_mm: Optional[float] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload an ultrasound video (MP4/AVI) or image file.
    Saves file to server storage and registers metadata in the database.
    """
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    # Save uploaded file safely
    dest_filename = f"scan_{patient_id}_{int(os.times().elapsed * 1000)}_{file.filename}"
    dest_path = UPLOAD_DIR / dest_filename

    file_size = 0
    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        file_size = dest_path.stat().st_size

    # Inspect video metadata with OpenCV if available
    total_frames = None
    fps = None
    duration = None
    width = None
    height = None

    try:
        import cv2
        if file_type.lower() == "video":
            cap = cv2.VideoCapture(str(dest_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if fps and fps > 0 and total_frames:
                    duration = total_frames / fps
                cap.release()
        elif file_type.lower() == "image":
            img = cv2.imread(str(dest_path))
            if img is not None:
                height, width = img.shape[:2]
                total_frames = 1
    except Exception:
        # Graceful fallback if video parsing fails
        pass

    scan = UltrasoundScan(
        patient_id=patient_id,
        title=title or file.filename,
        filename=file.filename,
        file_path=str(dest_path),
        file_type=file_type,
        mime_type=file.content_type,
        file_size_bytes=file_size,
        total_frames=total_frames,
        fps=fps,
        duration_seconds=duration,
        resolution_width=width,
        resolution_height=height,
        pixel_spacing_mm=pixel_spacing_mm,
        status="uploaded"
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan

@router.get("/", response_model=List[UltrasoundScanResponse])
def list_scans(
    patient_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List scans with optional filter by patient ID and processing status."""
    query = db.query(UltrasoundScan)
    if patient_id is not None:
        query = query.filter(UltrasoundScan.patient_id == patient_id)
    if status_filter:
        query = query.filter(UltrasoundScan.status == status_filter)
    return query.offset(skip).limit(limit).all()

@router.get("/{scan_id}", response_model=UltrasoundScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    """Retrieve details of a single ultrasound scan."""
    scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")
    return scan

@router.put("/{scan_id}", response_model=UltrasoundScanResponse)
def update_scan(scan_id: int, payload: UltrasoundScanUpdate, db: Session = Depends(get_db)):
    """Update scan status or calibration settings."""
    scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")
    
    update_data = payload.model_dump(exclude_unset=True)
    for k, v in update_data.items():
        setattr(scan, k, v)
        
    db.commit()
    db.refresh(scan)
    return scan

@router.delete("/{scan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    """Delete an ultrasound scan record and remove stored file if present."""
    scan = db.query(UltrasoundScan).filter(UltrasoundScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found.")
    
    # Try deleting local file
    try:
        p = Path(scan.file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    db.delete(scan)
    db.commit()
    return None
