from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from Backend.app.core.database import get_db, engine, Base
from Backend.app.core.config import settings
from Backend.app.api.v1.router import api_v1_router
import Backend.app.models  # Ensures all models are recognized by SQLAlchemy

# Automatically ensure tables are created
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Fetalytics AI-Assisted Fetal Ultrasound Video Analysis REST API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend applications (Vite, Next.js, React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routes
app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "api_v1": "/api/v1"
    }

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected (Neon PostgreSQL)",
            "version": settings.VERSION
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
