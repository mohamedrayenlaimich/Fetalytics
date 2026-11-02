from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from .config import settings

# Ensure compatibility with SQLAlchemy dialects & fallback when offline/in cloud
raw_url = (settings.DATABASE_URL or "").strip()
if not raw_url:
    # Fallback to local SQLite when DATABASE_URL is omitted (e.g. standalone cloud GPU training)
    db_url = "sqlite:///fetalytics_local.db"
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
    )
else:
    db_url = raw_url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
        # Default to psycopg2 driver for stability
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    # Neon serverless PostgreSQL connection engine
    engine = create_engine(
        db_url,
        pool_pre_ping=True,      # Automatically reconnects if connection was dropped/suspended
        pool_recycle=300,        # Recycle connections every 5 minutes
        pool_size=5,             # Pool size suitable for serverless pooler
        max_overflow=10,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session per request
    and closes it when the request is done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
