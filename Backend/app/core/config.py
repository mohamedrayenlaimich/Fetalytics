import os
from pathlib import Path
from dotenv import load_dotenv

# Locate project root (two levels up from Backend/app/core)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

# Fallback to local Backend/.env if root .env not found
if not ENV_PATH.exists():
    ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    PROJECT_NAME: str = "Fetalytics API"
    VERSION: str = "0.1.0"
    
    # Neon PostgreSQL database URL loaded securely from .env
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

settings = Settings()
