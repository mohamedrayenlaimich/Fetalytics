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
    VERSION: str = "1.0.0"

    # Neon PostgreSQL database URL loaded securely from .env
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Backblaze B2 S3-compatible configuration
    B2_KEY_ID: str = os.getenv("B2_KEY_ID", "")
    B2_KEY_NAME: str = os.getenv("B2_KEY_NAME", "fetalytics-backend")
    B2_APPLICATION_KEY: str = os.getenv("B2_APPLICATION_KEY", "")
    B2_ENDPOINT_URL: str = os.getenv("B2_ENDPOINT_URL", "s3.us-east-005.backblazeb2.com")
    B2_BUCKET_NAME: str = os.getenv("B2_BUCKET_NAME", "fetalytics-videos")
    B2_REGION_NAME: str = os.getenv("B2_REGION_NAME", "us-east-005")
    B2_PRESIGNED_URL_EXPIRY_SECONDS: int = int(os.getenv("B2_PRESIGNED_URL_EXPIRY_SECONDS", "900"))

    # Security, upload limits, and rate limits
    API_KEY_SECRET: str = os.getenv("API_KEY_SECRET", "fetalytics-dev-clinical-key-2026")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "fetalytics-jwt-secret-2026-research-support")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

    # Storage & Retention
    TEMP_STORAGE_DIR: Path = BASE_DIR / os.getenv("TEMP_STORAGE_DIR", "tmp_ingest")
    RETENTION_DAYS_UPLOADS: int = int(os.getenv("RETENTION_DAYS_UPLOADS", "30"))
    RETENTION_DAYS_ARTIFACTS: int = int(os.getenv("RETENTION_DAYS_ARTIFACTS", "90"))

    @property
    def b2_full_endpoint_url(self) -> str:
        url = self.B2_ENDPOINT_URL.strip()
        if url and not url.startswith("http://") and not url.startswith("https://"):
            return f"https://{url}"
        return url

    @property
    def b2_is_configured(self) -> bool:
        return bool(self.B2_KEY_ID and self.B2_APPLICATION_KEY and self.B2_ENDPOINT_URL)


settings = Settings()
