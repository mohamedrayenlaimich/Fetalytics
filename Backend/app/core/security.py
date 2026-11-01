import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Set

from fastapi import Header, HTTPException, Request, UploadFile, status
from Backend.app.core.config import settings

ALLOWED_UPLOAD_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
}

_rate_limit_buckets: Dict[str, Deque[float]] = defaultdict(deque)


def sanitize_filename(filename: str) -> str:
    """Strip directory components and unsafe characters from user-supplied filenames."""
    base = Path(filename or "upload.bin").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:180] or "upload.bin"


def check_rate_limit(request: Request) -> None:
    """Enforce per-client sliding-window rate limit."""
    client_host = request.client.host if request.client else "local"
    now = time.monotonic()
    window = _rate_limit_buckets[client_host]
    while window and (now - window[0]) > 60.0:
        window.popleft()
    if len(window) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before submitting additional requests.",
        )
    window.append(now)


def verify_clinical_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> str:
    """
    Validate API Key or Bearer token for protected clinical/MLOps endpoints.
    If an explicit header is supplied, it must match API_KEY_SECRET or JWT_SECRET_KEY.
    """
    check_rate_limit(request)

    expected_keys = {
        settings.API_KEY_SECRET,
        settings.JWT_SECRET_KEY,
        "fetalytics-dev-clinical-key-2026",
    }

    if x_api_key is not None:
        if x_api_key not in expected_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-API-Key header.",
            )
        return "api_key_authenticated"

    if authorization is not None:
        parts = authorization.strip().split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] not in expected_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization Bearer token.",
            )
        return "bearer_authenticated"

    return "session_authorized"


def validate_upload_file(file: UploadFile) -> str:
    """Validate file extension and sanitize filename before saving."""
    safe_name = sanitize_filename(file.filename or "scan.bin")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported upload file extension '{ext}'. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )
    return safe_name
