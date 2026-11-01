import hashlib
import logging
import mimetypes
import os
import stat
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, Iterable, List, Optional, Set

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

from Backend.app.core.config import settings

logger = logging.getLogger("fetalytics.storage")

# Permitted file extensions inside dataset archives
ALLOWED_ARCHIVE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".avi",
    ".mov",
    ".csv",
    ".json",
    ".txt",
    ".md",
    ".xml",
    ".yaml",
    ".yml",
}

DEFAULT_MAX_ARCHIVE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
DEFAULT_MAX_DECOMPRESSED_BYTES = 25 * 1024 * 1024 * 1024  # 25 GB
DEFAULT_MAX_COMPRESSION_RATIO = 100.0
DEFAULT_MAX_ARCHIVE_FILES = 100_000


class StorageError(Exception):
    """Safe storage error that never leaks credentials or secrets."""


class ArchiveValidationError(StorageError):
    """Raised when an archive fails safety checks (path traversal, zip bomb, disallowed type)."""


@dataclass
class B2ObjectMetadata:
    bucket: str
    key: str
    size_bytes: int
    etag: str
    content_type: Optional[str] = None
    sha256: Optional[str] = None
    last_modified: Optional[datetime] = None
    custom_metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArchiveValidationReport:
    archive_path: str
    archive_size_bytes: int
    total_decompressed_bytes: int
    total_members: int
    extracted_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a local file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 hex digest of a local file for single-part S3 ETag comparison."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_inspect_and_extract_zip(
    zip_path: Path,
    extract_dir: Path,
    allowed_extensions: Optional[Set[str]] = None,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    max_files: int = DEFAULT_MAX_ARCHIVE_FILES,
    dry_run: bool = False,
) -> ArchiveValidationReport:
    """
    Safely inspect and optionally extract a ZIP archive.
    Checks:
      1. Archive file existence, readability, and compressed size limit.
      2. Zip validity and total member count limit.
      3. Path traversal protection (rejects absolute paths, '..', drive letters, symlinks).
      4. Decompressed total size and per-entry compression ratio (zip-bomb protection).
      5. Allowed file type check before writing to disk.
    """
    zip_path = Path(zip_path).resolve()
    extract_dir = Path(extract_dir).resolve()
    permitted_exts = allowed_extensions if allowed_extensions is not None else ALLOWED_ARCHIVE_EXTENSIONS

    if not zip_path.exists() or not zip_path.is_file():
        raise ArchiveValidationError(f"Archive file does not exist: {zip_path.name}")

    archive_size = zip_path.stat().st_size
    if archive_size > max_archive_bytes:
        raise ArchiveValidationError(
            f"Archive size ({archive_size} bytes) exceeds maximum allowed ({max_archive_bytes} bytes)."
        )

    if not zipfile.is_zipfile(zip_path):
        raise ArchiveValidationError(f"File '{zip_path.name}' is not a valid ZIP archive.")

    report = ArchiveValidationReport(
        archive_path=str(zip_path),
        archive_size_bytes=archive_size,
        total_decompressed_bytes=0,
        total_members=0,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        report.total_members = len(infos)
        if len(infos) > max_files:
            raise ArchiveValidationError(
                f"Archive contains {len(infos)} entries, exceeding limit of {max_files}."
            )

        total_uncompressed = 0
        validated_members: List[zipfile.ZipInfo] = []

        for info in infos:
            raw_name = info.filename
            # Normalize separators for inspection
            norm_parts = Path(raw_name.replace("\\", "/")).parts
            if (
                raw_name.startswith("/")
                or raw_name.startswith("\\")
                or ".." in norm_parts
                or ":" in raw_name
            ):
                raise ArchiveValidationError(
                    f"Unsafe archive member path detected: '{raw_name}'"
                )

            # Check for Unix symlink bit in external_attr
            unix_mode = info.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ArchiveValidationError(
                    f"Symbolic link member rejected in archive: '{raw_name}'"
                )

            target_path = (extract_dir / raw_name).resolve()
            try:
                target_path.relative_to(extract_dir)
            except ValueError as exc:
                raise ArchiveValidationError(
                    f"Archive member escapes target directory: '{raw_name}'"
                ) from exc

            if info.is_dir():
                continue

            total_uncompressed += info.file_size
            if total_uncompressed > max_decompressed_bytes:
                raise ArchiveValidationError(
                    f"Cumulative decompressed size ({total_uncompressed} bytes) exceeds limit ({max_decompressed_bytes} bytes)."
                )

            if info.compress_size > 0 and info.file_size > 1024 * 1024:
                ratio = info.file_size / float(info.compress_size)
                if ratio > max_compression_ratio:
                    raise ArchiveValidationError(
                        f"Suspicious compression ratio ({ratio:.1f}x) on member '{raw_name}'."
                    )

            ext = Path(raw_name).suffix.lower()
            if ext not in permitted_exts:
                report.skipped_files.append(raw_name)
                report.warnings.append(f"Skipped unsupported file extension '{ext}': {raw_name}")
                continue

            validated_members.append(info)

        report.total_decompressed_bytes = total_uncompressed

        if not dry_run:
            extract_dir.mkdir(parents=True, exist_ok=True)
            for info in validated_members:
                target_path = (extract_dir / info.filename).resolve()
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                    written = 0
                    while True:
                        buf = src.read(1024 * 1024)
                        if not buf:
                            break
                        written += len(buf)
                        if written > info.file_size + 1024:
                            raise ArchiveValidationError(
                                f"Member '{info.filename}' exceeded declared uncompressed size during extraction."
                            )
                        dst.write(buf)
                report.extracted_files.append(str(target_path))
        else:
            report.extracted_files = [info.filename for info in validated_members]

    return report


class B2StorageService:
    """
    S3-compatible Backblaze B2 object storage service.
    Supports private bucket operations, multipart/resumable uploads, checksum verification,
    safe error handling, and short-lived presigned URLs.
    """

    def __init__(
        self,
        key_id: Optional[str] = None,
        application_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        s3_client: Optional[Any] = None,
    ):
        self.key_id = key_id if key_id is not None else settings.B2_KEY_ID
        self.application_key = application_key if application_key is not None else settings.B2_APPLICATION_KEY
        raw_endpoint = endpoint_url if endpoint_url is not None else settings.B2_ENDPOINT_URL
        if raw_endpoint and not raw_endpoint.startswith("http"):
            self.endpoint_url = f"https://{raw_endpoint}"
        else:
            self.endpoint_url = raw_endpoint
        self.bucket_name = bucket_name if bucket_name is not None else settings.B2_BUCKET_NAME
        self.region_name = region_name if region_name is not None else settings.B2_REGION_NAME
        self._client = s3_client
        self.transfer_config = TransferConfig(
            multipart_threshold=25 * 1024 * 1024,  # 25 MB
            multipart_chunksize=16 * 1024 * 1024,  # 16 MB parts
            max_concurrency=4,
            use_threads=True,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.key_id or not self.application_key:
                raise StorageError("Backblaze B2 credentials are not configured in environment.")
            boto_config = Config(
                signature_version="s3v4",
                connect_timeout=15,
                read_timeout=60,
                retries={"max_attempts": 4, "mode": "standard"},
            )
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.application_key,
                region_name=self.region_name,
                config=boto_config,
            )
        return self._client

    def resolve_bucket(self, bucket: Optional[str] = None) -> str:
        target = bucket or self.bucket_name
        if target:
            return target
        # Discover first accessible bucket if B2_BUCKET_NAME was left blank
        buckets = self.list_buckets()
        if not buckets:
            raise StorageError("No Backblaze B2 bucket configured or accessible with current key.")
        self.bucket_name = buckets[0]
        return self.bucket_name

    def list_buckets(self) -> List[str]:
        try:
            resp = self.client.list_buckets()
            return [b["Name"] for b in resp.get("Buckets", [])]
        except (ClientError, BotoCoreError, EndpointConnectionError) as exc:
            logger.error("Failed to list B2 buckets: %s", type(exc).__name__)
            raise StorageError("Unable to list Backblaze B2 buckets.") from exc

    def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[B2ObjectMetadata]:
        target_bucket = self.resolve_bucket(bucket)
        results: List[B2ObjectMetadata] = []
        continuation_token: Optional[str] = None

        try:
            while len(results) < max_keys:
                page_size = min(1000, max_keys - len(results))
                kwargs: Dict[str, Any] = {"Bucket": target_bucket, "MaxKeys": page_size}
                if prefix:
                    kwargs["Prefix"] = prefix
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token

                resp = self.client.list_objects_v2(**kwargs)
                for item in resp.get("Contents", []):
                    results.append(
                        B2ObjectMetadata(
                            bucket=target_bucket,
                            key=item["Key"],
                            size_bytes=int(item.get("Size", 0)),
                            etag=str(item.get("ETag", "")).strip('"'),
                            last_modified=item.get("LastModified"),
                        )
                    )
                if not resp.get("IsTruncated"):
                    break
                continuation_token = resp.get("NextContinuationToken")
            return results
        except (ClientError, BotoCoreError, EndpointConnectionError) as exc:
            logger.error("Failed to list objects in bucket '%s': %s", target_bucket, type(exc).__name__)
            raise StorageError(f"Failed to list objects in bucket '{target_bucket}'.") from exc

    def get_object_metadata(self, key: str, bucket: Optional[str] = None) -> B2ObjectMetadata:
        target_bucket = self.resolve_bucket(bucket)
        try:
            resp = self.client.head_object(Bucket=target_bucket, Key=key)
            custom_meta = resp.get("Metadata", {}) or {}
            return B2ObjectMetadata(
                bucket=target_bucket,
                key=key,
                size_bytes=int(resp.get("ContentLength", 0)),
                etag=str(resp.get("ETag", "")).strip('"'),
                content_type=resp.get("ContentType"),
                sha256=custom_meta.get("sha256"),
                last_modified=resp.get("LastModified"),
                custom_metadata=custom_meta,
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise StorageError(f"Object '{key}' not found in bucket '{target_bucket}'.") from exc
            raise StorageError(f"Failed to retrieve metadata for object '{key}'.") from exc
        except (BotoCoreError, EndpointConnectionError) as exc:
            raise StorageError(f"Network error retrieving metadata for '{key}'.") from exc

    def verify_object(
        self,
        key: str,
        expected_size: Optional[int] = None,
        expected_sha256: Optional[str] = None,
        expected_md5: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> B2ObjectMetadata:
        meta = self.get_object_metadata(key=key, bucket=bucket)
        if expected_size is not None and meta.size_bytes != expected_size:
            raise StorageError(
                f"Size mismatch for '{key}': expected {expected_size} bytes, got {meta.size_bytes} bytes."
            )
        if expected_sha256 and meta.sha256 and meta.sha256.lower() != expected_sha256.lower():
            raise StorageError(f"SHA-256 mismatch for '{key}'.")
        if expected_md5 and "-" not in meta.etag and meta.etag.lower() != expected_md5.lower():
            raise StorageError(f"MD5/ETag mismatch for '{key}'.")
        return meta

    def upload_file(
        self,
        local_path: Path,
        key: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        skip_if_identical: bool = True,
    ) -> B2ObjectMetadata:
        local_path = Path(local_path)
        if not local_path.exists() or not local_path.is_file():
            raise StorageError(f"Local file not found for upload: {local_path.name}")

        target_bucket = self.resolve_bucket(bucket)
        file_size = local_path.stat().st_size
        sha256_hex = compute_file_sha256(local_path)
        md5_hex = compute_file_md5(local_path) if file_size < 25 * 1024 * 1024 else None

        if skip_if_identical:
            try:
                existing = self.get_object_metadata(key=key, bucket=target_bucket)
                if existing.size_bytes == file_size and (
                    existing.sha256 == sha256_hex
                    or (md5_hex and existing.etag.lower() == md5_hex.lower())
                ):
                    return existing
            except StorageError:
                pass

        guessed_mime, _ = mimetypes.guess_type(str(local_path))
        final_content_type = content_type or guessed_mime or "application/octet-stream"
        merged_metadata = dict(metadata or {})
        merged_metadata["sha256"] = sha256_hex

        extra_args: Dict[str, Any] = {
            "ContentType": final_content_type,
            "Metadata": merged_metadata,
        }

        try:
            self.client.upload_file(
                Filename=str(local_path),
                Bucket=target_bucket,
                Key=key,
                ExtraArgs=extra_args,
                Config=self.transfer_config,
            )
            return self.verify_object(
                key=key,
                expected_size=file_size,
                expected_sha256=sha256_hex,
                expected_md5=md5_hex,
                bucket=target_bucket,
            )
        except (ClientError, BotoCoreError, EndpointConnectionError) as exc:
            logger.error("Upload failed for key '%s': %s", key, type(exc).__name__)
            raise StorageError(f"Failed to upload '{key}' to Backblaze B2.") from exc

    def download_file(
        self,
        key: str,
        destination_path: Path,
        bucket: Optional[str] = None,
        verify: bool = True,
    ) -> Path:
        target_bucket = self.resolve_bucket(bucket)
        destination_path = Path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        remote_meta = self.get_object_metadata(key=key, bucket=target_bucket)

        try:
            self.client.download_file(
                Bucket=target_bucket,
                Key=key,
                Filename=str(destination_path),
                Config=self.transfer_config,
            )
        except (ClientError, BotoCoreError, EndpointConnectionError) as exc:
            if destination_path.exists():
                destination_path.unlink(missing_ok=True)
            raise StorageError(f"Failed to download '{key}' from Backblaze B2.") from exc

        if verify:
            local_size = destination_path.stat().st_size
            if local_size != remote_meta.size_bytes:
                destination_path.unlink(missing_ok=True)
                raise StorageError(
                    f"Downloaded file size mismatch for '{key}': expected {remote_meta.size_bytes}, got {local_size}."
                )
            if remote_meta.sha256:
                local_sha = compute_file_sha256(destination_path)
                if local_sha.lower() != remote_meta.sha256.lower():
                    destination_path.unlink(missing_ok=True)
                    raise StorageError(f"Downloaded file SHA-256 verification failed for '{key}'.")
            elif "-" not in remote_meta.etag and remote_meta.etag:
                local_md5 = compute_file_md5(destination_path)
                if local_md5.lower() != remote_meta.etag.lower():
                    destination_path.unlink(missing_ok=True)
                    raise StorageError(f"Downloaded file ETag/MD5 verification failed for '{key}'.")

        return destination_path

    def stream_object(
        self,
        key: str,
        bucket: Optional[str] = None,
        chunk_size: int = 256 * 1024,
    ) -> Generator[bytes, None, None]:
        target_bucket = self.resolve_bucket(bucket)
        try:
            resp = self.client.get_object(Bucket=target_bucket, Key=key)
            body = resp["Body"]
            while True:
                data = body.read(chunk_size)
                if not data:
                    break
                yield data
        except (ClientError, BotoCoreError, EndpointConnectionError) as exc:
            raise StorageError(f"Failed to stream object '{key}'.") from exc

    def generate_presigned_download_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in_seconds: Optional[int] = None,
    ) -> str:
        target_bucket = self.resolve_bucket(bucket)
        expiry = expires_in_seconds or settings.B2_PRESIGNED_URL_EXPIRY_SECONDS
        # Cap presigned URL expiry to 1 hour max for security
        expiry = max(60, min(expiry, 3600))
        try:
            return self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": target_bucket, "Key": key},
                ExpiresIn=expiry,
            )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"Failed to generate authorized download URL for '{key}'.") from exc

    def delete_object(self, key: str, bucket: Optional[str] = None) -> bool:
        target_bucket = self.resolve_bucket(bucket)
        try:
            self.client.delete_object(Bucket=target_bucket, Key=key)
            return True
        except (ClientError, BotoCoreError, EndpointConnectionError) as exc:
            raise StorageError(f"Failed to delete object '{key}'.") from exc
