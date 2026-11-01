import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from Backend.app.core.database import Base
from Backend.app.core.migrations import run_schema_migrations
from Backend.app.models import (
    AnalysisJob,
    AnalysisReport,
    Annotation,
    Dataset,
    DatasetFile,
    Examination,
    ExtractedFrame,
    Measurement,
    ModelVersion,
    Patient,
    Subject,
    TrainingRun,
    UltrasoundVideo,
)
from Backend.app.services.b2_storage import (
    ArchiveValidationError,
    B2StorageService,
    StorageError,
    compute_file_md5,
    compute_file_sha256,
    safe_inspect_and_extract_zip,
)


class FakeS3Client:
    """In-memory S3-compatible mock client for deterministic unit tests."""

    def __init__(self) -> None:
        self.buckets: Dict[str, Dict[str, Dict[str, Any]]] = {"test-bucket": {}}
        self.upload_calls = 0

    def list_buckets(self) -> Dict[str, Any]:
        return {"Buckets": [{"Name": name} for name in self.buckets]}

    def list_objects_v2(self, Bucket: str, MaxKeys: int = 1000, Prefix: str = "", **kwargs: Any) -> Dict[str, Any]:
        bucket_objs = self.buckets.get(Bucket, {})
        contents = []
        for k, v in sorted(bucket_objs.items()):
            if k.startswith(Prefix):
                contents.append(
                    {
                        "Key": k,
                        "Size": len(v["bytes"]),
                        "ETag": f'"{v["etag"]}"',
                        "LastModified": v["last_modified"],
                    }
                )
        return {"Contents": contents[:MaxKeys], "IsTruncated": False}

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        from botocore.exceptions import ClientError

        bucket_objs = self.buckets.get(Bucket, {})
        if Key not in bucket_objs:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        obj = bucket_objs[Key]
        return {
            "ContentLength": len(obj["bytes"]),
            "ETag": f'"{obj["etag"]}"',
            "ContentType": obj["content_type"],
            "Metadata": obj["metadata"],
            "LastModified": obj["last_modified"],
        }

    def upload_file(
        self,
        Filename: str,
        Bucket: str,
        Key: str,
        ExtraArgs: Dict[str, Any] | None = None,
        Config: Any = None,
    ) -> None:
        self.upload_calls += 1
        p = Path(Filename)
        raw = p.read_bytes()
        etag = compute_file_md5(p)
        extra = ExtraArgs or {}
        self.buckets.setdefault(Bucket, {})[Key] = {
            "bytes": raw,
            "etag": etag,
            "content_type": extra.get("ContentType", "application/octet-stream"),
            "metadata": extra.get("Metadata", {}),
            "last_modified": datetime.utcnow(),
        }

    def download_file(self, Bucket: str, Key: str, Filename: str, Config: Any = None) -> None:
        obj = self.buckets[Bucket][Key]
        Path(Filename).write_bytes(obj["bytes"])

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        obj = self.buckets[Bucket][Key]
        return {"Body": io.BytesIO(obj["bytes"])}

    def generate_presigned_url(self, ClientMethod: str, Params: Dict[str, Any], ExpiresIn: int) -> str:
        return f"https://s3.us-east-005.backblazeb2.com/{Params['Bucket']}/{Params['Key']}?X-Amz-Expires={ExpiresIn}"

    def delete_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        self.buckets.get(Bucket, {}).pop(Key, None)
        return {}


def test_b2_storage_upload_download_verify_and_dedup(tmp_path: Path) -> None:
    fake_s3 = FakeS3Client()
    service = B2StorageService(
        key_id="test_id",
        application_key="test_secret",
        endpoint_url="s3.us-east-005.backblazeb2.com",
        bucket_name="test-bucket",
        s3_client=fake_s3,
    )

    sample_file = tmp_path / "ultrasound_sample.png"
    sample_file.write_bytes(b"\x89PNG\r\n\x1a\nfetal_ultrasound_payload_bytes")
    expected_sha256 = compute_file_sha256(sample_file)

    # 1. Upload and verify metadata
    meta = service.upload_file(sample_file, key="scans/ultrasound_sample.png")
    assert meta.bucket == "test-bucket"
    assert meta.key == "scans/ultrasound_sample.png"
    assert meta.size_bytes == sample_file.stat().st_size
    assert meta.sha256 == expected_sha256
    assert fake_s3.upload_calls == 1

    # 2. Duplicate upload skipped when identical
    meta2 = service.upload_file(sample_file, key="scans/ultrasound_sample.png", skip_if_identical=True)
    assert meta2.sha256 == expected_sha256
    assert fake_s3.upload_calls == 1

    # 3. List objects
    listed = service.list_objects(prefix="scans/")
    assert len(listed) == 1
    assert listed[0].key == "scans/ultrasound_sample.png"

    # 4. Download & verify checksum
    dest = tmp_path / "downloaded" / "ultrasound_sample.png"
    service.download_file("scans/ultrasound_sample.png", dest, verify=True)
    assert dest.exists()
    assert compute_file_sha256(dest) == expected_sha256

    # 5. Stream object
    streamed = b"".join(service.stream_object("scans/ultrasound_sample.png", chunk_size=8))
    assert streamed == sample_file.read_bytes()

    # 6. Presigned URL generation caps expiry
    url = service.generate_presigned_download_url("scans/ultrasound_sample.png", expires_in_seconds=99999)
    assert "X-Amz-Expires=3600" in url

    # 7. Checksum mismatch detection
    with pytest.raises(StorageError, match="SHA-256 mismatch"):
        service.verify_object("scans/ultrasound_sample.png", expected_sha256="0" * 64)

    # 8. Delete object
    assert service.delete_object("scans/ultrasound_sample.png") is True
    assert len(service.list_objects(prefix="scans/")) == 0


def test_safe_zip_inspection_and_extraction(tmp_path: Path) -> None:
    # 1. Valid archive with allowed medical/annotation files and one unsupported file
    valid_zip = tmp_path / "valid_dataset.zip"
    with zipfile.ZipFile(valid_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("val/seg/sample_1.png", b"\x89PNG\r\n\x1a\nmask_bytes")
        zf.writestr("val/seg/seg_info.csv", "filename,frame_count\ntest.avi,20\n")
        zf.writestr("val/scripts/autorun.exe", b"MZ_disallowed_binary")

    extract_dir = tmp_path / "extracted"
    report = safe_inspect_and_extract_zip(valid_zip, extract_dir)
    assert len(report.extracted_files) == 2
    assert len(report.skipped_files) == 1
    assert "val/scripts/autorun.exe" in report.skipped_files
    assert (extract_dir / "val" / "seg" / "sample_1.png").exists()

    # 2. Path traversal archive must be rejected before writing anything
    traversal_zip = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal_zip, "w") as zf:
        zf.writestr("../../outside.png", b"malicious")

    with pytest.raises(ArchiveValidationError, match="Unsafe archive member path"):
        safe_inspect_and_extract_zip(traversal_zip, tmp_path / "out_traversal")

    # 3. Decompressed size limit protection (zip bomb defense)
    large_zip = tmp_path / "large.zip"
    with zipfile.ZipFile(large_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.png", b"0" * 50_000)

    with pytest.raises(ArchiveValidationError, match="Cumulative decompressed size"):
        safe_inspect_and_extract_zip(large_zip, tmp_path / "out_large", max_decompressed_bytes=10_000)


def test_database_models_and_traceability_crud(tmp_path: Path) -> None:
    db_file = tmp_path / "test_fetalytics.db"
    test_engine = create_engine(f"sqlite:///{db_file}")
    run_schema_migrations(test_engine)
    TestingSession = sessionmaker(bind=test_engine)

    with TestingSession() as db:
        # 1. Create Dataset & Subject/Patient
        ds = Dataset(
            name="IUGC2024-Sample",
            version="3.0.0",
            b2_bucket="fetalytics-videos",
            b2_prefix="datasets/DatasetV3.zip",
            status="ready",
            supported_tasks=["standard_plane_classification", "fetal_head_segmentation", "fetal_biometry"],
        )
        subj = Subject(
            anonymized_id="SUBJ-IUGC-001",
            gestational_age_weeks=38.5,
            dataset_source="IUGC2024-Sample",
        )
        db.add_all([ds, subj])
        db.flush()

        # 2. Create Examination, DatasetFile, UltrasoundVideo, ExtractedFrame, Annotation
        exam = Examination(
            examination_code="EXAM-IUGC-001",
            patient_id=subj.id,
            dataset_id=ds.id,
            pixel_spacing_mm=0.25,
            standard_plane="intrapartum_transperineal",
        )
        db.add(exam)
        db.flush()

        dfile = DatasetFile(
            dataset_id=ds.id,
            subject_id=subj.id,
            examination_id=exam.id,
            b2_bucket="fetalytics-videos",
            b2_object_key="datasets/DatasetV3.zip#val/videos/test_1_20.avi",
            original_filename="test_1_20.avi",
            media_type="video",
            file_size_bytes=102400,
            sha256_checksum="a" * 64,
            split_subset="val",
            has_annotation=True,
            status="ingested",
        )
        db.add(dfile)
        db.flush()

        video = UltrasoundVideo(
            examination_id=exam.id,
            dataset_file_id=dfile.id,
            original_filename="test_1_20.avi",
            fps=24.0,
            total_frames=50,
            width=512,
            height=384,
            pixel_spacing_mm=0.25,
            processing_status="completed",
        )
        db.add(video)
        db.flush()

        frame = ExtractedFrame(
            video_id=video.id,
            examination_id=exam.id,
            dataset_file_id=dfile.id,
            frame_index=20,
            timestamp_ms=833.33,
            width=512,
            height=384,
            pixel_spacing_mm=0.25,
            is_standard_plane=True,
        )
        db.add(frame)
        db.flush()

        ann = Annotation(
            dataset_id=ds.id,
            dataset_file_id=dfile.id,
            frame_id=frame.id,
            annotation_type="segmentation_and_landmarks",
            task_name="fetal_head_segmentation",
            provenance="dataset_ground_truth",
            measurements_json={"aop_deg": 122.4, "hsd_px": 63.2},
        )
        db.add(ann)

        # 3. Create TrainingRun, ModelVersion, AnalysisJob, Measurement, AnalysisReport
        tr = TrainingRun(
            dataset_id=ds.id,
            dataset_version="3.0.0",
            task_name="fetal_head_segmentation",
            architecture="FetalHeadUNet",
            random_seed=42,
            status="completed",
            best_val_metric=0.91,
        )
        db.add(tr)
        db.flush()

        mv = ModelVersion(
            training_run_id=tr.id,
            model_name="Fetalytics-HeadSeg-UNet",
            version_tag="v1.0.0-seed42",
            task_name="fetal_head_segmentation",
            architecture="FetalHeadUNet",
            dataset_version="3.0.0",
            artifact_sha256="b" * 64,
        )
        db.add(mv)
        db.flush()

        job = AnalysisJob(
            examination_id=exam.id,
            video_id=video.id,
            model_version_id=mv.id,
            job_type="video_analysis",
            status="completed",
            progress_percent=100.0,
            source_filename="test_1_20.avi",
            source_sha256="a" * 64,
            pixel_spacing_mm=0.25,
        )
        db.add(job)
        db.flush()

        meas = Measurement(
            analysis_job_id=job.id,
            frame_id=frame.id,
            model_version_tag=mv.version_tag,
            measurement_type="HC",
            value=198.5,
            unit="mm",
            value_mm=198.5,
            value_px=794.0,
            pixel_spacing_mm=0.25,
            is_calibrated=True,
            confidence_score=0.92,
        )
        rep = AnalysisReport(
            analysis_job_id=job.id,
            examination_id=exam.id,
            model_version_tag=mv.version_tag,
            source_file_sha256="a" * 64,
            overall_confidence=0.92,
            measurements_summary={"HC_mm": 198.5},
        )
        db.add_all([meas, rep])
        db.commit()

        # Verify query and relationships
        fetched_job = db.query(AnalysisJob).filter_by(id=job.id).one()
        assert fetched_job.uuid is not None
        assert fetched_job.model_version.version_tag == "v1.0.0-seed42"
        assert len(fetched_job.measurements) == 1
        assert fetched_job.measurements[0].value_mm == 198.5
        assert len(fetched_job.reports) == 1
        assert "DECISION-SUPPORT ONLY" in fetched_job.reports[0].disclaimer
