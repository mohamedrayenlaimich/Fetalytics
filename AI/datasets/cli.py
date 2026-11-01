import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AI.datasets.ingestion import DatasetIngestionPipeline
from Backend.app.core.config import settings
from Backend.app.core.database import SessionLocal, engine
from Backend.app.core.migrations import run_schema_migrations
from Backend.app.services.b2_storage import B2StorageService, safe_inspect_and_extract_zip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetalytics Dataset Ingestion CLI (Backblaze B2 -> Controlled Storage -> Neon PostgreSQL)"
    )
    parser.add_argument("--bucket", default=settings.B2_BUCKET_NAME or "fetalytics-videos", help="B2 bucket name")
    parser.add_argument("--b2-key", default="datasets/DatasetV3.zip", help="B2 dataset archive key")
    parser.add_argument("--subset", default="val", choices=["val", "test", "train"], help="Inner dataset archive subset")
    parser.add_argument("--extracted-dir", default=None, help="Optional pre-extracted local directory path")
    parser.add_argument("--dataset-name", default="IUGC2024-Fetal-Ultrasound-V3", help="Dataset registry name")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos to ingest")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and validate without writing to PostgreSQL")
    parser.add_argument(
        "--report-out",
        default=str(PROJECT_ROOT / "results" / "reports" / "dataset_inventory_report.json"),
        help="Path to save JSON inventory report",
    )

    args = parser.parse_args(argv)
    run_schema_migrations(engine)

    storage = B2StorageService(bucket_name=args.bucket)
    db = None if args.dry_run else SessionLocal()

    try:
        pipeline = DatasetIngestionPipeline(db=db, storage=storage)
        if args.extracted_dir:
            extract_dir = Path(args.extracted_dir)
        else:
            local_zip = pipeline.ensure_sample_archive_from_b2(
                bucket=args.bucket,
                outer_key=args.b2_key,
                inner_subset=args.subset,
            )
            extract_dir = settings.TEMP_STORAGE_DIR / "DatasetV3" / "extracted"
            if not (extract_dir / args.subset).exists():
                safe_inspect_and_extract_zip(local_zip, extract_dir, dry_run=False)

        report = pipeline.ingest_extracted_directory(
            extracted_root=extract_dir,
            dataset_name=args.dataset_name,
            b2_bucket=args.bucket,
            b2_source_key=args.b2_key,
            dry_run=args.dry_run,
            max_videos=args.max_videos,
        )

        report_dict = asdict(report)
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

        print(f"[OK] Dataset Ingestion Completed (dry_run={args.dry_run})")
        print(f"  Dataset: {report.dataset_name} ({report.dataset_version})")
        print(f"  Total files discovered: {report.total_files_discovered}")
        print(f"  Valid videos: {report.videos_valid}, Valid masks/images: {report.images_or_masks_valid}")
        print(f"  Extracted frames: {report.extracted_frames_count}, Annotations linked: {report.annotations_parsed}")
        print(f"  Subject split distribution: {report.subject_counts_by_split}")
        print(f"  Subject leakage detected: {report.subject_leakage_detected}")
        print(f"  Report saved to: {out_path}")
        return 0
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
