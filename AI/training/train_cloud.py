"""
Fetalytics — Automated Cloud GPU Training Script
Designed to run in free cloud environments (Google Colab, Kaggle, Lightning AI, AWS/GCP GPU instances).
Automatically downloads authorized dataset partitions from Backblaze B2,
trains FetalHeadUNet-v2.0 on GPU, evaluates on the held-out test split,
uploads the checkpoint to Backblaze B2, and registers in Neon PostgreSQL.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from AI.datasets.ingestion import DatasetIngestionPipeline
from AI.training.trainer import run_reproducible_training_and_evaluation
from Backend.app.core.config import settings
from Backend.app.core.database import SessionLocal, engine
from Backend.app.core.migrations import run_schema_migrations
from Backend.app.services.b2_storage import B2StorageService, safe_inspect_and_extract_zip


def verify_gpu_environment() -> str:
    print("=" * 70)
    print("  FETALYTICS CLOUD TRAINING ENVIRONMENT CHECK")
    print("=" * 70)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [+] CUDA Hardware GPU Detected: {gpu_name}")
        print(f"  [+] Total VRAM Available:       {vram_gb:.2f} GB")
        print(f"  [+] PyTorch Version:            {torch.__version__}")
        print(f"  [+] CUDA Version:               {torch.version.cuda}")
        return "cuda"
    else:
        print("  [!] WARNING: No CUDA GPU detected. Running on CPU (slower).")
        print("      In Google Colab, go to: Runtime > Change runtime type > T4 GPU")
        return "cpu"


def ensure_cloud_dataset(
    extracted_dir: Path,
    drive_path: Optional[str] = None,
    bucket_name: str = "fetalytics-videos",
) -> Path:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    val_dir = extracted_dir / "val"
    test_dir = extracted_dir / "test"

    # 1. Check if user provided or has Google Drive dataset
    candidate_paths = []
    if drive_path:
        candidate_paths.append(Path(drive_path))
    # Standard Colab Google Drive mount search locations
    candidate_paths.extend([
        Path("/content/dataset_drive"),
        Path("/content/drive/MyDrive/dataset"),
        Path("/content/drive/MyDrive/DatasetV3"),
        Path("/content/drive/MyDrive/Fetalytics"),
    ])

    for cand in candidate_paths:
        if not cand.exists():
            continue
        print(f"  [+] Found dataset source in Drive: {cand}")
        if cand.is_file() and cand.suffix.lower() == ".zip":
            print(f"  [*] Extracting Drive ZIP archive {cand.name}...", flush=True)
            safe_inspect_and_extract_zip(cand, extracted_dir)
        elif cand.is_dir():
            # Check for inner zip files in Drive folder
            zip_files = list(cand.glob("*.zip"))
            if zip_files:
                for zf in zip_files:
                    print(f"  [*] Extracting Drive zip: {zf.name}...", flush=True)
                    safe_inspect_and_extract_zip(zf, extracted_dir)
            # Or if it's already an extracted folder with videos/
            if (cand / "videos").exists() or (cand / "val" / "videos").exists():
                print(f"  [+] Using Drive directory structure directly: {cand}")
                return cand

    val_vids = list((val_dir / "videos").glob("*.avi")) if (val_dir / "videos").exists() else []
    test_vids = list((test_dir / "videos").glob("*.avi")) if (test_dir / "videos").exists() else []

    if len(val_vids) >= 40 and len(test_vids) >= 300:
        print(f"  [+] Dataset already present on disk: {len(val_vids) + len(test_vids)} paired ultrasound cases.")
        return extracted_dir

    if (extracted_dir / "videos").exists():
        flat_count = len(list((extracted_dir / "videos").glob("*.avi")))
        if flat_count > 0:
            print(f"  [+] Dataset ready in flat directory: {flat_count} ultrasound videos.")
            return extracted_dir

    # 2. Fallback to Backblaze B2 if Drive folder did not supply all partitions
    print("  [*] Connecting to Backblaze B2 to fetch dataset partitions...")
    pipeline = DatasetIngestionPipeline(db=None, storage=B2StorageService(bucket_name=bucket_name))

    for subset in ("val", "test"):
        sub_vids = list((extracted_dir / subset / "videos").glob("*.avi")) if (extracted_dir / subset / "videos").exists() else []
        if len(sub_vids) > 0:
            print(f"  [+] Subset '{subset}' already extracted ({len(sub_vids)} videos). Skipping download.")
            continue

        print(f"  [*] Downloading {subset}.zip from Backblaze B2 (bucket: {bucket_name})...", flush=True)
        local_zip = pipeline.ensure_sample_archive_from_b2(
            bucket=bucket_name,
            outer_key="datasets/DatasetV3.zip",
            inner_subset=subset,
        )
        print(f"  [+] Downloaded {subset}.zip ({local_zip.stat().st_size / 1e6:.1f} MB). Extracting...", flush=True)
        report = safe_inspect_and_extract_zip(local_zip, extracted_dir)
        print(f"  [+] Extracted {len(report.extracted_files)} files for subset '{subset}'.", flush=True)

    val_count = len(list((extracted_dir / "val" / "videos").glob("*.avi")))
    test_count = len(list((extracted_dir / "test" / "videos").glob("*.avi")))
    print(f"  [+] Dataset Ready: {val_count} val videos + {test_count} test videos = {val_count + test_count} total cases.")
    return extracted_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetalytics Cloud GPU Training Runner")
    parser.add_argument("--data-path", type=str, default=None, help="Path to dataset ZIP file or folder in Google Drive")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16 on GPU)")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate (default: 0.003)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (default: 4)")
    parser.add_argument("--no-b2-upload", action="store_true", help="Skip uploading artifact to Backblaze B2")
    parser.add_argument("--activate-for-production", action="store_true", help="Set active flag on model version")
    args = parser.parse_args()

    dev_type = verify_gpu_environment()
    extracted_root = settings.TEMP_STORAGE_DIR / "DatasetV3" / "extracted"

    final_data_root = ensure_cloud_dataset(
        extracted_root,
        drive_path=args.data_path,
        bucket_name=settings.B2_BUCKET_NAME,
    )

    print("\n" + "=" * 70)
    print(f"  STARTING REPRODUCIBLE TRAINING & HELD-OUT EVALUATION ON {dev_type.upper()}")
    print("=" * 70)

    try:
        run_schema_migrations(engine)
    except Exception as e:
        print(f"  [*] Notice on schema migrations: {e}")

    db = None
    try:
        db = SessionLocal()
        db.execute(torch.text("SELECT 1") if hasattr(torch, "text") else "SELECT 1")
    except Exception:
        db = None

    report = run_reproducible_training_and_evaluation(
        extracted_root=final_data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        upload_to_b2=(not args.no_b2_upload),
        activate_for_production=bool(args.activate_for_production),
        early_stopping_patience=args.patience,
        db=db,
    )

    if db is not None:
        db.close()

    seg = report["evaluation_metrics"]["segmentation"]
    cls_m = report["evaluation_metrics"]["standard_plane_classification"]
    bio = report["evaluation_metrics"]["biometry_and_landmarks"]

    print("\n" + "=" * 70)
    print("  TRAINING & HELD-OUT EVALUATION COMPLETED")
    print("=" * 70)
    print(f"  Model Version Tag:      {report['version_tag']}")
    print(f"  Best Val Epoch:         Epoch {report.get('best_epoch', 'N/A')}")
    print(f"  Held-out Test Dice:     {seg['fetal_head_dice_mean']} (95% CI: {seg['fetal_head_dice_ci95']})")
    print(f"  Held-out Test IoU:      {seg['fetal_head_iou_mean']}")
    print(f"  Baseline Otsu Dice:     {seg['baseline_otsu_fetal_head_dice_mean']}")
    print(f"  Shallow Baseline Dice:  {seg['baseline_supervised_shallow_fetal_head_dice_mean']}")
    print(f"  Pubic Symphysis Dice:   {seg['pubic_symphysis_dice_mean']} (95% CI: {seg['pubic_symphysis_dice_ci95']})")
    print(f"  Standard Plane F1:      {cls_m['f1_score']} (Sensitivity: {cls_m['sensitivity']}, Specificity: {cls_m['specificity']})")
    print(f"  QC Accepted Contours:   {bio['qc_accepted_contours']} / {bio['qc_accepted_contours'] + bio['qc_rejected_contours']}")
    print(f"  B2 Model Artifact Key:  {report.get('b2_artifact_key')}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

