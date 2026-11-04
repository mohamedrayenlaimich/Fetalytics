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
import zipfile
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


def extract_all_nested_zips(target_dir: Path, max_depth: int = 3) -> None:
    """Recursively unpack any nested zip files (e.g. val.zip, test.zip inside DatasetV3.zip)."""
    for depth in range(max_depth):
        inner_zips = [z for z in target_dir.rglob("*.zip") if z.is_file() and z.stat().st_size > 1000]
        if not inner_zips:
            break
        print(f"  [*] Found {len(inner_zips)} archive(s) to unpack (depth {depth+1}): {[z.name for z in inner_zips]}", flush=True)
        for zf in inner_zips:
            print(f"  [*] Unpacking archive: {zf.name} ({zf.stat().st_size / 1e6:.1f} MB)...", flush=True)
            try:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(target_dir)
                zf.unlink()  # Remove zip to free disk space
                print(f"  [+] Unpacked {zf.name} successfully.", flush=True)
            except Exception as e:
                print(f"  [!] Notice unpacking {zf.name}: {e}")


def auto_discover_drive_dataset(target_extract_dir: Path) -> Optional[Path]:
    """Search for dataset in mounted Google Drive or download via gdown."""
    search_roots = [
        Path("/content/dataset_drive"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
    ]

    for root in search_roots:
        if not root.exists():
            continue
        if (root / "videos").exists() or (root / "val" / "videos").exists():
            return root

        try:
            for item in root.iterdir():
                if item.is_dir():
                    if (item / "videos").exists() or (item / "val" / "videos").exists():
                        return item
                    sub_zips = list(item.glob("*.zip"))
                    if sub_zips:
                        for sz in sub_zips:
                            print(f"  [*] Extracting found Drive zip: {sz.name}...", flush=True)
                            with zipfile.ZipFile(sz, "r") as z:
                                z.extractall(target_extract_dir)
                        extract_all_nested_zips(target_extract_dir)
                        return target_extract_dir
                elif item.is_file() and item.suffix.lower() == ".zip":
                    print(f"  [*] Extracting found Drive zip: {item.name}...", flush=True)
                    with zipfile.ZipFile(item, "r") as z:
                        z.extractall(target_extract_dir)
                    extract_all_nested_zips(target_extract_dir)
                    return target_extract_dir
        except Exception as e:
            print(f"  [!] Notice during Drive scan of {root}: {e}")

    # If Drive is empty or not mounted, try gdown on the user's Google Drive folder
    folder_id = "1am61JzoUFojmqvhQ1ztc2dllkmxtmlSD"
    try:
        import gdown
        out_gdown = Path("/content/dataset_drive")
        out_gdown.mkdir(parents=True, exist_ok=True)
        print(f"  [*] Attempting direct download of Google Drive folder ({folder_id})...", flush=True)
        gdown.download_folder(id=folder_id, output=str(out_gdown), quiet=False, use_cookies=False)
        extract_all_nested_zips(out_gdown)
        if list(out_gdown.rglob("*.avi")) or list(out_gdown.rglob("*.zip")):
            return out_gdown
    except Exception as ge:
        print(f"  [!] gdown automatic download note: {ge}")

    return None


def ensure_cloud_dataset(
    extracted_dir: Path,
    drive_path: Optional[str] = None,
    bucket_name: str = "fetalytics-videos",
) -> Path:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    val_dir = extracted_dir / "val"
    test_dir = extracted_dir / "test"

    # First unpack any existing nested zips that were already extracted
    extract_all_nested_zips(extracted_dir)

    # 1. Check if user passed an explicit --data-path
    if drive_path:
        cand = Path(drive_path)
        if cand.exists():
            print(f"  [+] Using specified dataset path: {cand}")
            if cand.is_file() and cand.suffix.lower() == ".zip":
                print(f"  [*] Extracting specified ZIP archive {cand.name}...", flush=True)
                safe_inspect_and_extract_zip(cand, extracted_dir)
                extract_all_nested_zips(extracted_dir)
            elif cand.is_dir():
                zip_files = list(cand.glob("*.zip"))
                if zip_files:
                    for zf in zip_files:
                        print(f"  [*] Extracting zip inside folder: {zf.name}...", flush=True)
                        safe_inspect_and_extract_zip(zf, extracted_dir)
                    extract_all_nested_zips(extracted_dir)
                if (cand / "videos").exists() or (cand / "val" / "videos").exists():
                    return cand

    # 2. Check if already extracted locally
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

    # 3. Auto-discover from mounted Google Drive or gdown
    print("  [*] Searching for dataset in Google Drive...", flush=True)
    discovered = auto_discover_drive_dataset(extracted_dir)
    extract_all_nested_zips(extracted_dir)

    if discovered and discovered != extracted_dir:
        extract_all_nested_zips(discovered)
        if (discovered / "val" / "videos").exists() or (discovered / "videos").exists():
            return discovered

    # 4. Fallback to Backblaze B2 only if credentials are set
    if settings.b2_is_configured:
        try:
            print("  [*] Connecting to Backblaze B2 to fetch dataset partitions...")
            pipeline = DatasetIngestionPipeline(db=None, storage=B2StorageService(bucket_name=bucket_name))

            for subset in ("val", "test"):
                sub_vids = list((extracted_dir / subset / "videos").glob("*.avi")) if (extracted_dir / subset / "videos").exists() else []
                if len(sub_vids) > 0:
                    continue
                local_zip = pipeline.ensure_sample_archive_from_b2(
                    bucket=bucket_name,
                    outer_key="datasets/DatasetV3.zip",
                    inner_subset=subset,
                )
                safe_inspect_and_extract_zip(local_zip, extracted_dir)
        except Exception as b2_err:
            print(f"  [!] Backblaze B2 notice: {b2_err}. Proceeding with local/Drive data.")

    extract_all_nested_zips(extracted_dir)

    val_count = len(list((extracted_dir / "val" / "videos").glob("*.avi")))
    test_count = len(list((extracted_dir / "test" / "videos").glob("*.avi")))
    flat_count = len(list((extracted_dir / "videos").glob("*.avi")))
    total_found = val_count + test_count + flat_count

    if total_found == 0:
        for sub in extracted_dir.iterdir():
            if sub.is_dir():
                sub_vids = len(list((sub / "val" / "videos").glob("*.avi"))) + len(list((sub / "test" / "videos").glob("*.avi"))) + len(list((sub / "videos").glob("*.avi")))
                if sub_vids > 0:
                    print(f"  [+] Found extracted dataset in subfolder: {sub} ({sub_vids} cases)")
                    return sub

    print(f"  [+] Total Ultrasound Cases Located: {total_found}")
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

    # Automatically back up trained checkpoint and diagnostics to Google Drive if mounted
    ckpt_path = Path(report.get("checkpoint_path", PROJECT_ROOT / "AI" / "models" / "weights" / f"{report['version_tag']}.pt"))
    drive_models_dir = Path("/content/drive/MyDrive/Fetalytics_Models")
    if Path("/content/drive/MyDrive").exists() and ckpt_path.exists():
        import shutil
        drive_models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_path, drive_models_dir / ckpt_path.name)
        report_json = PROJECT_ROOT / "results" / "reports" / "training_evaluation_report.json"
        if report_json.exists():
            shutil.copy2(report_json, drive_models_dir / report_json.name)
        print(f"  [+] Saved trained checkpoint to Google Drive: {drive_models_dir / ckpt_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

