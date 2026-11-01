import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AI.training.trainer import run_reproducible_training_and_evaluation
from Backend.app.core.config import settings
from Backend.app.core.database import SessionLocal, engine
from Backend.app.core.migrations import run_schema_migrations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetalytics Reproducible Model Training & Held-out Evaluation CLI"
    )
    parser.add_argument(
        "--extracted-dir",
        default=str(settings.TEMP_STORAGE_DIR / "DatasetV3" / "extracted"),
        help="Path to extracted IUGC2024 dataset directory",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience on validation Dice")
    parser.add_argument("--no-b2-upload", action="store_true", help="Skip uploading model artifact to B2")
    parser.add_argument(
        "--activate-for-production",
        action="store_true",
        help="Activate checkpoint for production inference (default: False until evaluation is reviewed)",
    )

    args = parser.parse_args(argv)
    run_schema_migrations(engine)

    db = SessionLocal()
    try:
        report = run_reproducible_training_and_evaluation(
            extracted_root=Path(args.extracted_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            upload_to_b2=(not args.no_b2_upload),
            activate_for_production=bool(args.activate_for_production),
            early_stopping_patience=args.patience,
            db=db,
        )
        print("=" * 65)
        print(f"[OK] Training & Evaluation Complete: {report['version_tag']}")
        print("=" * 65)
        seg = report["evaluation_metrics"]["segmentation"]
        cls_m = report["evaluation_metrics"]["standard_plane_classification"]
        print(f"  Fetal Head Dice: {seg['fetal_head_dice_mean']} (95% CI: {seg['fetal_head_dice_ci95']})")
        print(f"  Baseline Otsu Dice: {seg['baseline_otsu_fetal_head_dice_mean']}")
        print(f"  Pubic Symphysis Dice: {seg['pubic_symphysis_dice_mean']}")
        print(f"  Standard Plane F1: {cls_m['f1_score']} (Sens: {cls_m['sensitivity']}, Spec: {cls_m['specificity']})")
        print(f"  B2 Model Key: {report['b2_artifact_key']}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
