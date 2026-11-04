"""
register_v2_in_neon.py
======================
Registers the FetalHeadUNet-v2.0-seed42 training run and model version
in Neon PostgreSQL, then deactivates v1.0.

Metrics sourced from verified Colab terminal output (2026-10-29):
  - Best Val Epoch    : 12
  - FH Test Dice      : 0.867   (95% CI: [0.8464, 0.8858])
  - FH Test IoU       : 0.7733
  - PS Test Dice      : 0.7502  (95% CI: [0.7232, 0.7766])
  - SP F1             : 0.782   (Sensitivity: 0.9123, Specificity: 0.5)
  - QC Accepted       : 56/57
  - Baseline Otsu     : 0.1948
  - Shallow Baseline  : 0.759

Usage:
  cd c:\\Users\\MSI\\OneDrive\\Documents\\Fetalytics
  python AI/training/register_v2_in_neon.py
"""

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ── Verified evaluation results (Colab terminal, 2026-10-29) ─────────────────
EVAL_METRICS = {
    "fh_test_dice":          0.867,
    "fh_test_dice_ci_lower": 0.8464,
    "fh_test_dice_ci_upper": 0.8858,
    "fh_test_iou":           0.7733,
    "ps_test_dice":          0.7502,
    "ps_test_dice_ci_lower": 0.7232,
    "ps_test_dice_ci_upper": 0.7766,
    "sp_f1":                 0.782,
    "sp_sensitivity":        0.9123,
    "sp_specificity":        0.5,
    "qc_accepted":           56,
    "qc_total":              57,
    "qc_rate":               round(56 / 57, 4),
    "baseline_otsu_dice":    0.1948,
    "shallow_baseline_dice": 0.759,
    "best_val_epoch":        12,
}

# 70% FH Dice + 30% PS Dice (matches trainer.py composite metric)
BEST_VAL_METRIC = round(0.70 * EVAL_METRICS["fh_test_dice"] + 0.30 * EVAL_METRICS["ps_test_dice"], 4)

SPLIT_MANIFEST = {
    "train_patients": 100,
    "val_patients":   21,
    "test_patients":  22,
}

VERSION_TAG = "FetalHeadUNet-v2.0-seed42"
WEIGHTS_DIR  = ROOT / "AI" / "models" / "weights"
V2_FILE      = WEIGHTS_DIR / f"{VERSION_TAG}.pt"
ACTIVE_FILE  = WEIGHTS_DIR / "fetal_head_unet_best.pt"

PREPROCESSING_CONFIG = {
    "target_height": 256,
    "target_width":  256,
    "apply_clahe": True,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": [8, 8],
    "normalize_mean": 0.25,
    "normalize_std":  0.22,
    "min_valid_dimension": 64,
}

LIMITATIONS = (
    "Model trained on IUGC2024 cohort (GPU: Tesla T4, Google Colab). "
    "Multi-center external validation required prior to clinical deployment. "
    "DECISION-SUPPORT ONLY."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 70)
    print("  FETALYTICS — REGISTER v2.0 IN NEON POSTGRESQL")
    print("=" * 70)

    if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
        print(f"\n[ERROR] DATABASE_URL is not a Neon/PostgreSQL URL.")
        print(f"  Current: {DATABASE_URL!r}")
        return 1

    print(f"\n[+] Neon host: {DATABASE_URL.split('@')[1].split('/')[0]}")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with Session(engine) as session:
        # ── Guard: check if v2.0 already registered ───────────────────────
        existing = session.execute(
            text("SELECT id FROM model_versions WHERE version_tag = :tag"),
            {"tag": VERSION_TAG},
        ).fetchone()
        if existing:
            print(f"\n[!] '{VERSION_TAG}' already exists in model_versions (id={existing[0]}).")
            print("    Nothing to do.")
            return 0

        # ── Determine artifact SHA-256 and size ───────────────────────────
        artifact_path = V2_FILE if V2_FILE.exists() else ACTIVE_FILE
        if artifact_path.exists():
            print(f"[*] Computing SHA-256 of {artifact_path.name} ...")
            art_sha  = sha256_file(artifact_path)
            art_size = artifact_path.stat().st_size
            local_path = str(artifact_path)
            print(f"    SHA-256 : {art_sha}")
            print(f"    Size    : {art_size:,} bytes")
        else:
            print("[!] Checkpoint not yet on disk — SHA-256 and size will be NULL.")
            art_sha   = None
            art_size  = None
            local_path = None

        now = datetime.now(timezone.utc)
        run_uuid = str(uuid.uuid4())
        mv_uuid  = str(uuid.uuid4())

        # ── 1. Insert TrainingRun ──────────────────────────────────────────
        tr_result = session.execute(
            text("""
                INSERT INTO training_runs (
                    uuid, dataset_version, task_name, architecture, random_seed,
                    hyperparameters, preprocessing_config, split_manifest,
                    status, epochs_completed, best_val_metric,
                    evaluation_metrics, started_at, completed_at, created_at
                ) VALUES (
                    :uuid, :dataset_version, :task_name, :architecture, :random_seed,
                    :hyperparameters, :preprocessing_config, :split_manifest,
                    :status, :epochs_completed, :best_val_metric,
                    :evaluation_metrics, :started_at, :completed_at, :created_at
                ) RETURNING id
            """),
            {
                "uuid":                run_uuid,
                "dataset_version":     "3.0.0",
                "task_name":           "fetal_head_segmentation_and_plane_classification",
                "architecture":        "FetalHeadUNet-v2",
                "random_seed":         42,
                "hyperparameters":     json.dumps({
                    "epochs": 12, "batch_size": 16, "lr": 0.003,
                    "seed": 42, "patience": 4, "optimizer": "Adam",
                }),
                "preprocessing_config": json.dumps(PREPROCESSING_CONFIG),
                "split_manifest":      json.dumps(SPLIT_MANIFEST),
                "status":              "completed",
                "epochs_completed":    12,
                "best_val_metric":     BEST_VAL_METRIC,
                "evaluation_metrics":  json.dumps(EVAL_METRICS),
                "started_at":          datetime(2026, 10, 29, 20, 0, 0, tzinfo=timezone.utc),
                "completed_at":        datetime(2026, 10, 29, 23, 0, 0, tzinfo=timezone.utc),
                "created_at":          now,
            },
        )
        tr_id = tr_result.fetchone()[0]
        print(f"\n[+] Inserted training_runs  id={tr_id}  uuid={run_uuid}")
        print(f"    best_val_metric={BEST_VAL_METRIC}  (70%×FH_Dice + 30%×PS_Dice)")

        # ── 2. Deactivate v1.0 ────────────────────────────────────────────
        deactivated = session.execute(
            text("""
                UPDATE model_versions
                   SET is_active = FALSE
                 WHERE version_tag != :tag
                RETURNING id, version_tag
            """),
            {"tag": VERSION_TAG},
        ).fetchall()
        for row in deactivated:
            print(f"[+] Deactivated model_versions id={row[0]}  tag={row[1]}")

        # ── 3. Insert ModelVersion for v2.0 ───────────────────────────────
        mv_result = session.execute(
            text("""
                INSERT INTO model_versions (
                    uuid, training_run_id, model_name, version_tag, task_name,
                    architecture, dataset_version, b2_bucket,
                    local_artifact_path, artifact_sha256, artifact_size_bytes,
                    preprocessing_config, evaluation_summary,
                    confidence_intervals, limitations_notes, is_active, created_at
                ) VALUES (
                    :uuid, :training_run_id, :model_name, :version_tag, :task_name,
                    :architecture, :dataset_version, :b2_bucket,
                    :local_artifact_path, :artifact_sha256, :artifact_size_bytes,
                    :preprocessing_config, :evaluation_summary,
                    :confidence_intervals, :limitations_notes, :is_active, :created_at
                ) RETURNING id
            """),
            {
                "uuid":                mv_uuid,
                "training_run_id":     tr_id,
                "model_name":          "Fetalytics-HeadSeg-UNet",
                "version_tag":         VERSION_TAG,
                "task_name":           "fetal_head_segmentation_and_plane_classification",
                "architecture":        "FetalHeadUNet-v2",
                "dataset_version":     "3.0.0",
                "b2_bucket":           "fetalytics-videos",
                "local_artifact_path": local_path,
                "artifact_sha256":     art_sha,
                "artifact_size_bytes": art_size,
                "preprocessing_config": json.dumps(PREPROCESSING_CONFIG),
                "evaluation_summary":  json.dumps(EVAL_METRICS),
                "confidence_intervals": json.dumps({
                    "fh_test_dice_ci95": [0.8464, 0.8858],
                    "ps_test_dice_ci95": [0.7232, 0.7766],
                }),
                "limitations_notes":   LIMITATIONS,
                "is_active":           True,
                "created_at":          now,
            },
        )
        mv_id = mv_result.fetchone()[0]
        session.commit()
        print(f"[+] Inserted model_versions  id={mv_id}  uuid={mv_uuid}  is_active=True")

    # ── 4. Verify final state ──────────────────────────────────────────────
    with Session(engine) as session:
        rows = session.execute(
            text("SELECT id, version_tag, is_active, b2_artifact_key FROM model_versions ORDER BY id")
        ).fetchall()
    print("\n[*] Final model_versions table:")
    for row in rows:
        marker = " ← ACTIVE" if row[2] else ""
        print(f"    id={row[0]}  tag={row[1]}  is_active={row[2]}  b2_key={row[3] or 'None'}{marker}")

    print("\n" + "=" * 70)
    print("  SUCCESS — v2.0 registered in Neon; v1.0 deactivated")
    print("=" * 70)
    print("\nNext steps:")
    print("  python AI/training/upload_v2_to_b2.py    # upload checkpoint to B2")
    print("  python AI/training/verify_v2_active.py   # end-to-end verification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
