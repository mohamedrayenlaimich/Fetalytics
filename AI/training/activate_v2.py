"""
activate_v2.py
==============
Activates FetalHeadUNet-v2.0-seed42 for local production inference.

Steps:
  1. Verifies the v2.0 checkpoint is present.
  2. Creates a timestamped backup of the current fetal_head_unet_best.pt.
  3. Copies the v2.0 checkpoint → fetal_head_unet_best.pt.
  4. SHA-256 cross-checks source and destination.

Run AFTER downloading FetalHeadUNet-v2.0-seed42.pt into:
  AI/models/weights/FetalHeadUNet-v2.0-seed42.pt

Usage:
  cd c:\\Users\\MSI\\OneDrive\\Documents\\Fetalytics
  python AI/training/activate_v2.py
"""

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "models" / "weights"
V2_CHECKPOINT = WEIGHTS_DIR / "FetalHeadUNet-v2.0-seed42.pt"
ACTIVE_CHECKPOINT = WEIGHTS_DIR / "fetal_head_unet_best.pt"
V1_ORIGINAL = WEIGHTS_DIR / "FetalHeadUNet-v1.0-seed42_ORIGINAL.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 70)
    print("  FETALYTICS -- ACTIVATE v2.0 FOR PRODUCTION INFERENCE")
    print("=" * 70)

    # ── 1. Verify v2.0 checkpoint exists (also check Downloads folder) ─────
    if not V2_CHECKPOINT.exists():
        downloads_cand = Path.home() / "Downloads" / "FetalHeadUNet-v2.0-seed42.pt"
        if downloads_cand.exists():
            print(f"\n[*] Found {downloads_cand.name} in Downloads -> copying to {WEIGHTS_DIR} ...")
            shutil.copy2(downloads_cand, V2_CHECKPOINT)

    if not V2_CHECKPOINT.exists():
        print(f"\n[ERROR] v2.0 checkpoint not found at:\n  {V2_CHECKPOINT}")
        print("\nDownload it from Colab first:")
        print("  from google.colab import files")
        print('  files.download("/content/Fetalytics/AI/models/weights/FetalHeadUNet-v2.0-seed42.pt")')
        return 1

    v2_size = V2_CHECKPOINT.stat().st_size
    print(f"\n[+] v2.0 checkpoint found: {V2_CHECKPOINT.name}  ({v2_size:,} bytes)")

    # ── 2. Verify original v1.0 backup is intact ──────────────────────────
    if not V1_ORIGINAL.exists():
        print(f"[ERROR] v1.0 original backup missing: {V1_ORIGINAL}")
        return 1
    print(f"[+] v1.0 original backup intact: {V1_ORIGINAL.name}")

    # ── 3. Create a timestamped backup of current active checkpoint ────────
    if ACTIVE_CHECKPOINT.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        backup_path = WEIGHTS_DIR / f"fetal_head_unet_best_backup_{ts}.pt"
        shutil.copy2(ACTIVE_CHECKPOINT, backup_path)
        print(f"[+] Backed up current active checkpoint -> {backup_path.name}")
    else:
        print("[!] No existing active checkpoint to back up (first activation).")

    # ── 4. Compute SHA-256 of v2.0 source ─────────────────────────────────
    print("\n[*] Computing SHA-256 of v2.0 source (may take a few seconds)...")
    v2_sha = sha256(V2_CHECKPOINT)
    print(f"    v2.0 source SHA-256: {v2_sha}")

    # ── 5. Copy v2.0 -> fetal_head_unet_best.pt ────────────────────────────
    print(f"\n[*] Copying v2.0 checkpoint -> {ACTIVE_CHECKPOINT.name} ...")
    shutil.copy2(V2_CHECKPOINT, ACTIVE_CHECKPOINT)

    # ── 6. Verify destination SHA-256 matches ─────────────────────────────
    dest_sha = sha256(ACTIVE_CHECKPOINT)
    if v2_sha != dest_sha:
        print(f"[ERROR] SHA-256 mismatch after copy!")
        print(f"  source : {v2_sha}")
        print(f"  dest   : {dest_sha}")
        return 1

    print(f"[+] SHA-256 verified: {dest_sha}")
    print(f"[+] Active checkpoint updated: {ACTIVE_CHECKPOINT}")
    print(f"    Size: {ACTIVE_CHECKPOINT.stat().st_size:,} bytes")

    # ── 7. Sync SHA-256 and size into Neon PostgreSQL ─────────────────────
    try:
        import os
        from dotenv import load_dotenv
        from sqlalchemy import create_engine, text
        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        db_url = os.environ.get("DATABASE_URL", "")
        if db_url and not db_url.startswith("sqlite"):
            engine = create_engine(db_url, pool_pre_ping=True)
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE model_versions
                           SET artifact_sha256 = :sha,
                               artifact_size_bytes = :sz,
                               local_artifact_path = :lp,
                               is_active = TRUE
                         WHERE version_tag = 'FetalHeadUNet-v2.0-seed42'
                    """),
                    {"sha": dest_sha, "sz": v2_size, "lp": str(ACTIVE_CHECKPOINT)},
                )
            print("[+] Synced v2.0 SHA-256 and file size to Neon PostgreSQL.")
    except Exception as e:
        print(f"[!] Notice syncing to Neon: {e}")

    print("\n" + "=" * 70)
    print("  SUCCESS -- fetal_head_unet_best.pt is now FetalHeadUNet-v2.0-seed42")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

