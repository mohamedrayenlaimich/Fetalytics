"""
upload_v2_to_b2.py
==================
Uploads the FetalHeadUNet-v2.0-seed42.pt checkpoint to Backblaze B2
and updates the b2_artifact_key in Neon PostgreSQL.

Usage:
  cd c:\\Users\\MSI\\OneDrive\\Documents\\Fetalytics
  python AI/training/upload_v2_to_b2.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from Backend.app.core.database import DATABASE_URL
from Backend.app.services.b2_storage import B2StorageService

VERSION_TAG   = "FetalHeadUNet-v2.0-seed42"
WEIGHTS_DIR   = ROOT / "AI" / "models" / "weights"
CHECKPOINT    = WEIGHTS_DIR / f"{VERSION_TAG}.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print("=" * 70)
    print("  FETALYTICS — UPLOAD v2.0 CHECKPOINT TO BACKBLAZE B2")
    print("=" * 70)

    if not CHECKPOINT.exists():
        print(f"\n[ERROR] Checkpoint not found: {CHECKPOINT}")
        return 1

    print(f"\n[+] Found: {CHECKPOINT.name}  ({CHECKPOINT.stat().st_size:,} bytes)")
    print("[*] Computing SHA-256 ...")
    chk_sha = sha256(CHECKPOINT)
    print(f"    SHA-256: {chk_sha}")

    b2_key = f"models/{VERSION_TAG}_{chk_sha[:10]}.pt"
    print(f"[*] Target B2 key: {b2_key}")

    storage = B2StorageService()
    print("[*] Uploading to Backblaze B2 (this may take several minutes) ...")
    storage.upload_file(CHECKPOINT, b2_key)
    print(f"[+] Upload complete → {b2_key}")

    # ── Update Neon record ─────────────────────────────────────────────────
    if DATABASE_URL and not DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with Session(engine) as session:
            result = session.execute(
                text("""
                    UPDATE model_versions
                       SET b2_artifact_key = :key
                     WHERE version_tag = :tag
                    RETURNING id
                """),
                {"key": b2_key, "tag": VERSION_TAG},
            ).fetchone()
            session.commit()
            if result:
                print(f"[+] Neon model_versions.id={result[0]} → b2_artifact_key set to '{b2_key}'")
            else:
                print(f"[!] No Neon record found for tag='{VERSION_TAG}'. Run register_v2_in_neon.py first.")
    else:
        print("[!] DATABASE_URL not set — skipping Neon update.")

    print("\n" + "=" * 70)
    print("  SUCCESS — v2.0 checkpoint uploaded to B2")
    print("=" * 70)
    print("\nNext step: run  python AI/training/verify_v2_active.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

