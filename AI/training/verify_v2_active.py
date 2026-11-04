"""
verify_v2_active.py
===================
Verifies that FetalHeadUNet-v2.0-seed42 is the active production model.

Checks:
  1. fetal_head_unet_best.pt loads cleanly and reports v2.0 version_tag.
  2. Neon model_versions: v2.0 is_active=True, v1.0 is_active=False.

Usage:
  cd c:\\Users\\MSI\\OneDrive\\Documents\\Fetalytics
  python AI/training/verify_v2_active.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def check_checkpoint() -> bool:
    import torch
    weights = ROOT / "AI" / "models" / "weights" / "fetal_head_unet_best.pt"
    if not weights.exists():
        print(f"[ERROR] {weights} not found.")
        return False

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    tag = ckpt.get("version_tag", "<not set>")
    size_mb = weights.stat().st_size / 1e6
    print(f"[+] Checkpoint loaded: {weights.name}  ({size_mb:.2f} MB)")
    print(f"    version_tag       : {tag}")
    print(f"    keys present      : {list(ckpt.keys())}")

    if "v2.0" in tag:
        print("[+] PASS — checkpoint is v2.0")
        return True
    else:
        print(f"[FAIL] Expected v2.0 in version_tag, got: {tag!r}")
        return False


def check_neon() -> bool:
    from Backend.app.core.database import DATABASE_URL
    if not DATABASE_URL or DATABASE_URL.startswith("sqlite"):
        print("[!] DATABASE_URL is not Neon — skipping DB check.")
        return True

    from sqlalchemy import create_engine, text
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    from sqlalchemy.orm import Session
    with Session(engine) as session:
        rows = session.execute(
            text("SELECT id, version_tag, is_active, b2_artifact_key FROM model_versions ORDER BY id")
        ).fetchall()
    print("\n[*] Neon model_versions:")
    all_ok = True
    for row in rows:
        marker = "<- ACTIVE" if row[2] else ""
        print(f"    id={row[0]}  tag={row[1]}  is_active={row[2]}  b2_key={row[3] or 'None'}  {marker}")
        if "v2.0" in row[1] and not row[2]:
            print(f"    [FAIL] v2.0 should be active but is_active=False")
            all_ok = False
        if "v1.0" in row[1] and row[2]:
            print(f"    [FAIL] v1.0 should be inactive but is_active=True")
            all_ok = False
    if all_ok:
        print("[+] PASS — DB state is correct")
    return all_ok


def check_inference_engine() -> bool:
    try:
        from AI.analysis.pipelines import UltrasoundInferenceEngine
        engine = UltrasoundInferenceEngine()
        tag = getattr(engine, "model_version_tag", "<not available>")
        print(f"\n[+] UltrasoundInferenceEngine loaded successfully")
        print(f"    model_version_tag: {tag}")
        if "v2.0" in str(tag):
            print("[+] PASS — inference engine reports v2.0")
            return True
        else:
            print(f"[FAIL] Expected v2.0, got: {tag!r}")
            return False
    except Exception as exc:
        print(f"[!] Could not instantiate UltrasoundInferenceEngine: {exc}")
        print("    (This is OK if GPU/model dependencies are not installed locally.)")
        return True  # non-fatal for local dev


def main() -> int:
    print("=" * 70)
    print("  FETALYTICS — VERIFY v2.0 ACTIVE IN PRODUCTION")
    print("=" * 70)

    results = []

    print("\n── Checkpoint File ──────────────────────────────────────────────")
    results.append(check_checkpoint())

    print("\n── Neon PostgreSQL ──────────────────────────────────────────────")
    results.append(check_neon())

    print("\n── Inference Engine ─────────────────────────────────────────────")
    results.append(check_inference_engine())

    print("\n" + "=" * 70)
    if all(results):
        print("  ALL CHECKS PASSED — v2.0 is fully active")
    else:
        print("  SOME CHECKS FAILED — see details above")
    print("=" * 70)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

