import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AI.analysis.pipelines import UltrasoundInferenceEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetalytics Image & Video Ultrasound Inference CLI"
    )
    parser.add_argument("--input", required=True, help="Path to input ultrasound image (.png, .jpg) or video (.avi, .mp4)")
    parser.add_argument("--pixel-spacing-mm", type=float, default=None, help="Physical calibration (mm per pixel)")
    parser.add_argument("--frame-step", type=int, default=4, help="Frame extraction interval for videos")
    parser.add_argument("--checkpoint", default=None, help="Optional path to FetalHeadUNet checkpoint")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for annotated files")

    args = parser.parse_args(argv)
    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] Input file not found: {inp}", file=sys.stderr)
        return 1

    engine = UltrasoundInferenceEngine(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None
    )

    if inp.suffix.lower() in {".mp4", ".avi", ".mov"}:
        res = engine.analyze_video(
            video_path=inp,
            pixel_spacing_mm=args.pixel_spacing_mm,
            frame_step=args.frame_step,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print(json.dumps(res.to_dict(), indent=2))
    else:
        res_img = engine.analyze_image(
            image_path=inp,
            pixel_spacing_mm=args.pixel_spacing_mm,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        print(json.dumps(res_img.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
