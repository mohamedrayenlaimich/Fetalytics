import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from AI.measurements.biometry import AnatomicalExtractionResult, extract_landmarks_and_biometry
from AI.preprocessing.pipeline import (
    ImageQualityAssessment,
    PreprocessingConfig,
    assess_ultrasound_image_quality,
    preprocess_ultrasound_image,
)
from AI.segmentation.unet import FetalHeadUNet, StandardPlaneClassifier
from AI.tracking.tracker import TemporalStructureTracker
from Backend.app.core.config import BASE_DIR
from Backend.app.models.platform import CLINICAL_DISCLAIMER_TEXT
from Backend.app.services.b2_storage import compute_file_sha256


@dataclass
class ImageAnalysisOutput:
    source_file: str
    source_sha256: str
    model_version_tag: str
    preprocessing_config: Dict[str, Any]
    pixel_spacing_mm: Optional[float]
    is_calibrated: bool
    standard_plane_probability: float
    is_standard_plane: bool
    quality_assessment: Dict[str, Any]
    head_detected: bool
    ps_detected: bool
    overall_confidence: float
    measurements: List[Dict[str, Any]]
    landmarks_xy: Dict[str, Any]
    ellipse_params: Optional[Dict[str, float]]
    head_bbox: Optional[Dict[str, int]]
    ps_bbox: Optional[Dict[str, int]]
    head_contour_xy: List[List[int]]
    ps_contour_xy: List[List[int]]
    quality_warnings: List[str]
    annotated_image_path: Optional[str]
    processing_time_ms: float
    disclaimer: str = CLINICAL_DISCLAIMER_TEXT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnalysisOutput:
    source_file: str
    source_sha256: str
    model_version_tag: str
    preprocessing_config: Dict[str, Any]
    pixel_spacing_mm: Optional[float]
    is_calibrated: bool
    video_metadata: Dict[str, Any]
    frames_evaluated: int
    standard_plane_frames_count: int
    keyframe_index: Optional[int]
    keyframe_timestamp_ms: Optional[float]
    overall_confidence: float
    quality_score: float
    aggregated_measurements: List[Dict[str, Any]]
    tracking_metrics: Dict[str, Any]
    frame_observations: List[Dict[str, Any]]
    quality_warnings: List[str]
    annotated_video_path: Optional[str]
    annotated_keyframe_path: Optional[str]
    report_json_path: Optional[str]
    processing_time_ms: float
    disclaimer: str = CLINICAL_DISCLAIMER_TEXT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def render_annotated_frame(
    frame_bgr: np.ndarray,
    extraction: AnatomicalExtractionResult,
    plane_prob: float,
    model_version_tag: str,
    frame_index: Optional[int] = None,
    timestamp_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Draw clinical decision-support overlay on an ultrasound frame:
    - Semi-transparent contour shading for Fetal Head (cyan/green) and Pubic Symphysis (amber/orange)
    - Fitted BPD/OFD ellipse and axes
    - PS axis line, HSD distance line, and AOP progression angle line
    - Telemetry HUD with measurements and units
    """
    canvas = frame_bgr.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    overlay = canvas.copy()

    # Draw Fetal Head contour
    if extraction.head_contour_xy:
        fh_pts = np.array(extraction.head_contour_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(overlay, [fh_pts], -1, (0, 220, 110), thickness=-1)
        cv2.drawContours(canvas, [fh_pts], -1, (0, 255, 128), thickness=2, lineType=cv2.LINE_AA)

    # Draw Pubic Symphysis contour
    if extraction.ps_contour_xy:
        ps_pts = np.array(extraction.ps_contour_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(overlay, [ps_pts], -1, (255, 165, 0), thickness=-1)
        cv2.drawContours(canvas, [ps_pts], -1, (0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

    # Draw fitted ellipse
    if extraction.ellipse_params:
        ep = extraction.ellipse_params
        center = (int(round(ep["cx"])), int(round(ep["cy"])))
        axes = (int(round(ep["minor_axis_px"] / 2.0)), int(round(ep["major_axis_px"] / 2.0)))
        angle = float(ep["angle_deg"])
        if axes[0] > 0 and axes[1] > 0:
            cv2.ellipse(canvas, center, axes, angle, 0, 360, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw PS-to-FH landmarks (PS line, HSD segment, AOP tangency ray)
    lms = extraction.landmarks_xy
    if lms.get("ps_points_xy"):
        p1 = tuple(int(round(v)) for v in lms["ps_points_xy"][0])
        p2 = tuple(int(round(v)) for v in lms["ps_points_xy"][1])
        cv2.line(canvas, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, (0, 140, 255), -1)
        cv2.circle(canvas, p2, 4, (0, 140, 255), -1)

        if lms.get("hsd_point_xy"):
            hsd_pt = tuple(int(round(v)) for v in lms["hsd_point_xy"])
            cv2.line(canvas, p2, hsd_pt, (255, 100, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, hsd_pt, 4, (255, 100, 255), -1)

        if lms.get("aop_tangency_xy"):
            aop_pt = tuple(int(round(v)) for v in lms["aop_tangency_xy"])
            cv2.line(canvas, p2, aop_pt, (80, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, aop_pt, 4, (80, 220, 255), -1)

    # Draw HUD telemetry box
    hud_lines = [f"Fetalytics {model_version_tag} | SP Prob: {plane_prob:.2f}"]
    if frame_index is not None:
        ts_str = f" ({timestamp_ms:.0f} ms)" if timestamp_ms is not None else ""
        hud_lines[0] += f" | Frame {frame_index}{ts_str}"

    for m in extraction.measurements:
        hud_lines.append(f"{m.measurement_type}: {m.value:.1f} {m.unit} (conf {m.confidence_score:.2f})")

    y_pos = 22
    for line in hud_lines[:6]:
        cv2.putText(canvas, line, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 255, 240), 1, cv2.LINE_AA)
        y_pos += 20

    return canvas


def _iqr_aggregate(values: List[float]) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute robust median, IQR, and 95% confidence interval after IQR outlier filtering.
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, (0.0, 0.0)
    if len(arr) == 1:
        v = float(arr[0])
        return v, 0.0, (v, v)

    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    low_fence = q25 - 1.5 * iqr
    high_fence = q75 + 1.5 * iqr
    filtered = arr[(arr >= low_fence) & (arr <= high_fence)]
    if len(filtered) == 0:
        filtered = arr

    median_val = float(np.median(filtered))
    se = float(np.std(filtered) / np.sqrt(max(1, len(filtered))))
    ci_low = float(median_val - 1.96 * se)
    ci_high = float(median_val + 1.96 * se)
    return round(median_val, 2), round(iqr, 2), (round(ci_low, 2), round(ci_high, 2))


class UltrasoundInferenceEngine:
    """
    Modular medical computer-vision inference engine for single ultrasound images and videos.
    Uses identical PreprocessingConfig as model training.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        default_ckpt = BASE_DIR / "AI" / "models" / "weights" / "fetal_head_unet_best.pt"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_ckpt

        self.model = FetalHeadUNet(in_channels=1, num_classes=3, base_filters=24, use_context_stage=False).to(self.device)
        self.plane_classifier: Optional[StandardPlaneClassifier] = None
        self.preprocessing_config = PreprocessingConfig()
        self.model_version_tag = "FetalHeadUNet-untrained"

        if self.checkpoint_path.exists():
            self.load_checkpoint(self.checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        use_ctx = bool(ckpt.get("use_context_stage", "enc4.block.0.weight" in state_dict))
        base_filters = int(ckpt.get("base_filters", 24))
        self.model = FetalHeadUNet(
            in_channels=1,
            num_classes=3,
            base_filters=base_filters,
            use_context_stage=use_ctx,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        if "plane_classifier_state_dict" in ckpt and ckpt["plane_classifier_state_dict"]:
            self.plane_classifier = StandardPlaneClassifier(in_channels=1, base_filters=16).to(self.device)
            self.plane_classifier.load_state_dict(ckpt["plane_classifier_state_dict"])
            self.plane_classifier.eval()
        else:
            self.plane_classifier = None
        if "preprocessing_config" in ckpt and isinstance(ckpt["preprocessing_config"], dict):
            self.preprocessing_config = PreprocessingConfig.from_dict(ckpt["preprocessing_config"])
        self.model_version_tag = str(ckpt.get("version_tag", path.stem))
        self.model.eval()

    def predict_frame_arrays(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, ImageQualityAssessment]:
        """
        Run deterministic preprocessing and model forward pass on an image array.
        Returns:
          - class_mask: (orig_h, orig_w) uint8 in {0, 1, 2}
          - prob_map: (3, orig_h, orig_w) float32
          - plane_prob: float in [0, 1]
          - quality: ImageQualityAssessment
        """
        quality = assess_ultrasound_image_quality(
            frame_bgr,
            min_dimension=self.preprocessing_config.min_valid_dimension,
        )
        tensor_chw, (orig_h, orig_w) = preprocess_ultrasound_image(
            frame_bgr,
            config=self.preprocessing_config,
        )
        input_tensor = torch.from_numpy(tensor_chw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            seg_logits, plane_logit = self.model(input_tensor)
            probs_small = F.softmax(seg_logits, dim=1)
            probs_orig = F.interpolate(
                probs_small,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            prob_map = probs_orig.squeeze(0).cpu().numpy()
            class_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
            if self.plane_classifier is not None:
                sp_logit = self.plane_classifier(input_tensor)
                plane_prob = float(torch.sigmoid(sp_logit).item())
            else:
                plane_prob = float(torch.sigmoid(plane_logit).item())

        return class_mask, prob_map, round(plane_prob, 4), quality

    def analyze_image(
        self,
        image_path: Path,
        pixel_spacing_mm: Optional[float] = None,
        output_dir: Optional[Path] = None,
    ) -> ImageAnalysisOutput:
        t0 = time.perf_counter()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError(f"Corrupt or unreadable image file: {image_path.name}")

        sha256_hex = compute_file_sha256(image_path)
        class_mask, prob_map, plane_prob, quality = self.predict_frame_arrays(frame_bgr)
        extraction = extract_landmarks_and_biometry(
            class_mask=class_mask,
            prob_map=prob_map,
            pixel_spacing_mm=pixel_spacing_mm,
        )

        warnings = list(quality.warnings) + list(extraction.warnings)
        if plane_prob < 0.50:
            warnings.append(
                f"Low standard-plane probability ({plane_prob:.2f}); image may not represent an optimal biometric plane."
            )

        out_dir = Path(output_dir) if output_dir else (BASE_DIR / "results" / "annotated_videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = out_dir / f"annotated_{image_path.stem}_{sha256_hex[:8]}.png"
        annotated_bgr = render_annotated_frame(
            frame_bgr=frame_bgr,
            extraction=extraction,
            plane_prob=plane_prob,
            model_version_tag=self.model_version_tag,
        )
        cv2.imwrite(str(annotated_path), annotated_bgr)

        overall_conf = round(
            float(
                0.75 * extraction.head_confidence + 0.25 * plane_prob
                if (extraction.head_detected and extraction.qc_passed)
                else 0.25 * extraction.head_confidence
            ),
            4,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return ImageAnalysisOutput(
            source_file=image_path.name,
            source_sha256=sha256_hex,
            model_version_tag=self.model_version_tag,
            preprocessing_config=self.preprocessing_config.to_dict(),
            pixel_spacing_mm=pixel_spacing_mm if (pixel_spacing_mm and pixel_spacing_mm > 0) else None,
            is_calibrated=bool(pixel_spacing_mm and pixel_spacing_mm > 0),
            standard_plane_probability=plane_prob,
            is_standard_plane=(plane_prob >= 0.50 and extraction.head_detected),
            quality_assessment=asdict(quality),
            head_detected=extraction.head_detected,
            ps_detected=extraction.ps_detected,
            overall_confidence=overall_conf,
            measurements=[m.to_dict() for m in extraction.measurements],
            landmarks_xy=extraction.landmarks_xy,
            ellipse_params=extraction.ellipse_params,
            head_bbox=extraction.head_bbox,
            ps_bbox=extraction.ps_bbox,
            head_contour_xy=extraction.head_contour_xy,
            ps_contour_xy=extraction.ps_contour_xy,
            quality_warnings=warnings,
            annotated_image_path=str(annotated_path),
            processing_time_ms=elapsed_ms,
        )

    def analyze_video(
        self,
        video_path: Path,
        pixel_spacing_mm: Optional[float] = None,
        frame_step: int = 4,
        max_frames: int = 60,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> VideoAnalysisOutput:
        """
        Validate ultrasound video, extract frames at configurable intervals preserving timestamps,
        run segmentation + temporal tracking, aggregate standard-plane measurements using IQR
        filtering (avoiding independent-frame bias), and generate an annotated video + JSON report.
        """
        t0 = time.perf_counter()
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        sha256_hex = compute_file_sha256(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Unable to open or decode video file: {video_path.name}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        if fps <= 0:
            fps = 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip() or "unknown"
        duration_sec = round(total_frames / fps, 3) if total_frames > 0 else 0.0

        if total_frames <= 0 or width < 64 or height < 64:
            cap.release()
            raise ValueError(
                f"Invalid video dimensions or empty frame stream: {width}x{height}, frames={total_frames}"
            )

        out_root = Path(output_dir) if output_dir else (BASE_DIR / "results" / "annotated_videos")
        out_root.mkdir(parents=True, exist_ok=True)
        reports_dir = BASE_DIR / "results" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        annotated_video_path = out_root / f"annotated_{video_path.stem}_{sha256_hex[:8]}.mp4"
        annotated_keyframe_path = out_root / f"keyframe_{video_path.stem}_{sha256_hex[:8]}.png"
        report_json_path = reports_dir / f"video_report_{video_path.stem}_{sha256_hex[:8]}.json"

        out_fps = max(1.0, fps / max(1, frame_step))
        writer = cv2.VideoWriter(
            str(annotated_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (width, height),
        )

        tracker = TemporalStructureTracker(iou_threshold=0.25)
        frame_observations: List[Dict[str, Any]] = []
        quality_scores: List[float] = []
        measurements_by_type: Dict[str, List[Tuple[float, float, Optional[float], float]]] = {}
        # stores (value, value_px, value_mm, confidence) from valid standard-plane frames

        best_keyframe_score = -1.0
        best_keyframe_idx: Optional[int] = None
        best_keyframe_ts: Optional[float] = None
        best_keyframe_canvas: Optional[np.ndarray] = None

        frame_indices = list(range(0, total_frames, max(1, frame_step)))[:max_frames]
        for idx_pos, f_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            ts_ms = round((float(f_idx) / fps) * 1000.0, 2)
            class_mask, prob_map, plane_prob, quality = self.predict_frame_arrays(frame_bgr)
            extraction = extract_landmarks_and_biometry(
                class_mask=class_mask,
                prob_map=prob_map,
                pixel_spacing_mm=pixel_spacing_mm,
            )

            fh_track_id = tracker.update(f_idx, "fetal_head", extraction.head_bbox)
            ps_track_id = tracker.update(f_idx, "pubic_symphysis", extraction.ps_bbox)

            quality_scores.append(quality.quality_score)
            is_sp_frame = plane_prob >= 0.50 and extraction.head_detected

            annotated_canvas = render_annotated_frame(
                frame_bgr=frame_bgr,
                extraction=extraction,
                plane_prob=plane_prob,
                model_version_tag=self.model_version_tag,
                frame_index=f_idx,
                timestamp_ms=ts_ms,
            )
            writer.write(annotated_canvas)

            kf_score = (
                0.45 * extraction.head_confidence
                + 0.35 * plane_prob
                + 0.20 * (quality.quality_score / 100.0)
            )
            if extraction.head_detected and kf_score > best_keyframe_score:
                best_keyframe_score = kf_score
                best_keyframe_idx = f_idx
                best_keyframe_ts = ts_ms
                best_keyframe_canvas = annotated_canvas.copy()

            if is_sp_frame:
                for m in extraction.measurements:
                    measurements_by_type.setdefault(m.measurement_type, []).append(
                        (m.value, m.value_px, m.value_mm, m.confidence_score)
                    )

            frame_observations.append(
                {
                    "frame_index": f_idx,
                    "timestamp_ms": ts_ms,
                    "standard_plane_probability": plane_prob,
                    "is_standard_plane": is_sp_frame,
                    "quality_score": quality.quality_score,
                    "head_detected": extraction.head_detected,
                    "ps_detected": extraction.ps_detected,
                    "head_confidence": extraction.head_confidence,
                    "head_track_id": fh_track_id,
                    "ps_track_id": ps_track_id,
                    "head_bbox": extraction.head_bbox,
                    "ps_bbox": extraction.ps_bbox,
                    "measurements": [m.to_dict() for m in extraction.measurements],
                }
            )

            if progress_callback:
                progress_callback(round(((idx_pos + 1) / max(1, len(frame_indices))) * 100.0, 1))

        cap.release()
        writer.release()

        if best_keyframe_canvas is not None:
            cv2.imwrite(str(annotated_keyframe_path), best_keyframe_canvas)

        has_cal = bool(pixel_spacing_mm and pixel_spacing_mm > 0)
        warnings: List[str] = []
        if not has_cal:
            warnings.append(
                "Physical pixel_spacing_mm calibration not provided; linear biometric measurements "
                "(BPD, OFD, HC, HSD) are reported in pixels (px) only. Angle of Progression (AOP) in degrees is scale-invariant."
            )

        sp_count = sum(1 for fo in frame_observations if fo["is_standard_plane"])
        if sp_count == 0:
            warnings.append("No high-confidence standard plane frames detected across sampled video sequence.")

        # Aggregate measurements across standard-plane frames using IQR outlier rejection
        aggregated_measurements: List[Dict[str, Any]] = []
        for m_type, tuples_list in measurements_by_type.items():
            vals = [t[0] for t in tuples_list]
            vals_px = [t[1] for t in tuples_list]
            confs = [t[3] for t in tuples_list]
            med_val, iqr_val, ci95 = _iqr_aggregate(vals)
            med_px, iqr_px, _ = _iqr_aggregate(vals_px)
            unit = "deg" if m_type == "AOP" else ("mm" if has_cal else "px")
            aggregated_measurements.append(
                {
                    "measurement_type": m_type,
                    "value": med_val,
                    "unit": unit,
                    "value_px": med_px,
                    "value_mm": med_val if (has_cal and m_type != "AOP") else None,
                    "pixel_spacing_mm": pixel_spacing_mm if has_cal else None,
                    "is_calibrated": True if m_type == "AOP" else has_cal,
                    "iqr": iqr_val,
                    "ci95": list(ci95),
                    "uncertainty_px": round(max(1.0, iqr_px / 2.0), 2),
                    "uncertainty_mm": round(max(0.2, iqr_val / 2.0), 2)
                    if (has_cal and m_type != "AOP")
                    else None,
                    "confidence_score": round(float(np.mean(confs)), 4),
                    "num_correlated_frames_aggregated": len(tuples_list),
                    "method": "temporal_iqr_median_aggregation",
                }
            )

        tracking_metrics = tracker.get_summary_metrics(len(frame_observations))
        mean_qual = round(float(np.mean(quality_scores)), 2) if quality_scores else 0.0
        overall_conf = (
            round(float(np.mean([m["confidence_score"] for m in aggregated_measurements])), 4)
            if aggregated_measurements
            else 0.0
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        output = VideoAnalysisOutput(
            source_file=video_path.name,
            source_sha256=sha256_hex,
            model_version_tag=self.model_version_tag,
            preprocessing_config=self.preprocessing_config.to_dict(),
            pixel_spacing_mm=pixel_spacing_mm if has_cal else None,
            is_calibrated=has_cal,
            video_metadata={
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": duration_sec,
                "width": width,
                "height": height,
                "codec": codec_str,
                "frame_step": frame_step,
            },
            frames_evaluated=len(frame_observations),
            standard_plane_frames_count=sp_count,
            keyframe_index=best_keyframe_idx,
            keyframe_timestamp_ms=best_keyframe_ts,
            overall_confidence=overall_conf,
            quality_score=mean_qual,
            aggregated_measurements=aggregated_measurements,
            tracking_metrics=tracking_metrics,
            frame_observations=frame_observations,
            quality_warnings=warnings,
            annotated_video_path=str(annotated_video_path),
            annotated_keyframe_path=str(annotated_keyframe_path) if best_keyframe_canvas is not None else None,
            report_json_path=str(report_json_path),
            processing_time_ms=elapsed_ms,
        )

        report_json_path.write_text(json.dumps(output.to_dict(), indent=2), encoding="utf-8")
        return output
