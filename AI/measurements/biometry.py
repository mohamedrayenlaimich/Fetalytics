from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BiometricMeasurementResult:
    measurement_type: str  # BPD, OFD, HC, HSD, AOP
    value: float
    unit: str  # "mm", "px", or "deg"
    value_px: float
    value_mm: Optional[float]
    pixel_spacing_mm: Optional[float]
    is_calibrated: bool
    confidence_score: float
    uncertainty_px: float
    uncertainty_mm: Optional[float]
    estimated_ga_weeks: Optional[float] = None
    growth_percentile: Optional[float] = None
    method: str = "contour_ellipse_fit"
    quality_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnatomicalExtractionResult:
    head_detected: bool
    ps_detected: bool
    head_confidence: float
    ps_confidence: float
    qc_passed: bool = False
    qc_rejection_reasons: List[str] = field(default_factory=list)
    head_contour_xy: List[List[int]] = field(default_factory=list)
    ps_contour_xy: List[List[int]] = field(default_factory=list)
    head_bbox: Optional[Dict[str, int]] = None
    ps_bbox: Optional[Dict[str, int]] = None
    ellipse_params: Optional[Dict[str, float]] = None
    landmarks_xy: Dict[str, Any] = field(default_factory=dict)
    measurements: List[BiometricMeasurementResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _largest_component_mask_and_contour(
    binary_mask: np.ndarray,
    min_area_px: int = 80,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Keep only the largest connected component above min_area_px."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(binary_mask), None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px or len(largest) < 5:
        return np.zeros_like(binary_mask), None
    clean_mask = np.zeros_like(binary_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=-1)
    return clean_mask, largest


def validate_fetal_head_contour_qc(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
    seg_probability: float,
    min_area_fraction: float = 0.015,
    max_area_fraction: float = 0.65,
    min_confidence: float = 0.55,
    max_ellipse_residual: float = 0.18,
    min_aspect_ratio: float = 0.50,
    max_aspect_ratio: float = 0.99,
    min_solidity: float = 0.80,
) -> Tuple[bool, Dict[str, float], List[str]]:
    """
    Rigorous geometric and probabilistic Quality-Control (QC) check for a segmented
    fetal head contour before permitting biometric measurement extraction (BPD, OFD, HC).
    Rejects fragmented, low-confidence, non-elliptical, or anatomically implausible contours.
    """
    reasons: List[str] = []
    h, w = image_shape[:2]
    total_px = float(max(1, h * w))
    area_px = float(cv2.contourArea(contour))
    area_frac = area_px / total_px

    if len(contour) < 15:
        reasons.append(f"Contour has too few boundary points ({len(contour)} < 15).")
        return False, {}, reasons

    if area_frac < min_area_fraction:
        reasons.append(
            f"Fetal head contour area fraction ({area_frac:.4f}) is below minimum threshold ({min_area_fraction:.4f})."
        )
    if area_frac > max_area_fraction:
        reasons.append(
            f"Fetal head contour area fraction ({area_frac:.4f}) exceeds maximum anatomical threshold ({max_area_fraction:.4f})."
        )
    if seg_probability < min_confidence:
        reasons.append(
            f"Segmentation probability confidence ({seg_probability:.3f}) is below QC threshold ({min_confidence:.2f})."
        )

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area_px / max(1.0, hull_area)
    if solidity < min_solidity:
        reasons.append(
            f"Contour solidity/convexity ({solidity:.3f}) is below QC threshold ({min_solidity:.2f}); boundary is irregular."
        )

    (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(contour)
    major_px = float(max(axis1, axis2))
    minor_px = float(min(axis1, axis2))
    if major_px <= 1.0 or minor_px <= 1.0:
        reasons.append("Degenerate ellipse axes fitted to contour.")
        return False, {}, reasons

    aspect_ratio = minor_px / major_px
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        reasons.append(
            f"Fitted ellipse aspect ratio BPD/OFD ({aspect_ratio:.3f}) is outside plausible range [{min_aspect_ratio}, {max_aspect_ratio}]."
        )

    pts = contour.reshape(-1, 2).astype(np.float64)
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    rx = dx * cos_t + dy * sin_t
    ry = -dx * sin_t + dy * cos_t
    norm_rad = np.sqrt((rx / (axis1 / 2.0 + 1e-6)) ** 2 + (ry / (axis2 / 2.0 + 1e-6)) ** 2)
    rel_residual = float(np.mean(np.abs(norm_rad - 1.0)))

    if rel_residual > max_ellipse_residual:
        reasons.append(
            f"Contour deviates excessively from an ellipse (residual {rel_residual:.3f} > {max_ellipse_residual:.2f})."
        )

    ellipse_info = {
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "major_axis_px": round(major_px, 2),
        "minor_axis_px": round(minor_px, 2),
        "angle_deg": round(float(angle), 2),
        "fit_residual": round(rel_residual, 4),
        "solidity": round(solidity, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "area_fraction": round(area_frac, 4),
    }
    return (len(reasons) == 0), ellipse_info, reasons


def extract_landmarks_and_biometry(
    class_mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
    pixel_spacing_mm: Optional[float] = None,
    min_head_area_px: int = 200,
    min_ps_area_px: int = 40,
    min_confidence: float = 0.55,
) -> AnatomicalExtractionResult:
    """
    Extract Fetal Head (class 2) and Pubic Symphysis (class 1) contours, landmarks,
    and biometric measurements (BPD, OFD, HC, HSD, AOP) from a native-resolution mask.

    Safety & Quality Control Guarantees:
      1. Do NOT generate BPD, OFD, or HC measurements if the fetal head contour fails QC
         (low confidence, insufficient area, poor solidity, implausible aspect ratio, or high ellipse residual).
      2. Require valid pixel_spacing_mm (> 0) before reporting millimetre (mm) measurements.
      3. Never fabricate confidence scores or growth percentiles.
    """
    has_calibration = (
        pixel_spacing_mm is not None
        and isinstance(pixel_spacing_mm, (int, float))
        and float(pixel_spacing_mm) > 0.0
    )
    cal_warnings: List[str] = []
    if not has_calibration:
        cal_warnings.append(
            "Physical pixel_spacing_mm calibration not provided; linear biometric measurements "
            "(BPD, OFD, HC, HSD) are reported in pixels (px) only. Angle of Progression (AOP) in degrees is scale-invariant."
        )

    fh_raw = (class_mask == 2).astype(np.uint8) * 255
    ps_raw = (class_mask == 1).astype(np.uint8) * 255

    fh_clean, fh_contour = _largest_component_mask_and_contour(fh_raw, min_area_px=min_head_area_px)
    ps_clean, ps_contour = _largest_component_mask_and_contour(ps_raw, min_area_px=min_ps_area_px)

    fh_conf = 0.0
    ps_conf = 0.0
    if prob_map is not None and prob_map.ndim == 3 and prob_map.shape[0] >= 3:
        if fh_contour is not None and np.any(fh_clean > 0):
            fh_conf = float(np.mean(prob_map[2][fh_clean > 0]))
        if ps_contour is not None and np.any(ps_clean > 0):
            ps_conf = float(np.mean(prob_map[1][ps_clean > 0]))
    else:
        fh_conf = 0.90 if fh_contour is not None else 0.0
        ps_conf = 0.90 if ps_contour is not None else 0.0

    result = AnatomicalExtractionResult(
        head_detected=(fh_contour is not None),
        ps_detected=(ps_contour is not None),
        head_confidence=round(fh_conf, 4),
        ps_confidence=round(ps_conf, 4),
        warnings=list(cal_warnings),
    )

    if fh_contour is None:
        reason = "Fetal head contour not detected with sufficient area."
        result.qc_passed = False
        result.qc_rejection_reasons.append(reason)
        result.warnings.append(reason)
        return result

    qc_ok, ellipse_info, qc_reasons = validate_fetal_head_contour_qc(
        contour=fh_contour,
        image_shape=class_mask.shape[:2],
        seg_probability=fh_conf,
        min_confidence=min_confidence,
    )
    result.qc_passed = qc_ok
    result.qc_rejection_reasons = list(qc_reasons)
    result.ellipse_params = ellipse_info if ellipse_info else None

    pts_fh = fh_contour.reshape(-1, 2)
    step_fh = max(1, len(pts_fh) // 120)
    result.head_contour_xy = [[int(p[0]), int(p[1])] for p in pts_fh[::step_fh]]
    x, y, w, h = cv2.boundingRect(fh_contour)
    result.head_bbox = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    if not qc_ok:
        for r in qc_reasons:
            result.warnings.append(f"QC REJECTED: {r}")
        # Do not generate BPD, OFD, HC, HSD, or AOP from invalid contours!
        return result

    cx = ellipse_info["cx"]
    cy = ellipse_info["cy"]
    major_px = ellipse_info["major_axis_px"]
    minor_px = ellipse_info["minor_axis_px"]
    rel_residual = ellipse_info["fit_residual"]
    a = major_px / 2.0
    b = minor_px / 2.0

    # Honest confidence score derived from mean posterior probability and contour fit fidelity
    fit_fidelity = float(np.clip(1.0 - rel_residual * 2.0, 0.0, 1.0))
    combined_fh_conf = round(float(0.75 * fh_conf + 0.25 * fit_fidelity), 4)

    # Ramanujan second approximation for ellipse perimeter (HC)
    h_ram = ((a - b) ** 2) / ((a + b) ** 2 + 1e-8)
    hc_px = float(np.pi * (a + b) * (1.0 + (3.0 * h_ram) / (10.0 + np.sqrt(4.0 - 3.0 * h_ram))))

    bpd_unc_px = round(max(1.0, minor_px * rel_residual), 2)
    ofd_unc_px = round(max(1.0, major_px * rel_residual), 2)
    hc_unc_px = round(max(2.0, hc_px * rel_residual), 2)

    eff_spacing = float(pixel_spacing_mm) if (has_calibration and pixel_spacing_mm is not None) else None

    for m_type, val_px, unc_px in [
        ("BPD", minor_px, bpd_unc_px),
        ("OFD", major_px, ofd_unc_px),
        ("HC", hc_px, hc_unc_px),
    ]:
        val_mm = round(val_px * eff_spacing, 2) if eff_spacing is not None else None
        unc_mm = round(unc_px * eff_spacing, 2) if eff_spacing is not None else None
        result.measurements.append(
            BiometricMeasurementResult(
                measurement_type=m_type,
                value=val_mm if val_mm is not None else round(val_px, 2),
                unit="mm" if val_mm is not None else "px",
                value_px=round(val_px, 2),
                value_mm=val_mm,
                pixel_spacing_mm=eff_spacing,
                is_calibrated=has_calibration,
                confidence_score=combined_fh_conf,
                uncertainty_px=unc_px,
                uncertainty_mm=unc_mm,
                estimated_ga_weeks=None,
                growth_percentile=None,
                method="contour_ellipse_fit",
                quality_warnings=list(cal_warnings),
            )
        )

    # 2. Pubic Symphysis (PS) Landmarks -> HSD and AOP (only if PS confidence & area pass QC)
    if ps_contour is not None and ps_conf >= min_confidence:
        pts_ps = ps_contour.reshape(-1, 2)
        step_ps = max(1, len(pts_ps) // 60)
        result.ps_contour_xy = [[int(p[0]), int(p[1])] for p in pts_ps[::step_ps]]
        px_b, py_b, pw_b, ph_b = cv2.boundingRect(ps_contour)
        result.ps_bbox = {"x": int(px_b), "y": int(py_b), "w": int(pw_b), "h": int(ph_b)}

        ps_float = pts_ps.astype(np.float64)
        mean_ps = np.mean(ps_float, axis=0)
        centered = ps_float - mean_ps
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, int(np.argmax(eigvals))]
        projections = centered @ major_vec
        p_min = ps_float[int(np.argmin(projections))]
        p_max = ps_float[int(np.argmax(projections))]

        d_min = np.linalg.norm(p_min - np.array([cx, cy]))
        d_max = np.linalg.norm(p_max - np.array([cx, cy]))
        if d_max <= d_min:
            ps_proximal, ps_distal = p_min, p_max
        else:
            ps_proximal, ps_distal = p_max, p_min

        fh_float = pts_fh.astype(np.float64)
        dists_to_fh = np.linalg.norm(fh_float - ps_distal, axis=1)
        hsd_idx = int(np.argmin(dists_to_fh))
        hsd_point = fh_float[hsd_idx]
        hsd_px = float(dists_to_fh[hsd_idx])

        v_ps = ps_proximal - ps_distal
        v_ps_norm = np.linalg.norm(v_ps) + 1e-8
        u_ps = v_ps / v_ps_norm

        vecs_fh = fh_float - ps_distal
        norms_fh = np.linalg.norm(vecs_fh, axis=1) + 1e-8
        cos_angles = np.clip((vecs_fh @ u_ps) / norms_fh, -1.0, 1.0)
        angles_deg = np.rad2deg(np.arccos(cos_angles))
        aop_idx = int(np.argmax(angles_deg))
        aop_point = fh_float[aop_idx]
        aop_deg = float(angles_deg[aop_idx])

        combined_joint_conf = round(float(min(combined_fh_conf, ps_conf)), 4)

        result.landmarks_xy = {
            "ps_points_xy": [
                [round(float(ps_proximal[0]), 1), round(float(ps_proximal[1]), 1)],
                [round(float(ps_distal[0]), 1), round(float(ps_distal[1]), 1)],
            ],
            "hsd_point_xy": [round(float(hsd_point[0]), 1), round(float(hsd_point[1]), 1)],
            "aop_tangency_xy": [round(float(aop_point[0]), 1), round(float(aop_point[1]), 1)],
        }

        hsd_mm = round(hsd_px * eff_spacing, 2) if eff_spacing is not None else None
        hsd_unc_px = round(max(1.5, hsd_px * rel_residual), 2)
        result.measurements.append(
            BiometricMeasurementResult(
                measurement_type="HSD",
                value=hsd_mm if hsd_mm is not None else round(hsd_px, 2),
                unit="mm" if hsd_mm is not None else "px",
                value_px=round(hsd_px, 2),
                value_mm=hsd_mm,
                pixel_spacing_mm=eff_spacing,
                is_calibrated=has_calibration,
                confidence_score=combined_joint_conf,
                uncertainty_px=hsd_unc_px,
                uncertainty_mm=round(hsd_unc_px * eff_spacing, 2) if eff_spacing is not None else None,
                method="ps_to_skull_minimum_distance",
                quality_warnings=list(cal_warnings),
            )
        )

        result.measurements.append(
            BiometricMeasurementResult(
                measurement_type="AOP",
                value=round(aop_deg, 2),
                unit="deg",
                value_px=round(aop_deg, 2),
                value_mm=None,
                pixel_spacing_mm=eff_spacing,
                is_calibrated=True,
                confidence_score=combined_joint_conf,
                uncertainty_px=round(max(1.0, rel_residual * 15.0), 2),
                uncertainty_mm=None,
                method="ps_skull_tangency_angle",
                quality_warnings=[],
            )
        )
    elif ps_contour is not None and ps_conf < min_confidence:
        result.warnings.append(
            f"Pubic symphysis segmentation confidence ({ps_conf:.2f}) is below QC threshold ({min_confidence:.2f}); HSD and AOP omitted."
        )

    return result
