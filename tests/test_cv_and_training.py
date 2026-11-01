from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from AI.analysis.pipelines import UltrasoundInferenceEngine
from AI.datasets.ingestion import (
    assign_subject_splits,
    extract_subject_id_from_filename,
    verify_no_subject_leakage,
)
from AI.measurements.biometry import extract_landmarks_and_biometry
from AI.preprocessing.pipeline import (
    PreprocessingConfig,
    assess_ultrasound_image_quality,
    encode_segmentation_mask,
    preprocess_ultrasound_image,
)
from AI.tracking.tracker import TemporalStructureTracker
from AI.training.trainer import TrainingSample, UltrasoundMultiTaskDataset


def _make_synthetic_ultrasound_and_mask(height: int = 512, width: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Create a realistic synthetic ultrasound frame and 3-class ground-truth mask (0, 127, 255)."""
    rng = np.random.default_rng(123)
    img = rng.normal(loc=55.0, scale=28.0, size=(height, width, 3)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Pubic Symphysis (value 127)
    cv2.ellipse(img, (160, 140), (42, 16), -15, 0, 360, (210, 210, 210), -1)
    cv2.ellipse(mask, (160, 140), (42, 16), -15, 0, 360, 127, -1)

    # Fetal Head (value 255)
    cv2.ellipse(img, (290, 260), (110, 85), 20, 0, 360, (190, 190, 190), 5)
    cv2.ellipse(mask, (290, 260), (110, 85), 20, 0, 360, 255, -1)
    return img, mask


def test_training_and_inference_preprocessing_parity(tmp_path: Path) -> None:
    """
    Verify that model training (UltrasoundMultiTaskDataset with augment=False)
    and inference (UltrasoundInferenceEngine) apply the exact same preprocessing.
    """
    img, mask_raw = _make_synthetic_ultrasound_and_mask()
    cfg = PreprocessingConfig()
    mask_cls = encode_segmentation_mask(mask_raw, target_size=(cfg.target_height, cfg.target_width))

    sample = TrainingSample(
        subject_id="SUBJ-TEST-01",
        video_name="test_1_20.avi",
        frame_index=20,
        image_bgr=img,
        mask_classes=mask_cls,
        is_standard_plane=1,
    )
    ds = UltrasoundMultiTaskDataset([sample], preprocessing_config=cfg, augment=False, seed=42)
    train_tensor = ds[0]["image"].numpy()

    infer_tensor, orig_hw = preprocess_ultrasound_image(img, config=cfg)
    assert orig_hw == (512, 512)
    assert np.allclose(train_tensor, infer_tensor, atol=1e-7), "Training and inference preprocessing diverged!"


def test_calibration_gating_and_biometry_accuracy() -> None:
    """
    Verify that linear measurements (BPD, OFD, HC, HSD) are reported ONLY in pixels ('px')
    when pixel_spacing_mm is omitted, and converted to millimeters ('mm') when calibrated.
    """
    _, mask_raw = _make_synthetic_ultrasound_and_mask()
    class_mask = np.zeros_like(mask_raw, dtype=np.uint8)
    class_mask[mask_raw == 127] = 1
    class_mask[mask_raw == 255] = 2

    # 1. Uncalibrated extraction
    uncal = extract_landmarks_and_biometry(class_mask, prob_map=None, pixel_spacing_mm=None)
    assert uncal.head_detected is True
    assert uncal.ps_detected is True
    uncal_map = {m.measurement_type: m for m in uncal.measurements}
    assert set(uncal_map.keys()) == {"BPD", "OFD", "HC", "HSD", "AOP"}

    for linear_metric in ("BPD", "OFD", "HC", "HSD"):
        m = uncal_map[linear_metric]
        assert m.is_calibrated is False
        assert m.unit == "px"
        assert m.value_mm is None
        assert m.value_px > 0

    # AOP is an angle in degrees and is scale-invariant
    assert uncal_map["AOP"].unit == "deg"
    assert uncal_map["AOP"].is_calibrated is True
    assert 60.0 <= uncal_map["AOP"].value <= 180.0

    # Verify ellipse axes match synthetic ellipse (axes=(110, 85) -> major diameter=220, minor diameter=170)
    assert abs(uncal_map["BPD"].value_px - 170.0) < 3.0
    assert abs(uncal_map["OFD"].value_px - 220.0) < 3.0

    # 2. Calibrated extraction (0.25 mm/px)
    cal = extract_landmarks_and_biometry(class_mask, prob_map=None, pixel_spacing_mm=0.25)
    assert cal.qc_passed is True
    cal_map = {m.measurement_type: m for m in cal.measurements}
    for linear_metric in ("BPD", "OFD", "HC", "HSD"):
        m = cal_map[linear_metric]
        assert m.is_calibrated is True
        assert m.unit == "mm"
        assert m.value_mm == pytest.approx(m.value_px * 0.25, rel=1e-2)


def test_contour_qc_rejection_prevents_invalid_biometry() -> None:
    """
    Verify that invalid, tiny, fragmented, or low-confidence contours are rejected
    by quality control and NEVER produce BPD, OFD, or HC measurements.
    """
    bad_mask = np.zeros((512, 512), dtype=np.uint8)
    # Draw a tiny, highly elongated non-head sliver (area < 1.5% and extreme aspect ratio)
    cv2.rectangle(bad_mask, (50, 50), (140, 62), 2, -1)

    res = extract_landmarks_and_biometry(bad_mask, prob_map=None, pixel_spacing_mm=0.25)
    assert res.qc_passed is False
    assert len(res.qc_rejection_reasons) > 0
    assert len(res.measurements) == 0, "Invalid contour must not generate BPD, OFD, or HC measurements!"


def test_subject_level_splits_prevent_data_leakage() -> None:
    # Verify both test_X_Y.avi and multi-clip timestamped patient studies (I1, I2, I3)
    assert (
        extract_subject_id_from_filename("202101141947512000310I1.avi")
        == extract_subject_id_from_filename("202101141947512000310I2.avi")
    )
    filenames = [f"test_{i}_{frame}.avi" for i in range(1, 21) for frame in (10, 25, 40)]
    subject_ids = [extract_subject_id_from_filename(fn) for fn in filenames]
    split_map = assign_subject_splits(subject_ids, seed=42)

    subj_to_splits: dict[str, set[str]] = {}
    for fn in filenames:
        sid = extract_subject_id_from_filename(fn)
        sp = split_map[sid]
        subj_to_splits.setdefault(sid, set()).add(sp)

    assert verify_no_subject_leakage(subj_to_splits) is False
    assert set(split_map.values()) == {"train", "val", "test"}


def test_temporal_tracker_and_inference_pipelines(tmp_path: Path) -> None:
    tracker = TemporalStructureTracker(iou_threshold=0.3)
    t1 = tracker.update(0, "fetal_head", {"x": 100, "y": 100, "w": 150, "h": 150})
    t2 = tracker.update(4, "fetal_head", {"x": 104, "y": 102, "w": 150, "h": 150})
    assert t1 == t2
    summary = tracker.get_summary_metrics(2)
    assert summary["identity_switches"] == 0
    assert summary["mean_temporal_iou"] > 0.8

    # End-to-end image & video inference test
    img, _ = _make_synthetic_ultrasound_and_mask()
    img_path = tmp_path / "synthetic_frame.png"
    cv2.imwrite(str(img_path), img)

    engine = UltrasoundInferenceEngine()
    img_out = engine.analyze_image(img_path, pixel_spacing_mm=0.2, output_dir=tmp_path)
    assert img_out.source_file == "synthetic_frame.png"
    assert len(img_out.source_sha256) == 64
    assert Path(img_out.annotated_image_path).exists()
    assert "DECISION-SUPPORT ONLY" in img_out.disclaimer
