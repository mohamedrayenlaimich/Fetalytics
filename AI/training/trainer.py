import ast
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, Dataset as TorchDataset

from AI.analysis.pipelines import render_annotated_frame
from AI.datasets.ingestion import (
    assign_subject_splits,
    compute_mask_gt_biometry,
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
from AI.segmentation.unet import FetalHeadUNet, StandardPlaneClassifier
from Backend.app.core.config import BASE_DIR, settings
from Backend.app.models import Dataset, ModelVersion, TrainingRun
from Backend.app.services.b2_storage import B2StorageService, compute_file_sha256


def set_reproducible_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingSample:
    subject_id: str
    video_name: str
    frame_index: int
    image_bgr: np.ndarray
    mask_classes: Optional[np.ndarray]  # (H, W) int64 in {0, 1, 2}, None if classification-only frame
    is_standard_plane: int  # 1 for SP, 0 for NSP
    gt_biometry: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 75.0


class UltrasoundMultiTaskDataset(TorchDataset):
    """
    PyTorch Dataset for Ultrasound Segmentation and Standard-Plane Classification.
    Uses PreprocessingConfig identically to inference and ensures mask and image
    resolutions match BEFORE applying any spatial affine augmentation.
    """

    def __init__(
        self,
        samples: List[TrainingSample],
        preprocessing_config: PreprocessingConfig,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.config = preprocessing_config
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img = sample.image_bgr.copy()
        mask_raw = sample.mask_classes.copy() if sample.mask_classes is not None else None

        # CRITICAL FIX: Ensure mask_raw is at the exact same (H, W) resolution as img
        # BEFORE applying cv2.warpAffine((w, h)). In v1, mask_raw was (256, 256) while
        # img was (512, 512), which placed the mask into the top-left quadrant of a
        # 512x512 canvas and shrank it by 2x in each dimension upon resize!
        if mask_raw is not None and mask_raw.shape[:2] != img.shape[:2]:
            mask_raw = cv2.resize(
                mask_raw.astype(np.uint8),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

        if self.augment:
            angle = float(self.rng.uniform(-7.0, 7.0))
            scale = float(self.rng.uniform(0.95, 1.05))
            tx = float(self.rng.uniform(-10.0, 10.0))
            ty = float(self.rng.uniform(-10.0, 10.0))
            h, w = img.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty
            img = cv2.warpAffine(
                img,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if mask_raw is not None:
                mask_raw = cv2.warpAffine(
                    mask_raw.astype(np.uint8),
                    mat,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.int64)

            gain = float(self.rng.uniform(0.92, 1.08))
            bias = float(self.rng.uniform(-6.0, 6.0))
            img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        tensor_chw, _ = preprocess_ultrasound_image(img, config=self.config)
        has_seg = 1 if mask_raw is not None else 0
        if mask_raw is not None:
            if mask_raw.shape != (self.config.target_height, self.config.target_width):
                mask_resized = cv2.resize(
                    mask_raw.astype(np.uint8),
                    (self.config.target_width, self.config.target_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int64)
            else:
                mask_resized = mask_raw.astype(np.int64)
        else:
            mask_resized = np.zeros((self.config.target_height, self.config.target_width), dtype=np.int64)

        return {
            "image": torch.from_numpy(tensor_chw),
            "mask": torch.from_numpy(mask_resized),
            "has_seg": torch.tensor(has_seg, dtype=torch.float32),
            "is_sp": torch.tensor(float(sample.is_standard_plane), dtype=torch.float32),
            "subject_id": sample.subject_id,
            "video_name": sample.video_name,
        }


def per_class_balanced_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute Soft Dice loss independently per foreground class (Class 1: Pubic Symphysis,
    Class 2: Fetal Head) and per sample, then average with equal class weighting so the
    smaller Pubic Symphysis (1.74% area) is never drowned out by Fetal Head (13.08% area).
    """
    probs = F.softmax(logits, dim=1)
    one_hot = (
        F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )
    class_losses = []
    for cls_idx in range(1, num_classes):
        p_c = probs[:, cls_idx]
        t_c = one_hot[:, cls_idx]
        inter = torch.sum(p_c * t_c, dim=(1, 2))
        denom = torch.sum(p_c + t_c, dim=(1, 2))
        dice_c = (2.0 * inter + smooth) / (denom + smooth)
        class_losses.append(1.0 - torch.mean(dice_c))
    return torch.stack(class_losses).mean()


def multi_class_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    return per_class_balanced_dice_loss(logits, targets, num_classes=num_classes)


class IntensityThresholdEllipseBaseline:
    """Classical Otsu/Morphological Baseline Model."""

    def predict_mask(self, image_bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        resized = cv2.resize(gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        pred = np.zeros_like(closed, dtype=np.uint8)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(pred, [largest], -1, 2, thickness=-1)
        return pred


class SupervisedShallowSegBaseline(nn.Module):
    """
    Retrained supervised baseline segmentation model (Task 12: shallow 2-layer FCN
    with coordinate features) trained on the fixed preprocessing pipeline for comparison
    against both Otsu thresholding and FetalHeadUNet-v2.0.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 2, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.net(torch.cat([x, yy, xx], dim=1))


def compute_segmentation_metrics_detailed(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    cls_id: int,
) -> Dict[str, float]:
    """
    Calculate Dice, IoU, pixel-level Sensitivity (Recall), and Specificity for cls_id.
    """
    p = pred_mask == cls_id
    g = gt_mask == cls_id
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())

    union = tp + fp + fn
    denom = 2.0 * tp + fp + fn
    dice = (2.0 * tp / denom) if denom > 0 else (1.0 if union == 0 else 0.0)
    iou = (tp / union) if union > 0 else 1.0
    sens = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sens,
        "specificity": spec,
    }


def compute_segmentation_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, cls_id: int) -> Tuple[float, float]:
    m = compute_segmentation_metrics_detailed(pred_mask, gt_mask, cls_id)
    return m["dice"], m["iou"]


def bootstrap_ci95(values: List[float], n_boot: int = 400, seed: int = 42) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (round(values[0], 4), round(values[0], 4))
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    low, high = np.percentile(means, [2.5, 97.5])
    return (round(float(low), 4), round(float(high), 4))


def bland_altman_agreement(preds: List[float], gts: List[float]) -> Dict[str, float]:
    if not preds or len(preds) != len(gts):
        return {"mean_bias": 0.0, "sd_diff": 0.0, "loa_lower": 0.0, "loa_upper": 0.0}
    diffs = np.array(preds, dtype=np.float64) - np.array(gts, dtype=np.float64)
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs))
    return {
        "mean_bias": round(bias, 3),
        "sd_diff": round(sd, 3),
        "loa_lower": round(bias - 1.96 * sd, 3),
        "loa_upper": round(bias + 1.96 * sd, 3),
    }


def _load_subset_directory_samples(
    subset_dir: Path,
    preprocessing_config: PreprocessingConfig,
    max_videos: Optional[int] = None,
) -> List[TrainingSample]:
    """
    Load paired ultrasound frames and ground-truth masks (plus negative NSP frames for
    separate standard-plane classification) from an extracted IUGC2024 subset folder.
    """
    if not subset_dir.exists():
        return []

    seg_info: Dict[str, int] = {}
    seg_csv = subset_dir / "seg" / "seg_info.csv"
    if seg_csv.exists():
        with open(seg_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = (row.get("filename") or "").strip()
                idx_str = (row.get("labeled_frame_index") or row.get("labeled_index") or "0").strip()
                if "," in idx_str:
                    idx_str = idx_str.split(",")[0].strip()
                if fn and idx_str.isdigit():
                    seg_info[fn] = int(idx_str)

    sp_nsp_map: Dict[str, Tuple[List[int], List[int]]] = {}
    for info_name in ("val_info.csv", "test_info.csv", "train_info.csv"):
        info_csv = subset_dir / info_name
        if info_csv.exists():
            with open(info_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fn = (row.get("filename") or "").strip()
                    try:
                        sp_list = [int(x) for x in ast.literal_eval(row.get("SP_index") or "[]")]
                    except Exception:
                        sp_list = []
                    try:
                        nsp_list = [int(x) for x in ast.literal_eval(row.get("NSP_index") or "[]")]
                    except Exception:
                        nsp_list = []
                    sp_nsp_map[fn] = (sp_list, nsp_list)

    landmarks_data: Dict[str, Any] = {}
    for lm_name in ("landmarks.json", "landmark.json"):
        lm_path = subset_dir / "seg" / lm_name
        if lm_path.exists():
            landmarks_data.update(json.loads(lm_path.read_text(encoding="utf-8")))

    videos = sorted((subset_dir / "videos").glob("*.avi"))
    if max_videos is not None:
        videos = videos[:max_videos]

    samples: List[TrainingSample] = []
    for vpath in videos:
        vname = vpath.name
        stem = vpath.stem
        sid = extract_subject_id_from_filename(vname)
        labeled_fidx = seg_info.get(vname, 0)

        # Resolve mask path (<stem>_<fidx>.png or <stem>.png)
        cands = [
            subset_dir / "seg" / f"{stem}_{labeled_fidx}.png",
            subset_dir / "seg" / f"{stem}.png",
            subset_dir / "seg" / f"{stem}_{labeled_fidx}_6.png",
        ]
        mask_path = next((c for c in cands if c.exists()), None)

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, labeled_fidx)
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None and mask_path is not None:
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            # Keep mask_classes at native 512x512 resolution so image & mask share identical coordinate space!
            mask_cls = encode_segmentation_mask(
                mask_raw,
                target_size=(frame_bgr.shape[0], frame_bgr.shape[1]),
            )
            gt_bio = compute_mask_gt_biometry(mask_raw)
            lm_entry = landmarks_data.get(mask_path.name, {}) or landmarks_data.get(f"{stem}.png", {})
            if lm_entry:
                gt_bio["hsd_px"] = float(lm_entry.get("hsd", 0.0))
                gt_bio["aop_deg"] = float(lm_entry.get("aop", 0.0))
            qual = assess_ultrasound_image_quality(frame_bgr).quality_score
            samples.append(
                TrainingSample(
                    subject_id=sid,
                    video_name=vname,
                    frame_index=labeled_fidx,
                    image_bgr=frame_bgr,
                    mask_classes=mask_cls,
                    is_standard_plane=1,
                    gt_biometry=gt_bio,
                    quality_score=qual,
                )
            )

        _, nsp_indices = sp_nsp_map.get(vname, ([], []))
        if nsp_indices:
            nsp_fidx = nsp_indices[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, nsp_fidx)
            ok_neg, neg_bgr = cap.read()
            if ok_neg and neg_bgr is not None:
                qual_neg = assess_ultrasound_image_quality(neg_bgr).quality_score
                samples.append(
                    TrainingSample(
                        subject_id=sid,
                        video_name=vname,
                        frame_index=nsp_fidx,
                        image_bgr=neg_bgr,
                        mask_classes=None,
                        is_standard_plane=0,
                        gt_biometry={},
                        quality_score=qual_neg,
                    )
                )
        cap.release()

    return samples


def load_iugc_samples_from_extracted_dir(
    extracted_root: Path,
    preprocessing_config: PreprocessingConfig,
    seed: int = 42,
    max_train_videos: Optional[int] = None,
) -> Tuple[Dict[str, List[TrainingSample]], Dict[str, List[str]]]:
    """
    Load paired IUGC2024 samples with strict patient-level separation across train, val,
    and a completely held-out test set.
    - If both `extracted_root/test` (2020/2021 patient studies) and `extracted_root/val`
      (test_1..test_40 cohort) exist, `train` and `val` are partitioned by patient study ID
      from `extracted_root/test` plus the 28 training subjects of `extracted_root/val`,
      while the 12 held-out subjects (`val_6 + test_6`) in `extracted_root/val` AND
      held-out patient studies from `extracted_root/test` are strictly isolated in `test`
      (never seen during training or validation checkpoint selection!).
    """
    val_dir = extracted_root / "val"
    test_dir = extracted_root / "test"

    splits: Dict[str, List[TrainingSample]] = {"train": [], "val": [], "test": []}
    split_subjects: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    if val_dir.exists():
        val_samples = _load_subset_directory_samples(val_dir, preprocessing_config)
        val_sids = sorted({s.subject_id for s in val_samples})
        val_split_map = assign_subject_splits(val_sids, seed=seed, ratios=(0.70, 0.15, 0.15))

        for s in val_samples:
            sp = val_split_map[s.subject_id]
            splits[sp].append(s)
            if s.subject_id not in split_subjects[sp]:
                split_subjects[sp].append(s.subject_id)

    if test_dir.exists():
        extra_samples = _load_subset_directory_samples(test_dir, preprocessing_config, max_videos=max_train_videos)
        extra_sids = sorted({s.subject_id for s in extra_samples})
        # Partition patient studies by 14-digit patient timestamp into train (70%), val (15%), test (15%)
        extra_split_map = assign_subject_splits(extra_sids, seed=seed, ratios=(0.70, 0.15, 0.15))
        for s in extra_samples:
            sp = extra_split_map[s.subject_id]
            splits[sp].append(s)
            if s.subject_id not in split_subjects[sp]:
                split_subjects[sp].append(s.subject_id)

    # Verify strict patient-level disjointness across train, val, and held-out test splits
    subj_sets = {s: set(split_subjects[s]) for s in ("train", "val", "test")}
    assert subj_sets["train"].isdisjoint(subj_sets["val"]), "Patient leakage between train and val!"
    assert subj_sets["train"].isdisjoint(subj_sets["test"]), "Patient leakage between train and test!"
    assert subj_sets["val"].isdisjoint(subj_sets["test"]), "Patient leakage between val and test!"

    subj_to_splits: Dict[str, set] = {}
    for sp_name, sample_list in splits.items():
        for s in sample_list:
            subj_to_splits.setdefault(s.subject_id, set()).add(sp_name)
    assert not verify_no_subject_leakage(subj_to_splits), "Subject leakage detected across splits!"

    return splits, split_subjects


def _colorize_mask(mask_cls: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Render class mask {0, 1, 2} as a BGR visualization image (green=FH, orange=PS)."""
    m = cv2.resize(mask_cls.astype(np.uint8), (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    vis = np.zeros((size_hw[0], size_hw[1], 3), dtype=np.uint8)
    vis[m == 1] = (0, 165, 255)  # Orange: Pubic Symphysis
    vis[m == 2] = (0, 230, 90)   # Green: Fetal Head
    return vis


def run_reproducible_training_and_evaluation(
    extracted_root: Path,
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 3e-3,
    seed: int = 42,
    upload_to_b2: bool = True,
    activate_for_production: bool = False,
    early_stopping_patience: int = 4,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    1. Load patient-disjoint train, validation, and held-out test splits.
    2. Train Segmentation Model (FetalHeadUNet-v2.0) exclusively on paired segmented samples.
    3. Train Supervised Baseline Model (SupervisedShallowSegBaseline) for comparison.
    4. Train Dedicated Standard-Plane Classifier (StandardPlaneClassifier) separately from segmentation.
    5. Evaluate Otsu Baseline, Supervised Baseline, Preserved v1.0 Model, and Improved v2.0 Model
       exclusively on the completely held-out test set (never used for training or checkpoint selection).
    6. Generate side-by-side visual comparison sheets on the held-out test set.
    7. Upload versioned model checkpoint to Backblaze B2 and register in Neon PostgreSQL.
    """
    set_reproducible_seed(seed)
    pre_cfg = PreprocessingConfig()
    splits, split_subjects = load_iugc_samples_from_extracted_dir(
        extracted_root=extracted_root,
        preprocessing_config=pre_cfg,
        seed=seed,
    )

    seg_train_samples = [s for s in splits["train"] if s.mask_classes is not None]
    seg_val_samples = [s for s in splits["val"] if s.mask_classes is not None]
    seg_test_samples = [s for s in splits["test"] if s.mask_classes is not None]

    print(
        f"[Split Summary] Segmented pairs -> train={len(seg_train_samples)} ({len(split_subjects['train'])} patients), "
        f"val={len(seg_val_samples)} ({len(split_subjects['val'])} patients), "
        f"held-out test={len(seg_test_samples)} ({len(split_subjects['test'])} patients)",
        flush=True,
    )

    train_seg_ds = UltrasoundMultiTaskDataset(seg_train_samples, pre_cfg, augment=True, seed=seed)
    val_seg_ds = UltrasoundMultiTaskDataset(seg_val_samples, pre_cfg, augment=False, seed=seed)

    train_seg_loader = DataLoader(train_seg_ds, batch_size=batch_size, shuffle=True)
    val_seg_loader = DataLoader(val_seg_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # A. Train Improved FetalHeadUNet-v2.0 (4-stage context U-Net + coordinate priors)
    # --------------------------------------------------------------------------
    model = FetalHeadUNet(
        in_channels=1,
        num_classes=3,
        base_filters=24,
        use_context_stage=True,
    ).to(device)

    # Inverse-sqrt class weights for {0: Background (85%), 1: PS (1.7%), 2: FH (13.1%)}
    class_weights = torch.tensor([0.22, 4.2, 1.6], dtype=torch.float32, device=device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    weights_dir = BASE_DIR / "AI" / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    v1_preserved_path = weights_dir / "FetalHeadUNet-v1.0-seed42_ORIGINAL.pt"
    v2_version_tag = f"FetalHeadUNet-v2.0-seed{seed}"
    v2_ckpt_path = weights_dir / f"{v2_version_tag}.pt"
    active_best_path = weights_dir / "fetal_head_unet_best.pt"

    history: List[Dict[str, float]] = []
    best_val_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None
    started_at = datetime.utcnow()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_seg_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            seg_logits, plane_logits = model(imgs)
            ce_loss = ce_criterion(seg_logits, masks)
            dice_loss = per_class_balanced_dice_loss(seg_logits, masks, num_classes=3)
            # Train auxiliary plane head to output positive standard plane confidence on valid masks
            sp_targets = torch.ones_like(plane_logits)
            sp_aux_loss = F.binary_cross_entropy_with_logits(plane_logits, sp_targets)
            loss = 0.45 * ce_loss + 0.50 * dice_loss + 0.05 * sp_aux_loss

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        # Validation evaluation (ONLY on validation split, never on held-out test split!)
        model.eval()
        val_fh_dices: List[float] = []
        val_fh_ious: List[float] = []
        val_ps_dices: List[float] = []
        val_ps_ious: List[float] = []
        val_epoch_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_seg_loader:
                imgs = batch["image"].to(device)
                masks_t = batch["mask"].to(device)
                masks = masks_t.cpu().numpy()
                seg_logits, plane_logits = model(imgs)
                v_ce = ce_criterion(seg_logits, masks_t)
                v_dice = per_class_balanced_dice_loss(seg_logits, masks_t, num_classes=3)
                v_sp = F.binary_cross_entropy_with_logits(plane_logits, torch.ones_like(plane_logits))
                v_loss = 0.45 * v_ce + 0.50 * v_dice + 0.05 * v_sp
                val_epoch_loss += float(v_loss.item())
                n_val_batches += 1

                preds = torch.argmax(seg_logits, dim=1).cpu().numpy()
                for i in range(len(imgs)):
                    d_fh, iou_fh = compute_segmentation_dice_iou(preds[i], masks[i], cls_id=2)
                    d_ps, iou_ps = compute_segmentation_dice_iou(preds[i], masks[i], cls_id=1)
                    val_fh_dices.append(d_fh)
                    val_fh_ious.append(iou_fh)
                    val_ps_dices.append(d_ps)
                    val_ps_ious.append(iou_ps)

        mean_val_loss = val_epoch_loss / max(1, n_val_batches)
        mean_val_fh = float(np.mean(val_fh_dices)) if val_fh_dices else 0.0
        mean_val_fh_iou = float(np.mean(val_fh_ious)) if val_fh_ious else 0.0
        mean_val_ps = float(np.mean(val_ps_dices)) if val_ps_dices else 0.0
        mean_val_ps_iou = float(np.mean(val_ps_ious)) if val_ps_ious else 0.0
        combined_val = 0.70 * mean_val_fh + 0.30 * mean_val_ps

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss / max(1, n_batches), 4),
                "val_loss": round(mean_val_loss, 4),
                "val_fetal_head_dice": round(mean_val_fh, 4),
                "val_fetal_head_iou": round(mean_val_fh_iou, 4),
                "val_pubic_symphysis_dice": round(mean_val_ps, 4),
                "val_pubic_symphysis_iou": round(mean_val_ps_iou, 4),
                "val_combined_dice": round(combined_val, 4),
            }
        )
        print(
            f"  [UNet-v2] Epoch {epoch}/{epochs} | train_loss={epoch_loss / max(1, n_batches):.4f} "
            f"| val_loss={mean_val_loss:.4f} | val_fh_dice={mean_val_fh:.4f} (iou={mean_val_fh_iou:.4f}) "
            f"| val_ps_dice={mean_val_ps:.4f} (iou={mean_val_ps_iou:.4f})",
            flush=True,
        )

        if combined_val > best_val_score + 1e-4:
            best_val_score = combined_val
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"  [Early Stopping] Validation combined Dice did not improve for {early_stopping_patience} "
                    f"epochs (best={best_val_score:.4f} at epoch {best_epoch}). Halting training.",
                    flush=True,
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    # --------------------------------------------------------------------------
    # B. Train Supervised Shallow Baseline Model (Task 12) with val selection
    # --------------------------------------------------------------------------
    sup_baseline = SupervisedShallowSegBaseline(in_channels=1, num_classes=3).to(device)
    base_opt = torch.optim.AdamW(sup_baseline.parameters(), lr=5e-3, weight_decay=1e-4)
    best_base_val = -1.0
    best_base_state = None
    for _ in range(min(5, epochs)):
        sup_baseline.train()
        for batch in train_seg_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            base_opt.zero_grad()
            b_logits = sup_baseline(imgs)
            b_loss = 0.5 * ce_criterion(b_logits, masks) + 0.5 * per_class_balanced_dice_loss(b_logits, masks)
            b_loss.backward()
            base_opt.step()

        sup_baseline.eval()
        b_val_dices: List[float] = []
        with torch.no_grad():
            for batch in val_seg_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].numpy()
                b_preds = torch.argmax(sup_baseline(imgs), dim=1).cpu().numpy()
                for i in range(len(imgs)):
                    d_fh, _ = compute_segmentation_dice_iou(b_preds[i], masks[i], cls_id=2)
                    b_val_dices.append(d_fh)
        mean_b_val = float(np.mean(b_val_dices)) if b_val_dices else 0.0
        if mean_b_val >= best_base_val:
            best_base_val = mean_b_val
            best_base_state = {k: v.cpu().clone() for k, v in sup_baseline.state_dict().items()}

    if best_base_state is not None:
        sup_baseline.load_state_dict(best_base_state)
    sup_baseline.eval()

    # --------------------------------------------------------------------------
    # C. Train Dedicated Standard-Plane Classifier Separately (Task 16) with val selection
    # --------------------------------------------------------------------------
    plane_clf = StandardPlaneClassifier(in_channels=1, base_filters=16).to(device)
    clf_train_ds = UltrasoundMultiTaskDataset(splits["train"], pre_cfg, augment=True, seed=seed)
    clf_val_ds = UltrasoundMultiTaskDataset(splits["val"], pre_cfg, augment=False, seed=seed)
    clf_train_loader = DataLoader(clf_train_ds, batch_size=batch_size, shuffle=True)
    clf_val_loader = DataLoader(clf_val_ds, batch_size=batch_size, shuffle=False)
    clf_opt = torch.optim.AdamW(plane_clf.parameters(), lr=2e-3, weight_decay=1e-4)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    best_clf_val_loss = float("inf")
    best_clf_state = None
    for _ in range(min(8, epochs)):
        plane_clf.train()
        for batch in clf_train_loader:
            imgs = batch["image"].to(device)
            is_sp = batch["is_sp"].to(device)
            clf_opt.zero_grad()
            sp_logits = plane_clf(imgs)
            c_loss = bce_loss_fn(sp_logits, is_sp)
            c_loss.backward()
            clf_opt.step()

        plane_clf.eval()
        c_val_loss = 0.0
        c_val_batches = 0
        with torch.no_grad():
            for batch in clf_val_loader:
                imgs = batch["image"].to(device)
                is_sp = batch["is_sp"].to(device)
                sp_logits = plane_clf(imgs)
                c_val_loss += float(bce_loss_fn(sp_logits, is_sp).item())
                c_val_batches += 1
        mean_c_val = c_val_loss / max(1, c_val_batches)
        if mean_c_val <= best_clf_val_loss:
            best_clf_val_loss = mean_c_val
            best_clf_state = {k: v.cpu().clone() for k, v in plane_clf.state_dict().items()}

    if best_clf_state is not None:
        plane_clf.load_state_dict(best_clf_state)
    plane_clf.eval()

    # Save v2 checkpoint to its own versioned path (preserving v1_preserved_path and fetal_head_unet_best.pt untouched!)
    ckpt_payload = {
        "version_tag": v2_version_tag,
        "architecture": "FetalHeadUNet",
        "use_context_stage": True,
        "base_filters": 24,
        "random_seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_dice": round(best_val_score, 4),
        "preprocessing_config": pre_cfg.to_dict(),
        "model_state_dict": model.state_dict(),
        "plane_classifier_state_dict": plane_clf.state_dict(),
    }
    torch.save(ckpt_payload, str(v2_ckpt_path))
    if activate_for_production:
        torch.save(ckpt_payload, str(active_best_path))

    # Load preserved v1 model for direct comparison on the held-out test set
    v1_model: Optional[FetalHeadUNet] = None
    if v1_preserved_path.exists():
        v1_ckpt = torch.load(str(v1_preserved_path), map_location=device, weights_only=False)
        v1_model = FetalHeadUNet(in_channels=1, num_classes=3, base_filters=24, use_context_stage=False).to(device)
        v1_model.load_state_dict(v1_ckpt.get("model_state_dict", v1_ckpt))
        v1_model.eval()

    otsu_baseline = IntensityThresholdEllipseBaseline()

    # --------------------------------------------------------------------------
    # D. Evaluate Strictly on Completely Held-Out Test Set (Task 13, 14, 15)
    # --------------------------------------------------------------------------
    test_samples = splits["test"]
    v2_fh_metrics: List[Dict[str, float]] = []
    v2_ps_metrics: List[Dict[str, float]] = []
    v1_fh_metrics: List[Dict[str, float]] = []
    v1_ps_metrics: List[Dict[str, float]] = []
    otsu_fh_metrics: List[Dict[str, float]] = []
    sup_base_fh_metrics: List[Dict[str, float]] = []

    hc_preds_px, hc_gts_px = [], []
    bpd_preds_px, bpd_gts_px = [], []
    aop_preds_deg, aop_gts_deg = [], []
    hsd_preds_px, hsd_gts_px = [], []
    qc_accepted_count = 0
    qc_rejected_count = 0

    sp_y_true: List[int] = []
    sp_y_pred: List[int] = []
    sp_y_prob: List[float] = []

    high_qual_dices: List[float] = []
    mod_qual_dices: List[float] = []
    comparison_tiles: List[np.ndarray] = []

    for s in test_samples:
        tensor_chw, (orig_h, orig_w) = preprocess_ultrasound_image(s.image_bgr, config=pre_cfg)
        inp = torch.from_numpy(tensor_chw).unsqueeze(0).to(device)

        with torch.no_grad():
            seg_logits, _ = model(inp)
            prob_small = F.softmax(seg_logits, dim=1)
            pred_small = torch.argmax(prob_small, dim=1).squeeze(0).cpu().numpy()
            prob_orig = (
                F.interpolate(prob_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            pred_orig = np.argmax(prob_orig, axis=0).astype(np.uint8)

            sp_logit = plane_clf(inp)
            sp_prob = float(torch.sigmoid(sp_logit).item())

            sup_pred_small = torch.argmax(sup_baseline(inp), dim=1).squeeze(0).cpu().numpy()
            v1_pred_small = (
                torch.argmax(v1_model(inp)[0], dim=1).squeeze(0).cpu().numpy()
                if v1_model is not None
                else np.zeros_like(pred_small)
            )

        sp_y_true.append(int(s.is_standard_plane))
        sp_y_prob.append(sp_prob)
        sp_y_pred.append(1 if sp_prob >= 0.5 else 0)

        if s.mask_classes is not None:
            gt_small = cv2.resize(
                s.mask_classes.astype(np.uint8),
                (pre_cfg.target_width, pre_cfg.target_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

            m_fh_v2 = compute_segmentation_metrics_detailed(pred_small, gt_small, cls_id=2)
            m_ps_v2 = compute_segmentation_metrics_detailed(pred_small, gt_small, cls_id=1)
            v2_fh_metrics.append(m_fh_v2)
            v2_ps_metrics.append(m_ps_v2)

            m_fh_v1 = compute_segmentation_metrics_detailed(v1_pred_small, gt_small, cls_id=2)
            m_ps_v1 = compute_segmentation_metrics_detailed(v1_pred_small, gt_small, cls_id=1)
            v1_fh_metrics.append(m_fh_v1)
            v1_ps_metrics.append(m_ps_v1)

            otsu_pred = otsu_baseline.predict_mask(
                s.image_bgr,
                target_size=(pre_cfg.target_height, pre_cfg.target_width),
            )
            otsu_fh_metrics.append(compute_segmentation_metrics_detailed(otsu_pred, gt_small, cls_id=2))
            sup_base_fh_metrics.append(compute_segmentation_metrics_detailed(sup_pred_small, gt_small, cls_id=2))

            if s.quality_score >= 60.0:
                high_qual_dices.append(m_fh_v2["dice"])
            else:
                mod_qual_dices.append(m_fh_v2["dice"])

            extraction = extract_landmarks_and_biometry(pred_orig, prob_map=prob_orig, pixel_spacing_mm=None)
            if extraction.qc_passed:
                qc_accepted_count += 1
                meas_map = {m.measurement_type: m.value_px for m in extraction.measurements}
                if "HC" in meas_map and "hc_px" in s.gt_biometry:
                    hc_preds_px.append(meas_map["HC"])
                    hc_gts_px.append(s.gt_biometry["hc_px"])
                if "BPD" in meas_map and "bpd_px" in s.gt_biometry:
                    bpd_preds_px.append(meas_map["BPD"])
                    bpd_gts_px.append(s.gt_biometry["bpd_px"])
                if "AOP" in meas_map and "aop_deg" in s.gt_biometry and s.gt_biometry["aop_deg"] > 0:
                    aop_preds_deg.append(meas_map["AOP"])
                    aop_gts_deg.append(s.gt_biometry["aop_deg"])
                if "HSD" in meas_map and "hsd_px" in s.gt_biometry and s.gt_biometry["hsd_px"] > 0:
                    hsd_preds_px.append(meas_map["HSD"])
                    hsd_gts_px.append(s.gt_biometry["hsd_px"])
            else:
                qc_rejected_count += 1

            # Build side-by-side comparison tile for Task 14
            if len(comparison_tiles) < 20:
                tile_sz = (224, 224)
                orig_vis = cv2.resize(s.image_bgr, tile_sz)
                gt_vis = _colorize_mask(gt_small, tile_sz)
                v1_vis = _colorize_mask(v1_pred_small, tile_sz)
                v2_vis = _colorize_mask(pred_small, tile_sz)
                overlay_full = render_annotated_frame(
                    frame_bgr=s.image_bgr,
                    extraction=extraction,
                    plane_prob=sp_prob,
                    model_version_tag=v2_version_tag,
                    frame_index=s.frame_index,
                )
                ov_vis = cv2.resize(overlay_full, tile_sz)

                cv2.putText(orig_vis, f"{Path(s.video_name).stem[:14]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(gt_vis, "Ground Truth", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(v1_vis, f"v1.0 Dice={m_fh_v1['dice']:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
                cv2.putText(v2_vis, f"v2.0 Dice={m_fh_v2['dice']:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
                cv2.putText(ov_vis, "v2.0 Overlay+Biometry", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                comparison_tiles.append(np.hstack([orig_vis, gt_vis, v1_vis, v2_vis, ov_vis]))

    diag_dir = BASE_DIR / "results" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for sheet_idx in range(0, len(comparison_tiles), 10):
        sheet_img = np.vstack(comparison_tiles[sheet_idx : sheet_idx + 10])
        cv2.imwrite(str(diag_dir / f"heldout_test_comparison_sheet_{sheet_idx // 10 + 1}.png"), sheet_img)

    def _agg_seg(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
        if not metrics_list:
            return {"dice_mean": 0.0, "dice_ci95": [0.0, 0.0], "iou_mean": 0.0, "sensitivity_mean": 0.0, "specificity_mean": 0.0}
        dices = [m["dice"] for m in metrics_list]
        ious = [m["iou"] for m in metrics_list]
        sens = [m["sensitivity"] for m in metrics_list]
        spec = [m["specificity"] for m in metrics_list]
        return {
            "n_held_out_samples": len(metrics_list),
            "dice_mean": round(float(np.mean(dices)), 4),
            "dice_median": round(float(np.median(dices)), 4),
            "dice_ci95": list(bootstrap_ci95(dices)),
            "iou_mean": round(float(np.mean(ious)), 4),
            "iou_ci95": list(bootstrap_ci95(ious)),
            "sensitivity_mean": round(float(np.mean(sens)), 4),
            "specificity_mean": round(float(np.mean(spec)), 4),
        }

    def _mae_rmse(preds: List[float], gts: List[float]) -> Dict[str, Any]:
        if not preds or len(preds) != len(gts):
            return {"mae": None, "rmse": None, "n": 0}
        err = np.array(preds) - np.array(gts)
        return {
            "mae": round(float(np.mean(np.abs(err))), 3),
            "rmse": round(float(np.sqrt(np.mean(err**2))), 3),
            "n": len(preds),
            "bland_altman": bland_altman_agreement(preds, gts),
        }

    tp = sum(1 for yt, yp in zip(sp_y_true, sp_y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(sp_y_true, sp_y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(sp_y_true, sp_y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(sp_y_true, sp_y_pred) if yt == 1 and yp == 0)

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = sensitivity
    f1_score = (2 * precision * recall) / max(1e-8, precision + recall)
    brier_score = float(np.mean((np.array(sp_y_prob) - np.array(sp_y_true)) ** 2)) if sp_y_true else 0.0

    v2_fh_summary = _agg_seg(v2_fh_metrics)
    v2_ps_summary = _agg_seg(v2_ps_metrics)
    v1_fh_summary = _agg_seg(v1_fh_metrics)
    v1_ps_summary = _agg_seg(v1_ps_metrics)
    otsu_fh_summary = _agg_seg(otsu_fh_metrics)
    sup_base_fh_summary = _agg_seg(sup_base_fh_metrics)

    evaluation_metrics: Dict[str, Any] = {
        "segmentation": {
            "fetal_head_dice_mean": v2_fh_summary["dice_mean"],
            "fetal_head_dice_ci95": v2_fh_summary["dice_ci95"],
            "fetal_head_iou_mean": v2_fh_summary["iou_mean"],
            "fetal_head_sensitivity_mean": v2_fh_summary["sensitivity_mean"],
            "fetal_head_specificity_mean": v2_fh_summary["specificity_mean"],
            "pubic_symphysis_dice_mean": v2_ps_summary["dice_mean"],
            "pubic_symphysis_dice_ci95": v2_ps_summary["dice_ci95"],
            "pubic_symphysis_iou_mean": v2_ps_summary["iou_mean"],
            "pubic_symphysis_sensitivity_mean": v2_ps_summary["sensitivity_mean"],
            "pubic_symphysis_specificity_mean": v2_ps_summary["specificity_mean"],
            "baseline_otsu_fetal_head_dice_mean": otsu_fh_summary["dice_mean"],
            "baseline_supervised_shallow_fetal_head_dice_mean": sup_base_fh_summary["dice_mean"],
            "previous_v1_fetal_head_dice_mean": v1_fh_summary["dice_mean"],
        },
        "model_comparison_table": {
            "Otsu_Threshold_Baseline": otsu_fh_summary,
            "Supervised_Shallow_FCN_Baseline": sup_base_fh_summary,
            "FetalHeadUNet_v1.0_Original": {
                "fetal_head": v1_fh_summary,
                "pubic_symphysis": v1_ps_summary,
            },
            "FetalHeadUNet_v2.0_Improved": {
                "fetal_head": v2_fh_summary,
                "pubic_symphysis": v2_ps_summary,
            },
        },
        "biometry_and_landmarks": {
            "qc_accepted_contours": qc_accepted_count,
            "qc_rejected_contours": qc_rejected_count,
            "hc_px": _mae_rmse(hc_preds_px, hc_gts_px),
            "bpd_px": _mae_rmse(bpd_preds_px, bpd_gts_px),
            "hsd_px": _mae_rmse(hsd_preds_px, hsd_gts_px),
            "aop_deg": _mae_rmse(aop_preds_deg, aop_gts_deg),
        },
        "standard_plane_classification": {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "brier_calibration_score": round(brier_score, 4),
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        },
    }

    subgroup_metrics: Dict[str, Any] = {
        "by_image_quality": {
            "high_quality_score_ge_60": {
                "n": len(high_qual_dices),
                "fetal_head_dice_mean": round(float(np.mean(high_qual_dices)), 4) if high_qual_dices else None,
            },
            "moderate_or_shadowed_quality_lt_60": {
                "n": len(mod_qual_dices),
                "fetal_head_dice_mean": round(float(np.mean(mod_qual_dices)), 4) if mod_qual_dices else None,
            },
        },
        "clinical_limitations": (
            "Evaluated strictly on held-out patient studies from the IUGC2024 cohort. "
            "Linear measurements (BPD, OFD, HC, HSD) require calibrated pixel_spacing_mm from the ultrasound machine before reporting in millimetres."
        ),
    }

    artifact_sha256 = compute_file_sha256(v2_ckpt_path)
    artifact_size = v2_ckpt_path.stat().st_size
    b2_artifact_key = f"models/{v2_version_tag}_{artifact_sha256[:10]}.pt"

    if upload_to_b2 and settings.b2_is_configured:
        try:
            storage = B2StorageService()
            storage.upload_file(
                local_path=v2_ckpt_path,
                key=b2_artifact_key,
                metadata={"version_tag": v2_version_tag, "seed": str(seed)},
            )
        except Exception:
            b2_artifact_key = None

    completed_at = datetime.utcnow()

    if db is not None:
        ds_obj = db.query(Dataset).filter(Dataset.name == "IUGC2024-Fetal-Ultrasound-V3").first()
        tr_run = TrainingRun(
            dataset_id=ds_obj.id if ds_obj else None,
            dataset_version=ds_obj.version if ds_obj else "3.0.0",
            task_name="fetal_head_segmentation_and_plane_classification",
            architecture="FetalHeadUNet-v2",
            random_seed=seed,
            hyperparameters={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": "AdamW",
                "loss": "0.45*WeightedCE + 0.50*PerClassBalancedDice + 0.05*PlaneAux",
            },
            preprocessing_config=pre_cfg.to_dict(),
            split_manifest=split_subjects,
            status="completed",
            epochs_completed=len(history),
            best_val_metric=round(best_val_score, 4),
            training_history=history,
            evaluation_metrics=evaluation_metrics,
            subgroup_metrics=subgroup_metrics,
            started_at=started_at,
            completed_at=completed_at,
        )
        db.add(tr_run)
        db.flush()

        # Do NOT activate v2.0 for production inference unless activate_for_production=True
        if activate_for_production:
            for old_mv in db.query(ModelVersion).filter(ModelVersion.version_tag != v2_version_tag).all():
                old_mv.is_active = False

        existing_mv = db.query(ModelVersion).filter(ModelVersion.version_tag == v2_version_tag).first()
        if existing_mv:
            existing_mv.training_run_id = tr_run.id
            existing_mv.b2_artifact_key = b2_artifact_key
            existing_mv.local_artifact_path = str(v2_ckpt_path)
            existing_mv.artifact_sha256 = artifact_sha256
            existing_mv.artifact_size_bytes = artifact_size
            existing_mv.preprocessing_config = pre_cfg.to_dict()
            existing_mv.evaluation_summary = evaluation_metrics
            existing_mv.confidence_intervals = {
                "fetal_head_dice_ci95": v2_fh_summary["dice_ci95"],
                "pubic_symphysis_dice_ci95": v2_ps_summary["dice_ci95"],
            }
            existing_mv.limitations_notes = subgroup_metrics["clinical_limitations"]
            existing_mv.is_active = bool(activate_for_production)
        else:
            mv = ModelVersion(
                training_run_id=tr_run.id,
                model_name="Fetalytics-HeadSeg-UNet-v2",
                version_tag=v2_version_tag,
                task_name="fetal_head_segmentation_and_plane_classification",
                architecture="FetalHeadUNet-v2",
                dataset_version="3.0.0",
                b2_bucket=settings.B2_BUCKET_NAME or "fetalytics-videos",
                b2_artifact_key=b2_artifact_key,
                local_artifact_path=str(v2_ckpt_path),
                artifact_sha256=artifact_sha256,
                artifact_size_bytes=artifact_size,
                preprocessing_config=pre_cfg.to_dict(),
                evaluation_summary=evaluation_metrics,
                confidence_intervals={
                    "fetal_head_dice_ci95": v2_fh_summary["dice_ci95"],
                    "pubic_symphysis_dice_ci95": v2_ps_summary["dice_ci95"],
                },
                limitations_notes=subgroup_metrics["clinical_limitations"],
                is_active=bool(activate_for_production),
            )
            db.add(mv)
        db.commit()

    report_payload = {
        "version_tag": v2_version_tag,
        "preserved_v1_checkpoint": str(v1_preserved_path),
        "activated_for_production": bool(activate_for_production),
        "architecture": "FetalHeadUNet-v2",
        "random_seed": seed,
        "best_epoch": best_epoch,
        "checkpoint_path": str(v2_ckpt_path),
        "artifact_sha256": artifact_sha256,
        "b2_artifact_key": b2_artifact_key,
        "preprocessing_config": pre_cfg.to_dict(),
        "split_subjects": split_subjects,
        "training_history": history,
        "evaluation_metrics": evaluation_metrics,
        "subgroup_metrics": subgroup_metrics,
    }

    report_out = BASE_DIR / "results" / "reports" / "training_evaluation_report.json"
    report_v2_out = BASE_DIR / "results" / "reports" / "training_evaluation_report_v2.0.json"
    report_v1_bak = BASE_DIR / "results" / "reports" / "training_evaluation_report_v1.0_ORIGINAL.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    if report_out.exists() and not report_v1_bak.exists():
        report_v1_bak.write_text(report_out.read_text(encoding="utf-8"), encoding="utf-8")
    report_v2_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return report_payload
