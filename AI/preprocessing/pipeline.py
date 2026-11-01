from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    target_height: int = 256
    target_width: int = 256
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    normalize_mean: float = 0.25
    normalize_std: float = 0.22
    min_valid_dimension: int = 64

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clahe_tile_grid_size"] = list(self.clahe_tile_grid_size)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        grid = data.get("clahe_tile_grid_size", [8, 8])
        return cls(
            target_height=int(data.get("target_height", 256)),
            target_width=int(data.get("target_width", 256)),
            apply_clahe=bool(data.get("apply_clahe", True)),
            clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=(int(grid[0]), int(grid[1])),
            normalize_mean=float(data.get("normalize_mean", 0.25)),
            normalize_std=float(data.get("normalize_std", 0.22)),
            min_valid_dimension=int(data.get("min_valid_dimension", 64)),
        )


@dataclass
class ImageQualityAssessment:
    width: int
    height: int
    quality_score: float
    sharpness_laplacian: float
    contrast_std: float
    mean_intensity: float
    dark_shadow_ratio: float
    is_acceptable: bool
    warnings: List[str] = field(default_factory=list)


def assess_ultrasound_image_quality(
    image: np.ndarray,
    min_dimension: int = 64,
) -> ImageQualityAssessment:
    """
    Validate ultrasound image dimensions and compute an objective quality score (0-100)
    with explicit quality-control warnings for blur, low contrast, or acoustic shadowing.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid or empty ultrasound image array.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image array shape: {image.shape}")

    h, w = gray.shape[:2]
    warnings: List[str] = []

    if h < min_dimension or w < min_dimension:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are below minimum required ({min_dimension}x{min_dimension})."
        )

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_std = float(np.std(gray))
    mean_int = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 10))

    if lap_var < 15.0:
        warnings.append("Low image sharpness / potential motion blur detected.")
    if contrast_std < 20.0:
        warnings.append("Low acoustic dynamic range / contrast detected.")
    if dark_ratio > 0.75:
        warnings.append("High acoustic shadowing or narrow ultrasound sector fan (>75% dark pixels).")
    if mean_int > 220.0:
        warnings.append("Excessive gain / saturation detected in ultrasound image.")

    # Composite quality score in [0, 100]
    sharp_score = min(1.0, lap_var / 120.0) * 35.0
    contrast_score = min(1.0, contrast_std / 55.0) * 40.0
    shadow_score = max(0.0, 1.0 - max(0.0, dark_ratio - 0.35) / 0.55) * 25.0
    total_score = round(float(np.clip(sharp_score + contrast_score + shadow_score, 0.0, 100.0)), 2)

    return ImageQualityAssessment(
        width=w,
        height=h,
        quality_score=total_score,
        sharpness_laplacian=round(lap_var, 2),
        contrast_std=round(contrast_std, 2),
        mean_intensity=round(mean_int, 2),
        dark_shadow_ratio=round(dark_ratio, 4),
        is_acceptable=(total_score >= 30.0),
        warnings=warnings,
    )


def preprocess_ultrasound_image(
    image: np.ndarray,
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Deterministic preprocessing shared identically between model training and inference.
    Returns:
      - tensor_chw: float32 numpy array of shape (1, target_height, target_width)
      - original_hw: (orig_height, orig_width)
    """
    cfg = config or PreprocessingConfig()
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess empty image.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    orig_h, orig_w = gray.shape[:2]
    if orig_h < cfg.min_valid_dimension or orig_w < cfg.min_valid_dimension:
        raise ValueError(f"Image resolution {orig_w}x{orig_h} is smaller than {cfg.min_valid_dimension}.")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    if cfg.apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_grid_size,
        )
        gray = clahe.apply(gray)

    resized = cv2.resize(
        gray,
        (cfg.target_width, cfg.target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - cfg.normalize_mean) / (cfg.normalize_std + 1e-8)
    tensor_chw = np.expand_dims(normalized.astype(np.float32), axis=0)
    return tensor_chw, (orig_h, orig_w)


def encode_segmentation_mask(
    mask_raw: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Convert raw IUGC2024 segmentation mask (0=Background, 127=Pubic Symphysis, 255=Fetal Head)
    into integer class labels {0, 1, 2} of shape (target_height, target_width).
    """
    if mask_raw.ndim == 3:
        mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_raw

    resized = cv2.resize(
        mask_gray,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    classes = np.zeros_like(resized, dtype=np.int64)
    classes[(resized >= 80) & (resized < 190)] = 1  # Class 1: Pubic Symphysis (PS)
    classes[resized >= 190] = 2  # Class 2: Fetal Head (FH)
    return classes
