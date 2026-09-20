"""
TideTrace Autonomous Heuristic Adaptive Dark-Pixel Detector.
Deterministic, physics-based mathematical fallback detector executing when neural network
weights are uninitialized or unavailable. Operates via local contrast thresholding,
morphological spatial filtering, and shape-factor classification.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage

from app.config import (
    DARK_THRESHOLD_VV_DB,
    CONTRAST_DIFF_THRESHOLD_DB,
    MIN_SLICK_PIXELS,
    CLASS_SEA,
    CLASS_LOOKALIKE,
    CLASS_MINERAL_OIL,
)
from app.ml.preprocess import to_decibels


def compute_local_background_stats(
    vv_db: np.ndarray,
    kernel_size: int = 65,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute moving window mean and standard deviation across ambient sea backscatter.
    Uses fast uniform spatial filtering.
    """
    # Uniform filter calculates local mean
    local_mean = ndimage.uniform_filter(vv_db, size=kernel_size, mode="reflect")
    # Local variance: E[X^2] - (E[X])^2
    local_sq_mean = ndimage.uniform_filter(vv_db ** 2, size=kernel_size, mode="reflect")
    local_var = np.maximum(local_sq_mean - (local_mean ** 2), 1e-4)
    local_std = np.sqrt(local_var)
    return local_mean, local_std


def run_adaptive_heuristic_detector(
    vv_data: np.ndarray,
    vh_data: Optional[np.ndarray] = None,
    dark_threshold_db: float = DARK_THRESHOLD_VV_DB,
    contrast_diff_db: float = CONTRAST_DIFF_THRESHOLD_DB,
    min_area_px: int = MIN_SLICK_PIXELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute adaptive dark-pixel segmentation on SAR backscatter.
    
    Pipeline:
      1. Convert backscatter to dB if linear.
      2. Estimate local moving background mean (mu_sea) and standard deviation (sigma_sea).
      3. Flag candidate dark pixels: (sigma0_VV < -22 dB) AND (mu_sea - sigma0_VV > 4.5 dB).
      4. Morphological Opening (3x3) to remove single-pixel radar speckle.
      5. Morphological Closing (5x5) to bridge wave-fragmented slick filaments.
      6. Connected Component Analysis:
         - Discard tiny blobs (< min_area_px).
         - Calculate bounding box, aspect ratio, and compactness.
         - High aspect ratio / sharp contrast -> Class 2 (Mineral Oil).
         - Low aspect ratio / broad calm zone -> Class 1 (Look-alike).
      7. Return 3-class segmentation mask and estimated confidence map.

    Returns:
        class_mask: uint8 array of shape (H, W) with {0: Sea, 1: Lookalike, 2: Mineral Oil}.
        confidence_map: float32 array of shape (H, W) in [0.0, 1.0].
    """
    vv_db = to_decibels(vv_data)
    h, w = vv_db.shape

    # 1. Background statistics estimation (local window and regional sea floor)
    local_mean, local_std = compute_local_background_stats(vv_db, kernel_size=65)
    local_contrast = local_mean - vv_db
    sea_level = float(np.percentile(vv_db, 75))

    # 2. Adaptive dark thresholding
    # Candidate pixel must be below absolute dark threshold AND have significant contrast drop relative to sea
    candidate_mask = (vv_db < dark_threshold_db) & (
        (local_contrast > contrast_diff_db) | ((sea_level - vv_db) > contrast_diff_db)
    )

    # 3. Morphological Speckle Cleaning (Opening 3x3)
    k_open = np.ones((3, 3), dtype=bool)
    opened = ndimage.binary_opening(candidate_mask, structure=k_open)

    # 4. Morphological Filament Bridging (Closing 5x5)
    # Elliptical / circular structuring element for natural fluid boundary connection
    y, x = np.ogrid[-2:3, -2:3]
    k_close = (x**2 + y**2 <= 4)
    cleaned = ndimage.binary_closing(opened, structure=k_close)

    # 5. Connected Component Analysis
    labeled_array, num_features = ndimage.label(cleaned, structure=np.ones((3, 3), dtype=int))
    
    class_mask = np.zeros((h, w), dtype=np.uint8)
    confidence_map = np.zeros((h, w), dtype=np.float32)

    if num_features == 0:
        return class_mask, confidence_map

    # Extract component properties
    component_sizes = ndimage.sum(cleaned, labeled_array, range(1, num_features + 1))
    component_slices = ndimage.find_objects(labeled_array)

    for i, (size, slc) in enumerate(zip(component_sizes, component_slices), start=1):
        if size < min_area_px or slc is None:
            continue

        comp_mask = (labeled_array[slc] == i)
        
        # Bounding box dimensions
        r_slice, c_slice = slc
        box_h = r_slice.stop - r_slice.start
        box_w = c_slice.stop - c_slice.start
        aspect_ratio = max(box_h, box_w) / max(min(box_h, box_w), 1)

        # Local contrast strength within component
        comp_contrast = np.mean(local_contrast[slc][comp_mask])
        comp_vv = np.mean(vv_db[slc][comp_mask])

        # Confidence calculation based on contrast drop and absolute darkness
        # A true oil slick has VV < -24 dB and contrast > 6 dB
        contrast_score = np.clip((comp_contrast - 3.0) / 7.0, 0.5, 0.98)
        darkness_score = np.clip((-comp_vv - 18.0) / 10.0, 0.5, 0.98)
        comp_conf = float(0.5 * contrast_score + 0.5 * darkness_score)

        # Classification decision logic:
        # Mineral oil typically presents as narrow, high-aspect-ratio elongated filaments (AR >= 1.8),
        # or dark patches with sharp contrast damping. Look-alikes are broad, circular low-wind calm zones.
        is_mineral_oil = (
            (aspect_ratio >= 1.8 and comp_contrast >= 4.0) or
            (aspect_ratio >= 1.5 and comp_contrast >= 5.0 and size < 10000) or
            (aspect_ratio >= 2.5)
        ) and not (aspect_ratio < 1.5 and size > 5000)

        if is_mineral_oil:
            class_mask[slc][comp_mask] = CLASS_MINERAL_OIL
            confidence_map[slc][comp_mask] = comp_conf
        else:
            class_mask[slc][comp_mask] = CLASS_LOOKALIKE
            confidence_map[slc][comp_mask] = comp_conf * 0.85 # Lookalike confidence

    return class_mask, confidence_map
