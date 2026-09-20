"""
TideTrace SAR Preprocessing and Tiling Engine.
Ingests dual-polarization Sentinel-1 GeoTIFF / numpy arrays, performs decibel conversion,
radiometric clipping and normalization, and implements overlapping apodized tiling/stitching.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any

import numpy as np
import torch

from app.config import (
    VV_CLIP_MIN_DB,
    VV_CLIP_MAX_DB,
    VH_CLIP_MIN_DB,
    VH_CLIP_MAX_DB,
    TILE_SIZE,
    TILE_OVERLAP,
)


def to_decibels(linear_data: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Convert linear radar cross-section (sigma0) to logarithmic decibel (dB) scale.
    If input is already in dB (median < 5.0 and min < 0), returns input as-is.
    """
    data = np.asarray(linear_data, dtype=np.float32)
    # Heuristic check if array is already in dB
    if np.nanmedian(data) < 5.0 and np.nanmin(data) < 0.0:
        return data
    
    clean_data = np.nan_to_num(data, nan=eps, posinf=1.0, neginf=eps)
    clean_data = np.maximum(clean_data, eps)
    return 10.0 * np.log10(clean_data)


def normalize_sar_bands(
    vv_db: np.ndarray,
    vh_db: Optional[np.ndarray] = None,
    vv_range: Tuple[float, float] = (VV_CLIP_MIN_DB, VV_CLIP_MAX_DB),
    vh_range: Tuple[float, float] = (VH_CLIP_MIN_DB, VH_CLIP_MAX_DB),
) -> np.ndarray:
    """
    Normalize VV and VH backscatter channels to [0.0, 1.0] and compose 3-channel tensor.
    Channel composition:
      - Ch 0: Normalized VV
      - Ch 1: Normalized VH (or synthetic cross-pol if VH is None)
      - Ch 2: Channel difference |VV_norm - VH_norm| or replicated VV_norm
    
    Returns:
        Array of shape (3, H, W) normalized to [0.0, 1.0].
    """
    vv_db = to_decibels(vv_db)
    vv_min, vv_max = vv_range
    vv_norm = np.clip((vv_db - vv_min) / (vv_max - vv_min), 0.0, 1.0)

    if vh_db is not None:
        vh_db = to_decibels(vh_db)
        vh_min, vh_max = vh_range
        vh_norm = np.clip((vh_db - vh_min) / (vh_max - vh_min), 0.0, 1.0)
        diff_norm = np.abs(vv_norm - vh_norm)
    else:
        # Synthesize second channel via smoothed contrast and replicate
        vh_norm = vv_norm.copy()
        diff_norm = np.clip(vv_norm * 0.5, 0.0, 1.0)

    tensor_3ch = np.stack([vv_norm, vh_norm, diff_norm], axis=0).astype(np.float32)
    return tensor_3ch


def create_hann_window(tile_size: int = TILE_SIZE) -> np.ndarray:
    """
    Create a 2D Hanning/Cosine apodization window to eliminate seam artifacts at tile edges.
    Returns:
        Array of shape (tile_size, tile_size) with values in (0.001, 1.0].
    """
    w1d = np.hanning(tile_size)
    w2d = np.outer(w1d, w1d)
    return np.maximum(w2d, 1e-4).astype(np.float32)


def compute_tile_grid(
    height: int,
    width: int,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate grid coordinates (r_start, c_start, r_end, c_end) for sliding window tiling.
    Guarantees 100% full coverage including edges.
    """
    stride = tile_size - overlap
    assert stride > 0, "Tile size must be greater than overlap."

    r_starts = list(range(0, max(1, height - tile_size + 1), stride))
    if not r_starts or r_starts[-1] + tile_size < height:
        r_starts.append(max(0, height - tile_size))

    c_starts = list(range(0, max(1, width - tile_size + 1), stride))
    if not c_starts or c_starts[-1] + tile_size < width:
        c_starts.append(max(0, width - tile_size))

    # Remove duplicates while preserving order
    r_starts = sorted(list(set(r_starts)))
    c_starts = sorted(list(set(c_starts)))

    coords = []
    for r in r_starts:
        for c in c_starts:
            coords.append((r, c, min(height, r + tile_size), min(width, c + tile_size)))
    return coords


def extract_tiles(
    image_3ch: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Extract padded overlapping tiles of shape (N, 3, tile_size, tile_size).
    Args:
        image_3ch: Multi-channel array of shape (3, H, W)
    Returns:
        tiles: Array of shape (N, 3, tile_size, tile_size)
        coords: List of (r_start, c_start, r_end, c_end)
        orig_shape: (H, W)
    """
    _, h, w = image_3ch.shape
    coords = compute_tile_grid(h, w, tile_size, overlap)

    tiles_list = []
    for (r0, c0, r1, c1) in coords:
        patch = image_3ch[:, r0:r1, c0:c1]
        ph, pw = patch.shape[1], patch.shape[2]
        
        # Pad if tile is smaller than tile_size at edge
        if ph < tile_size or pw < tile_size:
            pad_h = tile_size - ph
            pad_w = tile_size - pw
            patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            
        tiles_list.append(patch)

    tiles = np.stack(tiles_list, axis=0).astype(np.float32)
    return tiles, coords, (h, w)


def stitch_tiles(
    tile_probs: np.ndarray,
    coords: List[Tuple[int, int, int, int]],
    out_shape: Tuple[int, int],
    num_classes: int = 3,
    tile_size: int = TILE_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stitch overlapping tile class probabilities using 2D Hanning apodization.
    Args:
        tile_probs: Array of shape (N, num_classes, tile_size, tile_size)
        coords: List of tile bounding coordinates (r0, c0, r1, c1)
        out_shape: Full image spatial shape (H, W)
        num_classes: Number of segmentation classes (3)
        tile_size: Tile dimension (512)
    Returns:
        prob_map: Stitched probability map (num_classes, H, W)
        class_mask: Argmax class segmentation mask (H, W) with values in {0, 1, 2}
        confidence_map: Maximum probability map (H, W)
    """
    h, w = out_shape
    accum_prob = np.zeros((num_classes, h, w), dtype=np.float32)
    accum_weight = np.zeros((1, h, w), dtype=np.float32)
    hann = create_hann_window(tile_size)[np.newaxis, ...] # (1, tile_size, tile_size)

    for i, (r0, c0, r1, c1) in enumerate(coords):
        ph = r1 - r0
        pw = c1 - c0
        prob_patch = tile_probs[i, :, :ph, :pw]
        weight_patch = hann[:, :ph, :pw]

        accum_prob[:, r0:r1, c0:c1] += prob_patch * weight_patch
        accum_weight[:, r0:r1, c0:c1] += weight_patch

    # Normalize by accumulated weights
    prob_map = accum_prob / np.maximum(accum_weight, 1e-7)
    
    # Argmax and confidence
    class_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
    confidence_map = np.max(prob_map, axis=0).astype(np.float32)

    return prob_map, class_mask, confidence_map


def load_sar_image(
    file_path: Union[str, Path]
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Ingest a SAR GeoTIFF or numpy raster file.
    Supports single-band (VV) or dual-band (VV, VH) images.
    Returns:
        vv_band: 2D array of VV backscatter
        vh_band: 2D array of VH backscatter (or None)
        metadata: Metadata dictionary (transform, crs, shape, etc.)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SAR file not found: {path}")

    metadata: Dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path),
    }

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            return arr.astype(np.float32), None, metadata
        elif arr.ndim == 3:
            return arr[0].astype(np.float32), arr[1].astype(np.float32), metadata
        else:
            raise ValueError(f"Invalid numpy array shape: {arr.shape}")

    # Try loading via tifffile
    try:
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            return arr.astype(np.float32), None, metadata
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 2, 3):
                vv = arr[0].astype(np.float32)
                vh = arr[1].astype(np.float32) if arr.shape[0] > 1 else None
            else:
                vv = arr[:, :, 0].astype(np.float32)
                vh = arr[:, :, 1].astype(np.float32) if arr.shape[2] > 1 else None
            return vv, vh, metadata
    except Exception:
        pass

    # Fallback to PIL
    try:
        from PIL import Image
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                return arr, None, metadata
            elif arr.ndim == 3:
                return arr[:, :, 0], arr[:, :, 1] if arr.shape[2] > 1 else None, metadata
    except Exception as e:
        raise IOError(f"Failed to read image at {path}: {e}")

    raise IOError(f"Could not parse image format for {path}")
