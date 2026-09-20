"""
TideTrace Unified SAR Segmentation Inference Engine.
Combines deep learning PyTorch U-Net with automatic, seamless fallback to the
adaptive heuristic detector when neural weights are uninitialized, missing, or on CPU.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
from scipy import ndimage

from app.config import (
    MODEL_PATH,
    DEVICE,
    TILE_SIZE,
    TILE_OVERLAP,
    MIN_SLICK_PIXELS,
    CLASS_SEA,
    CLASS_LOOKALIKE,
    CLASS_MINERAL_OIL,
)
from app.ml.model import TideTraceUNet
from app.ml.preprocess import (
    normalize_sar_bands,
    extract_tiles,
    stitch_tiles,
    to_decibels,
)
from app.ml.fallback import run_adaptive_heuristic_detector


class SARSegmenter:
    """
    Unified 3-Class SAR Segmentation Pipeline.
    Manages neural network lifecycle and fallback execution.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        force_fallback: bool = False,
    ):
        self.device = device or DEVICE
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.force_fallback = force_fallback
        self.model: Optional[TideTraceUNet] = None
        self.use_fallback = False
        self.status_message = ""

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Attempt to load trained PyTorch checkpoint; fall back gracefully if unavailable."""
        if self.force_fallback:
            self.use_fallback = True
            self.status_message = "Fallback active (manually forced)."
            return

        if not self.model_path.exists():
            self.use_fallback = True
            self.status_message = f"Weights not found at {self.model_path}; fallback active."
            return

        try:
            self.model = TideTraceUNet.load_checkpoint(self.model_path, device=self.device)
            self.use_fallback = False
            self.status_message = f"Loaded PyTorch U-Net checkpoint from {self.model_path} on {self.device}."
        except Exception as e:
            self.use_fallback = True
            self.status_message = f"Failed to load checkpoint ({e}); fallback active."

    def predict(
        self,
        vv_data: np.ndarray,
        vh_data: Optional[np.ndarray] = None,
        batch_size: int = 4,
        return_probs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Execute full-scene 3-class segmentation on SAR imagery.
        
        Args:
            vv_data: 2D numpy array of VV backscatter (linear or dB).
            vh_data: Optional 2D numpy array of VH backscatter.
            batch_size: Mini-batch size for GPU tile inference.
            return_probs: Whether to return full 3-channel probability map.

        Returns:
            class_mask: 2D uint8 array (H, W) where 0=Sea, 1=Lookalike, 2=Mineral Oil.
            confidence_map: 2D float32 array (H, W) with pixel-level confidence in [0.0, 1.0].
            prob_map: Optional 3D float32 array (3, H, W) containing class probabilities.
        """
        if self.use_fallback or self.model is None:
            mask, conf = run_adaptive_heuristic_detector(
                vv_data,
                vh_data,
                min_area_px=MIN_SLICK_PIXELS,
            )
            prob_map = None
            if return_probs:
                h, w = mask.shape
                prob_map = np.zeros((3, h, w), dtype=np.float32)
                for c in range(3):
                    c_mask = (mask == c)
                    prob_map[c, c_mask] = conf[c_mask] if c > 0 else 1.0 - conf[c_mask]
            return mask, conf, prob_map

        # Deep Learning Inference Pipeline
        norm_3ch = normalize_sar_bands(vv_data, vh_data)
        tiles, coords, orig_shape = extract_tiles(norm_3ch, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)

        num_tiles = tiles.shape[0]
        tile_probs_list = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, num_tiles, batch_size):
                batch_np = tiles[i : i + batch_size]
                batch_tensor = torch.from_numpy(batch_np).to(self.device)
                
                # Forward pass + softmax
                probs_tensor = self.model.predict_probabilities(batch_tensor)
                tile_probs_list.append(probs_tensor.cpu().numpy())

        tile_probs = np.concatenate(tile_probs_list, axis=0)
        prob_map, class_mask, confidence_map = stitch_tiles(
            tile_probs,
            coords,
            orig_shape,
            num_classes=3,
            tile_size=TILE_SIZE,
        )

        # Post-process: Filter connected components smaller than threshold
        class_mask, confidence_map = self._post_process_mask(class_mask, confidence_map)

        if not return_probs:
            prob_map = None

        return class_mask, confidence_map, prob_map

    def _post_process_mask(
        self,
        mask: np.ndarray,
        conf: np.ndarray,
        min_pixels: int = MIN_SLICK_PIXELS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove isolated speckle and tiny sub-threshold connected components."""
        clean_mask = mask.copy()
        clean_conf = conf.copy()

        for cls_id in [CLASS_LOOKALIKE, CLASS_MINERAL_OIL]:
            binary = (clean_mask == cls_id)
            labeled, num_comp = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
            if num_comp == 0:
                continue

            sizes = ndimage.sum(binary, labeled, range(1, num_comp + 1))
            for comp_idx, size in enumerate(sizes, start=1):
                if size < min_pixels:
                    small_blob = (labeled == comp_idx)
                    clean_mask[small_blob] = CLASS_SEA
                    clean_conf[small_blob] = 0.0

        return clean_mask, clean_conf

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic operational status of the segmentation engine."""
        return {
            "mode": "adaptive_heuristic_fallback" if self.use_fallback else "deep_learning_unet",
            "using_fallback": self.use_fallback,
            "device": self.device,
            "model_path": str(self.model_path),
            "status_message": self.status_message,
            "param_count": self.model.count_parameters() if self.model else 0,
            "model_size_mb": self.model.get_model_size_mb() if self.model else 0.0,
        }
