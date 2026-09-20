"""
TideTrace SAR Machine Learning and Detection Subpackage.
"""

from app.ml.model import TideTraceUNet, DoubleConv, Down, Up, OutConv
from app.ml.preprocess import (
    to_decibels,
    normalize_sar_bands,
    create_hann_window,
    compute_tile_grid,
    extract_tiles,
    stitch_tiles,
    load_sar_image,
)
from app.ml.fallback import run_adaptive_heuristic_detector
from app.ml.segmenter import SARSegmenter

__all__ = [
    "TideTraceUNet",
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
    "to_decibels",
    "normalize_sar_bands",
    "create_hann_window",
    "compute_tile_grid",
    "extract_tiles",
    "stitch_tiles",
    "load_sar_image",
    "run_adaptive_heuristic_detector",
    "SARSegmenter",
]
