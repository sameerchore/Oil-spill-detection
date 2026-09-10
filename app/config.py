"""
TideTrace Central Configuration
Air-gapped, offline-first Naval Forensic Oil Spill Attribution Console.
Defines all physical, scoring, detection, and pipeline parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# 1. BASE DIRECTORIES & FILE PATHS
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
EXTERNAL_DATA_DIR: Path = Path("D:/TTdata") if Path("D:/TTdata").exists() else DATA_DIR
SCENES_DIR: Path = DATA_DIR / "scenes"
MODELS_DIR: Path = DATA_DIR / "models"
METOCEAN_DIR: Path = DATA_DIR / "metocean"
AIS_DIR: Path = DATA_DIR / "ais"
JOBS_DIR: Path = DATA_DIR / "jobs"

MODEL_PATH: Path = MODELS_DIR / "oil_unet_best.pt"
AIS_DB_PATH: Path = AIS_DIR / "ais.sqlite"

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Live API Credentials & Tokens
ZENODO_API_TOKEN: str = os.environ.get("ZENODO_TOKEN", "")
ZENODO_USER_ID: str = os.environ.get("ZENODO_USER_ID", "Sid-V5")
AISSTREAM_API_KEY: str = os.environ.get("AISSTREAM_API_KEY", "")

# Ensure runtime directories exist
for directory in [DATA_DIR, SCENES_DIR, MODELS_DIR, METOCEAN_DIR, AIS_DIR, JOBS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. SAR SENSOR & DETECTION PARAMETERS (MODULE A)
# ==============================================================================
# Logarithmic backscatter thresholds
DARK_THRESHOLD_VV_DB: float = -22.0          # Maximum sigma0_VV for dark water / oil
CONTRAST_DIFF_THRESHOLD_DB: float = 4.5      # Minimum local contrast reduction (dB)
VV_CLIP_MIN_DB: float = -30.0               # Minimum clipping bound for VV dB
VV_CLIP_MAX_DB: float = 0.0                 # Maximum clipping bound for VV dB
VH_CLIP_MIN_DB: float = -35.0               # Minimum clipping bound for VH dB
VH_CLIP_MAX_DB: float = -5.0                # Maximum clipping bound for VH dB

# Tiling & inference parameters
TILE_SIZE: int = 512                         # Square tile dimension in pixels
TILE_OVERLAP: int = 64                       # Tile overlap stride boundary in pixels
MIN_SLICK_PIXELS: int = 50                  # Connected component minimum pixel area filter
MIN_SLICK_AREA_KM2: float = 0.05            # Minimum geographic area (km^2) threshold

# Class definitions
CLASS_SEA: int = 0
CLASS_LOOKALIKE: int = 1
CLASS_MINERAL_OIL: int = 2
NUM_CLASSES: int = 3
CLASS_NAMES: Dict[int, str] = {
    CLASS_SEA: "sea",
    CLASS_LOOKALIKE: "lookalike",
    CLASS_MINERAL_OIL: "mineral_oil",
}

# ==============================================================================
# 3. LAGRANGIAN METOCEAN DRIFT & PHYSICS PARAMETERS (MODULE B)
# ==============================================================================
WIND_FACTOR: float = 0.03                    # Leeway wind drift coefficient (alpha = 3%)
CORIOLIS_DEFLECTION_DEG: float = 15.0        # Northern Hemisphere Ekman/leeway deflection
DEFAULT_HINDCAST_HOURS: int = 48             # Default backward trajectory advection window
DEFAULT_FORECAST_HOURS: int = 36             # Default forward dispersion forecast window
DEFAULT_ENSEMBLE_PARTICLES: int = 50         # Monte Carlo particle ensemble size
CURRENT_TURBULENCE_SIGMA: float = 0.1        # Ocean current stochastic noise (m/s)
WIND_TURBULENCE_SIGMA: float = 1.0           # Atmospheric wind stochastic noise (m/s)
INTEGRATION_STEP_SECONDS: float = 3600.0     # Numerical integration time-step (1 hour)
ORIGIN_CONFIDENCE_PERCENTILE: float = 90.0   # Confidence contour envelope for origin zone

# ==============================================================================
# 4. AIS SPATIO-TEMPORAL JOIN & ATTRIBUTION SCORING (MODULE C)
# ==============================================================================
SEARCH_RADIUS_KM: float = 10.0               # Maximum spatial search radius around origin
TEMPORAL_WINDOW_HOURS: float = 3.0           # Temporal search window around origin [T_orig +- W]
GAP_THRESHOLD_MINUTES: float = 30.0          # Transponder silence anomaly threshold (minutes)
PROXIMITY_DECAY_D0_KM: float = 3.0           # Exponential decay scale distance for CPA score

# Attribution Formula Weights: S = 0.40*S_prox + 0.20*S_type + 0.15*S_traj + 0.25*S_beh
WEIGHT_PROX: float = 0.40                    # Proximity (CPA) weight
WEIGHT_TYPE: float = 0.20                    # Vessel risk classification weight
WEIGHT_TRAJ: float = 0.15                    # Trajectory & heading alignment weight
WEIGHT_BEH: float = 0.25                     # AIS silence / speed anomaly behavioral weight

# High-risk vessel types
TANKER_TYPES = [
    "Crude Oil Tanker",
    "Oil Products Tanker",
    "Chemical Tanker",
    "LNG Tanker",
    "LPG Tanker",
    "Bunker Vessel",
    "Tanker",
]

# ==============================================================================
# 5. HARDWARE & RUNTIME ENVIRONMENT
# ==============================================================================
try:
    import torch
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE: str = "cpu"

DEMO_METOCEAN: bool = os.getenv("DEMO_METOCEAN", "1").lower() in ("1", "true", "yes")
DEMO_AIS: bool = os.getenv("DEMO_AIS", "1").lower() in ("1", "true", "yes")
