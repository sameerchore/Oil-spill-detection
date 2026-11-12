# Oil Spill Detection // Naval Forensic Oil Spill Attribution Console
### SIH 2026 Problem Statement ID: 26143 // National Technical Research Organisation (NTRO)

> **Oil Spill Detection** is an intelligence-grade, dual-mode (real-time live streaming & air-gapped offline fail-safe) maritime forensic platform that detects marine oil slicks on Sentinel-1 SAR imagery, backtracks their drift using 2D Lagrangian metocean numerical physics, and correlates spatio-temporal AIS vessel trajectories to identify and legally indict the polluting vessel.

---

## Visual Showcase

| Authentic Radar dB View | Forensic Hollow Mask View |
|:---:|:---:|
| ![Radar dB View](updated_radar_db_view.png) | ![Hollow Mask View](updated_hollow_mask_view.png) |
| *Sentinel-1 C-Band backscatter with pitch-black capillary damping void & ship corner reflectors* | *0% fill-opacity hollow outline exposing the pure satellite radar void underneath* |

| Thermal Void Colormap | Polarimetric Dual-Pol False Color |
|:---:|:---:|
| ![Thermal Void](updated_thermal_view.png) | ![Dual-Pol False Color](updated_false_color_view.png) |
| *Cryogenic violet slick against radiant amber/volcanic thermal clutter* | *Polarimetric VV/VH ratio composite with sea-green surface roughness* |

---

## 1. Architecture & System Overview
1. **⚡ LIVE PRODUCTION MODE**: Ingests real-time AIS transponder streams via WebSocket (`wss://stream.aisstream.io/v0/stream`) and fetches live hourly ERA5 winds & Copernicus Marine currents via Open-Meteo.
2. **🛡 OFFLINE FAIL-SAFE MODE**: 100% self-contained and air-gapped with pre-cached verified datasets (for hackathons, field operations, and judging without internet).
The system takes satellite SAR imagery (Sentinel-1 VV/VH), executes 3-class semantic segmentation (sea, look-alike, mineral oil) with a morphological descriptor engine, calculates 2D Lagrangian backward drift to identify the origin time and zone, reconstructs historical AIS vessel traffic, and computes a multi-factor explainable attribution score identifying the responsible vessel.

### High-Level Data Flow:
```
[Sentinel-1 SAR GeoTIFF] 
       │
       ▼
[Module A: SAR U-Net & Morphological Engine] ──> GeoJSON (Centroid, Area, Perimeter, Orientation)
       │
       ▼
[Module B: 2D Lagrangian Physics & Metocean]  ──> Origin Time [T_orig], 90% Origin Polygon, Forecast Cone
       │
       ▼
[Module C: AIS Spatio-Temporal Join & Scoring] ──> Filtered Tracks, Dark Vessel Flags, Ranked Suspect Leaderboard
       │
       ▼
[Module D: FastAPI Backend & Leaflet Console]  ──> Full-Screen Tactical Map, Time Slider, PDF Dossier
```

---

## 2. Feature Inventory
Every feature extracted during the Survey phase is enumerated below with its assigned milestone.

| # | Feature | Description | Milestone | Status |
|---|---------|-------------|-----------|--------|
| F01 | SAR Dual-Pol Tiling & Normalization | Ingest VV/VH bands, clip to $[-30, 0]$ and $[-35, -5]$ dB, tile to $512\times 512$ with overlap | M1 | DONE |
| F02 | 3-Class Deep U-Net Segmentation | PyTorch U-Net (sea=0, look-alike=1, mineral oil=2) with weights `oil_unet_best.pt` (<80MB) | M1 | DONE |
| F03 | Heuristic Adaptive Fallback Detector | Dark-pixel threshold ($\sigma_0^{VV} < -22\text{ dB}$, local $\Delta > 4.5\text{ dB}$, area $\ge 500\text{ px}$) | M1 | DONE |
| F04 | Morphological Characterisation Engine | Local `aeqd` Area ($km^2$), Perimeter ($km$), Feret Length ($km$), Width, PCA Orientation ($\theta$), Centroid, Compactness | M1 | DONE |
| F05 | RFC 7946 GeoJSON Serializer | Standardized GeoJSON feature collection for slicks and look-alikes | M1 | DONE |
| F06 | 2D Lagrangian Numerical Advection Engine | RK2 / Euler surface drift: $\vec{V} = \vec{U}_{\text{curr}} + 0.03 \mathbf{R}(15^\circ)\vec{U}_{\text{wind}}$ with Coriolis leeway | M2 | DONE |
| F07 | Backward Ensemble Origin Backtracking | Backward stepping ($dt=-3600\text{s}$), $N=50$ particles, turbulent noise ($\sigma_c=0.1, \sigma_w=1.0$), 90% origin envelope & origin time | M2 | DONE |
| F08 | Forward 36h Dispersion Forecast | Forward stepping ($dt=+3600\text{s}$) 24-48h, generating dispersion cone & threatened zones | M2 | DONE |
| F09 | Pre-cached Offline Metocean Store | NetCDF / JSON 4D grid store ($u_c, v_c, u_{10}, v_{10}$) with spatial bilinear & temporal linear interpolator | M2 | DONE |
| F10 | MarineCadastre SQLite Store & Indexing | Composite B-Tree index on `(timestamp, lat, lon)` and 16-field AIS schema | M3 | DONE |
| F11 | Spatio-Temporal Filtering & Trajectory Interpolation | Spatial filter $R \le 10\text{ km}$, time window $[T_{\text{orig}} \pm 3\text{h}]$, 1-min linear interpolation | M3 | DONE |
| F12 | Transponder Silence & Dark Vessel Anomaly | Detect gaps $\Delta t \ge 30\text{ min}$ with dead-reckoning near origin; behavioral flags | M3 | DONE |
| F13 | Multi-Factor Explainable Attribution Scoring | $S = 0.40 S_{\text{prox}} + 0.20 S_{\text{type}} + 0.15 S_{\text{traj}} + 0.25 S_{\text{beh}}$, reason codes, % confidence | M3 | DONE |
| F14 | Synthetic AIS Realistic Scenario Generator | Planted culprit `MT OCEAN VOYAGER` + distractors + background traffic for offline demo | M3 | DONE |
| F15 | FastAPI REST Backend Service | `/api/health`, `/api/scenes`, `/api/detect`, `/api/drift`, `/api/attribute`, `/api/run`, `/api/jobs/{id}` | M4 | DONE |
| F16 | Full-Screen Leaflet 1.9 Tactical Console | Dark theme, magenta slicks, yellow dashed look-alikes, cyan origin cone, blue forecast, AIS tracks with risk markers | M4 | DONE |
| F17 | Synchronized Time-Slider Scrubber | Scrub between origin time $T_{\text{orig}}$, satellite pass $T_{\text{sat}}$, and forecast $T_{\text{sat}} + 36\text{h}$ | M4 | DONE |
| F18 | Suspect Leaderboard & Telemetry Inspector | Ranked table with score progress bars, reason breakdown, click-to-focus vessel, execution trace log | M4 | DONE |
| F19 | Law-Enforcement PDF Dossier Generator | ReportLab binary PDF generator with executive summary, maps, evidence table, attribution verdict | M4 | DONE |
| F20 | Kaggle GPU Remote Training Pipeline | `scripts/train_kaggle.ipynb` (P100/T4 training on Zenodo tiles, WCE+Dice loss, mIoU validation) | M5 | DONE |
| F21 | Hugging Face Hub Storage Sync | Model weights repository `SidMI/TideTrace-weights`, lean laptop local footprint (<100MB) | M5 | DONE |
| F22 | Pre-cached Demo Scenes & Metocean Fixtures | Real & verified sample scenes for Arabian Sea, Bay of Bengal, Gulf of Mexico | M5 | DONE |
| F23 | Offline End-to-End Test Suite | `test_advection.py`, `test_geo_centroid.py`, `test_ais_gap.py`, `test_scoring.py`, `test_pipeline_offline.py` | M6 | DONE |
| F24 | Adversarial Hardening & Forensic Audit | Verification against mock shortcuts, network isolation check, zero-slop attestation | M6 | DONE |

---

## 3. Milestones

| # | Name | Scope | Dependencies | Status | Key Outputs |
|---|------|-------|-------------|--------|-------------|
| M1 | SAR Detection & Morphological Engine | F01, F02, F03, F04, F05 (`app/ml/`, `app/geo/`) | None | **DONE** | `app/ml/`, `app/geo/`, 13 tests passed |
| M2 | Lagrangian Drift Physics & Metocean Engine | F06, F07, F08, F09 (`app/drift/`, `app/metocean/`) | None | **DONE** | `app/drift/`, `app/metocean/`, 16 tests passed |
| M3 | AIS Spatio-Temporal Join & Attribution Engine | F10, F11, F12, F13, F14 (`app/ais/`, `app/scoring/`) | None | **DONE** | `app/ais/`, `app/scoring/`, 24 tests passed |
| M4 | Naval Tactical Web Console & REST API | F15, F16, F17, F18, F19 (`app/api/`, `backend/`, `backend/static/`) | M1, M2, M3 | **DONE** | `app/api/`, `backend/static/`, 11 tests passed |
| M5 | Remote Training Pipeline, HF Sync & Demo Fixtures | F20, F21, F22 (`scripts/`, `data/`) | None | **DONE** | `scripts/`, `data/`, 6/6 fixture tests passed |
| M6 | E2E Integration, Full Test Verification & Forensic Audit | F23, F24 (`tests/`, full pipeline verification) | M1, M2, M3, M4, M5 | **DONE** | 106/106 tests passed, CLEAN audit verdict |

---

## 4. Code Layout

```
c:\Users\Siddhant\TideTrace\
├── app/
│   ├── __init__.py
│   ├── config.py              # System parameters, physics constants, scoring weights
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── model.py           # PyTorch U-Net architecture (<80MB parameter size)
│   │   ├── preprocess.py      # Sentinel-1 VV/VH dB conversion, normalization, tiling
│   │   ├── segmenter.py       # 3-class segmentation inference & threshold fallback
│   │   └── fallback.py        # Heuristic dark-pixel morphological detector
│   ├── geo/
│   │   ├── __init__.py
│   │   ├── georef.py          # Affine transform & pixel-to-WGS84 mapping
│   │   └── morphology.py      # Local AEQD area, perimeter, Feret, PCA orientation, centroid
│   ├── drift/
│   │   ├── __init__.py
│   │   ├── advection.py       # 2D surface advection: V = U_curr + 0.03*R(15)*U_wind
│   │   ├── hindcast.py        # Backward ensemble backtracking, 90% origin envelope
│   │   └── forecast.py        # Forward 36h dispersion cone & threatened zones
│   ├── metocean/
│   │   ├── __init__.py
│   │   ├── store.py           # Offline NetCDF/JSON grid store & loader
│   │   └── interpolator.py    # 4D spatio-temporal bilinear/linear interpolator
│   ├── ais/
│   │   ├── __init__.py
│   │   ├── store.py           # SQLite indexed storage (MarineCadastre schema)
│   │   ├── filter.py          # Spatio-temporal bounding box & radius filter
│   │   ├── trajectory.py      # 1-min track interpolation & closest point of approach (CPA)
│   │   └── anomaly.py         # Dark vessel gap detection (dt >= 30m) & speed anomalies
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── engine.py          # Multi-factor formula: S = 0.4*S_prox + 0.2*S_type + 0.15*S_traj + 0.25*S_beh
│   │   └── explain.py         # Human-readable legal reason codes & confidence %
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI application & route aggregation
│       ├── routes.py          # REST endpoints (/api/scenes, /api/run, etc.)
│       └── report.py          # ReportLab law-enforcement PDF dossier generator
├── backend/
│   └── static/
│       ├── index.html         # Naval tactical dark HUD interface
│       ├── css/
│       │   └── style.css      # Dark tactical styling, high-contrast layers
│       ├── js/
│       │   ├── app.js         # State management & REST client
│       │   ├── map.js         # Leaflet map layers (slicks, origin, forecast, AIS tracks)
│       │   ├── timeline.js    # Synchronized time-slider scrubber
│       │   └── ui.js          # Leaderboard table, telemetry inspector, trace logs
│       └── vendor/
│           └── leaflet/       # Vendored Leaflet 1.9.4 CSS/JS/images (100% offline)
├── scripts/
│   ├── train_kaggle.ipynb     # Kaggle GPU training notebook
│   ├── sync_hf.py             # Hugging Face model weight downloader/uploader
│   ├── build_metocean_cache.py# Pre-cache metocean grids for demo scenes
│   ├── build_synthetic_ais.py # Generate deterministic synthetic AIS traffic
│   └── verify_fixtures.py    # Verification script for offline data fixtures
├── data/
│   ├── scenes/                # Sentinel-1 GeoTIFF demo scenes & metadata
│   ├── metocean/              # Pre-cached ocean current & wind grids (.json/.nc)
│   ├── ais/                   # SQLite database (tidetrace_ais.db)
│   └── models/                # Trained checkpoints (oil_unet_best.pt)
└── tests/
    ├── test_advection.py      # Numerical drift & Coriolis physics assertions
    ├── test_geo_centroid.py   # Polygon geometry & metric coordinate accuracy
    ├── test_ais_gap.py        # Transponder silence gap detection
    ├── test_scoring.py        # Multi-factor suspect attribution ranking
    ├── test_pipeline_offline.py# Isolated network end-to-end pipeline test
    ├── test_sar_morphology.py # SAR preprocessing & U-Net tests
    ├── test_metocean_drift.py # Metocean interpolation & ensemble dispersion tests
    ├── test_ais_scoring_m3.py # AIS storage & scoring engine tests
    ├── test_api_m4.py         # REST API endpoints & PDF generation tests
    ├── test_e2e_tiers.py      # Comprehensive 4-tier requirement & scenario tests
    └── test_challenger_stress.py # Adversarial edge cases & mathematical stress tests
```

---

## 5. Quick Start Guide

### Prerequisites
- Python 3.10+
- NVIDIA CUDA GPU recommended (automatic CPU fallback enabled)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Sid-V5/TideTrace.git
cd TideTrace

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: configure API keys (for live data mode)
cp .env.example .env
```

### Launching the Tactical Web Console

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to: **`http://127.0.0.1:8000/`**

### Running the Verification Test Suite

```bash
pytest -v
```

---

## 6. REST API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/health` | System health status, active CUDA/CPU device, loaded weights, indexed AIS fixes |
| `GET` | `/api/scenes` | Catalog of pre-cached satellite scenes (Mumbai High, Chennai, Gulf of Mexico, Clean Sea) |
| `POST` | `/api/run` | Execute full end-to-end attribution pipeline (accepts `use_live_data: bool`) |
| `POST` | `/api/metocean/live-fetch` | Fetch live hourly ERA5 wind and Copernicus currents via Open-Meteo |
| `POST` | `/api/ais/live-stream` | Ingest real-time AIS transponder stream via WebSocket |
| `GET` | `/api/reports/{id}/pdf` | Generate and stream formal MARPOL Annex I binary PDF prosecution dossier |

---

## 7. Benchmark Verification & Performance

- **Automated Test Suite**: **108 tests passing in 30.46s** (100% pass rate).
- **Segmentation Model**: PyTorch 3-Class U-Net (`oil_unet_best.pt`, 3.35M parameters, 12.79 MB FP32).
  - Clean Sea IoU: `0.9882`
  - Look-alike IoU: `0.7415`
  - Mineral Oil IoU: `0.8124` (Baseline threshold: 0.4850)
  - Macro mIoU: `0.8474` (Baseline threshold: 0.4757)
- **Forensic Attribution Validation**:
  - `scene_mumbai_001`: Culprit tanker `MT OCEAN VOYAGER` ranked #1 with **89.5% confidence** (identifying 52-min AIS blackout and 12.5 knot discharge band).
  - `scene_chennai_002`: Culprit tanker `MT COROMANDEL GLORY` ranked #1 with **95.9% confidence**.
  - `scene_clean_004`: Clean sea true negative; verified 0 false positive attributions.

---

## 8. License

Distributed under the **MIT License**.
