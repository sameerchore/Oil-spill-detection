# E2E Test Infra: TideTrace

## 1. Test Philosophy
- Opaque-box, requirement-driven testing directly derived from `ORIGINAL_REQUEST.md` and `PROJECT.md`.
- Zero dependency on internal implementation details; tests exercise public interfaces and CLI/REST endpoints.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial + Real-World Workload Scenarios.
- 100% Offline Air-Gapped Verification: Network calls monkeypatched or isolated to guarantee failure-free offline execution.

---

## 2. Feature Inventory Mapping

| # | Feature | Requirement Source | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Scenario) |
|---|---------|-------------------|:-----------------:|:-----------------:|:----------------------:|:-----------------:|
| F01 | SAR Tiling & Preprocess | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F02 | U-Net 3-Class Segmentation | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F03 | Heuristic Fallback Detector | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F04 | Morphological Characterisation | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F05 | GeoJSON Output Format | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F06 | Lagrangian Numerical Advection | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F07 | Backward Origin Backtracking | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F08 | Forward Dispersion Forecast | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F09 | Offline Metocean Grid Store | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F10 | MarineCadastre SQLite Indexing | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F11 | Spatio-Temporal Join & Interpolation | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F12 | Dark Vessel Gap Detection | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F13 | Multi-Factor Attribution Scoring | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F14 | Synthetic AIS Scenarios | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F15 | FastAPI REST Endpoints | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F16 | Full-Screen Leaflet UI | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F17 | Time-Slider Scrubber | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F18 | Suspect Leaderboard & Logs | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F19 | ReportLab PDF Generator | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F20 | Kaggle Notebook Specification | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F21 | Hugging Face Sync Pipeline | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F22 | Pre-cached Demo Fixtures | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F23 | Offline Test Suite Execution | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |
| F24 | Anti-Slop Forensic Verification | PROJECT.md §2 | 5 tests | 5 tests | ✓ | ✓ |

---

## 3. Test Architecture & Runner
- Test runner invocation: `pytest -v tests/`
- Core Unit Test Files:
  - `tests/test_advection.py` (Advection speed, Coriolis angle, backward stepping)
  - `tests/test_geo_centroid.py` (WGS84 projection, AEQD metric area, PCA orientation, centroid error)
  - `tests/test_ais_gap.py` (AIS silence detection $\Delta t \ge 30\text{ min}$, dead-reckoning trajectory intersection)
  - `tests/test_scoring.py` (Proximity decay, vessel type risk, score monotonicity, rank ordering)
  - `tests/test_pipeline_offline.py` (End-to-end `/api/run` execution on offline scene with zero network)

---

## 4. Real-World Application Scenarios (Tier 4)

| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Dark Tanker in Arabian Sea Off Mumbai | F01-F05, F06-F09, F10-F14, F15, F19, F23 | High |
| 2 | Biogenic Natural Slick (Look-Alike Only) in Bay of Bengal | F01-F05, F06, F10-F14, F15, F23 (Empty attribution list returned) | Medium |
| 3 | Multi-Vessel Convoys in Singapore Strait | F06-F09, F10-F14, F15, F18 (High traffic density disambiguation) | High |
| 4 | Forward Dispersion Emergency Response (36h Drift to Coast) | F06, F08, F09, F15, F16, F19 | High |
| 5 | Air-Gapped Forensic PDF Dossier Export | F10-F14, F15, F19, F23 | Medium |
