"""
Metocean Data Store & Loader for TideTrace.
Provides offline access to ocean surface currents (u_c, v_c) and 10m atmospheric winds (u_10, v_10).
Supports pre-cached JSON, NPZ, and NetCDF formats covering T_sat +/- 48h,
with a deterministic, physically realistic synthetic fallback for air-gapped demo reliability.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def parse_timestamp(t: Union[str, datetime, int, float]) -> float:
    """Convert string ISO, datetime, or numerical timestamp to seconds since Unix epoch."""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        # Handle trailing Z or timezone offsets
        clean_str = t.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return float(t)
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.timestamp()
    raise TypeError(f"Cannot parse timestamp from type {type(t)}: {t}")


def format_iso_timestamp(ts: float) -> str:
    """Format Unix timestamp as ISO-8601 UTC string."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class MetoceanGrid:
    """
    4D Metocean dataset grid structure covering a temporal and spatial bounding domain.

    Arrays:
        times: 1D array of Unix timestamps in seconds (shape: [T]), strictly increasing.
        lats: 1D array of latitude coordinates in degrees (shape: [M]), strictly increasing.
        lons: 1D array of longitude coordinates in degrees (shape: [K]), strictly increasing.
        u_current: 3D array of eastward surface ocean current in m/s (shape: [T, M, K]).
        v_current: 3D array of northward surface ocean current in m/s (shape: [T, M, K]).
        u_wind: 3D array of eastward 10m wind velocity in m/s (shape: [T, M, K]).
        v_wind: 3D array of northward 10m wind velocity in m/s (shape: [T, M, K]).
        metadata: Dictionary with dataset provenance, scene ID, units, and timestamps.
    """
    times: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    u_current: np.ndarray
    v_current: np.ndarray
    u_wind: np.ndarray
    v_wind: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.lats = np.asarray(self.lats, dtype=np.float64)
        self.lons = np.asarray(self.lons, dtype=np.float64)
        self.u_current = np.asarray(self.u_current, dtype=np.float64)
        self.v_current = np.asarray(self.v_current, dtype=np.float64)
        self.u_wind = np.asarray(self.u_wind, dtype=np.float64)
        self.v_wind = np.asarray(self.v_wind, dtype=np.float64)

        # Validation
        if self.times.ndim != 1 or self.lats.ndim != 1 or self.lons.ndim != 1:
            raise ValueError("times, lats, and lons must be 1D arrays.")
        expected_shape = (len(self.times), len(self.lats), len(self.lons))
        for name, arr in [
            ("u_current", self.u_current),
            ("v_current", self.v_current),
            ("u_wind", self.u_wind),
            ("v_wind", self.v_wind),
        ]:
            if arr.shape != expected_shape:
                raise ValueError(f"{name} shape {arr.shape} does not match expected {expected_shape}")

    @property
    def time_range(self) -> Tuple[float, float]:
        """Return (min_timestamp, max_timestamp)."""
        return float(self.times[0]), float(self.times[-1])

    @property
    def lat_range(self) -> Tuple[float, float]:
        """Return (min_lat, max_lat)."""
        return float(self.lats[0]), float(self.lats[-1])

    @property
    def lon_range(self) -> Tuple[float, float]:
        """Return (min_lon, max_lon)."""
        return float(self.lons[0]), float(self.lons[-1])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (n_times, n_lats, n_lons)."""
        return (len(self.times), len(self.lats), len(self.lons))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grid to standard Python dictionary."""
        return {
            "metadata": self.metadata,
            "timestamps": self.times.tolist(),
            "times_iso": [format_iso_timestamp(ts) for ts in self.times],
            "lats": self.lats.tolist(),
            "lons": self.lons.tolist(),
            "u_current": self.u_current.tolist(),
            "v_current": self.v_current.tolist(),
            "u_wind": self.u_wind.tolist(),
            "v_wind": self.v_wind.tolist(),
        }


def create_synthetic_metocean_grid(
    center_lat: float = 18.9,
    center_lon: float = 72.8,
    t_center: Optional[Union[str, datetime, float]] = None,
    hours_before: int = 48,
    hours_after: int = 48,
    resolution_deg: float = 0.1,
    grid_radius_deg: float = 1.5,
    base_u_current: float = 0.22,
    base_v_current: float = -0.12,
    base_u_wind: float = 4.5,
    base_v_wind: float = -2.8,
    tidal_amplitude: float = 0.18,
    wind_gust_amplitude: float = 1.2,
    scene_id: Optional[str] = None,
) -> MetoceanGrid:
    """
    Generate a physically realistic synthetic metocean grid for offline testing and air-gapped demo runs.
    Includes:
    - Background geostrophic ocean current with mesoscale spatial gradient.
    - Semi-diurnal M2 tidal oscillation (period = 12.42 hours).
    - Synoptic atmospheric wind with diurnal sea-breeze oscillation (period = 24.0 hours).
    - Spatial vorticity variation across latitude and longitude.
    """
    if t_center is None:
        t_ref = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    else:
        t_ref = parse_timestamp(t_center)

    # Hourly timestamps from t_ref - hours_before to t_ref + hours_after
    num_hours = hours_before + hours_after + 1
    times = np.linspace(
        t_ref - hours_before * 3600.0,
        t_ref + hours_after * 3600.0,
        num_hours,
        dtype=np.float64,
    )

    # Spatial coordinates
    lats = np.arange(
        center_lat - grid_radius_deg,
        center_lat + grid_radius_deg + resolution_deg * 0.5,
        resolution_deg,
        dtype=np.float64,
    )
    lons = np.arange(
        center_lon - grid_radius_deg,
        center_lon + grid_radius_deg + resolution_deg * 0.5,
        resolution_deg,
        dtype=np.float64,
    )

    T = len(times)
    M = len(lats)
    K = len(lons)

    # Coordinate grids
    # lats_2d shape: [M, K], lons_2d shape: [M, K]
    lats_2d, lons_2d = np.meshgrid(lats, lons, indexing="ij")
    d_lat = lats_2d - center_lat
    d_lon = lons_2d - center_lon

    # Spatial modulation (mesoscale spatial variation)
    spatial_curl = np.sin(np.pi * d_lat / grid_radius_deg) * np.cos(np.pi * d_lon / grid_radius_deg)
    spatial_shear = np.cos(np.pi * d_lat / grid_radius_deg) * np.sin(np.pi * d_lon / grid_radius_deg)

    # Initialize 3D arrays [T, M, K]
    u_current = np.zeros((T, M, K), dtype=np.float64)
    v_current = np.zeros((T, M, K), dtype=np.float64)
    u_wind = np.zeros((T, M, K), dtype=np.float64)
    v_wind = np.zeros((T, M, K), dtype=np.float64)

    m2_period_sec = 12.4206 * 3600.0  # M2 semi-diurnal tide in seconds
    s1_period_sec = 24.0 * 3600.0      # Diurnal sea-breeze cycle in seconds

    for i, t in enumerate(times):
        dt_rel = t - t_ref

        # Tidal current phase
        tide_phase = 2.0 * math.pi * (dt_rel / m2_period_sec)
        u_tide = tidal_amplitude * math.cos(tide_phase)
        v_tide = tidal_amplitude * math.sin(tide_phase) * 0.75  # Elliptical tidal rotary

        # Diurnal wind cycle phase
        wind_phase = 2.0 * math.pi * (dt_rel / s1_period_sec)
        u_diurnal = wind_gust_amplitude * math.cos(wind_phase)
        v_diurnal = wind_gust_amplitude * math.sin(wind_phase) * 0.6

        # Assemble current field
        u_current[i] = base_u_current + u_tide + 0.05 * spatial_curl
        v_current[i] = base_v_current + v_tide + 0.05 * spatial_shear

        # Assemble wind field
        u_wind[i] = base_u_wind + u_diurnal + 0.5 * spatial_curl
        v_wind[i] = base_v_wind + v_diurnal + 0.5 * spatial_shear

    metadata = {
        "scene_id": scene_id or "synthetic_demo_scene",
        "source": "deterministic_synthetic_physics_engine",
        "center_lat": center_lat,
        "center_lon": center_lon,
        "t_center_iso": format_iso_timestamp(t_ref),
        "resolution_deg": resolution_deg,
        "is_synthetic": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return MetoceanGrid(
        times=times,
        lats=lats,
        lons=lons,
        u_current=u_current,
        v_current=v_current,
        u_wind=u_wind,
        v_wind=v_wind,
        metadata=metadata,
    )


def save_metocean_data(grid: MetoceanGrid, path: Union[str, Path]) -> str:
    """
    Save MetoceanGrid to disk as JSON or NPZ.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix.lower() == ".json":
        data = grid.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif file_path.suffix.lower() == ".npz":
        np.savez_compressed(
            file_path,
            times=grid.times,
            lats=grid.lats,
            lons=grid.lons,
            u_current=grid.u_current,
            v_current=grid.v_current,
            u_wind=grid.u_wind,
            v_wind=grid.v_wind,
            metadata_json=json.dumps(grid.metadata),
        )
    else:
        # Default to JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(grid.to_dict(), f)

    logger.info("Saved MetoceanGrid to %s", file_path)
    return str(file_path)


def load_metocean_from_file(file_path: Union[str, Path]) -> MetoceanGrid:
    """
    Load MetoceanGrid from JSON, NPZ, or NetCDF file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Metocean file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Parse times
        if "timestamps" in raw:
            times = np.array([parse_timestamp(t) for t in raw["timestamps"]], dtype=np.float64)
        elif "times" in raw:
            times = np.array([parse_timestamp(t) for t in raw["times"]], dtype=np.float64)
        elif "times_iso" in raw:
            times = np.array([parse_timestamp(t) for t in raw["times_iso"]], dtype=np.float64)
        else:
            raise KeyError("JSON metocean grid must contain 'timestamps', 'times', or 'times_iso'.")

        lats = np.array(raw["lats"], dtype=np.float64)
        lons = np.array(raw["lons"], dtype=np.float64)
        u_current = np.array(raw["u_current"], dtype=np.float64)
        v_current = np.array(raw["v_current"], dtype=np.float64)
        u_wind = np.array(raw["u_wind"], dtype=np.float64)
        v_wind = np.array(raw["v_wind"], dtype=np.float64)
        metadata = raw.get("metadata", {})

        return MetoceanGrid(
            times=times,
            lats=lats,
            lons=lons,
            u_current=u_current,
            v_current=v_current,
            u_wind=u_wind,
            v_wind=v_wind,
            metadata=metadata,
        )

    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            times = data["times"] if "times" in data else data["timestamps"]
            lats = data["lats"]
            lons = data["lons"]
            u_current = data["u_current"]
            v_current = data["v_current"]
            u_wind = data["u_wind"]
            v_wind = data["v_wind"]
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"]))
            elif "metadata" in data:
                try:
                    metadata = data["metadata"].item()
                except Exception:
                    metadata = {}

        return MetoceanGrid(
            times=times,
            lats=lats,
            lons=lons,
            u_current=u_current,
            v_current=v_current,
            u_wind=u_wind,
            v_wind=v_wind,
            metadata=metadata,
        )

    elif suffix in (".nc", ".nc4", ".netcdf"):
        # NetCDF reader via scipy.io.netcdf or fallback
        try:
            from scipy.io import netcdf_file
            with netcdf_file(str(path), "r", mmap=False) as nc:
                times_raw = nc.variables["time"][:].copy()
                lats = nc.variables["lat"][:].copy()
                lons = nc.variables["lon"][:].copy()
                u_current = nc.variables["u_current"][:].copy()
                v_current = nc.variables["v_current"][:].copy()
                u_wind = nc.variables["u_wind"][:].copy()
                v_wind = nc.variables["v_wind"][:].copy()
                times = np.array([parse_timestamp(t) for t in times_raw], dtype=np.float64)
                return MetoceanGrid(
                    times=times,
                    lats=lats,
                    lons=lons,
                    u_current=u_current,
                    v_current=v_current,
                    u_wind=u_wind,
                    v_wind=v_wind,
                    metadata={"source": "netcdf", "file": str(path)},
                )
        except Exception as e:
            logger.warning("Failed to parse NetCDF file %s with scipy: %s", path, e)
            raise RuntimeError(f"NetCDF parsing error for {path}: {e}")

    raise ValueError(f"Unsupported file extension for metocean data: {suffix}")


def load_metocean_data(
    scene_id: Optional[str] = None,
    file_path: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    center_lat: float = 18.9,
    center_lon: float = 72.8,
    t_sat: Optional[Union[str, datetime, float]] = None,
    force_synthetic: bool = False,
) -> MetoceanGrid:
    """
    Main loader function for Metocean datasets.
    Looks for cached files on disk; if absent or force_synthetic=True, generates
    a physically realistic synthetic dataset centered on (center_lat, center_lon, t_sat).

    Search order:
    1. file_path (if provided)
    2. data_dir / {scene_id}.json
    3. data_dir / {scene_id}.npz
    4. data_dir / {scene_id}.nc
    5. data/metocean/{scene_id}.json, .npz, .nc
    6. Synthetic generation fallback
    """
    if force_synthetic:
        logger.info("force_synthetic=True: generating synthetic metocean grid.")
        return create_synthetic_metocean_grid(
            center_lat=center_lat,
            center_lon=center_lon,
            t_center=t_sat,
            scene_id=scene_id,
        )

    # 1. Direct path check
    if file_path is not None:
        p = Path(file_path)
        if p.exists():
            try:
                return load_metocean_from_file(p)
            except Exception as e:
                logger.warning("Error reading %s: %s", p, e)

    # 2. Check search paths
    candidate_dirs: List[Path] = []
    if data_dir is not None:
        candidate_dirs.append(Path(data_dir))
    candidate_dirs.extend([
        Path("data/metocean"),
        Path("c:/Users/Siddhant/TideTrace/data/metocean"),
        Path.cwd() / "data" / "metocean",
    ])

    if scene_id:
        for cdir in candidate_dirs:
            for ext in (".json", ".npz", ".nc"):
                candidate = cdir / f"{scene_id}{ext}"
                if candidate.exists():
                    try:
                        logger.info("Found cached metocean dataset: %s", candidate)
                        return load_metocean_from_file(candidate)
                    except Exception as e:
                        logger.warning("Error reading candidate %s: %s", candidate, e)

    # Check default file
    for cdir in candidate_dirs:
        for ext in (".json", ".npz", ".nc"):
            candidate = cdir / f"default{ext}"
            if candidate.exists():
                try:
                    logger.info("Found default metocean dataset: %s", candidate)
                    return load_metocean_from_file(candidate)
                except Exception as e:
                    logger.warning("Error reading candidate %s: %s", candidate, e)

    # 3. Deterministic synthetic fallback
    logger.info(
        "No cached metocean file found for scene_id='%s'. Generating synthetic grid at (%.4f, %.4f).",
        scene_id,
        center_lat,
        center_lon,
    )
    return create_synthetic_metocean_grid(
        center_lat=center_lat,
        center_lon=center_lon,
        t_center=t_sat,
        scene_id=scene_id,
    )
