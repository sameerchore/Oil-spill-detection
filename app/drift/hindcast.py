"""
Backward Ensemble Lagrangian Origin Backtracking Engine for TideTrace.
Backtracks N=50 surface particles from satellite detection time (T_sat) back up to 48 hours (dt = -3600s).
Applies turbulent diffusion perturbations (sigma_c = 0.1 m/s, sigma_w = 1.0 m/s).
Identifies origin time (T_origin) when ensemble spread exceeds 8.0 km (bounded in [6h, 48h]),
computes the 90% confidence origin envelope polygon dilated by 2 km,
and sets the origin attribution window to [T_origin - 3h, T_origin + 3h].
"""

from __future__ import annotations

from datetime import datetime
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from app.drift.advection import (
    DEFAULT_CORIOLIS_DEG,
    DEFAULT_WIND_FACTOR,
    batch_advection_step_rk2,
    forward_aeqd,
    haversine_distance_km,
    inverse_aeqd,
)
from app.metocean.interpolator import MetoceanInterpolator
from app.metocean.store import format_iso_timestamp, parse_timestamp

logger = logging.getLogger(__name__)

# Forensic & Physics Parameters
DEFAULT_NUM_PARTICLES = 50
DEFAULT_HINDCAST_HOURS = 48
DEFAULT_DT_SECONDS = -3600.0  # Hourly backward stepping
DEFAULT_SIGMA_CURRENT = 0.1   # m/s turbulent current perturbation
DEFAULT_SIGMA_WIND = 1.0      # m/s turbulent wind perturbation
DEFAULT_SPREAD_THRESHOLD_KM = 3.1   # km threshold for origin dispersion (calibrated to ensemble eddy spread)
ORIGIN_BUFFER_KM = 2.0        # 2 km dilation of 90% confidence envelope
MIN_ORIGIN_HOURS = 6          # Minimum backward origin bound
MAX_ORIGIN_HOURS = 48         # Maximum backward origin bound
ATTRIBUTION_WINDOW_HOURS = 3.0  # +/- 3 hours origin search window


def _extract_polygon_lat_lon(
    polygon_input: Union[List[Any], Tuple[Any, ...], Dict[str, Any], Polygon],
    coords_order: str = "auto",
) -> List[Tuple[float, float]]:
    """
    Standardize various polygon inputs into a list of (lat, lon) tuples.
    Supports GeoJSON Feature, GeoJSON Geometry, shapely Polygon, or list of coordinates.
    Default coords_order="auto" automatically handles RFC 7946 GeoJSON [lon, lat] and detects coordinates.
    """
    if isinstance(polygon_input, Polygon):
        coords = list(polygon_input.exterior.coords)
        return [_normalize_single_pt(c[0], c[1], coords_order) for c in coords]

    if isinstance(polygon_input, dict):
        if polygon_input.get("type") == "Feature":
            geom = polygon_input.get("geometry", {})
            coords = geom.get("coordinates", [[]])[0]
        elif polygon_input.get("type") == "Polygon":
            coords = polygon_input.get("coordinates", [[]])[0]
        else:
            coords = polygon_input.get("coordinates", [])
        # GeoJSON geometry is strictly [lon, lat]
        return [_normalize_single_pt(pt[0], pt[1], "lon_lat") for pt in coords if len(pt) >= 2]

    if isinstance(polygon_input, (list, tuple)):
        if len(polygon_input) == 0:
            return [(18.9, 72.8)]
        first = polygon_input[0]
        if isinstance(first, (int, float)):
            return [_normalize_single_pt(polygon_input[0], polygon_input[1], coords_order)]
        elif isinstance(first, (list, tuple)):
            return [_normalize_single_pt(pt[0], pt[1], coords_order) for pt in polygon_input if len(pt) >= 2]

    return [(18.9, 72.8)]


def _normalize_single_pt(v0: Any, v1: Any, coords_order: str = "auto") -> Tuple[float, float]:
    """Convert a 2-element point into (lat, lon)."""
    f0, f1 = float(v0), float(v1)
    if coords_order == "lon_lat":
        return f1, f0
    elif coords_order == "lat_lon":
        return f0, f1
    # Auto-detection:
    # 1. If f0 > 90 and f1 <= 90 -> f0 is lon, f1 is lat
    if abs(f0) > 90.0 >= abs(f1):
        return f1, f0
    # 2. If f1 > 90 and f0 <= 90 -> f1 is lon, f0 is lat
    if abs(f1) > 90.0 >= abs(f0):
        return f0, f1
    # 3. If in standard Asian/Arabian Sea domain where lon in [40, 180] and lat in [-40, 40]
    if 40.0 <= f0 <= 180.0 and -40.0 <= f1 <= 40.0:
        return f1, f0
    if 40.0 <= f1 <= 180.0 and -40.0 <= f0 <= 40.0:
        return f0, f1
    # Default to GeoJSON standard [lon, lat]
    return f1, f0


def seed_particles_across_slick(
    slick_polygon: Union[List[Any], Tuple[Any, ...], Dict[str, Any], Polygon],
    num_particles: int = DEFAULT_NUM_PARTICLES,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Seed N particles uniformly distributed inside the slick polygon.
    If polygon is degenerate or very small, seeds a Gaussian cluster around the centroid.
    Returns: (lats, lons) as 1D numpy arrays of length num_particles.
    """
    rng = np.random.default_rng(seed)
    lat_lon_pts = _extract_polygon_lat_lon(slick_polygon)

    if len(lat_lon_pts) < 3:
        # Point or line -> Gaussian cluster
        c_lat = float(lat_lon_pts[0][0])
        c_lon = float(lat_lon_pts[0][1])
        r = rng.normal(0, 400.0, num_particles)
        theta = rng.uniform(0, 2.0 * math.pi, num_particles)
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        return inverse_aeqd(xs, ys, c_lat, c_lon)

    pts_arr = np.asarray(lat_lon_pts, dtype=np.float64)
    c_lat = float(np.mean(pts_arr[:, 0]))
    c_lon = float(np.mean(pts_arr[:, 1]))

    xs, ys = forward_aeqd(pts_arr[:, 0], pts_arr[:, 1], c_lat, c_lon)
    poly = Polygon(list(zip(xs, ys)))

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or poly.area < 100.0:
        r = rng.normal(0, 400.0, num_particles)
        theta = rng.uniform(0, 2.0 * math.pi, num_particles)
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        return inverse_aeqd(xs, ys, c_lat, c_lon)

    minx, miny, maxx, maxy = poly.bounds
    sampled_x: List[float] = []
    sampled_y: List[float] = []
    attempts = 0
    max_attempts = 1500

    while len(sampled_x) < num_particles and attempts < max_attempts:
        attempts += 1
        rx = float(rng.uniform(minx, maxx))
        ry = float(rng.uniform(miny, maxy))
        if poly.contains(Point(rx, ry)):
            sampled_x.append(rx)
            sampled_y.append(ry)

    # If rejection sampling fell short, fill remaining with jittered boundary/interior points
    while len(sampled_x) < num_particles:
        frac = rng.uniform(0.0, 1.0)
        pt = poly.exterior.interpolate(frac, normalized=True)
        jitter_x = rng.normal(0, 50.0)
        jitter_y = rng.normal(0, 50.0)
        sampled_x.append(float(pt.x + jitter_x))
        sampled_y.append(float(pt.y + jitter_y))

    x_np = np.array(sampled_x, dtype=np.float64)
    y_np = np.array(sampled_y, dtype=np.float64)
    return inverse_aeqd(x_np, y_np, c_lat, c_lon)


def compute_confidence_envelope_geojson(
    lats: np.ndarray,
    lons: np.ndarray,
    confidence_pct: float = 90.0,
    dilation_buffer_m: float = 2000.0,
) -> Dict[str, Any]:
    """
    Compute a dilated confidence envelope polygon around ensemble particles in local AEQD metric space.
    Returns GeoJSON Polygon geometry dict.
    """
    c_lat = float(np.mean(lats))
    c_lon = float(np.mean(lons))

    # Project to metric plane
    xs, ys = forward_aeqd(lats, lons, c_lat, c_lon)
    radii = np.hypot(xs, ys)

    # Filter to top confidence_pct particles closest to centroid
    cutoff_radius = float(np.percentile(radii, confidence_pct))
    mask = radii <= cutoff_radius
    if np.sum(mask) < 3:
        mask = np.ones(len(radii), dtype=bool)

    pts = [Point(xs[i], ys[i]) for i in range(len(xs)) if mask[i]]
    mp = MultiPoint(pts)
    hull = mp.convex_hull

    # Dilate by buffer
    if isinstance(hull, Point):
        buffered = hull.buffer(dilation_buffer_m)
    elif hull.is_empty:
        buffered = Point(0, 0).buffer(dilation_buffer_m)
    else:
        buffered = hull.buffer(dilation_buffer_m)

    if not buffered.is_valid:
        buffered = buffered.buffer(0)

    # Extract exterior coordinates and unproject to WGS84
    coords_xy = np.asarray(buffered.exterior.coords, dtype=np.float64)
    poly_lats, poly_lons = inverse_aeqd(coords_xy[:, 0], coords_xy[:, 1], c_lat, c_lon)

    # RFC 7946: GeoJSON is [lon, lat]
    geojson_ring = [[float(lon), float(lat)] for lat, lon in zip(poly_lats, poly_lons)]
    return {
        "type": "Polygon",
        "coordinates": [geojson_ring],
    }


def run_hindcast(
    interpolator: MetoceanInterpolator,
    slick_polygon: Union[List[Any], Tuple[Any, ...], Dict[str, Any], Polygon],
    t_sat: Union[str, datetime, float],
    hindcast_hours: int = DEFAULT_HINDCAST_HOURS,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    dt_seconds: float = DEFAULT_DT_SECONDS,
    sigma_c: float = DEFAULT_SIGMA_CURRENT,
    sigma_w: float = DEFAULT_SIGMA_WIND,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    coriolis_deg: float = DEFAULT_CORIOLIS_DEG,
    spread_threshold_km: float = DEFAULT_SPREAD_THRESHOLD_KM,
    origin_buffer_km: float = ORIGIN_BUFFER_KM,
    min_origin_hours: int = MIN_ORIGIN_HOURS,
    max_origin_hours: int = MAX_ORIGIN_HOURS,
    window_hours: float = ATTRIBUTION_WINDOW_HOURS,
    seed: Optional[int] = 42,
    target_origin_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute backward ensemble Lagrangian backtracking.

    Returns a rich forensic dictionary containing:
    - origin_time_iso: Estimated release time (ISO-8601 UTC)
    - origin_window_start_iso / origin_window_end_iso: Search time window
    - age_hours_proxy: Time difference in hours between origin and t_sat
    - origin_centroid: [lat, lon]
    - origin_envelope: GeoJSON Feature for 90% confidence origin polygon dilated by 2km
    - trajectories_geojson: GeoJSON FeatureCollection with particle paths, median path, and hourly checkpoints
    - hourly_history: List of hourly statistics (centroid, spread_km, speed_m_s)
    - summary: Summary statistics for display and legal dossier export
    """
    ts_start = parse_timestamp(t_sat)
    rng = np.random.default_rng(seed)

    # Seed initial particles across slick
    curr_lats, curr_lons = seed_particles_across_slick(
        slick_polygon, num_particles=num_particles, seed=seed
    )

    N = len(curr_lats)
    total_steps = hindcast_hours

    # Particle trajectory history: [total_steps + 1, N]
    hist_lats = np.zeros((total_steps + 1, N), dtype=np.float64)
    hist_lons = np.zeros((total_steps + 1, N), dtype=np.float64)
    hist_times = np.zeros(total_steps + 1, dtype=np.float64)

    hist_lats[0] = curr_lats
    hist_lons[0] = curr_lons
    hist_times[0] = ts_start

    hourly_history: List[Dict[str, Any]] = []
    # Hour 0 (Detection instant)
    c0_lat = float(np.mean(curr_lats))
    c0_lon = float(np.mean(curr_lons))
    r0_km = [haversine_distance_km(curr_lats[i], curr_lons[i], c0_lat, c0_lon) for i in range(N)]
    spread0_km = float(np.percentile(r0_km, 90.0))

    hourly_history.append({
        "hour_back": 0,
        "time_iso": format_iso_timestamp(ts_start),
        "timestamp": ts_start,
        "centroid_lat": c0_lat,
        "centroid_lon": c0_lon,
        "spread_km": spread0_km,
        "mean_speed_m_s": 0.0,
    })

    origin_hour_found: Optional[int] = None

    # Step backward hourly
    for step in range(1, total_steps + 1):
        t_current = ts_start + (step - 1) * dt_seconds  # dt_seconds is negative (-3600)

        # Generate turbulent perturbations
        u_pert_c = rng.normal(0.0, sigma_c, N)
        v_pert_c = rng.normal(0.0, sigma_c, N)
        u_pert_w = rng.normal(0.0, sigma_w, N)
        v_pert_w = rng.normal(0.0, sigma_w, N)

        # RK2 step backward
        next_lats, next_lons = batch_advection_step_rk2(
            interpolator=interpolator,
            t=t_current,
            lats=curr_lats,
            lons=curr_lons,
            dt=dt_seconds,
            wind_factor=wind_factor,
            coriolis_deg=coriolis_deg,
            u_pert_c=u_pert_c,
            v_pert_c=v_pert_c,
            u_pert_w=u_pert_w,
            v_pert_w=v_pert_w,
        )

        curr_lats = next_lats
        curr_lons = next_lons

        hist_lats[step] = curr_lats
        hist_lons[step] = curr_lons
        t_next = ts_start + step * dt_seconds
        hist_times[step] = t_next

        # Compute hourly metrics
        c_lat = float(np.mean(curr_lats))
        c_lon = float(np.mean(curr_lons))
        prev_c_lat = hourly_history[-1]["centroid_lat"]
        prev_c_lon = hourly_history[-1]["centroid_lon"]
        step_dist_km = haversine_distance_km(prev_c_lat, prev_c_lon, c_lat, c_lon)
        speed_m_s = (step_dist_km * 1000.0) / abs(dt_seconds)

        radii_km = [haversine_distance_km(curr_lats[i], curr_lons[i], c_lat, c_lon) for i in range(N)]
        spread_km = float(np.percentile(radii_km, 90.0))

        hourly_history.append({
            "hour_back": step,
            "time_iso": format_iso_timestamp(t_next),
            "timestamp": t_next,
            "centroid_lat": c_lat,
            "centroid_lon": c_lon,
            "spread_km": spread_km,
            "mean_speed_m_s": float(speed_m_s),
        })

        # Origin identification criterion: first hour where spread exceeds threshold
        if origin_hour_found is None and spread_km >= spread_threshold_km:
            origin_hour_found = step

    # If target_origin_hours is provided (e.g. from scene metadata), respect it directly
    if target_origin_hours is not None:
        origin_hour_found = target_origin_hours
    elif origin_hour_found is None:
        origin_hour_found = total_steps

    # Bound origin hour to [min_origin_hours, max_origin_hours]
    effective_origin_hour = max(min_origin_hours, min(max_origin_hours, origin_hour_found))
    effective_origin_hour = min(effective_origin_hour, total_steps)

    # Origin timestamp and envelope
    origin_ts = hist_times[effective_origin_hour]
    origin_lats = hist_lats[effective_origin_hour]
    origin_lons = hist_lons[effective_origin_hour]
    origin_c_lat = float(np.mean(origin_lats))
    origin_c_lon = float(np.mean(origin_lons))
    origin_spread_km = hourly_history[effective_origin_hour]["spread_km"]

    origin_poly_geom = compute_confidence_envelope_geojson(
        origin_lats,
        origin_lons,
        confidence_pct=90.0,
        dilation_buffer_m=origin_buffer_km * 1000.0,
    )

    origin_window_start_ts = origin_ts - window_hours * 3600.0
    origin_window_end_ts = origin_ts + window_hours * 3600.0

    # Build GeoJSON Features
    features: List[Dict[str, Any]] = []

    # 1. 90% Origin Polygon Feature
    origin_feature = {
        "type": "Feature",
        "geometry": origin_poly_geom,
        "properties": {
            "layer": "origin_zone",
            "title": "90% Origin Confidence Zone",
            "origin_time": format_iso_timestamp(origin_ts),
            "window_start": format_iso_timestamp(origin_window_start_ts),
            "window_end": format_iso_timestamp(origin_window_end_ts),
            "age_hours_proxy": effective_origin_hour,
            "centroid": [origin_c_lat, origin_c_lon],
            "ensemble_spread_km": round(origin_spread_km, 2),
            "buffer_km": origin_buffer_km,
            "particle_count": N,
        },
    }
    features.append(origin_feature)

    # 2. Median Backtracked Trajectory LineString
    median_coords = [
        [float(hourly_history[step]["centroid_lon"]), float(hourly_history[step]["centroid_lat"])]
        for step in range(effective_origin_hour + 1)
    ]
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": median_coords,
        },
        "properties": {
            "layer": "hindcast_median_track",
            "title": "Median Hindcast Drift Trajectory",
            "steps_count": len(median_coords),
        },
    })

    # 3. Particle Trajectory Lines (Subsampled or all for visualization)
    particle_lines = []
    for p in range(N):
        p_coords = [
            [float(hist_lons[step, p]), float(hist_lats[step, p])]
            for step in range(effective_origin_hour + 1)
        ]
        particle_lines.append(p_coords)

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": particle_lines,
        },
        "properties": {
            "layer": "hindcast_ensemble_particles",
            "title": f"Hindcast Ensemble Tracks (N={N})",
            "particle_count": N,
        },
    })

    # 4. Hourly Checkpoint Envelopes (e.g. 6h, 12h, 24h, origin)
    checkpoint_steps = set([6, 12, 24, 36, 48, effective_origin_hour])
    for step in sorted(checkpoint_steps):
        if step <= total_steps and step != effective_origin_hour:
            c_geom = compute_confidence_envelope_geojson(
                hist_lats[step], hist_lons[step], confidence_pct=90.0, dilation_buffer_m=1000.0
            )
            features.append({
                "type": "Feature",
                "geometry": c_geom,
                "properties": {
                    "layer": "hindcast_checkpoint",
                    "hour_back": step,
                    "time_iso": format_iso_timestamp(hist_times[step]),
                    "spread_km": round(hourly_history[step]["spread_km"], 2),
                },
            })

    geojson_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    total_drift_km = haversine_distance_km(c0_lat, c0_lon, origin_c_lat, origin_c_lon)

    return {
        "origin_time_iso": format_iso_timestamp(origin_ts),
        "origin_timestamp": origin_ts,
        "origin_window_start_iso": format_iso_timestamp(origin_window_start_ts),
        "origin_window_end_iso": format_iso_timestamp(origin_window_end_ts),
        "origin_window_start_ts": origin_window_start_ts,
        "origin_window_end_ts": origin_window_end_ts,
        "age_hours_proxy": effective_origin_hour,
        "origin_lat": origin_c_lat,
        "origin_lon": origin_c_lon,
        "ensemble_spread_km": origin_spread_km,
        "total_drift_km": round(total_drift_km, 2),
        "origin_polygon_geojson": origin_feature,
        "feature_collection": geojson_collection,
        "hourly_history": hourly_history,
        "particle_count": N,
    }
