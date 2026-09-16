"""
Forward Dispersion Forecast Engine for TideTrace.
Forecasts future oil slick advection and dispersion for 24h to 48h (default 36h, dt = +3600s).
Propagates an ensemble of N=50 surface particles under 4D metocean forcing with turbulent diffusion.
Generates forward trajectory dispersion cones, checkpoint polygons at +12h, +24h, +36h,
and computes the threatened marine/coastal bounding box.
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
from app.drift.hindcast import (
    DEFAULT_NUM_PARTICLES,
    DEFAULT_SIGMA_CURRENT,
    DEFAULT_SIGMA_WIND,
    compute_confidence_envelope_geojson,
    seed_particles_across_slick,
)
from app.metocean.interpolator import MetoceanInterpolator
from app.metocean.store import format_iso_timestamp, parse_timestamp

logger = logging.getLogger(__name__)

DEFAULT_FORECAST_HOURS = 36
DEFAULT_CHECKPOINTS = [12, 24, 36]
FORECAST_BUFFER_KM = 1.5  # 1.5 km dilation of checkpoint forecast envelopes


def compute_dispersion_cone_geojson(
    all_lats: np.ndarray,
    all_lons: np.ndarray,
    dilation_buffer_m: float = 2000.0,
) -> Dict[str, Any]:
    """
    Compute the composite forward dispersion cone encompassing all particle positions across time.
    """
    c_lat = float(np.mean(all_lats))
    c_lon = float(np.mean(all_lons))

    xs, ys = forward_aeqd(all_lats.ravel(), all_lons.ravel(), c_lat, c_lon)
    pts = [Point(xs[i], ys[i]) for i in range(len(xs))]
    mp = MultiPoint(pts)
    hull = mp.convex_hull

    if isinstance(hull, Point):
        buffered = hull.buffer(dilation_buffer_m)
    elif hull.is_empty:
        buffered = Point(0, 0).buffer(dilation_buffer_m)
    else:
        buffered = hull.buffer(dilation_buffer_m)

    if not buffered.is_valid:
        buffered = buffered.buffer(0)

    coords_xy = np.asarray(buffered.exterior.coords, dtype=np.float64)
    poly_lats, poly_lons = inverse_aeqd(coords_xy[:, 0], coords_xy[:, 1], c_lat, c_lon)
    geojson_ring = [[float(lon), float(lat)] for lat, lon in zip(poly_lats, poly_lons)]

    return {
        "type": "Polygon",
        "coordinates": [geojson_ring],
    }


def run_forecast(
    interpolator: MetoceanInterpolator,
    slick_polygon: Union[List[Any], Tuple[Any, ...], Dict[str, Any], Polygon],
    t_sat: Union[str, datetime, float],
    forecast_hours: int = DEFAULT_FORECAST_HOURS,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    dt_seconds: float = 3600.0,
    sigma_c: float = DEFAULT_SIGMA_CURRENT,
    sigma_w: float = DEFAULT_SIGMA_WIND,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    coriolis_deg: float = DEFAULT_CORIOLIS_DEG,
    checkpoint_hours: Optional[List[int]] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Execute forward Lagrangian ensemble dispersion forecast.

    Returns:
    - forecast_end_time_iso: ISO-8601 UTC timestamp of forecast horizon
    - total_forecast_drift_km: Total forward displacement of ensemble median in km
    - threatened_bbox: [min_lon, min_lat, max_lon, max_lat]
    - checkpoint_polygons: Dict of checkpoint polygons (e.g. +12h, +24h, +36h)
    - feature_collection: GeoJSON FeatureCollection containing cone, checkpoints, and tracks
    - hourly_history: Hourly centroid coordinates, spread, and speed
    """
    if checkpoint_hours is None:
        checkpoint_hours = DEFAULT_CHECKPOINTS

    ts_start = parse_timestamp(t_sat)
    rng = np.random.default_rng(seed)

    curr_lats, curr_lons = seed_particles_across_slick(
        slick_polygon, num_particles=num_particles, seed=seed
    )

    N = len(curr_lats)
    total_steps = forecast_hours

    hist_lats = np.zeros((total_steps + 1, N), dtype=np.float64)
    hist_lons = np.zeros((total_steps + 1, N), dtype=np.float64)
    hist_times = np.zeros(total_steps + 1, dtype=np.float64)

    hist_lats[0] = curr_lats
    hist_lons[0] = curr_lons
    hist_times[0] = ts_start

    hourly_history: List[Dict[str, Any]] = []
    c0_lat = float(np.mean(curr_lats))
    c0_lon = float(np.mean(curr_lons))
    r0_km = [haversine_distance_km(curr_lats[i], curr_lons[i], c0_lat, c0_lon) for i in range(N)]
    spread0_km = float(np.percentile(r0_km, 90.0))

    hourly_history.append({
        "hour_fwd": 0,
        "time_iso": format_iso_timestamp(ts_start),
        "timestamp": ts_start,
        "centroid_lat": c0_lat,
        "centroid_lon": c0_lon,
        "spread_km": spread0_km,
        "mean_speed_m_s": 0.0,
    })

    # Step forward hourly
    for step in range(1, total_steps + 1):
        t_current = ts_start + (step - 1) * dt_seconds

        u_pert_c = rng.normal(0.0, sigma_c, N)
        v_pert_c = rng.normal(0.0, sigma_c, N)
        u_pert_w = rng.normal(0.0, sigma_w, N)
        v_pert_w = rng.normal(0.0, sigma_w, N)

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

        c_lat = float(np.mean(curr_lats))
        c_lon = float(np.mean(curr_lons))
        prev_c_lat = hourly_history[-1]["centroid_lat"]
        prev_c_lon = hourly_history[-1]["centroid_lon"]
        step_dist_km = haversine_distance_km(prev_c_lat, prev_c_lon, c_lat, c_lon)
        speed_m_s = (step_dist_km * 1000.0) / abs(dt_seconds)

        radii_km = [haversine_distance_km(curr_lats[i], curr_lons[i], c_lat, c_lon) for i in range(N)]
        spread_km = float(np.percentile(radii_km, 90.0))

        hourly_history.append({
            "hour_fwd": step,
            "time_iso": format_iso_timestamp(t_next),
            "timestamp": t_next,
            "centroid_lat": c_lat,
            "centroid_lon": c_lon,
            "spread_km": spread_km,
            "mean_speed_m_s": float(speed_m_s),
        })

    # Composite forward dispersion cone
    cone_poly_geom = compute_dispersion_cone_geojson(
        hist_lats, hist_lons, dilation_buffer_m=2000.0
    )

    # Threatened bounding box
    min_lat = float(np.min(hist_lats))
    max_lat = float(np.max(hist_lats))
    min_lon = float(np.min(hist_lons))
    max_lon = float(np.max(hist_lons))

    # Add 0.05 degree margin to threatened bbox
    threatened_bbox = [
        round(min_lon - 0.05, 4),
        round(min_lat - 0.05, 4),
        round(max_lon + 0.05, 4),
        round(max_lat + 0.05, 4),
    ]

    features: List[Dict[str, Any]] = []

    # 1. Composite Forecast Cone Feature
    features.append({
        "type": "Feature",
        "geometry": cone_poly_geom,
        "properties": {
            "layer": "forecast_cone",
            "title": f"Forward {forecast_hours}h Dispersion Cone",
            "start_time": format_iso_timestamp(ts_start),
            "end_time": format_iso_timestamp(hist_times[-1]),
            "threatened_bbox": threatened_bbox,
            "particle_count": N,
        },
    })

    # 2. Checkpoint Polygons (e.g. +12h, +24h, +36h)
    checkpoint_polygons: Dict[str, Any] = {}
    for chk in checkpoint_hours:
        if 1 <= chk <= total_steps:
            chk_geom = compute_confidence_envelope_geojson(
                hist_lats[chk],
                hist_lons[chk],
                confidence_pct=90.0,
                dilation_buffer_m=FORECAST_BUFFER_KM * 1000.0,
            )
            chk_feature = {
                "type": "Feature",
                "geometry": chk_geom,
                "properties": {
                    "layer": "forecast_checkpoint",
                    "checkpoint_hours": chk,
                    "valid_time": format_iso_timestamp(hist_times[chk]),
                    "centroid": [
                        round(hourly_history[chk]["centroid_lat"], 4),
                        round(hourly_history[chk]["centroid_lon"], 4),
                    ],
                    "spread_km": round(hourly_history[chk]["spread_km"], 2),
                },
            }
            features.append(chk_feature)
            checkpoint_polygons[f"plus_{chk}h"] = chk_feature

    # 3. Median Trajectory LineString
    median_coords = [
        [float(hourly_history[s]["centroid_lon"]), float(hourly_history[s]["centroid_lat"])]
        for s in range(total_steps + 1)
    ]
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": median_coords,
        },
        "properties": {
            "layer": "forecast_median_track",
            "title": "Forecast Median Drift Track",
            "steps_count": len(median_coords),
        },
    })

    # 4. Ensemble Particle Lines
    particle_lines = []
    for p in range(N):
        p_coords = [
            [float(hist_lons[s, p]), float(hist_lats[s, p])]
            for s in range(total_steps + 1)
        ]
        particle_lines.append(p_coords)

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "MultiLineString",
            "coordinates": particle_lines,
        },
        "properties": {
            "layer": "forecast_ensemble_particles",
            "title": f"Forecast Ensemble Tracks (N={N})",
            "particle_count": N,
        },
    })

    geojson_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    final_c_lat = hourly_history[-1]["centroid_lat"]
    final_c_lon = hourly_history[-1]["centroid_lon"]
    total_drift_km = haversine_distance_km(c0_lat, c0_lon, final_c_lat, final_c_lon)

    # Compute dominant drift heading from start to end
    d_lat_deg = final_c_lat - c0_lat
    d_lon_deg = final_c_lon - c0_lon
    heading_rad = math.atan2(
        d_lon_deg * math.cos(math.radians(c0_lat)), d_lat_deg
    )
    heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0

    return {
        "start_time_iso": format_iso_timestamp(ts_start),
        "forecast_end_time_iso": format_iso_timestamp(hist_times[-1]),
        "forecast_hours": forecast_hours,
        "total_drift_km": round(total_drift_km, 2),
        "dominant_heading_deg": round(heading_deg, 1),
        "threatened_bbox": threatened_bbox,
        "checkpoint_polygons": checkpoint_polygons,
        "feature_collection": geojson_collection,
        "hourly_history": hourly_history,
        "particle_count": N,
    }
