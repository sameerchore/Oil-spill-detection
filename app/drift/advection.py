"""
2D Lagrangian Surface Drift Advection Engine for TideTrace.
Implements surface slick velocity integration:
    V_slick = U_current + 0.03 * R(15 deg) * U_wind
where R(15 deg) deflects atmospheric wind drift 15 deg clockwise (to the right in Northern Hemisphere).
Provides Euler and 2nd-order Runge-Kutta (RK2 midpoint) numerical integrators
with local metric geodesic and Azimuthal Equidistant (AEQD) coordinate transformations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from app.metocean.interpolator import MetoceanInterpolator

# Physical and Geodetic Constants
EARTH_RADIUS_M = 6371000.0  # Mean volumetric Earth radius in meters
DEFAULT_WIND_FACTOR = 0.03   # 3% wind leeway drift factor
DEFAULT_CORIOLIS_DEG = 15.0  # 15 deg Coriolis / leeway deflection


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, R: float = EARTH_RADIUS_M
) -> float:
    """
    Calculate great-circle distance between two points in meters using the Haversine formula.
    """
    p1, l1, p2, l2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dp = p2 - p1
    dl = l2 - l1
    a = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two points in kilometers."""
    return haversine_distance(lat1, lon1, lat2, lon2) / 1000.0


def translate_geodesic(
    lat: float, lon: float, dx_meters: float, dy_meters: float, R: float = EARTH_RADIUS_M
) -> Tuple[float, float]:
    """
    Translate a geographic coordinate (lat, lon) by (dx, dy) meters
    where dx is eastward and dy is northward, using exact midpoint geodesic metric projection.
    Guarantees reversible forward/backward numerical trajectory stepping.
    """
    dlat = (dy_meters / R) * (180.0 / math.pi)
    lat_mid = lat + dlat * 0.5
    lat_mid_rad = math.radians(lat_mid)
    cos_mid = math.cos(lat_mid_rad)
    if abs(cos_mid) < 1e-12:
        cos_mid = 1e-12 if cos_mid >= 0 else -1e-12
    dlon = (dx_meters / (R * cos_mid)) * (180.0 / math.pi)

    out_lat = lat + dlat
    out_lon = ((lon + dlon + 180.0) % 360.0) - 180.0
    return float(out_lat), float(out_lon)


def forward_aeqd(
    lats: Union[float, np.ndarray],
    lons: Union[float, np.ndarray],
    lat0: float,
    lon0: float,
    R: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project geographic coordinates (lats, lons) in degrees to local Azimuthal Equidistant (AEQD)
    metric plane coordinates (x, y) in meters centered at (lat0, lon0).
    """
    p0 = math.radians(lat0)
    l0 = math.radians(lon0)

    p = np.radians(np.asarray(lats, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lons, dtype=np.float64))
    dl = lon_rad - l0

    cos_c = np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(dl)
    cos_c = np.clip(cos_c, -1.0, 1.0)
    c = np.arccos(cos_c)

    sin_c = np.sin(c)
    # Handle singularity at center (c = 0)
    k = np.where(sin_c == 0.0, 1.0, c / np.where(sin_c == 0.0, 1.0, sin_c))

    x = R * k * np.cos(p) * np.sin(dl)
    y = R * k * (math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(dl))

    return x, y


def inverse_aeqd(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    lat0: float,
    lon0: float,
    R: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject local Azimuthal Equidistant (AEQD) metric coordinates (x, y) in meters
    back to geographic coordinates (lats, lons) in degrees relative to (lat0, lon0).
    """
    p0 = math.radians(lat0)
    l0 = math.radians(lon0)

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    rho = np.hypot(x_arr, y_arr)
    c = rho / R
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_rho = np.where(rho == 0.0, 1.0, rho)
        p = np.where(
            rho == 0.0,
            p0,
            np.arcsin(cos_c * math.sin(p0) + (y_arr * sin_c * math.cos(p0)) / safe_rho),
        )
        denom = safe_rho * math.cos(p0) * cos_c - y_arr * math.sin(p0) * sin_c
        num = x_arr * sin_c
        lon_rad = np.where(rho == 0.0, l0, l0 + np.arctan2(num, denom))

    lats_deg = np.degrees(p)
    lons_deg = ((np.degrees(lon_rad) + 180.0) % 360.0) - 180.0
    return lats_deg, lons_deg


def apply_coriolis_deflection(
    u_w: float,
    v_w: float,
    deflection_deg: float = DEFAULT_CORIOLIS_DEG,
    lat: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Apply Coriolis deflection to 10m wind vector.
    In Northern Hemisphere (lat >= 0 or None), deflection is clockwise (+15 deg):
        u_w_eff =  u_w * cos(15 deg) + v_w * sin(15 deg)
        v_w_eff = -u_w * sin(15 deg) + v_w * cos(15 deg)
    In Southern Hemisphere (lat < 0), deflection is counter-clockwise (-15 deg).
    """
    # Determine rotation sign based on hemisphere
    if lat is not None and lat < 0.0:
        angle_rad = -math.radians(deflection_deg)
    else:
        angle_rad = math.radians(deflection_deg)

    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # Standard 2D rotation for clockwise deflection in compass coordinates
    u_eff = u_w * cos_theta + v_w * sin_theta
    v_eff = -u_w * sin_theta + v_w * cos_theta
    return float(u_eff), float(v_eff)


def compute_slick_velocity(
    u_c: float,
    v_c: float,
    u_w: float,
    v_w: float,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    deflection_deg: float = DEFAULT_CORIOLIS_DEG,
    lat: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute net surface slick drift velocity:
        V_slick = U_current + wind_factor * R(deflection_deg) * U_wind
    """
    u_w_eff, v_w_eff = apply_coriolis_deflection(
        u_w, v_w, deflection_deg=deflection_deg, lat=lat
    )
    u_slick = u_c + wind_factor * u_w_eff
    v_slick = v_c + wind_factor * v_w_eff
    return float(u_slick), float(v_slick)


def advect_point(
    lat: float,
    lon: float,
    u_curr: float,
    v_curr: float,
    u_wind: float,
    v_wind: float,
    dt_sec: float,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    coriolis_deg: float = DEFAULT_CORIOLIS_DEG,
) -> Tuple[float, float]:
    """
    Advect a single point (lat, lon) with given velocity vectors over duration dt_sec.
    """
    u_s, v_s = compute_slick_velocity(
        u_curr, v_curr, u_wind, v_wind,
        wind_factor=wind_factor,
        deflection_deg=coriolis_deg,
        lat=lat,
    )
    dx = u_s * dt_sec
    dy = v_s * dt_sec
    return translate_geodesic(lat, lon, dx, dy)


def advection_step_euler(
    interpolator: MetoceanInterpolator,
    t: float,
    lat: float,
    lon: float,
    dt: float,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    coriolis_deg: float = DEFAULT_CORIOLIS_DEG,
    u_pert_c: float = 0.0,
    v_pert_c: float = 0.0,
    u_pert_w: float = 0.0,
    v_pert_w: float = 0.0,
) -> Tuple[float, float]:
    """
    Perform a single 1st-order Euler numerical advection step:
        x_{n+1} = x_n + V_slick(t_n, x_n) * dt
    """
    vec = interpolator.interpolate(t, lat, lon)
    u_c = vec.u_current + u_pert_c
    v_c = vec.v_current + v_pert_c
    u_w = vec.u_wind + u_pert_w
    v_w = vec.v_wind + v_pert_w

    u_s, v_s = compute_slick_velocity(
        u_c, v_c, u_w, v_w, wind_factor=wind_factor, deflection_deg=coriolis_deg, lat=lat
    )

    dx = u_s * dt
    dy = v_s * dt
    return translate_geodesic(lat, lon, dx, dy)


def advection_step_rk2(
    interpolator: MetoceanInterpolator,
    t: float,
    lat: float,
    lon: float,
    dt: float,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    coriolis_deg: float = DEFAULT_CORIOLIS_DEG,
    u_pert_c: float = 0.0,
    v_pert_c: float = 0.0,
    u_pert_w: float = 0.0,
    v_pert_w: float = 0.0,
) -> Tuple[float, float]:
    """
    Perform a single 2nd-order Runge-Kutta (RK2 / Midpoint) numerical advection step:
        k1 = V_slick(t_n, x_n)
        x_{mid} = x_n + k1 * (dt / 2)
        k2 = V_slick(t_n + dt / 2, x_{mid})
        x_{n+1} = x_n + k2 * dt
    """
    # Step 1: Initial velocity k1
    vec1 = interpolator.interpolate(t, lat, lon)
    u_c1 = vec1.u_current + u_pert_c
    v_c1 = vec1.v_current + v_pert_c
    u_w1 = vec1.u_wind + u_pert_w
    v_w1 = vec1.v_wind + v_pert_w

    u_s1, v_s1 = compute_slick_velocity(
        u_c1, v_c1, u_w1, v_w1, wind_factor=wind_factor, deflection_deg=coriolis_deg, lat=lat
    )

    # Step 2: Midpoint position
    dx_mid = u_s1 * (dt * 0.5)
    dy_mid = v_s1 * (dt * 0.5)
    lat_mid, lon_mid = translate_geodesic(lat, lon, dx_mid, dy_mid)
    t_mid = t + dt * 0.5

    # Step 3: Midpoint velocity k2
    vec_mid = interpolator.interpolate(t_mid, lat_mid, lon_mid)
    u_cm = vec_mid.u_current + u_pert_c
    v_cm = vec_mid.v_current + v_pert_c
    u_wm = vec_mid.u_wind + u_pert_w
    v_wm = vec_mid.v_wind + v_pert_w

    u_sm, v_sm = compute_slick_velocity(
        u_cm, v_cm, u_wm, v_wm, wind_factor=wind_factor, deflection_deg=coriolis_deg, lat=lat_mid
    )

    # Step 4: Full step displacement
    dx_full = u_sm * dt
    dy_full = v_sm * dt
    return translate_geodesic(lat, lon, dx_full, dy_full)


def batch_advection_step_rk2(
    interpolator: MetoceanInterpolator,
    t: float,
    lats: np.ndarray,
    lons: np.ndarray,
    dt: float,
    wind_factor: float = DEFAULT_WIND_FACTOR,
    coriolis_deg: float = DEFAULT_CORIOLIS_DEG,
    u_pert_c: Optional[np.ndarray] = None,
    v_pert_c: Optional[np.ndarray] = None,
    u_pert_w: Optional[np.ndarray] = None,
    v_pert_w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch RK2 advection step for N particles simultaneously.
    """
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    N = len(lats_arr)

    # Stage 1: Interpolate at initial coordinates
    u_c1, v_c1, u_w1, v_w1 = interpolator.interpolate_batch(t, lats_arr, lons_arr)
    if u_pert_c is not None:
        u_c1 = u_c1 + u_pert_c
        v_c1 = v_c1 + v_pert_c
    if u_pert_w is not None:
        u_w1 = u_w1 + u_pert_w
        v_w1 = v_w1 + v_pert_w

    # Coriolis deflection on wind
    ang_rad = np.where(lats_arr < 0.0, -math.radians(coriolis_deg), math.radians(coriolis_deg))
    cos_t = np.cos(ang_rad)
    sin_t = np.sin(ang_rad)

    u_w1_eff = u_w1 * cos_t + v_w1 * sin_t
    v_w1_eff = -u_w1 * sin_t + v_w1 * cos_t

    u_s1 = u_c1 + wind_factor * u_w1_eff
    v_s1 = v_c1 + wind_factor * v_w1_eff

    # Midpoint translation
    dx_mid = u_s1 * (dt * 0.5)
    dy_mid = v_s1 * (dt * 0.5)
    lat_mid = np.empty(N, dtype=np.float64)
    lon_mid = np.empty(N, dtype=np.float64)
    for i in range(N):
        lat_mid[i], lon_mid[i] = translate_geodesic(lats_arr[i], lons_arr[i], dx_mid[i], dy_mid[i])

    # Stage 2: Interpolate at midpoint coordinates
    t_mid = t + dt * 0.5
    u_cm, v_cm, u_wm, v_wm = interpolator.interpolate_batch(t_mid, lat_mid, lon_mid)
    if u_pert_c is not None:
        u_cm = u_cm + u_pert_c
        v_cm = v_cm + v_pert_c
    if u_pert_w is not None:
        u_wm = u_wm + u_pert_w
        v_wm = v_wm + v_pert_w

    ang_rad_m = np.where(lat_mid < 0.0, -math.radians(coriolis_deg), math.radians(coriolis_deg))
    cos_tm = np.cos(ang_rad_m)
    sin_tm = np.sin(ang_rad_m)

    u_wm_eff = u_wm * cos_tm + v_wm * sin_tm
    v_wm_eff = -u_wm * sin_tm + v_wm * cos_tm

    u_sm = u_cm + wind_factor * u_wm_eff
    v_sm = v_cm + wind_factor * v_wm_eff

    # Full step translation
    dx_full = u_sm * dt
    dy_full = v_sm * dt
    lat_out = np.empty(N, dtype=np.float64)
    lon_out = np.empty(N, dtype=np.float64)
    for i in range(N):
        lat_out[i], lon_out[i] = translate_geodesic(lats_arr[i], lons_arr[i], dx_full[i], dy_full[i])

    return lat_out, lon_out
