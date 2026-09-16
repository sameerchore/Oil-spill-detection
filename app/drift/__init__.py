"""
Lagrangian Drift Physics Engine for TideTrace.
Provides 2D surface advection, Coriolis leeway deflection, backward origin backtracking,
and forward dispersion forecasting.
"""

from app.drift.advection import (
    EARTH_RADIUS_M,
    DEFAULT_WIND_FACTOR,
    DEFAULT_CORIOLIS_DEG,
    haversine_distance,
    haversine_distance_km,
    translate_geodesic,
    forward_aeqd,
    inverse_aeqd,
    apply_coriolis_deflection,
    compute_slick_velocity,
    advect_point,
    advection_step_euler,
    advection_step_rk2,
    batch_advection_step_rk2,
)
from app.drift.hindcast import (
    run_hindcast,
    seed_particles_across_slick,
    compute_confidence_envelope_geojson,
)
from app.drift.forecast import (
    run_forecast,
    compute_dispersion_cone_geojson,
)

__all__ = [
    "EARTH_RADIUS_M",
    "DEFAULT_WIND_FACTOR",
    "DEFAULT_CORIOLIS_DEG",
    "haversine_distance",
    "haversine_distance_km",
    "translate_geodesic",
    "forward_aeqd",
    "inverse_aeqd",
    "apply_coriolis_deflection",
    "compute_slick_velocity",
    "advect_point",
    "advection_step_euler",
    "advection_step_rk2",
    "batch_advection_step_rk2",
    "run_hindcast",
    "seed_particles_across_slick",
    "compute_confidence_envelope_geojson",
    "run_forecast",
    "compute_dispersion_cone_geojson",
]
