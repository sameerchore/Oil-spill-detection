"""
AIS Subsystem for TideTrace.
Provides MarineCadastre SQLite store, spatio-temporal filtering, 1-min trajectory reconstruction,
and dark vessel / behavioral anomaly detection.
"""

from app.ais.anomaly import (
    AnomalyDetector,
    AnomalyReport,
    GapDetail,
    analyze_vessel_trajectory,
    angular_difference_deg,
    detect_ais_gaps,
    point_to_segment_distance_km,
)
from app.ais.filter import (
    SpatioTemporalFilter,
    compute_bounding_box_from_origin,
    haversine_distance_km,
)
from app.ais.store import AISStore, MARINECADASTRE_COLUMNS, parse_timestamp
from app.ais.trajectory import (
    InterpolatedPoint,
    TrajectoryEngine,
    TrajectoryResult,
    interpolate_angle_deg,
)

__all__ = [
    "AISStore",
    "MARINECADASTRE_COLUMNS",
    "parse_timestamp",
    "SpatioTemporalFilter",
    "compute_bounding_box_from_origin",
    "haversine_distance_km",
    "TrajectoryEngine",
    "TrajectoryResult",
    "InterpolatedPoint",
    "interpolate_angle_deg",
    "AnomalyDetector",
    "AnomalyReport",
    "GapDetail",
    "detect_ais_gaps",
    "analyze_vessel_trajectory",
    "point_to_segment_distance_km",
    "angular_difference_deg",
]
