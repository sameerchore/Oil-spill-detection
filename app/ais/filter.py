"""
High-Performance Spatio-Temporal AIS Filter.
Filters vessel fixes intersecting search radius R <= 10 km around origin zone during origin window [T_origin +/- 3h].
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Polygon, shape

from app.ais.store import AISStore, parse_timestamp

try:
    from app.config import SEARCH_RADIUS_KM, TEMPORAL_WINDOW_HOURS
except ImportError:
    SEARCH_RADIUS_KM = 10.0
    TEMPORAL_WINDOW_HOURS = 3.0

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute great-circle distance between coordinates in kilometers using Haversine formula.
    Supports scalar floats and vectorized NumPy arrays.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    )
    # Clip for floating point stability
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def compute_bounding_box_from_origin(
    origin_zone: Union[Tuple[float, float], List[float], Dict[str, Any], Polygon],
    radius_km: float = SEARCH_RADIUS_KM,
) -> Tuple[float, float, float, float, Tuple[float, float]]:
    """
    Calculate (min_lat, max_lat, min_lon, max_lon, (centroid_lat, centroid_lon))
    for a given origin centroid or polygon expanded by radius_km.
    """
    if isinstance(origin_zone, (tuple, list)):
        if len(origin_zone) == 2:
            c_lat, c_lon = float(origin_zone[0]), float(origin_zone[1])
            min_lat, max_lat = c_lat, c_lat
            min_lon, max_lon = c_lon, c_lon
        else:
            raise ValueError(f"Expected 2 coordinates for origin (lat, lon), got {origin_zone}")
    elif isinstance(origin_zone, dict):
        if "type" in origin_zone and origin_zone["type"] in ("Polygon", "MultiPolygon", "Feature"):
            geom = shape(origin_zone["geometry"] if origin_zone.get("type") == "Feature" else origin_zone)
            min_lon_g, min_lat_g, max_lon_g, max_lat_g = geom.bounds
            c_lon, c_lat = geom.centroid.x, geom.centroid.y
            min_lat, max_lat = min_lat_g, max_lat_g
            min_lon, max_lon = min_lon_g, max_lon_g
        elif "centroid" in origin_zone:
            c_lat, c_lon = origin_zone["centroid"]
            min_lat, max_lat = c_lat, c_lat
            min_lon, max_lon = c_lon, c_lon
        elif "lat" in origin_zone and "lon" in origin_zone:
            c_lat, c_lon = float(origin_zone["lat"]), float(origin_zone["lon"])
            min_lat, max_lat = c_lat, c_lat
            min_lon, max_lon = c_lon, c_lon
        else:
            raise ValueError(f"Unrecognized origin_zone dict format: {origin_zone}")
    elif isinstance(origin_zone, Polygon):
        min_lon_g, min_lat_g, max_lon_g, max_lat_g = origin_zone.bounds
        c_lon, c_lat = origin_zone.centroid.x, origin_zone.centroid.y
        min_lat, max_lat = min_lat_g, max_lat_g
        min_lon, max_lon = min_lon_g, max_lon_g
    else:
        raise TypeError(f"Unsupported origin_zone type: {type(origin_zone)}")

    # Degrees per km
    deg_lat = radius_km / 111.32
    # Adjust for latitude cosine
    cos_lat = max(math.cos(math.radians(c_lat)), 0.01)
    deg_lon = radius_km / (111.32 * cos_lat)

    bbox_min_lat = min_lat - deg_lat
    bbox_max_lat = max_lat + deg_lat
    bbox_min_lon = min_lon - deg_lon
    bbox_max_lon = max_lon + deg_lon

    return bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon, (c_lat, c_lon)


class SpatioTemporalFilter:
    """
    High-performance Spatio-Temporal Query Filter for AIS Tracks.
    """

    def __init__(
        self,
        store: AISStore,
        search_radius_km: float = SEARCH_RADIUS_KM,
        temporal_window_hours: float = TEMPORAL_WINDOW_HOURS,
    ):
        self.store = store
        self.search_radius_km = float(search_radius_km)
        self.temporal_window_hours = float(temporal_window_hours)

    def filter_vessels(
        self,
        origin_zone: Union[Tuple[float, float], List[float], Dict[str, Any], Polygon],
        origin_time: Union[str, dt.datetime, float],
        radius_km: Optional[float] = None,
        window_hours: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter candidate vessels in SQLite whose trajectory intersects or approaches within
        search radius R around the origin zone in window [T_origin - W, T_origin + W].

        Returns a list of candidate vessel dictionaries containing:
        - 'mmsi': int
        - 'vessel_name': str
        - 'vessel_type': int / str
        - 'raw_records': List[dict] (sorted by timestamp)
        - 'min_raw_distance_km': float
        - 'closest_raw_time': str
        - 'centroid': Tuple[float, float]
        """
        r_km = self.search_radius_km if radius_km is None else float(radius_km)
        w_hrs = self.temporal_window_hours if window_hours is None else float(window_hours)

        iso_orig, epoch_orig = parse_timestamp(origin_time)
        start_epoch = epoch_orig - (w_hrs * 3600.0)
        end_epoch = epoch_orig + (w_hrs * 3600.0)

        min_lat, max_lat, min_lon, max_lon, centroid = compute_bounding_box_from_origin(
            origin_zone, radius_km=r_km
        )
        c_lat, c_lon = centroid

        # Query candidates from store
        raw_rows = self.store.query_bounding_box(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            start_time=start_epoch,
            end_time=end_epoch,
        )

        if not raw_rows:
            return []

        # Group fixes by MMSI
        vessels_map: Dict[int, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            mmsi = row["MMSI"]
            vessels_map.setdefault(mmsi, []).append(row)

        candidates: List[Dict[str, Any]] = []

        for mmsi, fixes in vessels_map.items():
            # Sort chronologically
            fixes.sort(key=lambda x: x["timestamp_epoch"])

            # Compute min distance among raw fixes to centroid
            lats = np.array([f["LAT"] for f in fixes], dtype=np.float64)
            lons = np.array([f["LON"] for f in fixes], dtype=np.float64)
            distances = haversine_distance_km(lats, lons, c_lat, c_lon)
            min_dist = float(np.min(distances))
            min_idx = int(np.argmin(distances))

            # Fetch vessel metadata
            metadata = self.store.get_vessel_metadata(mmsi) or {}
            v_name = metadata.get("VesselName") or fixes[0].get("VesselName") or f"MMSI_{mmsi}"
            v_type = metadata.get("VesselType") or fixes[0].get("VesselType")

            # Also fetch full vessel track in the expanded time window if needed
            expanded_track = self.store.get_vessel_records(
                mmsi=mmsi,
                start_time=start_epoch,
                end_time=end_epoch,
            )
            track_to_use = expanded_track if len(expanded_track) >= len(fixes) else fixes

            candidates.append({
                "mmsi": mmsi,
                "vessel_name": str(v_name),
                "vessel_type": v_type,
                "metadata": metadata,
                "raw_records": track_to_use,
                "min_raw_distance_km": round(min_dist, 4),
                "closest_raw_time": fixes[min_idx]["BaseDateTime"],
                "origin_centroid": (c_lat, c_lon),
                "origin_time": iso_orig,
            })

        # Sort candidate vessels by min raw distance
        candidates.sort(key=lambda x: x["min_raw_distance_km"])
        return candidates
