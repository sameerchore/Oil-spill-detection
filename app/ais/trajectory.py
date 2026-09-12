"""
Trajectory Reconstruction Engine.
Interpolates vessel fixes to 1-minute intervals and computes Closest Point of Approach (CPA)
distance (d_km) and CPA timestamp (T_CPA) to the origin zone centroid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.ais.filter import haversine_distance_km
from app.ais.store import parse_timestamp


def interpolate_angle_deg(a1: float, a2: float, alpha: float) -> float:
    """
    Circular shortest-path interpolation between two angles in degrees [0, 360).
    """
    diff = (a2 - a1 + 180.0) % 360.0 - 180.0
    res = (a1 + alpha * diff) % 360.0
    return float(res)


@dataclass
class InterpolatedPoint:
    timestamp: str
    timestamp_epoch: float
    lat: float
    lon: float
    sog: float
    cog: float
    heading: Optional[float]
    distance_to_origin_km: float


@dataclass
class TrajectoryResult:
    mmsi: int
    vessel_name: str
    vessel_type: Optional[Union[int, str]]
    cpa_distance_km: float
    cpa_time: str
    cpa_epoch: float
    cpa_lat: float
    cpa_lon: float
    cpa_sog: float
    cpa_cog: float
    cpa_heading: Optional[float]
    raw_points_count: int
    interpolated_points_count: int
    interpolated_track: List[Dict[str, Any]]
    geojson: Dict[str, Any]


class TrajectoryEngine:
    """
    Trajectory Reconstruction and CPA Analysis Engine.
    """

    def __init__(self, step_seconds: int = 60):
        self.step_seconds = int(step_seconds)

    def interpolate_track(
        self,
        records: List[Dict[str, Any]],
        origin_centroid: Tuple[float, float],
        start_epoch: Optional[float] = None,
        end_epoch: Optional[float] = None,
    ) -> List[InterpolatedPoint]:
        """
        Interpolate raw vessel fixes to fixed step_seconds (default 1-minute / 60s) intervals.
        """
        if not records:
            return []

        # Ensure sorted by epoch
        cleaned_records: List[Tuple[float, float, float, float, float, Optional[float]]] = []
        for r in records:
            t_epoch = float(r.get("timestamp_epoch", 0.0))
            if t_epoch <= 0:
                dt_str = r.get("BaseDateTime") or r.get("datetime")
                if dt_str:
                    _, t_epoch = parse_timestamp(dt_str)
                else:
                    continue

            lat = float(r.get("LAT") if r.get("LAT") is not None else r.get("lat", 0.0))
            lon = float(r.get("LON") if r.get("LON") is not None else r.get("lon", 0.0))
            sog = float(r.get("SOG") if r.get("SOG") is not None else r.get("sog", 0.0))
            cog = float(r.get("COG") if r.get("COG") is not None else r.get("cog", 0.0))
            hdg_val = r.get("Heading") if r.get("Heading") is not None else r.get("heading")
            heading = float(hdg_val) if hdg_val is not None and not (isinstance(hdg_val, float) and np.isnan(hdg_val)) else None

            cleaned_records.append((t_epoch, lat, lon, sog, cog, heading))

        cleaned_records.sort(key=lambda x: x[0])
        if not cleaned_records:
            return []

        # Deduplicate identical consecutive timestamps
        deduped: List[Tuple[float, float, float, float, float, Optional[float]]] = [cleaned_records[0]]
        for rec in cleaned_records[1:]:
            if rec[0] > deduped[-1][0]:
                deduped.append(rec)
            else:
                # Update latest fix at same timestamp
                deduped[-1] = rec

        if len(deduped) == 1:
            t_epoch, lat, lon, sog, cog, hdg = deduped[0]
            d_km = float(haversine_distance_km(lat, lon, origin_centroid[0], origin_centroid[1]))
            iso_time, _ = parse_timestamp(t_epoch)
            return [
                InterpolatedPoint(
                    timestamp=iso_time,
                    timestamp_epoch=t_epoch,
                    lat=round(lat, 6),
                    lon=round(lon, 6),
                    sog=round(sog, 2),
                    cog=round(cog, 2),
                    heading=round(hdg, 2) if hdg is not None else None,
                    distance_to_origin_km=round(d_km, 4),
                )
            ]

        t_min = deduped[0][0] if start_epoch is None else max(deduped[0][0], start_epoch)
        t_max = deduped[-1][0] if end_epoch is None else min(deduped[-1][0], end_epoch)

        if t_min > t_max:
            t_min, t_max = deduped[0][0], deduped[-1][0]

        # Generate target sampling timestamps
        # Align to 60s boundaries
        first_step = math.ceil(t_min / self.step_seconds) * self.step_seconds
        sample_times = np.arange(first_step, t_max + 1e-3, self.step_seconds)

        if len(sample_times) == 0:
            sample_times = np.array([t_min])

        # Also guarantee boundary points are included
        sample_times = np.unique(np.concatenate(([t_min], sample_times, [t_max])))
        sample_times.sort()

        epochs = np.array([r[0] for r in deduped], dtype=np.float64)
        lats = np.array([r[1] for r in deduped], dtype=np.float64)
        lons = np.array([r[2] for r in deduped], dtype=np.float64)
        sogs = np.array([r[3] for r in deduped], dtype=np.float64)
        cogs = np.array([r[4] for r in deduped], dtype=np.float64)

        # Linear lat/lon/sog interpolation
        interp_lats = np.interp(sample_times, epochs, lats)
        interp_lons = np.interp(sample_times, epochs, lons)
        interp_sogs = np.interp(sample_times, epochs, sogs)

        # Angular COG interpolation using sin/cos unwrap
        rad_cogs = np.radians(cogs)
        sin_cogs = np.sin(rad_cogs)
        cos_cogs = np.cos(rad_cogs)
        interp_sin = np.interp(sample_times, epochs, sin_cogs)
        interp_cos = np.interp(sample_times, epochs, cos_cogs)
        interp_cogs = (np.degrees(np.arctan2(interp_sin, interp_cos)) + 360.0) % 360.0

        # Calculate distances to origin centroid
        c_lat, c_lon = origin_centroid
        distances = haversine_distance_km(interp_lats, interp_lons, c_lat, c_lon)

        result_points: List[InterpolatedPoint] = []
        for i, t in enumerate(sample_times):
            iso_t, _ = parse_timestamp(t)
            result_points.append(
                InterpolatedPoint(
                    timestamp=iso_t,
                    timestamp_epoch=float(t),
                    lat=round(float(interp_lats[i]), 6),
                    lon=round(float(interp_lons[i]), 6),
                    sog=round(float(interp_sogs[i]), 2),
                    cog=round(float(interp_cogs[i]), 2),
                    heading=round(float(interp_cogs[i]), 2),
                    distance_to_origin_km=round(float(distances[i]), 4),
                )
            )

        return result_points

    def compute_cpa(
        self,
        mmsi: int,
        records: List[Dict[str, Any]],
        origin_centroid: Tuple[float, float],
        vessel_name: Optional[str] = None,
        vessel_type: Optional[Union[int, str]] = None,
        start_epoch: Optional[float] = None,
        end_epoch: Optional[float] = None,
    ) -> TrajectoryResult:
        """
        Reconstruct 1-minute trajectory and compute CPA (Closest Point of Approach).
        """
        interp_pts = self.interpolate_track(
            records=records,
            origin_centroid=origin_centroid,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
        )

        if not interp_pts:
            raise ValueError(f"No valid fixes to compute trajectory for MMSI {mmsi}")

        # Find CPA
        min_idx = int(np.argmin([p.distance_to_origin_km for p in interp_pts]))
        cpa_pt = interp_pts[min_idx]

        name = vessel_name or str(records[0].get("VesselName") or f"MMSI_{mmsi}")
        vtype = vessel_type if vessel_type is not None else records[0].get("VesselType")

        # Create GeoJSON LineString representation
        coordinates = [[p.lon, p.lat] for p in interp_pts]
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "mmsi": mmsi,
                "vessel_name": name,
                "vessel_type": vtype,
                "cpa_distance_km": cpa_pt.distance_to_origin_km,
                "cpa_time": cpa_pt.timestamp,
                "cpa_sog": cpa_pt.sog,
                "cpa_cog": cpa_pt.cog,
                "points_count": len(interp_pts),
            },
        }

        return TrajectoryResult(
            mmsi=mmsi,
            vessel_name=name,
            vessel_type=vtype,
            cpa_distance_km=cpa_pt.distance_to_origin_km,
            cpa_time=cpa_pt.timestamp,
            cpa_epoch=cpa_pt.timestamp_epoch,
            cpa_lat=cpa_pt.lat,
            cpa_lon=cpa_pt.lon,
            cpa_sog=cpa_pt.sog,
            cpa_cog=cpa_pt.cog,
            cpa_heading=cpa_pt.heading,
            raw_points_count=len(records),
            interpolated_points_count=len(interp_pts),
            interpolated_track=[asdict(p) for p in interp_pts],
            geojson=geojson,
        )
