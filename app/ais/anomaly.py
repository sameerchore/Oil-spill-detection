"""
Dark Vessel and Behavioral Anomaly Detector.
Detects transponder silence gaps (dt >= 30 min) with dead-reckoning near origin (<= 5 km),
operational discharge speed profile (8 <= SOG <= 16 kn), abrupt course changes (|dCOG| > 30 deg),
and sudden speed drops (> 5 kn).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.ais.filter import haversine_distance_km
from app.ais.store import parse_timestamp

try:
    from app.config import GAP_THRESHOLD_MINUTES
except ImportError:
    GAP_THRESHOLD_MINUTES = 30.0

GAP_ORIGIN_PROXIMITY_KM = 5.0


def point_to_segment_distance_km(
    p_lat: float,
    p_lon: float,
    seg_start_lat: float,
    seg_start_lon: float,
    seg_end_lat: float,
    seg_end_lon: float,
) -> Tuple[float, float, float]:
    """
    Calculate the minimum geodesic distance in km from a point (p_lat, p_lon)
    to a line segment (seg_start -> seg_end) via local projected coordinates,
    and return (min_dist_km, closest_lat, closest_lon).
    """
    cos_lat = math.cos(math.radians(p_lat))
    x1 = (seg_start_lon - p_lon) * 111.32 * cos_lat
    y1 = (seg_start_lat - p_lat) * 111.32
    x2 = (seg_end_lon - p_lon) * 111.32 * cos_lat
    y2 = (seg_end_lat - p_lat) * 111.32

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-9:
        d_km = float(haversine_distance_km(p_lat, p_lon, seg_start_lat, seg_start_lon))
        return d_km, seg_start_lat, seg_start_lon

    u = (-x1 * dx - y1 * dy) / seg_len_sq
    u_clamped = max(0.0, min(1.0, u))

    x_close = x1 + u_clamped * dx
    y_close = y1 + u_clamped * dy

    close_lat = p_lat + (y_close / 111.32)
    close_lon = p_lon + (x_close / (111.32 * max(cos_lat, 0.01)))

    exact_d_km = float(haversine_distance_km(p_lat, p_lon, close_lat, close_lon))
    return exact_d_km, close_lat, close_lon


def angular_difference_deg(deg1: float, deg2: float) -> float:
    """
    Compute minimal angular difference in degrees [0, 180].
    """
    diff = abs(deg1 - deg2) % 360.0
    return min(diff, 360.0 - diff)


@dataclass
class GapDetail:
    gap_duration_minutes: float
    start_time: str
    end_time: str
    start_epoch: float
    end_epoch: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    min_distance_to_origin_km: float
    closest_lat: float
    closest_lon: float
    crosses_origin_zone: bool


@dataclass
class AnomalyReport:
    mmsi: int
    vessel_name: str
    has_gap: bool
    max_gap_minutes: float
    gaps: List[Dict[str, Any]]
    gap_crosses_origin: bool
    critical_gap: Optional[Dict[str, Any]]
    is_operational_discharge_speed: bool
    cpa_sog: float
    has_course_change: bool
    max_course_change_deg: float
    has_speed_drop: bool
    max_speed_drop_kn: float
    behavior_score: float
    flags: List[str]


def detect_ais_gaps(
    fixes: List[Dict[str, Any]],
    gap_threshold_minutes: float = GAP_THRESHOLD_MINUTES,
) -> Dict[str, Any]:
    """
    Analyzes time deltas between successive AIS position fixes.
    Returns dictionary with has_gap, max_gap_min, and list of detected gap objects.
    """
    if len(fixes) < 2:
        return {"has_gap": False, "max_gap_min": 0.0, "gaps": []}

    clean_fixes = []
    for f in fixes:
        t_raw = f.get("timestamp") or f.get("BaseDateTime") or f.get("timestamp_epoch")
        if t_raw is None:
            continue
        iso_str, epoch = parse_timestamp(t_raw)
        lat = float(f.get("LAT") if f.get("LAT") is not None else f.get("lat", 0.0))
        lon = float(f.get("LON") if f.get("LON") is not None else f.get("lon", 0.0))
        clean_fixes.append({"iso": iso_str, "epoch": epoch, "lat": lat, "lon": lon})

    clean_fixes.sort(key=lambda x: x["epoch"])
    detected_gaps = []
    max_gap = 0.0

    for i in range(len(clean_fixes) - 1):
        dt_min = (clean_fixes[i + 1]["epoch"] - clean_fixes[i]["epoch"]) / 60.0
        if dt_min > max_gap:
            max_gap = dt_min
        if dt_min >= gap_threshold_minutes:
            detected_gaps.append({
                "start_time": clean_fixes[i]["iso"],
                "end_time": clean_fixes[i + 1]["iso"],
                "duration_min": round(dt_min, 1),
                "start_pos": (clean_fixes[i]["lat"], clean_fixes[i]["lon"]),
                "end_pos": (clean_fixes[i + 1]["lat"], clean_fixes[i + 1]["lon"]),
            })

    return {
        "has_gap": len(detected_gaps) > 0,
        "max_gap_min": round(max_gap, 1),
        "gaps": detected_gaps,
    }


def analyze_vessel_trajectory(
    fixes: List[Dict[str, Any]],
    origin_poly: Any,
    origin_time_iso: str,
    gap_threshold_min: float = GAP_THRESHOLD_MINUTES,
) -> Dict[str, Any]:
    """
    Analyzes vessel trajectory against origin polygon and origin time.
    """
    gap_info = detect_ais_gaps(fixes, gap_threshold_min)
    s_beh = 0.10
    critical_gap = None

    _, t_orig_epoch = parse_timestamp(origin_time_iso)

    for gap in gap_info["gaps"]:
        _, t_start_epoch = parse_timestamp(gap["start_time"])
        _, t_end_epoch = parse_timestamp(gap["end_time"])

        # Check temporal overlap [T_start - 30m, T_end + 30m]
        if (t_start_epoch - 1800.0) <= t_orig_epoch <= (t_end_epoch + 1800.0):
            critical_gap = gap
            s_beh = 0.95
            break

    return {
        "has_ais_gap": gap_info["has_gap"],
        "max_gap_min": gap_info["max_gap_min"],
        "critical_gap": critical_gap,
        "s_beh": s_beh,
    }


class AnomalyDetector:
    """
    Behavioral Anomaly & Dark Vessel Detector.
    """

    def __init__(
        self,
        gap_threshold_minutes: float = GAP_THRESHOLD_MINUTES,
        gap_origin_proximity_km: float = GAP_ORIGIN_PROXIMITY_KM,
    ):
        self.gap_threshold_minutes = float(gap_threshold_minutes)
        self.gap_origin_proximity_km = float(gap_origin_proximity_km)

    def detect_transponder_gaps(
        self,
        records: List[Dict[str, Any]],
        origin_centroid: Tuple[float, float],
        origin_epoch: Optional[float] = None,
        window_seconds: float = 3.0 * 3600.0,
    ) -> List[GapDetail]:
        """
        Identify consecutive fixes with dt >= gap_threshold_minutes and check dead-reckoning
        intercept to origin zone.
        """
        if len(records) < 2:
            return []

        clean_fixes = []
        for r in records:
            t_epoch = float(r.get("timestamp_epoch", 0.0))
            if t_epoch <= 0:
                dt_str = r.get("BaseDateTime") or r.get("datetime") or r.get("timestamp")
                if dt_str:
                    _, t_epoch = parse_timestamp(dt_str)
                else:
                    continue
            lat = float(r.get("LAT") if r.get("LAT") is not None else r.get("lat", 0.0))
            lon = float(r.get("LON") if r.get("LON") is not None else r.get("lon", 0.0))
            clean_fixes.append((t_epoch, lat, lon))

        clean_fixes.sort(key=lambda x: x[0])
        gaps: List[GapDetail] = []
        c_lat, c_lon = origin_centroid

        for i in range(len(clean_fixes) - 1):
            t1, lat1, lon1 = clean_fixes[i]
            t2, lat2, lon2 = clean_fixes[i + 1]

            dt_min = (t2 - t1) / 60.0
            if dt_min >= self.gap_threshold_minutes:
                is_temporally_relevant = True
                if origin_epoch is not None:
                    if t2 < (origin_epoch - window_seconds) or t1 > (origin_epoch + window_seconds):
                        is_temporally_relevant = False

                if not is_temporally_relevant:
                    continue

                min_d_km, cl_lat, cl_lon = point_to_segment_distance_km(
                    c_lat, c_lon, lat1, lon1, lat2, lon2
                )
                crosses = min_d_km <= self.gap_origin_proximity_km

                t1_iso, _ = parse_timestamp(t1)
                t2_iso, _ = parse_timestamp(t2)

                gaps.append(
                    GapDetail(
                        gap_duration_minutes=round(dt_min, 1),
                        start_time=t1_iso,
                        end_time=t2_iso,
                        start_epoch=t1,
                        end_epoch=t2,
                        start_lat=round(lat1, 6),
                        start_lon=round(lon1, 6),
                        end_lat=round(lat2, 6),
                        end_lon=round(lon2, 6),
                        min_distance_to_origin_km=round(min_d_km, 4),
                        closest_lat=round(cl_lat, 6),
                        closest_lon=round(cl_lon, 6),
                        crosses_origin_zone=crosses,
                    )
                )

        return gaps

    def analyze_vessel_behavior(
        self,
        mmsi: int,
        records: List[Dict[str, Any]],
        origin_centroid: Tuple[float, float],
        origin_time: Optional[Union[str, dt.datetime, float]] = None,
        cpa_sog: Optional[float] = None,
        vessel_name: Optional[str] = None,
    ) -> AnomalyReport:
        """
        Perform comprehensive behavioral anomaly analysis for a vessel.
        """
        name = vessel_name or str(records[0].get("VesselName") or f"MMSI_{mmsi}")
        origin_epoch = None
        if origin_time is not None:
            _, origin_epoch = parse_timestamp(origin_time)

        # 1. Detect gaps
        gap_details = self.detect_transponder_gaps(
            records=records,
            origin_centroid=origin_centroid,
            origin_epoch=origin_epoch,
        )

        has_gap = len(gap_details) > 0
        max_gap = max([g.gap_duration_minutes for g in gap_details], default=0.0)
        gap_crosses_origin = any(g.crosses_origin_zone for g in gap_details)

        critical_gap_obj = None
        for g in gap_details:
            if g.crosses_origin_zone:
                critical_gap_obj = asdict(g)
                break

        # 2. Operational discharge speed check (8 <= SOG <= 16 kn)
        if cpa_sog is None:
            sogs = [
                float(r.get("SOG") or r.get("sog") or 0.0)
                for r in records
                if (r.get("SOG") is not None or r.get("sog") is not None)
            ]
            cpa_sog = float(np.mean(sogs)) if sogs else 0.0

        is_op_discharge = (8.0 <= cpa_sog <= 16.0)

        # 3. Course change check (|dCOG| > 30 deg)
        max_course_change = 0.0
        sorted_fixes = sorted(records, key=lambda x: float(x.get("timestamp_epoch", 0.0)))

        cogs = [
            (
                float(r.get("timestamp_epoch", 0.0)),
                float(r.get("COG") if r.get("COG") is not None else r.get("cog", 0.0)),
            )
            for r in sorted_fixes
            if (r.get("COG") is not None or r.get("cog") is not None)
        ]

        for i in range(len(cogs) - 1):
            t_i, cog_i = cogs[i]
            t_j, cog_j = cogs[i + 1]
            if abs(t_j - t_i) <= 1800:
                ang_diff = angular_difference_deg(cog_i, cog_j)
                if ang_diff > max_course_change:
                    max_course_change = ang_diff

        has_course_change = max_course_change > 30.0

        # 4. Speed drop check (> 5 kn)
        max_speed_drop = 0.0
        sog_series = [
            (
                float(r.get("timestamp_epoch", 0.0)),
                float(r.get("SOG") if r.get("SOG") is not None else r.get("sog", 0.0)),
            )
            for r in sorted_fixes
            if (r.get("SOG") is not None or r.get("sog") is not None)
        ]

        for i in range(len(sog_series) - 1):
            t_i, sog_i = sog_series[i]
            t_j, sog_j = sog_series[i + 1]
            if abs(t_j - t_i) <= 3600:
                drop = sog_i - sog_j
                if drop > max_speed_drop:
                    max_speed_drop = drop

        has_speed_drop = max_speed_drop > 5.0

        # Compile flags
        flags: List[str] = []
        if gap_crosses_origin:
            flags.append(f"ais_gap_{int(max_gap)}min")
        elif has_gap:
            flags.append(f"transponder_silence_{int(max_gap)}min")

        if is_op_discharge:
            flags.append(f"sog_{round(cpa_sog, 1)}kn_discharge_band")

        if has_course_change:
            flags.append(f"course_change_{round(max_course_change, 1)}deg")

        if has_speed_drop:
            flags.append(f"speed_drop_{round(max_speed_drop, 1)}kn")

        scores: List[float] = [0.10]
        if gap_crosses_origin:
            scores.append(0.95)
        if is_op_discharge:
            scores.append(0.70)
        if has_course_change:
            scores.append(0.60)
        if has_speed_drop:
            scores.append(0.50)

        behavior_score = float(max(scores))

        return AnomalyReport(
            mmsi=mmsi,
            vessel_name=name,
            has_gap=has_gap,
            max_gap_minutes=round(max_gap, 1),
            gaps=[asdict(g) for g in gap_details],
            gap_crosses_origin=gap_crosses_origin,
            critical_gap=critical_gap_obj,
            is_operational_discharge_speed=is_op_discharge,
            cpa_sog=round(cpa_sog, 2),
            has_course_change=has_course_change,
            max_course_change_deg=round(max_course_change, 1),
            has_speed_drop=has_speed_drop,
            max_speed_drop_kn=round(max_speed_drop, 1),
            behavior_score=round(behavior_score, 3),
            flags=flags,
        )
