"""
Multi-Factor Attribution Scoring Engine for TideTrace.
Calculates S_total = 0.40*S_prox + 0.20*S_type + 0.15*S_traj + 0.25*S_beh in [0, 1.0]
and produces ranked suspect leaderboards with forensic explanation bullets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.ais.anomaly import AnomalyDetector, angular_difference_deg
from app.ais.filter import SpatioTemporalFilter
from app.ais.store import AISStore
from app.ais.trajectory import TrajectoryEngine
from app.scoring.explain import (
    format_forensic_summary,
    generate_reason_bullets,
    generate_reasons,
)

try:
    from app.config import (
        PROXIMITY_DECAY_D0_KM,
        SEARCH_RADIUS_KM,
        TEMPORAL_WINDOW_HOURS,
        WEIGHT_BEH,
        WEIGHT_PROX,
        WEIGHT_TRAJ,
        WEIGHT_TYPE,
    )
except ImportError:
    WEIGHT_PROX = 0.40
    WEIGHT_TYPE = 0.20
    WEIGHT_TRAJ = 0.15
    WEIGHT_BEH = 0.25
    PROXIMITY_DECAY_D0_KM = 3.0
    SEARCH_RADIUS_KM = 10.0
    TEMPORAL_WINDOW_HOURS = 3.0


# Standard Vessel Type Scoring dictionary mapping
VESSEL_TYPE_SCORES: Dict[str, float] = {
    "crude_oil_tanker": 1.00,
    "crude_tanker": 1.00,
    "oil_tanker": 1.00,
    "oil_products_tanker": 0.95,
    "chemical_tanker": 0.90,
    "generic_tanker": 0.85,
    "tanker": 0.85,
    "bunkering_vessel": 0.80,
    "gas_tanker": 0.85,
    "lng_tanker": 0.85,
    "lpg_tanker": 0.85,
    "bulk_cargo": 0.55,
    "bulk_carrier": 0.55,
    "cargo": 0.50,
    "general_cargo": 0.50,
    "container": 0.45,
    "container_ship": 0.45,
    "fishing": 0.20,
    "fishing_vessel": 0.20,
    "passenger": 0.05,
    "passenger_ship": 0.05,
    "cruise": 0.05,
    "ferry": 0.05,
    "tug": 0.30,
    "towing": 0.30,
    "dredger": 0.30,
    "military": 0.30,
    "pleasure_craft": 0.10,
    "sailing": 0.10,
    "unknown": 0.30,
    "other": 0.30,
}


def map_vessel_type_to_score(vessel_type: Optional[Union[str, int, float]]) -> Tuple[float, str]:
    """
    Resolve numeric MarineCadastre VesselType code or string to (s_type, normalized_name).
    """
    if vessel_type is None:
        return 0.30, "Unknown"

    if isinstance(vessel_type, (int, float)) or (isinstance(vessel_type, str) and vessel_type.strip().isdigit()):
        code = int(vessel_type)
        if code in (80, 81):
            return 1.00, "Crude Oil Tanker"
        elif code == 82:
            return 0.90, "Chemical Tanker"
        elif 80 <= code <= 89:
            return 0.85, "Generic Tanker"
        elif code in (71, 72):
            return 0.55, "Bulk Cargo"
        elif code in (73, 74):
            return 0.45, "Container Ship"
        elif 70 <= code <= 79:
            return 0.50, "Cargo"
        elif code == 30:
            return 0.20, "Fishing"
        elif 60 <= code <= 69:
            return 0.05, "Passenger"
        elif code in (52, 31, 32):
            return 0.30, "Tug"
        else:
            return 0.30, f"Other (Code {code})"

    s_val = str(vessel_type).strip().lower().replace("-", "_").replace(" ", "_")

    if s_val in VESSEL_TYPE_SCORES:
        score = VESSEL_TYPE_SCORES[s_val]
        name = str(vessel_type).strip()
        return score, name

    if "crude" in s_val:
        return 1.00, "Crude Oil Tanker"
    if "chemical" in s_val:
        return 0.90, "Chemical Tanker"
    if "tanker" in s_val:
        return 0.85, "Generic Tanker"
    if "bulk" in s_val:
        return 0.55, "Bulk Carrier"
    if "container" in s_val:
        return 0.45, "Container Ship"
    if "cargo" in s_val:
        return 0.50, "Cargo"
    if "fish" in s_val:
        return 0.20, "Fishing"
    if "passenger" in s_val or "cruise" in s_val or "ferry" in s_val:
        return 0.05, "Passenger"
    if "pleasure" in s_val or "sailing" in s_val:
        return 0.10, "Pleasure Craft"
    if "tug" in s_val or "tow" in s_val:
        return 0.30, "Tug"

    return 0.30, str(vessel_type).strip()


def calculate_proximity_score(distance_km: float, d0: float = PROXIMITY_DECAY_D0_KM) -> float:
    """
    S_prox = exp(-d_km / d0). If inside origin zone (d_km <= 0), S_prox = 1.0.
    """
    d = max(0.0, float(distance_km))
    return float(np.exp(-d / d0))


def calculate_trajectory_score(
    vessel_cog: float,
    drift_angle_deg: Optional[float] = None,
    threshold_deg: float = 45.0,
) -> Tuple[float, bool]:
    """
    S_traj = 1.0 if |COG_CPA - theta_drift| <= 45 deg, else 0.40.
    Returns (score, is_aligned).
    """
    if drift_angle_deg is None:
        return 1.0, True

    diff = angular_difference_deg(vessel_cog, drift_angle_deg)
    is_aligned = (diff <= threshold_deg)
    score = 1.00 if is_aligned else 0.40
    return score, is_aligned


def compute_attribution_score(vessel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct scoring function for dictionary-based candidate records.
    Conforms to test_scoring.py and PROJECT.md §5.3.
    """
    # 1. Proximity exponential decay: S_prox = exp(-d / 3.0)
    d = float(vessel_data.get("distance_km", 10.0))
    s_prox = calculate_proximity_score(d, d0=PROXIMITY_DECAY_D0_KM)

    # 2. Vessel type risk score
    vtype = vessel_data.get("vessel_type", "Unknown")
    if "risk_type_score" in vessel_data and vessel_data["risk_type_score"] is not None:
        s_type = float(vessel_data["risk_type_score"])
        vtype_name = str(vtype)
    else:
        s_type, vtype_name = map_vessel_type_to_score(vtype)

    # 3. Trajectory alignment
    heading_deg = vessel_data.get("heading_alignment_deg")
    if heading_deg is not None:
        if heading_deg <= 15.0:
            s_traj = 1.00
            heading_aligned = True
        elif heading_deg <= 45.0:
            s_traj = 0.60
            heading_aligned = False
        else:
            s_traj = 0.20
            heading_aligned = False
    elif "vessel_cog" in vessel_data:
        s_traj, heading_aligned = calculate_trajectory_score(
            vessel_data["vessel_cog"],
            vessel_data.get("drift_angle_deg"),
            threshold_deg=45.0,
        )
    else:
        s_traj = 1.00
        heading_aligned = True

    # 4. Behavioral anomaly (AIS transponder silence)
    has_gap = bool(vessel_data.get("has_ais_gap", False))
    gap_duration = float(vessel_data.get("gap_duration_min", 0.0))
    if has_gap and gap_duration >= 30.0:
        s_beh = 0.95
    elif "s_beh" in vessel_data:
        s_beh = float(vessel_data["s_beh"])
    else:
        s_beh = 0.10

    # Total weighted score: S = 0.4*S_prox + 0.2*S_type + 0.15*S_traj + 0.25*S_beh
    s_total = (WEIGHT_PROX * s_prox) + (WEIGHT_TYPE * s_type) + (WEIGHT_TRAJ * s_traj) + (WEIGHT_BEH * s_beh)
    s_total = max(0.0, min(1.0, float(s_total)))

    reasons = generate_reasons(
        vessel_name=vessel_data.get("vessel_name", "Unknown"),
        vessel_type=vtype_name,
        distance_km=d,
        gap_duration_min=gap_duration,
        has_gap=has_gap,
        heading_aligned=heading_aligned,
    )

    return {
        "mmsi": vessel_data.get("mmsi", 0),
        "vessel_name": vessel_data.get("vessel_name", "Unknown"),
        "vessel_type": vtype_name,
        "distance_km": round(d, 2),
        "closest_time": vessel_data.get("closest_time", ""),
        "s_prox": round(s_prox, 4),
        "s_type": round(s_type, 4),
        "s_traj": round(s_traj, 4),
        "s_beh": round(s_beh, 4),
        "s_total": round(s_total, 4),
        "confidence_pct": round(s_total * 100.0, 1),
        "reasons": reasons,
    }


def rank_suspects(vessels_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score all candidate vessels and rank in descending order of s_total.
    """
    scored = [compute_attribution_score(v) for v in vessels_list]
    scored.sort(key=lambda s: s["s_total"], reverse=True)
    for rank_idx, suspect in enumerate(scored, start=1):
        suspect["rank"] = rank_idx
    return scored


@dataclass
class ScoredSuspect:
    mmsi: int
    vessel_name: str
    vessel_type: str
    distance_km: float
    closest_time: str
    s_prox: float
    s_type: float
    s_traj: float
    s_beh: float
    s_total: float
    confidence_pct: float
    rank: int
    reasons: List[str]
    cpa_sog: float
    cpa_cog: float
    anomaly_flags: List[str]
    trajectory_points_count: int
    geojson: Dict[str, Any]
    forensic_summary: str
    cpa_lat: float = 0.0
    cpa_lon: float = 0.0
    interpolated_track: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi,
            "vessel_name": self.vessel_name,
            "vessel_type": self.vessel_type,
            "distance_km": round(self.distance_km, 2),
            "closest_time": self.closest_time,
            "closest_approach_time": self.closest_time,
            "s_prox": round(self.s_prox, 3),
            "s_type": round(self.s_type, 3),
            "s_traj": round(self.s_traj, 3),
            "s_beh": round(self.s_beh, 3),
            "s_total": round(self.s_total, 3),
            "score": round(self.s_total, 3),
            "confidence_pct": round(self.confidence_pct, 1),
            "rank": self.rank,
            "reasons": self.reasons,
            "cpa_sog": round(self.cpa_sog, 2),
            "cpa_cog": round(self.cpa_cog, 2),
            "cpa_lat": round(self.cpa_lat, 5),
            "cpa_lon": round(self.cpa_lon, 5),
            "anomaly_flags": self.anomaly_flags,
            "trajectory_points_count": self.trajectory_points_count,
            "geojson": self.geojson,
            "interpolated_track": self.interpolated_track,
            "forensic_summary": self.forensic_summary,
        }


class AttributionScoringEngine:
    """
    Multi-Factor Attribution Scoring Engine.
    Executes end-to-end AIS spatiotemporal filtering, trajectory reconstruction,
    behavioral anomaly detection, and multi-factor ranking.
    """

    def __init__(
        self,
        weight_prox: float = WEIGHT_PROX,
        weight_type: float = WEIGHT_TYPE,
        weight_traj: float = WEIGHT_TRAJ,
        weight_beh: float = WEIGHT_BEH,
        d0_km: float = PROXIMITY_DECAY_D0_KM,
    ):
        self.w_prox = float(weight_prox)
        self.w_type = float(weight_type)
        self.w_traj = float(weight_traj)
        self.w_beh = float(weight_beh)
        self.d0_km = float(d0_km)

        self.trajectory_engine = TrajectoryEngine(step_seconds=60)
        self.anomaly_detector = AnomalyDetector()

    def score_single_vessel(
        self,
        mmsi: int,
        records: List[Dict[str, Any]],
        origin_centroid: Tuple[float, float],
        origin_time: str,
        drift_angle_deg: Optional[float] = None,
        vessel_name: Optional[str] = None,
        vessel_type: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute multi-factor score for a single candidate vessel's AIS records.
        """
        # 1. Trajectory reconstruction & CPA
        traj_res = self.trajectory_engine.compute_cpa(
            mmsi=mmsi,
            records=records,
            origin_centroid=origin_centroid,
            vessel_name=vessel_name,
            vessel_type=vessel_type,
        )

        d_km = traj_res.cpa_distance_km
        s_prox = calculate_proximity_score(d_km, d0=self.d0_km)

        # 2. Vessel Type Score
        s_type, v_type_name = map_vessel_type_to_score(traj_res.vessel_type)

        # 3. Trajectory / Drift Alignment Score
        s_traj, is_traj_aligned = calculate_trajectory_score(
            vessel_cog=traj_res.cpa_cog,
            drift_angle_deg=drift_angle_deg,
            threshold_deg=45.0,
        )

        # 4. Behavioral Anomaly Score
        anomaly_rep = self.anomaly_detector.analyze_vessel_behavior(
            mmsi=mmsi,
            records=records,
            origin_centroid=origin_centroid,
            origin_time=origin_time,
            cpa_sog=traj_res.cpa_sog,
            vessel_name=traj_res.vessel_name,
        )
        s_beh = anomaly_rep.behavior_score

        # 5. Total Multi-Factor Score: S = 0.4*S_prox + 0.2*S_type + 0.15*S_traj + 0.25*S_beh
        s_total = (
            self.w_prox * s_prox
            + self.w_type * s_type
            + self.w_traj * s_traj
            + self.w_beh * s_beh
        )
        s_total = max(0.0, min(1.0, float(s_total)))
        confidence_pct = s_total * 100.0

        # Generate reasons
        reasons = generate_reason_bullets(
            cpa_distance_km=d_km,
            closest_time=traj_res.cpa_time,
            vessel_type_name=v_type_name,
            s_type=s_type,
            cog_cpa=traj_res.cpa_cog,
            drift_angle=drift_angle_deg,
            traj_aligned=is_traj_aligned,
            anomaly_report=asdict(anomaly_rep),
            s_beh=s_beh,
        )

        return {
            "mmsi": mmsi,
            "vessel_name": traj_res.vessel_name,
            "vessel_type": v_type_name,
            "distance_km": round(d_km, 4),
            "closest_time": traj_res.cpa_time,
            "s_prox": round(s_prox, 4),
            "s_type": round(s_type, 4),
            "s_traj": round(s_traj, 4),
            "s_beh": round(s_beh, 4),
            "s_total": round(s_total, 4),
            "confidence_pct": round(confidence_pct, 2),
            "reasons": reasons,
            "cpa_sog": traj_res.cpa_sog,
            "cpa_cog": traj_res.cpa_cog,
            "cpa_lat": traj_res.cpa_lat,
            "cpa_lon": traj_res.cpa_lon,
            "anomaly_flags": anomaly_rep.flags,
            "anomaly_report": asdict(anomaly_rep),
            "trajectory_points_count": traj_res.interpolated_points_count,
            "interpolated_track": traj_res.interpolated_track,
            "geojson": traj_res.geojson,
        }

    def score_and_rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        origin_centroid: Tuple[float, float],
        origin_time: str,
        drift_angle_deg: Optional[float] = None,
    ) -> List[ScoredSuspect]:
        """
        Score all candidate vessels and rank them in descending order of S_total.
        """
        scored_list: List[Dict[str, Any]] = []
        for cand in candidates:
            mmsi = cand["mmsi"]
            records = cand.get("raw_records") or []
            if not records:
                continue

            v_name = cand.get("vessel_name")
            v_type = cand.get("vessel_type")

            score_dict = self.score_single_vessel(
                mmsi=mmsi,
                records=records,
                origin_centroid=origin_centroid,
                origin_time=origin_time,
                drift_angle_deg=drift_angle_deg,
                vessel_name=v_name,
                vessel_type=v_type,
            )
            scored_list.append(score_dict)

        scored_list.sort(key=lambda x: (-x["s_total"], x["distance_km"]))

        ranked_suspects: List[ScoredSuspect] = []
        for rank_idx, s in enumerate(scored_list, start=1):
            narrative = format_forensic_summary(
                rank=rank_idx,
                mmsi=s["mmsi"],
                vessel_name=s["vessel_name"],
                vessel_type=s["vessel_type"],
                confidence_pct=s["confidence_pct"],
                s_total=s["s_total"],
                cpa_distance_km=s["distance_km"],
                closest_time=s["closest_time"],
                reasons=s["reasons"],
            )

            suspect = ScoredSuspect(
                mmsi=s["mmsi"],
                vessel_name=s["vessel_name"],
                vessel_type=s["vessel_type"],
                distance_km=s["distance_km"],
                closest_time=s["closest_time"],
                s_prox=s["s_prox"],
                s_type=s["s_type"],
                s_traj=s["s_traj"],
                s_beh=s["s_beh"],
                s_total=s["s_total"],
                confidence_pct=s["confidence_pct"],
                rank=rank_idx,
                reasons=s["reasons"],
                cpa_sog=s["cpa_sog"],
                cpa_cog=s["cpa_cog"],
                anomaly_flags=s["anomaly_flags"],
                trajectory_points_count=s["trajectory_points_count"],
                geojson=s["geojson"],
                forensic_summary=narrative,
                cpa_lat=s.get("cpa_lat", 0.0),
                cpa_lon=s.get("cpa_lon", 0.0),
                interpolated_track=s.get("interpolated_track", []),
            )
            ranked_suspects.append(suspect)

        return ranked_suspects

    def run_attribution_pipeline(
        self,
        store: AISStore,
        origin_zone: Union[Tuple[float, float], List[float], Dict[str, Any]],
        origin_time: str,
        drift_angle_deg: Optional[float] = None,
        radius_km: Optional[float] = None,
        window_hours: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute full attribution pipeline from SQLite store -> Filter -> Trajectory -> Anomaly -> Scoring -> Leaderboard.
        """
        ais_filter = SpatioTemporalFilter(
            store=store,
            search_radius_km=radius_km or SEARCH_RADIUS_KM,
            temporal_window_hours=window_hours or TEMPORAL_WINDOW_HOURS,
        )

        candidates = ais_filter.filter_vessels(
            origin_zone=origin_zone,
            origin_time=origin_time,
            radius_km=radius_km,
            window_hours=window_hours,
        )

        if not candidates:
            return []

        c_lat, c_lon = candidates[0]["origin_centroid"]

        ranked_suspects = self.score_and_rank_candidates(
            candidates=candidates,
            origin_centroid=(c_lat, c_lon),
            origin_time=origin_time,
            drift_angle_deg=drift_angle_deg,
        )

        return [s.to_dict() for s in ranked_suspects]
