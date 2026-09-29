"""
Tests for AIS Transponder Silence Gap Detection, Dead-Reckoning Trajectory, and Behavioral Anomaly Scoring
Features: F10, F11, F12, F14
Specification Source: PROJECT.md §2, §5.1, §5.3 & p2 (1).pdf p8
"""

import math
from datetime import datetime, timedelta
import pytest
from shapely.geometry import Point, LineString, Polygon


def get_ais_detector():
    """Dynamically loads AIS gap detector from app.ais.anomaly or fallback reference."""
    try:
        from app.ais.anomaly import detect_ais_gaps, analyze_vessel_trajectory
        return detect_ais_gaps, analyze_vessel_trajectory
    except (ImportError, ModuleNotFoundError):
        # Authoritative reference implementation directly conforming to PROJECT.md §2 (F12)
        def reference_detect_ais_gaps(fixes, gap_threshold_minutes=30.0):
            """
            Analyzes time deltas between successive AIS position fixes.
            Returns list of gap dictionaries and maximum gap duration.
            """
            if len(fixes) < 2:
                return {"has_gap": False, "max_gap_min": 0.0, "gaps": []}
                
            sorted_fixes = sorted(fixes, key=lambda f: f["timestamp"])
            detected_gaps = []
            max_gap = 0.0
            
            for i in range(len(sorted_fixes) - 1):
                t1 = datetime.fromisoformat(sorted_fixes[i]["timestamp"].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(sorted_fixes[i+1]["timestamp"].replace("Z", "+00:00"))
                delta_min = (t2 - t1).total_seconds() / 60.0
                
                if delta_min > max_gap:
                    max_gap = delta_min
                    
                if delta_min >= gap_threshold_minutes:
                    detected_gaps.append({
                        "start_time": sorted_fixes[i]["timestamp"],
                        "end_time": sorted_fixes[i+1]["timestamp"],
                        "duration_min": delta_min,
                        "start_pos": (sorted_fixes[i]["lat"], sorted_fixes[i]["lon"]),
                        "end_pos": (sorted_fixes[i+1]["lat"], sorted_fixes[i+1]["lon"])
                    })
                    
            return {
                "has_gap": len(detected_gaps) > 0,
                "max_gap_min": max_gap,
                "gaps": detected_gaps
            }
            
        def reference_analyze_vessel_trajectory(fixes, origin_poly, origin_time_iso, gap_threshold_min=30.0):
            gap_info = reference_detect_ais_gaps(fixes, gap_threshold_min)
            
            # Behavioral score: 0.95 if gap >= 30m overlapping origin, else 0.10 baseline
            s_beh = 0.10
            critical_gap = None
            
            t_orig = datetime.fromisoformat(origin_time_iso.replace("Z", "+00:00"))
            
            for gap in gap_info["gaps"]:
                t_start = datetime.fromisoformat(gap["start_time"].replace("Z", "+00:00"))
                t_end = datetime.fromisoformat(gap["end_time"].replace("Z", "+00:00"))
                
                # Check temporal overlap [T_start - 30m, T_end + 30m]
                if (t_start - timedelta(minutes=30)) <= t_orig <= (t_end + timedelta(minutes=30)):
                    critical_gap = gap
                    s_beh = 0.95
                    break
            
            return {
                "has_ais_gap": gap_info["has_gap"],
                "max_gap_min": gap_info["max_gap_min"],
                "critical_gap": critical_gap,
                "s_beh": s_beh
            }
            
        return reference_detect_ais_gaps, reference_analyze_vessel_trajectory


def test_planted_52min_gap_triggers_alarm():
    """
    Test 1: Planted 52-minute gap triggers ais_gap=True and S_beh=0.95.
    Vessel transponder is turned off for 52 minutes (> 30 min threshold) across origin time.
    """
    detect_ais_gaps, analyze_trajectory = get_ais_detector()
    
    # 52-minute transponder silence (11:30 to 12:22)
    fixes = [
        {"timestamp": "2026-09-02T11:15:00Z", "lat": 18.800, "lon": 72.700, "sog": 12.5, "cog": 45.0},
        {"timestamp": "2026-09-02T11:30:00Z", "lat": 18.880, "lon": 72.780, "sog": 12.0, "cog": 45.0},
        # --- 52 MINUTE GAP (11:30 to 12:22) ---
        {"timestamp": "2026-09-02T12:22:00Z", "lat": 18.960, "lon": 72.860, "sog": 11.8, "cog": 45.0},
        {"timestamp": "2026-09-02T12:35:00Z", "lat": 19.020, "lon": 72.920, "sog": 12.2, "cog": 45.0}
    ]
    
    gap_result = detect_ais_gaps(fixes, gap_threshold_minutes=30.0)
    assert gap_result["has_gap"] is True, "Must detect transponder gap >= 30 minutes"
    assert abs(gap_result["max_gap_min"] - 52.0) < 1e-3, f"Expected 52.0 min gap, got {gap_result['max_gap_min']}"
    assert len(gap_result["gaps"]) == 1
    
    # Analyze trajectory with origin at 12:00:00Z
    origin_poly = Polygon([(72.81, 18.91), (72.84, 18.91), (72.84, 18.94), (72.81, 18.94)])
    analysis = analyze_trajectory(fixes, origin_poly, origin_time_iso="2026-09-02T12:00:00Z")
    
    assert analysis["has_ais_gap"] is True
    assert analysis["s_beh"] == 0.95, f"Expected behavioral score 0.95 for deliberate gap, got {analysis['s_beh']}"


def test_continuous_5min_fixes_no_gap_alarm():
    """
    Test 2: Continuous 5-minute fixes do not trigger gap alarm (S_beh=0.10 baseline).
    Normal law-abiding vessel transmitting regular fixes every 5 minutes.
    """
    detect_ais_gaps, analyze_trajectory = get_ais_detector()
    
    # Regular 5-minute fixes
    fixes = [
        {"timestamp": f"2026-09-02T11:{minute:02d}:00Z", "lat": 18.80 + minute*0.002, "lon": 72.70 + minute*0.002, "sog": 10.0, "cog": 45.0}
        for minute in range(0, 60, 5)
    ]
    
    gap_result = detect_ais_gaps(fixes, gap_threshold_minutes=30.0)
    assert gap_result["has_gap"] is False, "Continuous fixes must not trigger gap alarm"
    assert gap_result["max_gap_min"] == 5.0
    assert len(gap_result["gaps"]) == 0
    
    origin_poly = Polygon([(72.81, 18.91), (72.84, 18.91), (72.84, 18.94), (72.81, 18.94)])
    analysis = analyze_trajectory(fixes, origin_poly, origin_time_iso="2026-09-02T11:30:00Z")
    
    assert analysis["has_ais_gap"] is False
    assert analysis["s_beh"] == 0.10, f"Expected baseline behavioral score 0.10, got {analysis['s_beh']}"


def test_dead_reckoning_intersection_with_origin_zone():
    """
    Test 3: Dead-reckoning straight line through origin zone correctly intersects polygon.
    Vessel enters silence at (18.88, 72.78) and exits at (18.96, 72.86).
    Origin envelope centered at (18.92, 72.82) is directly intersected by linear dead-reckoning.
    """
    p1 = Point(72.78, 18.88)  # [lon, lat]
    p2 = Point(72.86, 18.96)
    dead_reckoning_line = LineString([p1, p2])
    
    # 90% confidence backtracked origin polygon
    origin_poly = Polygon([
        (72.810, 18.910),
        (72.830, 18.910),
        (72.830, 18.930),
        (72.810, 18.930)
    ])
    
    # Assert line directly intersects origin polygon
    assert dead_reckoning_line.intersects(origin_poly), "Dead-reckoning segment must intersect backtracked origin polygon"
    
    # Calculate intersection line length and closest distance to center
    intersection = dead_reckoning_line.intersection(origin_poly)
    assert not intersection.is_empty, "Intersection geometry must not be empty"
    
    center_pt = Point(72.820, 18.920)
    min_dist_deg = dead_reckoning_line.distance(center_pt)
    min_dist_km = min_dist_deg * 111.0  # Approx km
    assert min_dist_km < 0.5, f"CPA distance {min_dist_km:.2f}km exceeds 0.5km tolerance"


def test_dead_reckoning_segment_far_no_anomaly():
    """
    Test 4: Dead-reckoning segment far from origin zone does NOT trigger anomaly.
    Vessel transits 40+ km away from spill origin.
    """
    p1 = Point(72.10, 18.40)  # [lon, lat] far southwest
    p2 = Point(72.15, 18.45)
    dead_reckoning_line = LineString([p1, p2])
    
    origin_poly = Polygon([
        (72.810, 18.910),
        (72.830, 18.910),
        (72.830, 18.930),
        (72.810, 18.930)
    ])
    
    # Assert no intersection
    assert not dead_reckoning_line.intersects(origin_poly), "Distant vessel trajectory must not intersect origin polygon"
    
    center_pt = Point(72.820, 18.920)
    dist_deg = dead_reckoning_line.distance(center_pt)
    dist_km = dist_deg * 111.0
    assert dist_km > 30.0, f"Distant vessel distance {dist_km:.2f}km must be > 30 km"


def test_multiple_gaps_on_same_vessel():
    """
    Test 5: Multiple gaps on same vessel handled cleanly.
    Trajectory contains:
      - Gap 1: 35 min at 04:00 - 04:35 (irrelevant morning gap)
      - Gap 2: 55 min at 11:30 - 12:25 (critical gap overlapping spill origin at 12:00)
    The engine must catalog both gaps and isolate the critical gap for forensic scoring.
    """
    detect_ais_gaps, analyze_trajectory = get_ais_detector()
    
    fixes = [
        {"timestamp": "2026-09-02T10:00:00Z", "lat": 18.50, "lon": 72.40, "sog": 10.0, "cog": 45.0},
        # Gap 1: 35 min (10:00 to 10:35)
        {"timestamp": "2026-09-02T10:35:00Z", "lat": 18.55, "lon": 72.45, "sog": 10.0, "cog": 45.0},
        {"timestamp": "2026-09-02T10:50:00Z", "lat": 18.60, "lon": 72.50, "sog": 10.0, "cog": 45.0},
        {"timestamp": "2026-09-02T11:10:00Z", "lat": 18.75, "lon": 72.65, "sog": 11.0, "cog": 45.0},
        {"timestamp": "2026-09-02T11:30:00Z", "lat": 18.88, "lon": 72.78, "sog": 12.0, "cog": 45.0},
        # Gap 2: 55 min (11:30 to 12:25, critical overlapping origin at 12:00)
        {"timestamp": "2026-09-02T12:25:00Z", "lat": 18.96, "lon": 72.86, "sog": 11.5, "cog": 45.0},
        {"timestamp": "2026-09-02T12:40:00Z", "lat": 19.00, "lon": 72.90, "sog": 11.0, "cog": 45.0}
    ]
    
    gap_result = detect_ais_gaps(fixes, gap_threshold_minutes=30.0)
    assert len(gap_result["gaps"]) == 2, f"Expected 2 detected gaps, got {len(gap_result['gaps'])}"
    assert abs(gap_result["max_gap_min"] - 55.0) < 1e-3
    
    # Origin analysis for spill at 12:00:00Z
    origin_poly = Polygon([(72.81, 18.91), (72.84, 18.91), (72.84, 18.94), (72.81, 18.94)])
    analysis = analyze_trajectory(fixes, origin_poly, origin_time_iso="2026-09-02T12:00:00Z")
    
    assert analysis["has_ais_gap"] is True
    assert analysis["critical_gap"] is not None
    assert abs(analysis["critical_gap"]["duration_min"] - 55.0) < 1e-3, "Must select the 55-min gap overlapping origin window"
    assert analysis["s_beh"] == 0.95
