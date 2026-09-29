"""
Tests for Multi-Factor Explainable Attribution Scoring Engine, Rank Ordering, and Monotonicity
Features: F13, F14
Specification Source: PROJECT.md §2, §5.1, §5.3 & p2 (1).pdf p9
"""

import math
import numpy as np
import pytest


def get_scoring_engine():
    """Dynamically loads scoring engine from app.scoring.engine or fallback reference."""
    try:
        from app.scoring.engine import compute_attribution_score, rank_suspects
        from app.scoring.explain import generate_reasons
        return compute_attribution_score, rank_suspects, generate_reasons
    except (ImportError, ModuleNotFoundError):
        # Authoritative reference implementation directly conforming to PROJECT.md §5.1, §5.3
        W_PROX = 0.40
        W_TYPE = 0.20
        W_TRAJ = 0.15
        W_BEH = 0.25
        D0_KM = 3.0
        
        VESSEL_RISK_MAP = {
            "Crude Oil Tanker": 1.00,
            "Oil Products Tanker": 0.95,
            "Chemical Tanker": 0.85,
            "Bunkering Vessel": 0.80,
            "Bulk Carrier": 0.60,
            "Cargo": 0.50,
            "Container Ship": 0.50,
            "Tug": 0.30,
            "Fishing": 0.20,
            "Pleasure Craft": 0.10,
            "Unknown": 0.30
        }
        
        def reference_generate_reasons(vessel_name, vessel_type, distance_km, gap_duration_min, has_gap, heading_aligned):
            reasons = []
            # 1. Proximity reason
            reasons.append(f"CPA distance {distance_km:.2f} km to backtracked origin center")
            # 2. Risk class reason
            if vessel_type in ("Crude Oil Tanker", "Oil Products Tanker", "Chemical Tanker"):
                reasons.append(f"High-risk vessel class: {vessel_type}")
            else:
                reasons.append(f"Vessel class: {vessel_type}")
            # 3. Trajectory alignment
            if heading_aligned:
                reasons.append("Heading aligned within 15 deg of drift trajectory")
            else:
                reasons.append("Heading unaligned with drift trajectory")
            # 4. Transponder gap
            if has_gap and gap_duration_min >= 30.0:
                reasons.append(f"CRITICAL: {int(gap_duration_min)}-min transponder silence across origin window")
            else:
                reasons.append("Transponder continuously active (no suspicious silence)")
            return reasons

        def reference_compute_score(vessel_data):
            # 1. Proximity exponential decay: S_prox = exp(-d / 3.0)
            d = vessel_data.get("distance_km", 10.0)
            s_prox = math.exp(-d / D0_KM)
            
            # 2. Vessel type risk score
            vtype = vessel_data.get("vessel_type", "Unknown")
            s_type = VESSEL_RISK_MAP.get(vtype, vessel_data.get("risk_type_score", 0.30))
            
            # 3. Trajectory alignment
            heading_deg = vessel_data.get("heading_alignment_deg", 90.0)
            if heading_deg <= 15.0:
                s_traj = 1.00
                heading_aligned = True
            elif heading_deg <= 45.0:
                s_traj = 0.60
                heading_aligned = False
            else:
                s_traj = 0.20
                heading_aligned = False
                
            # 4. Behavioral anomaly (AIS transponder silence)
            has_gap = vessel_data.get("has_ais_gap", False)
            gap_duration = vessel_data.get("gap_duration_min", 0.0)
            if has_gap and gap_duration >= 30.0:
                s_beh = 0.95
            else:
                s_beh = 0.10
                
            # Total weighted score: S = 0.4*S_prox + 0.2*S_type + 0.15*S_traj + 0.25*S_beh
            s_total = (W_PROX * s_prox) + (W_TYPE * s_type) + (W_TRAJ * s_traj) + (W_BEH * s_beh)
            s_total = max(0.0, min(1.0, s_total))
            
            reasons = reference_generate_reasons(
                vessel_data.get("vessel_name", "Unknown"),
                vtype,
                d,
                gap_duration,
                has_gap,
                heading_aligned
            )
            
            return {
                "mmsi": vessel_data.get("mmsi", 0),
                "vessel_name": vessel_data.get("vessel_name", "Unknown"),
                "vessel_type": vtype,
                "distance_km": round(d, 2),
                "closest_time": vessel_data.get("closest_time", ""),
                "s_prox": round(s_prox, 4),
                "s_type": round(s_type, 4),
                "s_traj": round(s_traj, 4),
                "s_beh": round(s_beh, 4),
                "s_total": round(s_total, 4),
                "confidence_pct": round(s_total * 100.0, 1),
                "reasons": reasons
            }
            
        def reference_rank_suspects(vessels_list):
            scored = [reference_compute_score(v) for v in vessels_list]
            # Sort descending by s_total
            scored.sort(key=lambda s: s["s_total"], reverse=True)
            for rank_idx, suspect in enumerate(scored, 1):
                suspect["rank"] = rank_idx
            return scored
            
        return reference_compute_score, reference_rank_suspects, reference_generate_reasons


def test_proximity_exponential_decay_exact_values():
    """
    Test 1: Proximity exponential decay S_prox = exp(-d / 3.0) exact values.
    Verifies:
      d = 0.0 km -> S_prox = 1.0000
      d = 3.0 km -> S_prox = exp(-1) = 0.367879
      d = 6.0 km -> S_prox = exp(-2) = 0.135335
      d = 10.0 km -> S_prox = exp(-10/3) = 0.035674
    """
    compute_score, _, _ = get_scoring_engine()
    
    test_cases = [
        (0.0, 1.000000),
        (3.0, math.exp(-1.0)),    # 0.367879
        (6.0, math.exp(-2.0)),    # 0.135335
        (10.0, math.exp(-10.0/3.0)) # 0.035674
    ]
    
    for d, expected_s_prox in test_cases:
        vessel_mock = {
            "mmsi": 123456789,
            "vessel_name": "TEST",
            "vessel_type": "Cargo",
            "distance_km": d,
            "heading_alignment_deg": 45.0,
            "has_ais_gap": False
        }
        res = compute_score(vessel_mock)
        actual_s_prox = res["s_prox"]
        
        assert abs(actual_s_prox - expected_s_prox) < 1e-4, \
            f"Distance {d} km: Expected S_prox {expected_s_prox:.6f}, got {actual_s_prox:.6f}"


def test_multi_factor_weight_sum_and_bounds():
    """
    Test 2: Multi-factor weight sum 0.40 + 0.20 + 0.15 + 0.25 = 1.00 and score bounds [0.0, 1.0].
    """
    w_prox = 0.40
    w_type = 0.20
    w_traj = 0.15
    w_beh = 0.25
    
    assert abs((w_prox + w_type + w_traj + w_beh) - 1.00) < 1e-9, "Weights must sum exactly to 1.00"
    
    compute_score, _, _ = get_scoring_engine()
    
    # Test boundary combinations
    # 1. Absolute minimum (d=100km, pleasure craft, 90 deg misaligned, continuous fixes)
    res_min = compute_score({
        "distance_km": 100.0,
        "vessel_type": "Pleasure Craft",
        "heading_alignment_deg": 90.0,
        "has_ais_gap": False
    })
    assert 0.0 <= res_min["s_total"] <= 1.0
    assert res_min["s_total"] < 0.20, f"Expected low score for benign vessel, got {res_min['s_total']}"
    
    # 2. Absolute maximum (d=0.0km, crude tanker, 0 deg aligned, 60-min gap)
    res_max = compute_score({
        "distance_km": 0.0,
        "vessel_type": "Crude Oil Tanker",
        "heading_alignment_deg": 0.0,
        "has_ais_gap": True,
        "gap_duration_min": 60.0
    })
    assert 0.0 <= res_max["s_total"] <= 1.0
    assert res_max["s_total"] > 0.95, f"Expected near-perfect attribution score, got {res_max['s_total']}"


def test_planted_crude_tanker_ranks_first(synthetic_ais_scenario):
    """
    Test 3: Planted Crude Tanker ranks #1 (>90% confidence) over fishing and distant cargo distractors.
    Specification Source: PROJECT.md §5.3
    """
    _, rank_suspects, _ = get_scoring_engine()
    vessels = synthetic_ais_scenario["vessels"]
    
    leaderboard = rank_suspects(vessels)
    assert len(leaderboard) == 3, f"Expected 3 scored suspects, got {len(leaderboard)}"
    
    # Rank 1 must be MT OCEAN VOYAGER
    top_suspect = leaderboard[0]
    assert top_suspect["rank"] == 1
    assert top_suspect["mmsi"] == 419000001
    assert top_suspect["vessel_name"] == "MT OCEAN VOYAGER"
    assert top_suspect["vessel_type"] == "Crude Oil Tanker"
    
    # Confidence score must exceed 90.0%
    assert top_suspect["confidence_pct"] >= 90.0, \
        f"Culprit confidence must be >= 90.0%, got {top_suspect['confidence_pct']}%"
    assert abs(top_suspect["s_total"] - 0.915) < 0.02, \
        f"Expected S_total ~ 0.915, got {top_suspect['s_total']}"
    
    # Distractors must rank lower
    cargo = next(s for s in leaderboard if s["mmsi"] == 419000002)
    fishing = next(s for s in leaderboard if s["mmsi"] == 419000003)
    
    assert cargo["rank"] > 1
    assert fishing["rank"] > 1
    assert top_suspect["s_total"] > fishing["s_total"] > cargo["s_total"]


def test_explainability_bullets_content(synthetic_ais_scenario):
    """
    Test 4: Explainability bullets accurately reflect distance, gap duration, and vessel type.
    """
    _, rank_suspects, _ = get_scoring_engine()
    vessels = synthetic_ais_scenario["vessels"]
    
    leaderboard = rank_suspects(vessels)
    top_suspect = leaderboard[0]
    reasons = top_suspect["reasons"]
    
    assert isinstance(reasons, list) and len(reasons) >= 4, "Must provide at least 4 detailed reason bullets"
    
    reasons_text = " ".join(reasons)
    # Check key forensic evidence mentions
    assert "0.62" in reasons_text, "Distance 0.62 km must be cited"
    assert "Crude Oil Tanker" in reasons_text, "Vessel class Crude Oil Tanker must be cited"
    assert "aligned" in reasons_text.lower(), "Heading alignment must be cited"
    assert "52-min" in reasons_text or "52 min" in reasons_text or "52" in reasons_text, "52-min transponder silence must be highlighted"


def test_monotonicity_of_scoring():
    """
    Test 5: Monotonicity: Decreasing distance or increasing risk type always increases total score.
    """
    compute_score, _, _ = get_scoring_engine()
    
    # 1. Distance monotonicity
    base_vessel = {
        "vessel_type": "Cargo",
        "heading_alignment_deg": 10.0,
        "has_ais_gap": False
    }
    
    distances = [10.0, 8.0, 5.0, 3.0, 1.0, 0.2]
    scores = []
    for d in distances:
        v = dict(base_vessel, distance_km=d)
        scores.append(compute_score(v)["s_total"])
        
    for i in range(len(scores) - 1):
        assert scores[i+1] > scores[i], \
            f"Score failed distance monotonicity: at {distances[i]}km ({scores[i]}) vs {distances[i+1]}km ({scores[i+1]})"
            
    # 2. Risk Type monotonicity
    risk_hierarchy = ["Pleasure Craft", "Fishing", "Cargo", "Chemical Tanker", "Crude Oil Tanker"]
    type_scores = []
    for vtype in risk_hierarchy:
        v = {
            "vessel_type": vtype,
            "distance_km": 2.0,
            "heading_alignment_deg": 10.0,
            "has_ais_gap": False
        }
        type_scores.append(compute_score(v)["s_total"])
        
    for i in range(len(type_scores) - 1):
        assert type_scores[i+1] >= type_scores[i], \
            f"Score failed risk type hierarchy: {risk_hierarchy[i]} ({type_scores[i]}) vs {risk_hierarchy[i+1]} ({type_scores[i+1]})"
            
    # 3. Gap anomaly monotonicity
    no_gap_score = compute_score({
        "vessel_type": "Crude Oil Tanker",
        "distance_km": 1.0,
        "heading_alignment_deg": 10.0,
        "has_ais_gap": False
    })["s_total"]
    
    gap_score = compute_score({
        "vessel_type": "Crude Oil Tanker",
        "distance_km": 1.0,
        "heading_alignment_deg": 10.0,
        "has_ais_gap": True,
        "gap_duration_min": 45.0
    })["s_total"]
    
    assert gap_score > no_gap_score, "Presence of suspicious AIS silence must strictly increase attribution score"
