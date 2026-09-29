"""
Comprehensive Unit and Integration Tests for Worker M3:
AIS Spatio-Temporal Join & Multi-Factor Scoring Subsystem.
"""

import os
import tempfile
from datetime import datetime, timezone
from shapely.geometry import Polygon

from app.ais.store import AISStore, parse_timestamp
from app.ais.filter import (
    SpatioTemporalFilter,
    compute_bounding_box_from_origin,
    haversine_distance_km,
)
from app.ais.trajectory import (
    TrajectoryEngine,
    interpolate_angle_deg,
)
from app.ais.anomaly import (
    AnomalyDetector,
    point_to_segment_distance_km,
)
from app.scoring.engine import (
    AttributionScoringEngine,
    calculate_trajectory_score,
    map_vessel_type_to_score,
)


class TestAISStore:
    """Tests for SQLite AIS storage engine."""

    def test_init_and_composite_indexes(self):
        with AISStore(":memory:") as store:
            cursor = store._conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = [r[0] for r in cursor.fetchall()]
            assert "idx_ais_time_lat_lon" in indexes
            assert "idx_ais_epoch_lat_lon" in indexes
            assert "idx_ais_mmsi_epoch" in indexes

    def test_parse_timestamp_formats(self):
        # Epoch
        iso, ep = parse_timestamp(1700000000)
        assert iso == "2023-11-14T22:13:20Z"
        assert ep == 1700000000.0

        # ISO String with Z
        iso2, ep2 = parse_timestamp("2026-09-02T12:00:00Z")
        assert ep2 == datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp()

        # Space separated
        iso3, ep3 = parse_timestamp("2026-09-02 12:00:00")
        assert iso3 == "2026-09-02T12:00:00Z"

    def test_insert_and_query_records(self):
        store = AISStore(":memory:")
        records = [
            {
                "MMSI": 419000001,
                "BaseDateTime": "2026-09-02T12:00:00Z",
                "LAT": 18.925,
                "LON": 72.825,
                "SOG": 12.0,
                "COG": 45.0,
                "Heading": 45.0,
                "VesselName": "MT OCEAN VOYAGER",
                "IMO": "IMO9123456",
                "CallSign": "AT1234",
                "VesselType": 81,
                "Status": 0,
                "Length": 250.0,
                "Width": 44.0,
                "Draft": 14.5,
                "Cargo": 80,
            },
            {
                "MMSI": 419000002,
                "BaseDateTime": "2026-09-02T12:10:00Z",
                "LAT": 19.500,
                "LON": 73.500,
                "SOG": 10.0,
                "COG": 180.0,
                "Heading": 180.0,
                "VesselName": "MV CARGO",
                "VesselType": 70,
            },
        ]
        inserted = store.insert_records(records)
        assert inserted == 2
        assert store.count_records() == 2

        # Query bounding box
        res = store.query_bounding_box(
            min_lat=18.8,
            max_lat=19.0,
            min_lon=72.7,
            max_lon=72.9,
            start_time="2026-09-02T11:00:00Z",
            end_time="2026-09-02T13:00:00Z",
        )
        assert len(res) == 1
        assert res[0]["MMSI"] == 419000001
        assert res[0]["VesselName"] == "MT OCEAN VOYAGER"

        meta = store.get_vessel_metadata(419000001)
        assert meta["VesselName"] == "MT OCEAN VOYAGER"
        assert meta["VesselType"] == 81

        mmsis = store.get_distinct_mmsi_in_window(
            18.8, 19.0, 72.7, 72.9, "2026-09-02T11:00:00Z", "2026-09-02T13:00:00Z"
        )
        assert mmsis == [419000001]
        store.close()

    def test_csv_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,VesselType\n")
            f.write("419000001,2026-09-02 12:00:00,18.925,72.825,12.0,45.0,45.0,MT TEST,80\n")
            f.write("419000002,2026-09-02 12:05:00,18.930,72.830,11.5,45.0,45.0,MV TEST2,70\n")
            tmp_path = f.name

        try:
            store = AISStore(":memory:")
            count = store.load_csv(tmp_path)
            assert count == 2
            assert store.count_records() == 2
            store.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSpatioTemporalFilter:
    """Tests for spatial bounding box and spatiotemporal filtering."""

    def test_haversine_accuracy(self):
        # 1 deg latitude difference ~ 111.195 km
        d = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
        assert abs(d - 111.195) < 0.2

    def test_bounding_box_from_polygon(self):
        poly = Polygon([(72.80, 18.90), (72.85, 18.90), (72.85, 18.95), (72.80, 18.95)])
        min_lat, max_lat, min_lon, max_lon, centroid = compute_bounding_box_from_origin(poly, radius_km=10.0)
        assert min_lat < 18.90
        assert max_lat > 18.95
        assert min_lon < 72.80
        assert max_lon > 72.85
        assert abs(centroid[0] - 18.925) < 1e-3
        assert abs(centroid[1] - 72.825) < 1e-3

    def test_filter_candidate_vessels(self):
        store = AISStore(":memory:")
        t_orig = "2026-09-02T12:00:00Z"
        # Close vessel
        store.insert_record({
            "MMSI": 111,
            "BaseDateTime": "2026-09-02T12:05:00Z",
            "LAT": 18.926,
            "LON": 72.826,
            "SOG": 12.0,
            "COG": 45.0,
            "VesselName": "CLOSE TANKER",
            "VesselType": 80,
        })
        # Distant vessel
        store.insert_record({
            "MMSI": 222,
            "BaseDateTime": "2026-09-02T12:05:00Z",
            "LAT": 19.500,
            "LON": 73.500,
            "SOG": 12.0,
            "COG": 45.0,
            "VesselName": "FAR VESSEL",
            "VesselType": 70,
        })

        filt = SpatioTemporalFilter(store, search_radius_km=10.0, temporal_window_hours=3.0)
        candidates = filt.filter_vessels(
            origin_zone=(18.925, 72.825),
            origin_time=t_orig,
        )
        assert len(candidates) == 1
        assert candidates[0]["mmsi"] == 111
        assert candidates[0]["vessel_name"] == "CLOSE TANKER"
        store.close()


class TestTrajectoryReconstruction:
    """Tests for 1-minute trajectory interpolation and CPA."""

    def test_interpolate_angle_deg(self):
        # Shortest circular wrap around 360/0
        interp = interpolate_angle_deg(350.0, 10.0, 0.5)
        assert abs(interp - 0.0) < 1e-4

    def test_compute_cpa_and_track_geojson(self):
        engine = TrajectoryEngine(step_seconds=60)
        fixes = [
            {"timestamp_epoch": 1000, "LAT": 18.00, "LON": 72.00, "SOG": 12.0, "COG": 45.0, "Heading": 45.0},
            {"timestamp_epoch": 1600, "LAT": 18.10, "LON": 72.10, "SOG": 12.0, "COG": 45.0, "Heading": 45.0},
        ]
        res = engine.compute_cpa(
            mmsi=419000001,
            records=fixes,
            origin_centroid=(18.05, 72.05),
            vessel_name="TEST SHIP",
            vessel_type="Tanker",
        )
        assert res.mmsi == 419000001
        assert res.cpa_distance_km < 0.6
        assert res.interpolated_points_count >= 10
        assert res.geojson["type"] == "Feature"
        assert res.geojson["geometry"]["type"] == "LineString"
        assert len(res.geojson["geometry"]["coordinates"]) >= 10


class TestAnomalyDetector:
    """Tests for transponder silence gap and behavioral anomaly detection."""

    def test_point_to_segment_distance(self):
        # Point (0, 1) to segment from (-1, 0) to (1, 0) -> closest point is (0, 0), dist ~ 111.32 km
        d_km, cl_lat, cl_lon = point_to_segment_distance_km(1.0, 0.0, 0.0, -1.0, 0.0, 1.0)
        assert abs(d_km - 111.195) < 1.0
        assert abs(cl_lat - 0.0) < 1e-3
        assert abs(cl_lon - 0.0) < 1e-3

    def test_course_change_and_speed_drop_anomalies(self):
        detector = AnomalyDetector()
        fixes = [
            {"timestamp_epoch": 1000, "LAT": 18.0, "LON": 72.0, "SOG": 14.0, "COG": 10.0},
            {"timestamp_epoch": 1300, "LAT": 18.02, "LON": 72.02, "SOG": 8.0, "COG": 70.0},  # 60 deg turn, 6 kn drop
        ]
        rep = detector.analyze_vessel_behavior(
            mmsi=123,
            records=fixes,
            origin_centroid=(18.01, 72.01),
            origin_time="1970-01-01T00:20:00Z",
        )
        assert rep.has_course_change is True
        assert rep.max_course_change_deg == 60.0
        assert rep.has_speed_drop is True
        assert rep.max_speed_drop_kn == 6.0
        assert rep.behavior_score == 0.70  # operational discharge speed + course change


class TestAttributionScoringEngine:
    """Tests for multi-factor formula, risk mappings, and end-to-end pipeline."""

    def test_vessel_type_risk_mappings(self):
        score_crude, _ = map_vessel_type_to_score(81)
        assert score_crude == 1.00

        score_chem, _ = map_vessel_type_to_score("Chemical Tanker")
        assert score_chem == 0.90

        score_cargo, _ = map_vessel_type_to_score(70)
        assert score_cargo == 0.50

        score_fish, _ = map_vessel_type_to_score(30)
        assert score_fish == 0.20

        score_pass, _ = map_vessel_type_to_score(60)
        assert score_pass == 0.05

    def test_trajectory_score_alignment(self):
        s_align, is_al = calculate_trajectory_score(45.0, drift_angle_deg=40.0)
        assert s_align == 1.00
        assert is_al is True

        s_misalign, is_mis = calculate_trajectory_score(45.0, drift_angle_deg=180.0)
        assert s_misalign == 0.40
        assert is_mis is False

    def test_full_pipeline_ranking(self):
        store = AISStore(":memory:")
        t_orig = "2026-09-02T12:00:00Z"
        
        # 1. Primary Culprit: MT OCEAN VOYAGER (Crude Tanker, 52-min silence across origin, 0.6 km away)
        store.insert_records([
            {"MMSI": 419000001, "BaseDateTime": "2026-09-02T11:30:00Z", "LAT": 18.88, "LON": 72.78, "SOG": 12.0, "COG": 45.0, "VesselName": "MT OCEAN VOYAGER", "VesselType": 81},
            {"MMSI": 419000001, "BaseDateTime": "2026-09-02T12:22:00Z", "LAT": 18.96, "LON": 72.86, "SOG": 12.0, "COG": 45.0, "VesselName": "MT OCEAN VOYAGER", "VesselType": 81},
        ])
        # 2. Fishing boat close but low risk and no silence
        store.insert_records([
            {"MMSI": 419000002, "BaseDateTime": "2026-09-02T11:55:00Z", "LAT": 18.92, "LON": 72.82, "SOG": 5.0, "COG": 90.0, "VesselName": "SAGAR KANYA", "VesselType": 30},
            {"MMSI": 419000002, "BaseDateTime": "2026-09-02T12:05:00Z", "LAT": 18.93, "LON": 72.83, "SOG": 5.0, "COG": 90.0, "VesselName": "SAGAR KANYA", "VesselType": 30},
        ])
        # 3. Cargo vessel 5 km away
        store.insert_records([
            {"MMSI": 419000003, "BaseDateTime": "2026-09-02T11:50:00Z", "LAT": 18.87, "LON": 72.80, "SOG": 10.0, "COG": 45.0, "VesselName": "MV CARGO", "VesselType": 70},
            {"MMSI": 419000003, "BaseDateTime": "2026-09-02T12:10:00Z", "LAT": 18.89, "LON": 72.82, "SOG": 10.0, "COG": 45.0, "VesselName": "MV CARGO", "VesselType": 70},
        ])

        engine = AttributionScoringEngine()
        leaderboard = engine.run_attribution_pipeline(
            store=store,
            origin_zone=(18.925, 72.825),
            origin_time=t_orig,
            drift_angle_deg=45.0,
        )

        assert len(leaderboard) == 3
        top = leaderboard[0]
        assert top["rank"] == 1
        assert top["mmsi"] == 419000001
        assert top["vessel_name"] == "MT OCEAN VOYAGER"
        assert top["confidence_pct"] >= 88.0
        assert "reasons" in top and len(top["reasons"]) >= 3
        assert "forensic_summary" in top
        store.close()
