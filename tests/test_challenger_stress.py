
import math
import socket
import pytest
import numpy as np
from shapely.geometry import Polygon, Point
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import get_active_ais_store
from app.config import (
    WIND_FACTOR,
    CORIOLIS_DEFLECTION_DEG,
    SEARCH_RADIUS_KM,
    TEMPORAL_WINDOW_HOURS,
    GAP_THRESHOLD_MINUTES,
    PROXIMITY_DECAY_D0_KM,
)
from app.drift.advection import (
    translate_geodesic,
    haversine_distance,
    haversine_distance_km,
    forward_aeqd,
    inverse_aeqd,
    apply_coriolis_deflection,
    compute_slick_velocity,
    advect_point,
    advection_step_euler,
    advection_step_rk2,
    batch_advection_step_rk2,
)
from app.geo.morphology import (
    compute_polygon_morphology,
    forward_aeqd as geo_forward_aeqd,
    inverse_aeqd as geo_inverse_aeqd,
)
from app.metocean.interpolator import MetoceanInterpolator
from app.metocean.store import MetoceanGrid
from app.ais.trajectory import TrajectoryEngine, interpolate_angle_deg
from app.ais.anomaly import (
    AnomalyDetector,
    detect_ais_gaps,
    point_to_segment_distance_km,
    angular_difference_deg,
)
from app.ais.filter import SpatioTemporalFilter
from app.scoring.engine import (
    AttributionScoringEngine,
    calculate_proximity_score,
    calculate_trajectory_score,
    compute_attribution_score,
    map_vessel_type_to_score,
)


class MockMetoceanInterpolator:
    def __init__(self, u_curr=0.0, v_curr=0.0, u_wind=0.0, v_wind=0.0, fn=None):
        self.u_c = u_curr
        self.v_c = v_curr
        self.u_w = u_wind
        self.v_w = v_wind
        self.fn = fn

    def interpolate(self, t, lat, lon):
        from dataclasses import dataclass
        @dataclass
        class MetoceanVector:
            u_current: float
            v_current: float
            u_wind: float
            v_wind: float

        if self.fn is not None:
            uc, vc, uw, vw = self.fn(t, lat, lon)
            return MetoceanVector(uc, vc, uw, vw)
        return MetoceanVector(self.u_c, self.v_c, self.u_w, self.v_w)

    def interpolate_batch(self, t, lats, lons):
        N = len(lats)
        if self.fn is not None:
            ucs = np.zeros(N)
            vcs = np.zeros(N)
            uws = np.zeros(N)
            vws = np.zeros(N)
            for i in range(N):
                uc, vc, uw, vw = self.fn(t, lats[i], lons[i])
                ucs[i] = uc
                vcs[i] = vc
                uws[i] = uw
                vws[i] = vw
            return ucs, vcs, uws, vws
        return (
            np.full(N, self.u_c, dtype=np.float64),
            np.full(N, self.v_c, dtype=np.float64),
            np.full(N, self.u_w, dtype=np.float64),
            np.full(N, self.v_w, dtype=np.float64),
        )


class TestPhysicalEdgeCases:
    def test_zero_wind_and_zero_current(self):
        lat0, lon0 = 18.95, 72.82
        interp = MockMetoceanInterpolator(u_curr=0.0, v_curr=0.0, u_wind=0.0, v_wind=0.0)
        lat1, lon1 = advection_step_rk2(interp, t=0.0, lat=lat0, lon=lon0, dt=3600.0)
        dist = haversine_distance(lat0, lon0, lat1, lon1)
        assert dist == pytest.approx(0.0, abs=1e-6)
        assert not math.isnan(lat1)
        assert not math.isnan(lon1)

        lats_in = np.full(100, lat0)
        lons_in = np.full(100, lon0)
        lats_out, lons_out = batch_advection_step_rk2(interp, t=0.0, lats=lats_in, lons=lons_in, dt=3600.0)
        assert np.allclose(lats_out, lat0, atol=1e-8)
        assert np.allclose(lons_out, lon0, atol=1e-8)

    def test_gale_force_winds_30ms_to_50ms(self):
        lat0, lon0 = 20.0, 70.0
        u_w = 30.0
        v_w = 0.0
        u_c = 0.0
        v_c = 0.0

        u_s, v_s = compute_slick_velocity(u_c, v_c, u_w, v_w, wind_factor=0.03, deflection_deg=15.0, lat=lat0)
        expected_speed = 30.0 * 0.03
        actual_speed = math.hypot(u_s, v_s)
        assert actual_speed == pytest.approx(expected_speed, rel=1e-5)

        interp = MockMetoceanInterpolator(u_curr=0.0, v_curr=0.0, u_wind=30.0, v_wind=0.0)
        lat1, lon1 = advection_step_rk2(interp, t=0.0, lat=lat0, lon=lon0, dt=3600.0)
        dist_km = haversine_distance_km(lat0, lon0, lat1, lon1)
        expected_dist_km = (0.90 * 3600.0) / 1000.0
        assert dist_km == pytest.approx(expected_dist_km, rel=1e-3)

        interp50 = MockMetoceanInterpolator(u_curr=1.5, v_curr=-0.5, u_wind=50.0, v_wind=30.0)
        lat_50, lon_50 = advection_step_rk2(interp50, t=0.0, lat=lat0, lon=lon0, dt=86400.0)
        assert not math.isnan(lat_50)
        assert not math.isnan(lon_50)
        assert -90.0 <= lat_50 <= 90.0
        assert -180.0 <= lon_50 <= 180.0

    def test_polar_coordinates_stability(self):
        polar_lats = [85.0, -85.0, 89.9, -89.9]
        for p_lat in polar_lats:
            p_lon = 45.0
            lat_t, lon_t = translate_geodesic(p_lat, p_lon, dx_meters=1000.0, dy_meters=1000.0)
            assert not math.isnan(lat_t)
            assert not math.isnan(lon_t)
            assert -90.0 <= lat_t <= 90.0
            assert -180.0 <= lon_t <= 180.0

            x, y = forward_aeqd(p_lat, p_lon, lat0=p_lat, lon0=p_lon)
            assert abs(x) < 1e-4 and abs(y) < 1e-4
            rev_lat, rev_lon = inverse_aeqd(1000.0, 2000.0, lat0=p_lat, lon0=p_lon)
            x_back, y_back = forward_aeqd(rev_lat, rev_lon, lat0=p_lat, lon0=p_lon)
            assert x_back == pytest.approx(1000.0, abs=1.0)
            assert y_back == pytest.approx(2000.0, abs=1.0)

            ring = [
                [p_lon - 0.05, p_lat - 0.01],
                [p_lon + 0.05, p_lat - 0.01],
                [p_lon + 0.05, p_lat + 0.01],
                [p_lon - 0.05, p_lat + 0.01],
                [p_lon - 0.05, p_lat - 0.01],
            ]
            poly = Polygon(ring)
            props = compute_polygon_morphology(poly)
            assert props['area_km2'] > 0.0
            assert props['perimeter_km'] > 0.0
            assert 0.0 <= props['orientation_deg'] <= 180.0
            assert not math.isnan(props['aspect_ratio'])

    def test_equator_crossing_and_coriolis_deflection(self):
        u_w = 0.0
        v_w = 20.0

        u_eff_nh, v_eff_nh = apply_coriolis_deflection(u_w, v_w, deflection_deg=15.0, lat=1.0)
        assert u_eff_nh > 0.0
        assert v_eff_nh > 0.0

        u_eff_sh, v_eff_sh = apply_coriolis_deflection(u_w, v_w, deflection_deg=15.0, lat=-1.0)
        assert u_eff_sh < 0.0
        assert v_eff_sh > 0.0

        interp = MockMetoceanInterpolator(u_curr=0.0, v_curr=1.0, u_wind=0.0, v_wind=0.0)
        lat_eq = -0.05
        lon_eq = 80.0
        lat_cross, lon_cross = advection_step_rk2(interp, t=0.0, lat=lat_eq, lon=lon_eq, dt=3 * 3600.0)
        assert lat_cross > 0.0
        assert not math.isnan(lat_cross)


class TestMathematicalStability:
    def test_rk2_integration_under_sharp_velocity_gradients(self):
        def sharp_shear_field(t, lat, lon):
            u_c = 5.0 * math.tanh(500.0 * (lat - 19.0))
            v_c = 0.5 * math.sin(lon)
            return u_c, v_c, 0.0, 0.0

        interp = MockMetoceanInterpolator(fn=sharp_shear_field)
        lats_init = np.linspace(18.99, 19.01, 50)
        lons_init = np.full(50, 72.0)

        for step in range(10):
            lats_out, lons_out = batch_advection_step_rk2(
                interp, t=step * 300.0, lats=lats_init, lons=lons_init, dt=300.0
            )
            assert not np.isnan(lats_out).any()
            assert not np.isnan(lons_out).any()
            assert np.all(lats_out > 18.0) and np.all(lats_out < 20.0)
            lats_init, lons_init = lats_out, lons_out

    def test_backward_stepping_reversibility_uniform_and_multistep(self):
        lat0, lon0 = 18.925, 72.825

        # 1. Uniform velocity field: analytical reversibility is exact (0.000 mm)
        interp_uni = MockMetoceanInterpolator(u_curr=1.2, v_curr=-0.8, u_wind=10.0, v_wind=5.0)
        lat_fwd_u, lon_fwd_u = advection_step_rk2(interp_uni, t=0.0, lat=lat0, lon=lon0, dt=3600.0)
        lat_rev_u, lon_rev_u = advection_step_rk2(interp_uni, t=3600.0, lat=lat_fwd_u, lon=lon_fwd_u, dt=-3600.0)
        err_u = haversine_distance(lat0, lon0, lat_rev_u, lon_rev_u)
        assert err_u < 1e-5, f'Uniform field reversibility error: {err_u:.8f} meters'

        # 2. Smooth non-uniform field with multi-step integration (dt=60s)
        def smooth_field(t, lat, lon):
            u = 0.5 + 0.2 * math.sin(lat * 10.0)
            v = -0.3 + 0.1 * math.cos(lon * 10.0)
            return u, v, 5.0, -2.0

        interp_smooth = MockMetoceanInterpolator(fn=smooth_field)
        dt = 60.0
        steps = 60
        c_lat, c_lon = lat0, lon0
        for s in range(steps):
            c_lat, c_lon = advection_step_rk2(interp_smooth, t=s*dt, lat=c_lat, lon=c_lon, dt=dt)
        for s in range(steps, 0, -1):
            c_lat, c_lon = advection_step_rk2(interp_smooth, t=s*dt, lat=c_lat, lon=c_lon, dt=-dt)

        err_smooth = haversine_distance(lat0, lon0, c_lat, c_lon)
        assert err_smooth < 1e-4, f'Non-uniform multi-step reversibility error: {err_smooth:.8f} meters'

    def test_covariance_matrix_singularity_in_pca(self):
        line_ring_h = [[72.80, 18.90], [72.82, 18.90], [72.84, 18.90], [72.82, 18.90], [72.80, 18.90]]
        poly_h = Polygon(line_ring_h).buffer(0.0001)
        props_h = compute_polygon_morphology(poly_h)
        assert 0.0 <= props_h['orientation_deg'] <= 180.0
        assert not math.isnan(props_h['aspect_ratio'])

        line_ring_v = [[72.80, 18.90], [72.80, 18.92], [72.80, 18.94], [72.80, 18.92], [72.80, 18.90]]
        poly_v = Polygon(line_ring_v).buffer(0.0001)
        props_v = compute_polygon_morphology(poly_v)
        assert 0.0 <= props_v['orientation_deg'] <= 180.0
        assert not math.isnan(props_v['aspect_ratio'])

        square_ring = [[72.80, 18.90], [72.82, 18.90], [72.82, 18.92], [72.80, 18.92], [72.80, 18.90]]
        poly_sq = Polygon(square_ring)
        props_sq = compute_polygon_morphology(poly_sq)
        assert 0.0 <= props_sq['orientation_deg'] <= 180.0
        assert props_sq['compactness'] > 0.70


class TestAISBoundaryCases:
    def test_vessel_with_zero_fixes(self):
        traj_engine = TrajectoryEngine()
        origin_centroid = (18.95, 72.82)

        pts = traj_engine.interpolate_track([], origin_centroid)
        assert pts == []

        gap_res = detect_ais_gaps([], gap_threshold_minutes=30.0)
        assert gap_res['has_gap'] is False
        assert gap_res['gaps'] == []

        detector = AnomalyDetector()
        gaps = detector.detect_transponder_gaps([], origin_centroid)
        assert gaps == []

        with pytest.raises(ValueError, match='No valid fixes'):
            traj_engine.compute_cpa(mmsi=999999999, records=[], origin_centroid=origin_centroid)

    def test_vessel_with_overlapping_timestamps(self):
        traj_engine = TrajectoryEngine(step_seconds=60)
        origin_centroid = (18.95, 72.82)

        records = [
            {'timestamp_epoch': 1000.0, 'LAT': 18.90, 'LON': 72.80, 'SOG': 12.0, 'COG': 45.0, 'Heading': 45.0},
            {'timestamp_epoch': 1000.0, 'LAT': 18.901, 'LON': 72.801, 'SOG': 12.0, 'COG': 45.0, 'Heading': 45.0},
            {'timestamp_epoch': 1060.0, 'LAT': 18.91, 'LON': 72.81, 'SOG': 12.0, 'COG': 45.0, 'Heading': 45.0},
            {'timestamp_epoch': 1060.0, 'LAT': 18.91, 'LON': 72.81, 'SOG': 12.0, 'COG': 45.0, 'Heading': 45.0},
            {'timestamp_epoch': 1120.0, 'LAT': 18.92, 'LON': 72.82, 'SOG': 12.0, 'COG': 45.0, 'Heading': 45.0},
        ]

        res = traj_engine.compute_cpa(mmsi=123456789, records=records, origin_centroid=origin_centroid)
        assert res.cpa_distance_km > 0.0
        assert len(res.interpolated_track) >= 3
        assert res.raw_points_count == 5

    def test_exact_silence_gap_threshold_29_9_vs_30_0_vs_30_1_minutes(self):
        t0 = 1788350400.0

        fixes_29_9 = [
            {'timestamp_epoch': t0, 'LAT': 18.90, 'LON': 72.80},
            {'timestamp_epoch': t0 + (29.9 * 60.0), 'LAT': 18.92, 'LON': 72.82},
        ]
        res_29_9 = detect_ais_gaps(fixes_29_9, gap_threshold_minutes=30.0)
        assert res_29_9['has_gap'] is False
        assert len(res_29_9['gaps']) == 0

        fixes_30_0 = [
            {'timestamp_epoch': t0, 'LAT': 18.90, 'LON': 72.80},
            {'timestamp_epoch': t0 + (30.0 * 60.0), 'LAT': 18.92, 'LON': 72.82},
        ]
        res_30_0 = detect_ais_gaps(fixes_30_0, gap_threshold_minutes=30.0)
        assert res_30_0['has_gap'] is True
        assert len(res_30_0['gaps']) == 1
        assert res_30_0['gaps'][0]['duration_min'] == 30.0

        fixes_30_1 = [
            {'timestamp_epoch': t0, 'LAT': 18.90, 'LON': 72.80},
            {'timestamp_epoch': t0 + (30.1 * 60.0), 'LAT': 18.92, 'LON': 72.82},
        ]
        res_30_1 = detect_ais_gaps(fixes_30_1, gap_threshold_minutes=30.0)
        assert res_30_1['has_gap'] is True
        assert len(res_30_1['gaps']) == 1
        assert res_30_1['gaps'][0]['duration_min'] == 30.1

    def test_cpa_exactly_at_polygon_boundary_and_zero_distance(self):
        s_prox_zero = calculate_proximity_score(distance_km=0.0, d0=3.0)
        assert s_prox_zero == pytest.approx(1.0000, rel=1e-6)

        s_prox_d0 = calculate_proximity_score(distance_km=3.0, d0=3.0)
        assert s_prox_d0 == pytest.approx(math.exp(-1.0), rel=1e-5)

        d_km, cl_lat, cl_lon = point_to_segment_distance_km(
            p_lat=18.91, p_lon=72.81,
            seg_start_lat=18.90, seg_start_lon=72.80,
            seg_end_lat=18.92, seg_end_lon=72.82,
        )
        assert d_km == pytest.approx(0.0, abs=0.02)


class TestAirGappedNetworkIsolation:
    def test_zero_network_traffic_during_full_api_run(self, monkeypatch):
        orig_connect = socket.socket.connect

        def forbid_outbound_network(self, addr):
            host = addr[0] if isinstance(addr, tuple) else addr
            if host in ('127.0.0.1', 'localhost', '::1', 0):
                return orig_connect(self, addr)
            raise RuntimeError(f'AIR-GAP BREACH ATTEMPTED: Outbound network connection to {addr}!')

        monkeypatch.setattr(socket.socket, 'connect', forbid_outbound_network)

        client = TestClient(app)

        # 1. Health check
        r_health = client.get('/api/health')
        assert r_health.status_code == 200
        assert r_health.json()['air_gapped'] is True

        # 2. Scenes catalog
        r_scenes = client.get('/api/scenes')
        assert r_scenes.status_code == 200

        # 3. Master /api/run execution
        r_run = client.post('/api/run', json={'scene_id': 'scene_mumbai_001', 'hindcast_hours': 48})
        assert r_run.status_code == 200
        run_data = r_run.json()
        assert run_data['status'] == 'completed'
        assert 'detection' in run_data
        assert 'drift' in run_data
        assert 'attribution' in run_data
        job_id = run_data['job_id']

        # 4. Job status polling
        r_job = client.get(f'/api/jobs/{job_id}')
        assert r_job.status_code == 200

        # 5. PDF Dossier Report Download (/api/report/{job_id})
        r_pdf = client.get(f'/api/report/{job_id}')
        assert r_pdf.status_code == 200
        assert r_pdf.headers.get('content-type') == 'application/pdf'
        assert len(r_pdf.content) > 1000
