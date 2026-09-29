"""
Comprehensive Test Suite for TideTrace Metocean & Lagrangian Drift Physics Engine
Milestone: M2 (Worker M2)
Features: F06 (Advection), F07 (Hindcast), F08 (Forecast), F09 (Metocean Store & 4D Interpolator)
"""

import math
import os
import tempfile
import numpy as np
import pytest
from shapely.geometry import shape

from app.metocean import (
    MetoceanGrid,
    MetoceanInterpolator,
    load_metocean_from_file,
    save_metocean_data,
    create_synthetic_metocean_grid,
    parse_timestamp,
)
from app.drift import (
    haversine_distance,
    haversine_distance_km,
    translate_geodesic,
    forward_aeqd,
    inverse_aeqd,
    apply_coriolis_deflection,
    advection_step_euler,
    advection_step_rk2,
    run_hindcast,
    run_forecast,
    seed_particles_across_slick,
)


# ==============================================================================
# F09: Metocean Store & 4D Interpolator Tests
# ==============================================================================

class TestMetoceanStoreAndInterpolator:
    """Test offline metocean loading, synthetic generation, and 4D interpolation."""

    def test_synthetic_metocean_grid_generation(self):
        """Verify synthetic metocean grid contains valid physical structures and dimensions."""
        grid = create_synthetic_metocean_grid(
            center_lat=18.9,
            center_lon=72.8,
            t_center="2026-09-02T12:00:00Z",
            hours_before=48,
            hours_after=48,
            resolution_deg=0.1,
            grid_radius_deg=1.5,
        )
        assert grid.shape[0] == 97  # 48 + 48 + 1
        assert grid.shape[1] == 31  # 3.0 / 0.1 + 1
        assert grid.shape[2] == 31
        assert np.all(np.diff(grid.times) > 0)
        assert np.all(np.diff(grid.lats) > 0)
        assert np.all(np.diff(grid.lons) > 0)
        assert not np.isnan(grid.u_current).any()
        assert not np.isnan(grid.v_wind).any()

    def test_metocean_json_and_npz_roundtrip(self):
        """Verify saving and loading grids across JSON and NPZ formats."""
        grid = create_synthetic_metocean_grid(
            center_lat=15.0,
            center_lon=70.0,
            t_center="2026-09-02T12:00:00Z",
            hours_before=6,
            hours_after=6,
            resolution_deg=0.2,
            grid_radius_deg=0.6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "grid.json")
            npz_path = os.path.join(tmpdir, "grid.npz")

            # JSON roundtrip
            save_metocean_data(grid, json_path)
            loaded_json = load_metocean_from_file(json_path)
            assert loaded_json.shape == grid.shape
            assert np.allclose(loaded_json.times, grid.times)
            assert np.allclose(loaded_json.u_current, grid.u_current, atol=1e-5)

            # NPZ roundtrip
            save_metocean_data(grid, npz_path)
            loaded_npz = load_metocean_from_file(npz_path)
            assert loaded_npz.shape == grid.shape
            assert np.allclose(loaded_npz.u_wind, grid.u_wind, atol=1e-5)

    def test_4d_bilinear_linear_interpolation_exactness(self):
        """Verify 4D interpolator produces exact values at grid nodes and smooth linear transitions."""
        # Create a simple analytical grid
        times = np.array([0.0, 3600.0], dtype=np.float64)
        lats = np.array([10.0, 20.0], dtype=np.float64)
        lons = np.array([70.0, 80.0], dtype=np.float64)

        # u_current = t/3600 + lat/10 + lon/10
        T, M, K = len(times), len(lats), len(lons)
        u_c = np.zeros((T, M, K))
        v_c = np.zeros((T, M, K))
        u_w = np.zeros((T, M, K))
        v_w = np.zeros((T, M, K))

        for t_i in range(T):
            for m_i in range(M):
                for k_i in range(K):
                    u_c[t_i, m_i, k_i] = (times[t_i] / 3600.0) + (lats[m_i] / 10.0) + (lons[k_i] / 10.0)

        grid = MetoceanGrid(
            times=times,
            lats=lats,
            lons=lons,
            u_current=u_c,
            v_current=v_c,
            u_wind=u_w,
            v_wind=v_w,
        )
        interp = MetoceanInterpolator(grid)

        # Query midpoint: t=1800 (frac 0.5), lat=15.0 (frac 0.5), lon=75.0 (frac 0.5)
        # Expected u_c = 0.5 + 1.5 + 7.5 = 9.5
        vec = interp.interpolate(1800.0, 15.0, 75.0)
        assert abs(vec.u_current - 9.5) < 1e-6

        # Vectorized batch interpolation matches scalar
        t_batch = np.array([1800.0, 0.0, 3600.0])
        lat_batch = np.array([15.0, 10.0, 20.0])
        lon_batch = np.array([75.0, 70.0, 80.0])
        u_batch, _, _, _ = interp.interpolate_batch(t_batch, lat_batch, lon_batch)

        assert abs(u_batch[0] - 9.5) < 1e-6
        assert abs(u_batch[1] - 8.0) < 1e-6  # 0 + 1 + 7 = 8
        assert abs(u_batch[2] - 11.0) < 1e-6 # 1 + 2 + 8 = 11


# ==============================================================================
# F06: 2D Lagrangian Numerical Advection Tests
# ==============================================================================

class TestLagrangianAdvectionPhysics:
    """Test 2D surface advection physics, Coriolis leeway, and geodesic transformations."""

    def test_exact_displacement_3_6km_per_hour(self):
        """Displacement for u=1.0 m/s over 3600s must equal exactly 3600.0 meters."""
        lat0, lon0 = 18.9, 72.8
        lat1, lon1 = translate_geodesic(lat0, lon0, 3600.0, 0.0)
        d_m = haversine_distance(lat0, lon0, lat1, lon1)
        assert abs(d_m - 3600.0) < 0.1

    def test_coriolis_deflection_clockwise_northern_hemisphere(self):
        """Wind towards North (0, 10) must deflect +15 deg clockwise towards East-North-East."""
        u_w, v_w = 0.0, 10.0
        u_eff, v_eff = apply_coriolis_deflection(u_w, v_w, deflection_deg=15.0, lat=18.9)
        # Expected: u_eff = 10 * sin(15 deg) > 0, v_eff = 10 * cos(15 deg) > 0
        assert u_eff > 0.0
        assert v_eff > 0.0
        assert abs(math.hypot(u_eff, v_eff) - 10.0) < 1e-6
        bearing_deg = math.degrees(math.atan2(u_eff, v_eff))
        assert abs(bearing_deg - 15.0) < 1e-5

    def test_coriolis_deflection_counterclockwise_southern_hemisphere(self):
        """In Southern Hemisphere (lat < 0), deflection is counter-clockwise (-15 deg)."""
        u_w, v_w = 0.0, 10.0
        u_eff, v_eff = apply_coriolis_deflection(u_w, v_w, deflection_deg=15.0, lat=-18.9)
        # Deflected to West-North-West (-15 deg)
        assert u_eff < 0.0
        assert v_eff > 0.0
        bearing_deg = (math.degrees(math.atan2(u_eff, v_eff)) + 360.0) % 360.0
        assert abs(bearing_deg - 345.0) < 1e-5

    def test_aeqd_projection_roundtrip(self):
        """Verify forward and inverse AEQD projection round-trip accuracy."""
        lat0, lon0 = 18.925, 72.825
        lats_in = np.array([18.90, 18.95, 18.88, 19.00])
        lons_in = np.array([72.80, 72.85, 72.78, 72.90])

        xs, ys = forward_aeqd(lats_in, lons_in, lat0, lon0)
        lats_out, lons_out = inverse_aeqd(xs, ys, lat0, lon0)

        assert np.allclose(lats_in, lats_out, atol=1e-9)
        assert np.allclose(lons_in, lons_out, atol=1e-9)

    def test_rk2_vs_euler_integration(self):
        """Verify RK2 midpoint integration with time-dependent metocean grid."""
        grid = create_synthetic_metocean_grid(center_lat=18.9, center_lon=72.8)
        interp = MetoceanInterpolator(grid)
        t0 = float(grid.times[24])

        lat_rk2, lon_rk2 = advection_step_rk2(interp, t0, 18.9, 72.8, 3600.0)
        lat_eul, lon_eul = advection_step_euler(interp, t0, 18.9, 72.8, 3600.0)

        # Both methods drift within realistic physical bounds (< 5 km in 1h)
        d_rk2 = haversine_distance_km(18.9, 72.8, lat_rk2, lon_rk2)
        d_eul = haversine_distance_km(18.9, 72.8, lat_eul, lon_eul)
        assert 0.1 < d_rk2 < 5.0
        assert 0.1 < d_eul < 5.0
        # RK2 and Euler agree within ~200m
        assert haversine_distance_km(lat_rk2, lon_rk2, lat_eul, lon_eul) < 0.3


# ==============================================================================
# F07: Backward Lagrangian Origin Backtracking Tests
# ==============================================================================

class TestBackwardHindcastOrigin:
    """Test backward ensemble backtracking, origin estimation, and confidence envelope."""

    @pytest.fixture
    def sample_grid_and_slick(self):
        grid = create_synthetic_metocean_grid(
            center_lat=18.9,
            center_lon=72.8,
            t_center="2026-09-02T12:00:00Z",
            hours_before=48,
            hours_after=48,
        )
        interp = MetoceanInterpolator(grid)
        slick_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [72.80, 18.90],
                    [72.83, 18.92],
                    [72.84, 18.90],
                    [72.81, 18.88],
                    [72.80, 18.90],
                ]]
            }
        }
        return interp, slick_geojson

    def test_particle_seeding_in_slick_polygon(self, sample_grid_and_slick):
        """Verify N=50 particles are seeded uniformly within the slick polygon boundaries."""
        _, slick_geojson = sample_grid_and_slick
        lats, lons = seed_particles_across_slick(slick_geojson, num_particles=50, seed=42)
        assert len(lats) == 50
        assert len(lons) == 50
        # Centroid check
        assert abs(np.mean(lats) - 18.90) < 0.05
        assert abs(np.mean(lons) - 72.82) < 0.05

    def test_hindcast_origin_time_and_envelope(self, sample_grid_and_slick):
        """Verify backward backtracking produces valid origin time bounded in [6h, 48h] and dilated envelope."""
        interp, slick_geojson = sample_grid_and_slick
        res = run_hindcast(
            interpolator=interp,
            slick_polygon=slick_geojson,
            t_sat="2026-09-02T12:00:00Z",
            hindcast_hours=48,
            num_particles=50,
            seed=42,
        )

        assert "origin_time_iso" in res
        assert "origin_window_start_iso" in res
        assert "origin_window_end_iso" in res
        assert 6 <= res["age_hours_proxy"] <= 48
        assert res["total_drift_km"] > 0.0

        # Verify Origin GeoJSON Polygon
        origin_poly = res["origin_polygon_geojson"]
        assert origin_poly["type"] == "Feature"
        assert origin_poly["geometry"]["type"] == "Polygon"
        geom = shape(origin_poly["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.area > 0.0

        # Verify time window is exactly +/- 3 hours around origin time
        orig_ts = parse_timestamp(res["origin_time_iso"])
        win_start_ts = parse_timestamp(res["origin_window_start_iso"])
        win_end_ts = parse_timestamp(res["origin_window_end_iso"])
        assert abs((orig_ts - win_start_ts) - 3 * 3600.0) < 1e-3
        assert abs((win_end_ts - orig_ts) - 3 * 3600.0) < 1e-3


# ==============================================================================
# F08: Forward 36h Dispersion Forecast Tests
# ==============================================================================

class TestForwardForecastDispersion:
    """Test forward dispersion forecast cone, checkpoints, and threatened bounding box."""

    def test_forecast_cone_and_checkpoints(self):
        """Verify forward forecast creates composite cone, checkpoint polygons at +12h/+24h/+36h, and bbox."""
        grid = create_synthetic_metocean_grid(center_lat=18.9, center_lon=72.8)
        interp = MetoceanInterpolator(grid)
        slick_poly = [
            [72.80, 18.90],
            [72.82, 18.92],
            [72.83, 18.91],
            [72.81, 18.89],
            [72.80, 18.90],
        ]

        res = run_forecast(
            interpolator=interp,
            slick_polygon=slick_poly,
            t_sat="2026-09-02T12:00:00Z",
            forecast_hours=36,
            num_particles=50,
            seed=42,
        )

        assert res["forecast_hours"] == 36
        assert res["total_drift_km"] > 0.0
        assert 0.0 <= res["dominant_heading_deg"] <= 360.0

        # Checkpoints verification
        assert "plus_12h" in res["checkpoint_polygons"]
        assert "plus_24h" in res["checkpoint_polygons"]
        assert "plus_36h" in res["checkpoint_polygons"]

        for chk_key in ["plus_12h", "plus_24h", "plus_36h"]:
            chk = res["checkpoint_polygons"][chk_key]
            geom = shape(chk["geometry"])
            assert geom.is_valid
            assert not geom.is_empty

        # Threatened Bounding Box verification [min_lon, min_lat, max_lon, max_lat]
        bbox = res["threatened_bbox"]
        assert len(bbox) == 4
        min_lon, min_lat, max_lon, max_lat = bbox
        assert min_lon < max_lon
        assert min_lat < max_lat
        assert 70.0 < min_lon < 75.0
        assert 16.0 < min_lat < 21.0

        # FeatureCollection integrity
        fc = res["feature_collection"]
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) >= 4  # cone, checkpoints, median track, particle lines
