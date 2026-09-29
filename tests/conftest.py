"""
Shared pytest fixtures and test configuration for TideTrace test suite.
Guarantees 100% air-gapped offline test execution and authentic synthetic fixtures.
"""

import math
import socket
import urllib.request
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def air_gapped_network_guard(monkeypatch):
    """
    Enforces strict air-gapped testing by monkeypatching low-level socket connections.
    Prevents any accidental external internet or cloud API calls.
    """
    orig_connect = socket.socket.connect

    def forbidden_connect(self, address):
        # Allow local inter-process loopback connections if needed (127.0.0.1, localhost)
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "localhost", "::1"):
            return orig_connect(self, address)
        raise RuntimeError(f"AIR-GAP VIOLATION: Attempted external socket connection to {address}")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


@pytest.fixture
def synthetic_circular_mask():
    """
    Generates a 512x512 binary raster containing a centered circular slick
    with radius R = 100 pixels.
    At 10 m/pixel resolution, radius = 1000 m (1.0 km).
    Theoretical True Area = pi * 1.0^2 = 3.14159265 km^2.
    """
    H, W = 512, 512
    y, x = np.ogrid[:H, :W]
    center_y, center_x = 256, 256
    radius = 100
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = (dist_from_center <= radius).astype(np.uint8)
    
    # Geotransform: [origin_lon, pixel_width_deg, 0, origin_lat, 0, -pixel_height_deg]
    # At ~18.925 N, 1 deg lat ~ 110.574 km, 1 deg lon ~ 105.150 km
    # 10 m/px -> deg_per_px_lat = 10 / 110574 = 9.0437e-5
    # 10 m/px -> deg_per_px_lon = 10 / 105150 = 9.5102e-5
    lat_center = 18.92500
    lon_center = 72.82500
    deg_per_px_lat = 10.0 / 110574.0
    deg_per_px_lon = 10.0 / 105150.0
    
    origin_lon = lon_center - center_x * deg_per_px_lon
    origin_lat = lat_center + center_y * deg_per_px_lat
    
    transform = {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "pixel_size_lat_deg": deg_per_px_lat,
        "pixel_size_lon_deg": deg_per_px_lon,
        "center_lat": lat_center,
        "center_lon": lon_center,
        "radius_px": radius,
        "expected_area_km2": math.pi * 1.0**2,
        "expected_perimeter_km": 2 * math.pi * 1.0
    }
    return mask, transform


@pytest.fixture
def synthetic_rotated_ellipse_mask():
    """
    Generates rotated ellipse binary masks with known geometric rotation angles.
    """
    def _create_ellipse(angle_deg=45.0, semi_major=120, semi_minor=30):
        H, W = 512, 512
        y, x = np.ogrid[:H, :W]
        cx, cy = 256, 256
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        # Rotated coordinates
        xr = (x - cx) * cos_t + (y - cy) * sin_t
        yr = -(x - cx) * sin_t + (y - cy) * cos_t
        
        mask = ((xr / semi_major) ** 2 + (yr / semi_minor) ** 2 <= 1.0).astype(np.uint8)
        return mask, angle_deg, semi_major, semi_minor
        
    return _create_ellipse


@pytest.fixture
def synthetic_ais_scenario():
    """
    Provides deterministic AIS traffic for forensic attribution verification:
    - Culprit (MMSI 419000001): MT OCEAN VOYAGER (Crude Oil Tanker) with 52-min silence across origin
    - Distractor 1 (MMSI 419000002): MV GLOBAL CARGO (Cargo) 4.5 km away, no silence
    - Distractor 2 (MMSI 419000003): SAGAR KANYA (Fishing) 1.2 km away, low risk type, no silence
    """
    origin_time = "2026-09-02T12:00:00Z"
    origin_lat = 18.925
    origin_lon = 72.825
    
    vessels = [
        {
            "mmsi": 419000001,
            "vessel_name": "MT OCEAN VOYAGER",
            "vessel_type": "Crude Oil Tanker",
            "risk_type_score": 1.0,
            "distance_km": 0.62,
            "heading_alignment_deg": 12.0,
            "gap_duration_min": 52.0,
            "has_ais_gap": True,
            "closest_time": "2026-09-02T12:05:00Z"
        },
        {
            "mmsi": 419000002,
            "vessel_name": "MV GLOBAL CARGO",
            "vessel_type": "Cargo",
            "risk_type_score": 0.5,
            "distance_km": 4.50,
            "heading_alignment_deg": 45.0,
            "gap_duration_min": 5.0,
            "has_ais_gap": False,
            "closest_time": "2026-09-02T12:40:00Z"
        },
        {
            "mmsi": 419000003,
            "vessel_name": "SAGAR KANYA",
            "vessel_type": "Fishing",
            "risk_type_score": 0.2,
            "distance_km": 1.20,
            "heading_alignment_deg": 80.0,
            "gap_duration_min": 5.0,
            "has_ais_gap": False,
            "closest_time": "2026-09-02T11:50:00Z"
        }
    ]
    return {
        "origin_time": origin_time,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "vessels": vessels
    }
