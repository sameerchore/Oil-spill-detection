"""
Unit tests for Geodesic Geometry, Centroid calculation, and Local AEQD metric projections.
Validates exact coordinate invariance, Shoelace area, and PCA orientation.
"""

import math
from typing import Any, Dict
import numpy as np
import pytest
from shapely.geometry import Polygon, Point, mapping

from app.geo.georef import AffineTransform, GeoReference, create_default_georef
from app.geo.morphology import (
    forward_aeqd,
    inverse_aeqd,
    compute_polygon_morphology,
    vectorize_mask_to_polygons,
)
import app.config as config


def get_morphology_extractor():
    """Helper factory returning a callable to extract morphology feature from mask and transform."""
    def extract_morphology(mask: np.ndarray, transform: Any, slick_id: str = "slick_001") -> Dict[str, Any]:
        h, w = mask.shape
        if isinstance(transform, dict):
            deg_lat = transform.get("pixel_size_lat_deg", 10.0 / 110574.0)
            deg_lon = transform.get("pixel_size_lon_deg", 10.0 / 105150.0)
            orig_lon = transform.get("origin_lon", 72.825 - (w / 2) * deg_lon)
            orig_lat = transform.get("origin_lat", 18.925 + (h / 2) * deg_lat)
            affine = AffineTransform(deg_lon, 0.0, orig_lon, 0.0, -deg_lat, orig_lat)
            georef = GeoReference(transform=affine, height=h, width=w)
        elif isinstance(transform, GeoReference):
            georef = transform
        elif isinstance(transform, AffineTransform):
            georef = GeoReference(transform=transform, height=h, width=w)
        else:
            georef = create_default_georef(height=h, width=w)

        # Target class: if mask has 2, use 2, else if mask has 1, use 1
        target_cls = 2 if 2 in mask else (1 if 1 in mask else 0)
        slicks = vectorize_mask_to_polygons(mask, georef, target_class=target_cls, min_pixels=10)
        if slicks:
            poly, props = slicks[0]
            props["slick_id"] = slick_id
            return {
                "type": "Feature",
                "id": slick_id,
                "geometry": mapping(poly),
                "properties": props,
            }

        return {
            "type": "Feature",
            "id": slick_id,
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {"slick_id": slick_id, "area_km2": 0.0, "centroid": [18.925, 72.825]},
        }

    return extract_morphology


def test_geodesic_centroid_accuracy():
    """Verify centroid invariance under forward and inverse AEQD projection."""
    lat_center, lon_center = 19.12345, 71.98765
    
    # Create circular polygon around center in metric meters (radius = 5000m)
    angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
    radius = 5000.0
    metric_x = radius * np.cos(angles)
    metric_y = radius * np.sin(angles)
    
    # Convert to WGS84
    wgs84_pts = []
    for mx, my in zip(metric_x, metric_y):
        lon, lat = inverse_aeqd(mx, my, lon_center, lat_center)
        wgs84_pts.append([float(lon), float(lat)])
    wgs84_pts.append(wgs84_pts[0]) # close ring
    
    poly = Polygon(wgs84_pts)
    props = compute_polygon_morphology(poly, slick_id="circle_01")
    
    # Area of circle: pi * r^2 = pi * 25 = ~78.54 km^2
    expected_area = math.pi * (5.0 ** 2)
    assert props["area_km2"] == pytest.approx(expected_area, rel=0.01)
    
    # Perimeter: 2 * pi * r = ~31.41 km
    expected_perimeter = 2.0 * math.pi * 5.0
    assert props["perimeter_km"] == pytest.approx(expected_perimeter, rel=0.01)
    
    # Compactness of circle must be ~1.0
    assert props["compactness"] == pytest.approx(1.0, abs=0.02)
    
    # Centroid must match origin exactly
    assert props["centroid"][0] == pytest.approx(lat_center, abs=1e-5)
    assert props["centroid"][1] == pytest.approx(lon_center, abs=1e-5)


def test_feret_diameter_and_aspect_ratio():
    """Verify max Feret length, width, and aspect ratio on an elongated polygon."""
    lat_c, lon_c = 0.0, 80.0
    # Length 12 km along East-West, Width 2 km along North-South
    half_l = 6000.0
    half_w = 1000.0
    
    box_pts = [
        [-half_l, -half_w],
        [half_l, -half_w],
        [half_l, half_w],
        [-half_l, half_w],
        [-half_l, -half_w],
    ]
    
    wgs84_pts = []
    for x, y in box_pts:
        lon, lat = inverse_aeqd(x, y, lon_c, lat_c)
        wgs84_pts.append([float(lon), float(lat)])
        
    poly = Polygon(wgs84_pts)
    props = compute_polygon_morphology(poly, slick_id="elongated_01")
    
    assert props["area_km2"] == pytest.approx(24.0, rel=0.02)
    assert props["length_km"] == pytest.approx(12.0, rel=0.02)
    assert props["width_km"] == pytest.approx(2.0, rel=0.05)
    assert props["aspect_ratio"] == pytest.approx(6.0, rel=0.05)
    # Orientation of East-West rectangle relative to North is 90 degrees
    assert props["orientation_deg"] == pytest.approx(90.0, abs=2.0)
