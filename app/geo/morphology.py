"""
TideTrace Morphological Characterisation Engine.
Vectorizes SAR segmentation masks into geo-referenced polygons, projects geometries to
local Azimuthal Equidistant (AEQD) metric space, and computes exact geometric descriptors:
  - Area (km^2) & Perimeter (km)
  - Max Feret Length (km) & Width (km)
  - PCA Orientation Angle (deg in [0, 180] from North)
  - Centroid (lat_c, lon_c)
  - Compactness / Circularity (4*pi*A / P^2)
  - Standard RFC 7946 GeoJSON FeatureCollection Serialization
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import transform as shapely_transform

from app.config import (
    CLASS_SEA,
    CLASS_LOOKALIKE,
    CLASS_MINERAL_OIL,
    MIN_SLICK_PIXELS,
    MIN_SLICK_AREA_KM2,
)
from app.geo.georef import GeoReference, create_default_georef

# WGS84 Ellipsoid Mean Radius (IUGG R1) in meters
WGS84_R_METERS = 6371008.8


# ==============================================================================
# 1. LOCAL AZIMUTHAL EQUIDISTANT (AEQD) PROJECTION
# ==============================================================================

def forward_aeqd(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    lon_0: float,
    lat_0: float,
) -> Tuple[Any, Any]:
    """
    Project WGS84 (lon, lat) to local Azimuthal Equidistant (AEQD) metric plane (x, y) in meters
    centered at origin (lon_0, lat_0).
    Guarantees true distance isometry from center point.
    """
    phi0 = math.radians(lat_0)
    lam0 = math.radians(lon_0)

    phi = np.radians(lat)
    lam = np.radians(lon)

    dlam = lam - lam0

    cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dlam)
    cos_c = np.clip(cos_c, -1.0, 1.0)
    c = np.arccos(cos_c)

    # Avoid division by zero at center
    sin_c = np.sin(c)
    k = np.where(sin_c > 1e-7, c / sin_c, 1.0)

    x = WGS84_R_METERS * k * np.cos(phi) * np.sin(dlam)
    y = WGS84_R_METERS * k * (np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlam))

    return x, y


def inverse_aeqd(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    lon_0: float,
    lat_0: float,
) -> Tuple[Any, Any]:
    """
    Inverse project local AEQD metric coordinates (x, y) in meters back to WGS84 (lon, lat).
    """
    phi0 = math.radians(lat_0)
    lam0 = math.radians(lon_0)

    rho = np.sqrt(x**2 + y**2)
    c = rho / WGS84_R_METERS

    sin_c = np.sin(c)
    cos_c = np.cos(c)

    # Lat calculation
    sin_phi = cos_c * math.sin(phi0) + np.where(
        rho > 1e-7,
        (y * sin_c * math.cos(phi0)) / np.maximum(rho, 1e-7),
        0.0,
    )
    sin_phi = np.clip(sin_phi, -1.0, 1.0)
    phi = np.arcsin(sin_phi)

    # Lon calculation
    denom = rho * math.cos(phi0) * cos_c - y * math.sin(phi0) * sin_c
    num = x * sin_c
    dlam = np.where(rho > 1e-7, np.arctan2(num, denom), 0.0)
    lam = lam0 + dlam

    return np.degrees(lam), np.degrees(phi)


# ==============================================================================
# 2. MORPHOLOGY CALCULATION & DESCRIPTORS
# ==============================================================================

def compute_polygon_morphology(
    wgs84_polygon: Polygon,
    confidence_val: float = 0.90,
    pixel_count: int = 100,
    slick_id: str = "slick_001",
    class_id: int = CLASS_MINERAL_OIL,
) -> Dict[str, Any]:
    """
    Compute rigorous metric morphological descriptors for a WGS84 polygon by projecting
    to a local AEQD metric coordinate system centered at its centroid.
    """
    if wgs84_polygon.is_empty or not wgs84_polygon.is_valid:
        wgs84_polygon = wgs84_polygon.buffer(0)

    # 1. Geographic centroid
    centroid_lon_raw, centroid_lat_raw = wgs84_polygon.centroid.x, wgs84_polygon.centroid.y
    lon_0, lat_0 = float(centroid_lon_raw), float(centroid_lat_raw)

    # 2. Project polygon to local AEQD metric plane
    def _proj(lon, lat):
        return forward_aeqd(lon, lat, lon_0, lat_0)

    metric_polygon = shapely_transform(_proj, wgs84_polygon)

    # Metric area in km^2 and perimeter in km
    area_m2 = float(metric_polygon.area)
    perimeter_m = float(metric_polygon.length)

    area_km2 = area_m2 / 1.0e6
    perimeter_km = perimeter_m / 1.0e3

    # 3. Centroid verification in metric coordinates
    metric_cx, metric_cy = float(metric_polygon.centroid.x), float(metric_polygon.centroid.y)
    true_lon_c, true_lat_c = inverse_aeqd(metric_cx, metric_cy, lon_0, lat_0)
    true_lon_c, true_lat_c = float(true_lon_c), float(true_lat_c)

    # 4. Max Feret Diameter (Length km) via Convex Hull pairwise max distance
    hull = metric_polygon.convex_hull
    if hull.geom_type == "Polygon" and len(hull.exterior.coords) > 2:
        hull_coords = np.array(hull.exterior.coords)
        # Pairwise distance matrix
        diff = hull_coords[:, np.newaxis, :] - hull_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        max_dist_m = float(np.max(dist_matrix))
    else:
        # Fallback to bounding box diagonal
        minx, miny, maxx, maxy = metric_polygon.bounds
        max_dist_m = math.sqrt((maxx - minx)**2 + (maxy - miny)**2)

    length_km = max(max_dist_m / 1.0e3, 0.001)

    # 5. Width (km) and Aspect Ratio
    # Width computed as minimum rotated rectangle minor axis, or hydraulic width (Area / Length)
    min_rect = metric_polygon.minimum_rotated_rectangle
    if min_rect.geom_type == "Polygon" and len(min_rect.exterior.coords) >= 4:
        r_coords = np.array(min_rect.exterior.coords)[:4]
        edge1 = np.linalg.norm(r_coords[0] - r_coords[1])
        edge2 = np.linalg.norm(r_coords[1] - r_coords[2])
        width_m = min(edge1, edge2)
        length_rect_m = max(edge1, edge2)
        width_km = max(float(width_m) / 1.0e3, 0.001)
        length_km = max(float(length_rect_m) / 1.0e3, length_km)
    else:
        width_km = max(area_km2 / length_km, 0.001)

    aspect_ratio = float(length_km / max(width_km, 1e-4))

    # 6. Orientation Angle via Principal Component Analysis (PCA) of boundary coordinates
    # Measures major axis bearing relative to True North (0 deg to 180 deg)
    ext_coords = np.array(metric_polygon.exterior.coords)
    x_c = ext_coords[:, 0] - metric_cx
    y_c = ext_coords[:, 1] - metric_cy

    if len(x_c) >= 3:
        cov = np.cov(x_c, y_c)
        if cov.shape == (2, 2) and not np.isnan(cov).any():
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Principal component vector (largest eigenvalue is index 1)
            major_vec = eigenvectors[:, 1]
            vx, vy = major_vec[0], major_vec[1]
            # Bearing from North (+y is North, +x is East): azimuth = atan2(vx, vy)
            angle_rad = math.atan2(vx, vy)
            angle_deg = math.degrees(angle_rad) % 180.0
            if angle_deg < 0:
                angle_deg += 180.0
        else:
            angle_deg = 0.0
    else:
        angle_deg = 0.0

    # 7. Compactness / Circularity (4*pi*A / P^2)
    if perimeter_m > 0:
        compactness = float((4.0 * math.pi * area_m2) / (perimeter_m ** 2))
        compactness = max(0.0, min(1.0, compactness))
    else:
        compactness = 0.0

    class_name = "mineral_oil" if class_id == CLASS_MINERAL_OIL else "lookalike"

    return {
        "slick_id": slick_id,
        "class_id": class_id,
        "class_name": class_name,
        "area_km2": round(area_km2, 4),
        "perimeter_km": round(perimeter_km, 4),
        "length_km": round(length_km, 4),
        "width_km": round(width_km, 4),
        "aspect_ratio": round(aspect_ratio, 2),
        "orientation_deg": round(float(angle_deg), 1),
        "centroid": [round(true_lat_c, 5), round(true_lon_c, 5)],
        "centroid_lat": round(true_lat_c, 5),
        "centroid_lon": round(true_lon_c, 5),
        "compactness": round(compactness, 3),
        "confidence": round(float(confidence_val), 3),
        "pixel_count": int(pixel_count),
    }


# ==============================================================================
# 3. VECTORIZATION & GEOJSON SERIALIZATION
# ==============================================================================

def vectorize_mask_to_polygons(
    class_mask: np.ndarray,
    georef: GeoReference,
    confidence_map: Optional[np.ndarray] = None,
    target_class: int = CLASS_MINERAL_OIL,
    min_pixels: int = MIN_SLICK_PIXELS,
    min_area_km2: float = MIN_SLICK_AREA_KM2,
) -> List[Tuple[Polygon, Dict[str, Any]]]:
    """
    Extract vector polygons for a specific target class from a segmentation mask,
    transform pixel vertices to WGS84 (lon, lat), and compute morphological properties.
    """
    binary_mask = (class_mask == target_class).astype(np.uint8)
    if not np.any(binary_mask):
        return []

    # Find external contours
    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    results = []
    prefix = "slick" if target_class == CLASS_MINERAL_OIL else "lookalike"

    feature_counter = 1
    for contour in contours:
        if len(contour) < 3:
            continue

        # Check pixel area
        px_area = cv2.contourArea(contour)
        if px_area < min_pixels:
            continue

        # Contour points are (col, row)
        pts = contour.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue

        cols = pts[:, 0]
        rows = pts[:, 1]

        # Convert pixel to WGS84
        lons, lats = georef.pixel_to_wgs84(rows, cols)
        ring_coords = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]
        
        # Close ring if necessary
        if ring_coords[0] != ring_coords[-1]:
            ring_coords.append(ring_coords[0])

        poly = Polygon(ring_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        # Calculate mean confidence inside component
        if confidence_map is not None:
            # Create sub-mask to sample confidence
            comp_mask = np.zeros(class_mask.shape, dtype=np.uint8)
            cv2.drawContours(comp_mask, [contour], -1, 1, thickness=-1)
            conf_vals = confidence_map[comp_mask == 1]
            mean_conf = float(np.mean(conf_vals)) if len(conf_vals) > 0 else 0.90
        else:
            mean_conf = 0.90

        slick_id = f"{prefix}_{feature_counter:03d}"
        props = compute_polygon_morphology(
            wgs84_polygon=poly,
            confidence_val=mean_conf,
            pixel_count=int(px_area),
            slick_id=slick_id,
            class_id=target_class,
        )

        if props["area_km2"] >= min_area_km2 or px_area >= min_pixels:
            results.append((poly, props))
            feature_counter += 1

    return results


def generate_geojson_feature_collection(
    slicks: List[Tuple[Polygon, Dict[str, Any]]],
    lookalikes: Optional[List[Tuple[Polygon, Dict[str, Any]]]] = None,
    scene_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble slick and lookalike polygons into a standardized RFC 7946 GeoJSON FeatureCollection.
    """
    lookalikes = lookalikes or []
    scene_metadata = scene_metadata or {}

    features = []

    # Add mineral oil slicks
    total_oil_area = sum(props["area_km2"] for _, props in slicks)
    for poly, props in slicks:
        feature = {
            "type": "Feature",
            "id": props["slick_id"],
            "geometry": mapping(poly),
            "properties": props,
        }
        features.append(feature)

    # Add lookalikes
    total_lookalike_area = sum(props["area_km2"] for _, props in lookalikes)
    for poly, props in lookalikes:
        feature = {
            "type": "Feature",
            "id": props["slick_id"],
            "geometry": mapping(poly),
            "properties": props,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "scene_id": scene_metadata.get("scene_id", "SAR_SCENE_001"),
            "acquisition_time_utc": scene_metadata.get("acquisition_time_utc") or scene_metadata.get("t_sat") or "2026-09-02T12:00:00Z",
            "sensor": scene_metadata.get("sensor", "Sentinel-1 C-SAR"),
            "detection_mode": scene_metadata.get("detection_mode", "unet_segmenter"),
            "total_oil_area_km2": round(total_oil_area, 4),
            "total_lookalike_area_km2": round(total_lookalike_area, 4),
            "num_oil_slicks": len(slicks),
            "num_lookalikes": len(lookalikes),
        },
        "features": features,
    }

    return geojson
