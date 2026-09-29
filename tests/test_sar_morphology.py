"""
Comprehensive Test Suite for TideTrace Module M1:
  - Configuration constants and physical thresholds
  - PyTorch 3-Class U-Net architecture & memory footprint
  - Radiometric preprocessing, clipping, tiling & apodized stitching
  - Adaptive heuristic fallback dark-pixel detector
  - Unified SARSegmenter inference engine
  - 6-Parameter Affine georeferencing & WGS84 mapping
  - Local AEQD projection, geodesic metrics & PCA morphology
  - RFC 7946 GeoJSON serialization
"""

import math
import numpy as np
import pytest
import torch
from shapely.geometry import Polygon, box

import app.config as config
from app.ml.model import TideTraceUNet
from app.ml.preprocess import (
    to_decibels,
    normalize_sar_bands,
    create_hann_window,
    compute_tile_grid,
    extract_tiles,
    stitch_tiles,
)
from app.ml.fallback import run_adaptive_heuristic_detector
from app.ml.segmenter import SARSegmenter
from app.geo.georef import AffineTransform, GeoReference, create_default_georef
from app.geo.morphology import (
    forward_aeqd,
    inverse_aeqd,
    compute_polygon_morphology,
    vectorize_mask_to_polygons,
    generate_geojson_feature_collection,
)


# ==============================================================================
# 1. CONFIGURATION TESTS
# ==============================================================================

def test_config_physical_and_scoring_constants():
    """Verify all physical and scoring constants match the master specification."""
    assert config.WIND_FACTOR == 0.03
    assert config.CORIOLIS_DEFLECTION_DEG == 15.0
    assert config.SEARCH_RADIUS_KM == 10.0
    assert config.GAP_THRESHOLD_MINUTES == 30.0
    assert config.WEIGHT_PROX == 0.40
    assert config.WEIGHT_TYPE == 0.20
    assert config.WEIGHT_TRAJ == 0.15
    assert config.WEIGHT_BEH == 0.25
    assert config.WEIGHT_PROX + config.WEIGHT_TYPE + config.WEIGHT_TRAJ + config.WEIGHT_BEH == 1.0
    assert config.DARK_THRESHOLD_VV_DB == -22.0
    assert config.CONTRAST_DIFF_THRESHOLD_DB == 4.5
    assert config.TILE_SIZE == 512
    assert config.TILE_OVERLAP == 64


# ==============================================================================
# 2. PYTORCH U-NET ARCHITECTURE TESTS
# ==============================================================================

def test_unet_architecture_and_size():
    """Verify U-Net initialization, parameter count, memory footprint (<80MB), and forward pass."""
    model = TideTraceUNet(in_channels=3, num_classes=3, base_features=32)
    model.eval()

    # Parameter footprint
    size_mb = model.get_model_size_mb()
    assert size_mb < 80.0, f"Model size {size_mb:.2f} MB exceeds 80 MB limit"
    assert size_mb > 5.0, f"Model size {size_mb:.2f} MB is unrealistically small"
    
    # Forward pass on standard 512x512 tile
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        logits = model(x)
        assert logits.shape == (2, 3, 512, 512)

        probs = model.predict_probabilities(x)
        assert probs.shape == (2, 3, 512, 512)
        # Probabilities must sum to 1 across channels
        prob_sum = probs.sum(dim=1)
        torch.testing.assert_close(prob_sum, torch.ones_like(prob_sum), atol=1e-5, rtol=1e-5)


def test_unet_checkpoint_save_and_load(tmp_path):
    """Verify saving and loading model weights."""
    model = TideTraceUNet(in_channels=3, num_classes=3)
    ckpt_path = tmp_path / "test_unet.pt"
    model.save_checkpoint(ckpt_path, extra_meta={"author": "TideTrace"})

    assert ckpt_path.exists()
    loaded = TideTraceUNet.load_checkpoint(ckpt_path)
    assert loaded.num_classes == 3
    assert loaded.in_channels == 3


# ==============================================================================
# 3. PREPROCESSING & APODIZED TILING TESTS
# ==============================================================================

def test_radiometric_normalization():
    """Verify dB conversion and clipping to [-30, 0] dB for VV and [-35, -5] dB for VH."""
    # Synthetic linear backscatter
    vv_linear = np.array([[1e-4, 1e-2], [1.0, 1e-1]], dtype=np.float32)
    vv_db = to_decibels(vv_linear)
    assert vv_db[0, 0] == pytest.approx(-40.0, abs=0.5)
    assert vv_db[1, 0] == pytest.approx(0.0, abs=0.1)

    # 3-channel normalization
    h, w = 600, 600
    vv = np.random.uniform(-35.0, 5.0, (h, w)).astype(np.float32)
    vh = np.random.uniform(-40.0, 0.0, (h, w)).astype(np.float32)
    norm_3ch = normalize_sar_bands(vv, vh)

    assert norm_3ch.shape == (3, h, w)
    assert np.all(norm_3ch >= 0.0) and np.all(norm_3ch <= 1.0)


def test_tile_extraction_and_apodized_stitching():
    """Verify seamless tiling, Hann windowing, and stitching reconstruction."""
    h, w = 1000, 1000
    dummy_input = np.ones((3, h, w), dtype=np.float32)
    tiles, coords, orig_shape = extract_tiles(dummy_input, tile_size=512, overlap=64)

    assert orig_shape == (h, w)
    assert tiles.shape[1:] == (3, 512, 512)
    assert len(coords) == tiles.shape[0]

    # Create synthetic tile probabilities where class 0 is 0.7, class 1 is 0.2, class 2 is 0.1
    tile_probs = np.zeros((tiles.shape[0], 3, 512, 512), dtype=np.float32)
    tile_probs[:, 0, :, :] = 0.7
    tile_probs[:, 1, :, :] = 0.2
    tile_probs[:, 2, :, :] = 0.1

    prob_map, class_mask, confidence_map = stitch_tiles(
        tile_probs, coords, orig_shape, num_classes=3, tile_size=512
    )

    assert prob_map.shape == (3, h, w)
    assert class_mask.shape == (h, w)
    assert np.all(class_mask == 0)
    assert np.allclose(confidence_map, 0.7, atol=1e-4)


# ==============================================================================
# 4. ADAPTIVE HEURISTIC FALLBACK DETECTOR TESTS
# ==============================================================================

def test_heuristic_fallback_detector():
    """Verify dark pixel detection, speckle removal, and class separation (oil vs lookalike)."""
    h, w = 512, 512
    # Ambient sea at -12 dB (rough water)
    vv_scene = np.full((h, w), -12.0, dtype=np.float32)

    # 1. Plant an elongated mineral oil slick (narrow filament, -26 dB, aspect ratio > 4)
    vv_scene[200:220, 100:350] = -26.0  # 20 x 250 = 5000 px, aspect ratio ~ 12.5

    # 2. Plant a broad circular calm zone look-alike (-24 dB, 120 x 120 = 14400 px, aspect ratio 1.0)
    vv_scene[320:440, 320:440] = -24.0

    # 3. Add single-pixel speckle noise (-28 dB)
    vv_scene[50, 50] = -28.0

    class_mask, conf_map = run_adaptive_heuristic_detector(vv_scene)

    assert class_mask.shape == (h, w)
    assert conf_map.shape == (h, w)

    # Speckle at (50, 50) must be rejected by opening
    assert class_mask[50, 50] == config.CLASS_SEA

    # Elongated slick must be classified as Mineral Oil (Class 2)
    oil_patch = class_mask[205:215, 120:330]
    assert np.mean(oil_patch == config.CLASS_MINERAL_OIL) > 0.85

    # Broad calm zone must be classified as Look-alike (Class 1)
    lookalike_patch = class_mask[330:430, 330:430]
    assert np.mean(lookalike_patch == config.CLASS_LOOKALIKE) > 0.85


# ==============================================================================
# 5. UNIFIED SEGMENTER TESTS
# ==============================================================================

def test_unified_segmenter_execution():
    """Verify SARSegmenter automatic fallback and full prediction lifecycle."""
    segmenter = SARSegmenter(force_fallback=True)
    status = segmenter.get_status()
    assert status["using_fallback"] is True
    assert status["mode"] == "adaptive_heuristic_fallback"

    vv_test = np.full((512, 512), -10.0, dtype=np.float32)
    vv_test[150:170, 150:300] = -25.0

    mask, conf, probs = segmenter.predict(vv_test, return_probs=True)
    assert mask.shape == (512, 512)
    assert conf.shape == (512, 512)
    assert probs.shape == (3, 512, 512)


# ==============================================================================
# 6. GEOREFERENCING & AFFINE TRANSFORMATION TESTS
# ==============================================================================

def test_affine_transform_forward_and_inverse():
    """Verify pixel to WGS84 and WGS84 to pixel round-trip precision."""
    georef = create_default_georef(height=1000, width=1000, center_lat=18.925, center_lon=72.825)
    
    row, col = 250.0, 750.0
    lon, lat = georef.pixel_to_wgs84(row, col)
    
    row_rec, col_rec = georef.wgs84_to_pixel(lon, lat)
    assert row_rec == pytest.approx(row, abs=1e-5)
    assert col_rec == pytest.approx(col, abs=1e-5)


# ==============================================================================
# 7. LOCAL AEQD PROJECTION & MORPHOLOGY TESTS
# ==============================================================================

def test_aeqd_projection_isometry_and_roundtrip():
    """Verify local AEQD forward and inverse roundtrip and distance isometry."""
    lon0, lat0 = 72.825, 18.925
    # Point exactly ~10 km North
    # 1 deg latitude is approximately 111.195 km, so 10 km is ~0.08993 deg
    lat1 = lat0 + 0.08993216
    lon1 = lon0

    x, y = forward_aeqd(lon1, lat1, lon0, lat0)
    assert abs(x) < 1.0  # Due north -> x ~ 0
    assert y == pytest.approx(10000.0, abs=15.0)  # ~10,000 meters

    # Inverse projection
    lon_rec, lat_rec = inverse_aeqd(x, y, lon0, lat0)
    assert lon_rec == pytest.approx(lon1, abs=1e-6)
    assert lat_rec == pytest.approx(lat1, abs=1e-6)


def test_morphology_descriptors_on_synthetic_slick():
    """
    Construct a rectangular slick aligned at 45 degrees of known dimensions (10 km x 2 km),
    and verify calculated area, perimeter, length, width, and PCA orientation.
    """
    center_lon, center_lat = 72.825, 18.925

    # Build rectangular polygon in local metric space (10 km x 2 km) oriented at 45 degrees
    length_m = 10000.0
    width_m = 2000.0
    half_l = length_m / 2.0
    half_w = width_m / 2.0

    # Corners unrotated
    corners = np.array([
        [-half_w, -half_l],
        [half_w, -half_l],
        [half_w, half_l],
        [-half_w, half_l],
        [-half_w, -half_l],
    ])

    # Rotate clockwise by 45 degrees towards East (North-East, bearing 45 deg): +y is North, +x is East
    theta = math.radians(45.0)
    rot_mat = np.array([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)],
    ])
    rotated_corners = corners @ rot_mat.T

    # Convert rotated metric corners to WGS84
    wgs84_coords = []
    for x, y in rotated_corners:
        lon, lat = inverse_aeqd(x, y, center_lon, center_lat)
        wgs84_coords.append([float(lon), float(lat)])

    poly = Polygon(wgs84_coords)
    props = compute_polygon_morphology(poly, slick_id="test_slick_01")

    # Verification assertions:
    # Expected Area: 10 km * 2 km = 20 km^2
    assert props["area_km2"] == pytest.approx(20.0, rel=0.03)
    # Expected Perimeter: 2 * (10 + 2) = 24 km
    assert props["perimeter_km"] == pytest.approx(24.0, rel=0.03)
    # Expected Length: ~10 km
    assert props["length_km"] == pytest.approx(10.0, rel=0.05)
    # Expected Width: ~2 km
    assert props["width_km"] == pytest.approx(2.0, rel=0.10)
    # Expected Orientation: ~45 degrees
    assert props["orientation_deg"] == pytest.approx(45.0, abs=5.0)
    # Expected Centroid
    assert props["centroid"][0] == pytest.approx(center_lat, abs=1e-4)
    assert props["centroid"][1] == pytest.approx(center_lon, abs=1e-4)


def test_mask_vectorization_and_geojson_export():
    """Verify full raster-to-GeoJSON pipeline and RFC 7946 compliance."""
    h, w = 500, 500
    class_mask = np.zeros((h, w), dtype=np.uint8)
    # Add mineral oil slick
    class_mask[100:150, 100:300] = config.CLASS_MINERAL_OIL
    # Add look-alike
    class_mask[350:400, 350:400] = config.CLASS_LOOKALIKE

    georef = create_default_georef(height=h, width=w, center_lat=18.925, center_lon=72.825)
    
    slicks = vectorize_mask_to_polygons(class_mask, georef, target_class=config.CLASS_MINERAL_OIL)
    lookalikes = vectorize_mask_to_polygons(class_mask, georef, target_class=config.CLASS_LOOKALIKE)

    assert len(slicks) == 1
    assert len(lookalikes) == 1

    geojson = generate_geojson_feature_collection(slicks, lookalikes, scene_metadata={"scene_id": "TEST_SCENE"})

    assert geojson["type"] == "FeatureCollection"
    assert geojson["properties"]["num_oil_slicks"] == 1
    assert geojson["properties"]["num_lookalikes"] == 1
    assert len(geojson["features"]) == 2

    # Check feature properties
    f_oil = geojson["features"][0]
    assert f_oil["properties"]["class_name"] == "mineral_oil"
    assert f_oil["properties"]["area_km2"] > 0
    assert "centroid" in f_oil["properties"]
    assert len(f_oil["geometry"]["coordinates"][0]) >= 4
