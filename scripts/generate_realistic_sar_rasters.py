#!/usr/bin/env python3
"""
TideTrace Realistic SAR Radar Backscatter & High-Contrast Imagery Generator
SIH 26143 / NTRO Naval Forensic Oil Spill Attribution Console

Generates high-resolution, physically authentic Sentinel-1 SAR dual-pol (VV, VH in dB)
radar swaths, GeoTIFFs, and multi-palette tactical map overlays:
1. Mode 'radar_db': Deep oceanic slate-navy clutter with gravity wave swells,
   wind streaks, multiplicative Gamma speckle, brilliant white specular vessel corner
   reflectors with cruciform radar sidelobes, and a stark, jet-black petroleum damping void.
2. Mode 'thermal': Radiometric cold-void thermal palette (cryogenic dark-purple oil void,
   radiant warm amber/volcanic sea clutter, white-hot ship reflections).
3. Mode 'false_color': Sentinel-1 dual-pol polarimetric composite (VV, VH, VV/VH ratio)
   with vibrant sea-green clutter, ink-black petroleum void, and glowing neon ships.
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("tidetrace.sar_generator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = PROJECT_ROOT / "data" / "scenes"


def add_ship_reflector(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    cy: int,
    cx: int,
    heading_deg: float = 0.0,
    length_px: int = 12,
    width_px: int = 4,
    intensity_db: float = 0.0,
):
    """
    Render a metallic ship hull as an intense specular radar corner reflector
    with range/azimuth cross-sidelobe diffraction spikes and trailing wake.
    Sentinel-1 calibrated GRD saturates at 0.0 dB for metallic targets.
    """
    H, W = vv_db.shape
    rad_hdg = math.radians(heading_deg)
    cos_h, sin_h = math.cos(rad_hdg), math.sin(rad_hdg)

    # 1. Ship metallic hull
    half_l = max(2, length_px // 2)
    half_w = max(1, width_px // 2)
    
    y_min = max(0, cy - half_l - 2)
    y_max = min(H, cy + half_l + 3)
    x_min = max(0, cx - half_l - 2)
    x_max = min(W, cx + half_l + 3)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dy = y - cy
            dx = x - cx
            # Rotate along ship heading
            long_axis = -dx * sin_h + dy * cos_h
            lat_axis = dx * cos_h + dy * sin_h
            if abs(long_axis) <= half_l and abs(lat_axis) <= half_w:
                dist_ratio = 1.0 - math.sqrt((long_axis / half_l)**2 + (lat_axis / half_w)**2) / math.sqrt(2)
                val_vv = intensity_db - 2.5 * (1.0 - max(0.6, dist_ratio))
                val_vh = (intensity_db - 5.0) - 2.0 * (1.0 - max(0.5, dist_ratio))
                vv_db[y, x] = max(vv_db[y, x], val_vv)
                vh_db[y, x] = max(vh_db[y, x], val_vh)

    # 2. SAR Cardinal Cruciform Sidelobes (radar antenna beam pattern artifact)
    spike_len = 16
    for offset in range(-spike_len, spike_len + 1):
        if offset == 0:
            continue
        decay = math.exp(-abs(offset) / 5.5)
        # Horizontal range sidelobe
        sx = cx + offset
        if 0 <= sx < W and 0 <= cy < H:
            vv_db[cy, sx] = max(vv_db[cy, sx], (intensity_db - 3.5) - (1.0 - decay) * 7.0)
            vh_db[cy, sx] = max(vh_db[cy, sx], (intensity_db - 8.0) - (1.0 - decay) * 7.0)
        # Vertical azimuth sidelobe
        sy = cy + offset
        if 0 <= sy < H and 0 <= cx < W:
            vv_db[sy, cx] = max(vv_db[sy, cx], (intensity_db - 3.5) - (1.0 - decay) * 7.0)
            vh_db[sy, cx] = max(vh_db[sy, cx], (intensity_db - 8.0) - (1.0 - decay) * 7.0)

    # 3. Hydrodynamic turbulent wake trailing vessel (behind stern)
    stern_y = cy - int(half_l * cos_h)
    stern_x = cx + int(half_l * sin_h)
    wake_len = 35
    for step in range(1, wake_len):
        wy = int(stern_y - step * cos_h)
        wx = int(stern_x + step * sin_h)
        if 0 <= wy < H and 0 <= wx < W:
            # Turbulent center (slightly suppressed backscatter due to foam/damping)
            vv_db[wy, wx] = min(vv_db[wy, wx], -18.0)
            # Diverging wake arms (brighter backscatter from breaking wake waves)
            for arm_sign in [-1, 1]:
                arm_y = int(wy + arm_sign * (step * 0.22) * sin_h)
                arm_x = int(wx + arm_sign * (step * 0.22) * cos_h)
                if 0 <= arm_y < H and 0 <= arm_x < W:
                    vv_db[arm_y, arm_x] = max(vv_db[arm_y, arm_x], -11.0)


def add_platform_reflector(vv_db: np.ndarray, vh_db: np.ndarray, cy: int, cx: int, intensity_db: float = 0.0):
    """Render an offshore oil drilling rig / production platform corner reflector."""
    H, W = vv_db.shape
    r = 3
    y_min, y_max = max(0, cy - r), min(H, cy + r + 1)
    x_min, x_max = max(0, cx - r), min(W, cx + r + 1)
    vv_db[y_min:y_max, x_min:x_max] = intensity_db
    vh_db[y_min:y_max, x_min:x_max] = intensity_db - 5.0

    # Hexagonal / cross truss diffraction
    for offset in range(-12, 13):
        if offset == 0:
            continue
        decay = math.exp(-abs(offset) / 4.0)
        if 0 <= cx + offset < W:
            vv_db[cy, cx + offset] = max(vv_db[cy, cx + offset], (intensity_db - 3.0) - (1.0 - decay) * 7.0)
        if 0 <= cy + offset < H:
            vv_db[cy + offset, cx] = max(vv_db[cy + offset, cx], (intensity_db - 3.0) - (1.0 - decay) * 7.0)


def synthesize_realistic_sar_channels(
    scene_id: str,
    mask: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic 2048x2048 dual-pol Sentinel-1 SAR arrays (VV, VH in dB)
    with genuine ocean wave swell patterns, capillary clutter, and a stark
    pitch-black petroleum damping void matching the ground truth mask.
    """
    H, W = mask.shape
    rng = np.random.default_rng(seed)
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # 1. Base clean ocean background
    grad_vv = 0.6 * (x_coords / W - 0.5) - 0.4 * (y_coords / H - 0.5)

    # Ocean primary gravity swell (wavelength ~ 22 px, ~220m)
    swell_deg = 52.0 if "mumbai" in scene_id else (28.0 if "chennai" in scene_id else 122.0)
    swell_rad = math.radians(swell_deg)
    k_x = 2.0 * np.pi / 22.0 * np.cos(swell_rad)
    k_y = 2.0 * np.pi / 22.0 * np.sin(swell_rad)
    swell = 2.0 * np.sin(k_x * x_coords + k_y * y_coords)

    # Secondary cross-swell (wavelength ~ 9 px, ~90m)
    cross_rad = swell_rad + math.radians(38.0)
    k2_x = 2.0 * np.pi / 9.5 * np.cos(cross_rad)
    k2_y = 2.0 * np.pi / 9.5 * np.sin(cross_rad)
    cross_swell = 0.9 * np.sin(k2_x * x_coords + k2_y * y_coords + 1.2)

    # Wind roll vortices / streaks (wavelength ~ 240 px, ~2.4 km)
    kw_rad = swell_rad - math.radians(14.0)
    kw_x = 2.0 * np.pi / 240.0 * np.cos(kw_rad)
    kw_y = 2.0 * np.pi / 240.0 * np.sin(kw_rad)
    wind_streaks = 1.3 * np.sin(kw_x * x_coords + kw_y * y_coords)

    # Multiplicative SAR Gamma speckle (4-look ENL)
    speckle_vv = rng.gamma(shape=4.0, scale=0.25, size=(H, W)) - 1.0
    speckle_vh = rng.gamma(shape=4.0, scale=0.25, size=(H, W)) - 1.0

    # Ambient sea backscatter
    vv_sea = -14.2 + grad_vv + swell + cross_swell + wind_streaks + 1.25 * speckle_vv
    vh_sea = -24.5 + grad_vv * 0.8 + 0.5 * (swell + cross_swell) + wind_streaks * 0.6 + 1.35 * speckle_vh

    vv_db = vv_sea.copy()
    vh_db = vh_sea.copy()

    # 2. Mineral Oil Slick Damping (Class 2)
    oil_mask = (mask == 2)
    if np.any(oil_mask):
        dist_inside = ndimage.distance_transform_edt(oil_mask)
        dist_max = max(1.0, float(dist_inside.max()))
        core_weight = np.clip(dist_inside / (dist_max * 0.35), 0.0, 1.0)

        # Damping magnitude: drops VV down to -28.0 dB ... -31.5 dB (NESZ floor)
        damping_vv = 13.8 + 3.8 * core_weight
        damping_vh = 10.0 + 3.0 * core_weight

        # Wave flattening: oil surface tension eradicates gravity-capillary swell ripples
        flatten_factor = 0.88 * core_weight
        vv_damped = (vv_sea - (swell + cross_swell) * flatten_factor) - damping_vv
        vh_damped = (vh_sea - 0.5 * (swell + cross_swell) * flatten_factor) - damping_vh

        vv_db[oil_mask] = vv_damped[oil_mask]
        vh_db[oil_mask] = vh_damped[oil_mask]

    # 3. Look-alike Biogenic Film (Class 1)
    look_mask = (mask == 1)
    if np.any(look_mask):
        dist_inside = ndimage.distance_transform_edt(look_mask)
        core_weight = np.clip(dist_inside / 16.0, 0.0, 1.0)
        damping_vv = 5.2 + 2.0 * core_weight
        damping_vh = 3.8 + 1.5 * core_weight

        flatten_factor = 0.45 * core_weight
        vv_damped = (vv_sea - (swell + cross_swell) * flatten_factor) - damping_vv
        vh_damped = (vh_sea - 0.5 * (swell + cross_swell) * flatten_factor) - damping_vh

        vv_db[look_mask] = vv_damped[look_mask]
        vh_db[look_mask] = vh_damped[look_mask]

    # 4. Specular Corner Reflectors (Ships & Platforms)
    # Calibrated to physical Sentinel-1 GRD saturation bounds (VV <= 0.0 dB, VH <= -5.0 dB)
    if "mumbai" in scene_id:
        # MT OCEAN VOYAGER at (1024, 1024) (Culprit Crude Oil Tanker)
        add_ship_reflector(vv_db, vh_db, cy=1024, cx=1024, heading_deg=65.0, length_px=14, width_px=4, intensity_db=0.0)
        # FV SEA HARVEST at (850, 1150) (Fishing Trawler)
        add_ship_reflector(vv_db, vh_db, cy=850, cx=1150, heading_deg=140.0, length_px=6, width_px=3, intensity_db=-2.0)
        # Mumbai High North offshore platform rig at (1180, 920)
        add_platform_reflector(vv_db, vh_db, cy=1180, cx=920, intensity_db=0.0)
    elif "chennai" in scene_id:
        # Offending bunker discharge vessel at (1024, 1024)
        add_ship_reflector(vv_db, vh_db, cy=1024, cx=1024, heading_deg=18.0, length_px=12, width_px=4, intensity_db=0.0)
        # Commercial cargo ship at (1300, 950)
        add_ship_reflector(vv_db, vh_db, cy=1300, cx=950, heading_deg=198.0, length_px=8, width_px=3, intensity_db=-2.5)
    elif "gom" in scene_id:
        # Deepwater tanker at (1024, 1024)
        add_ship_reflector(vv_db, vh_db, cy=1024, cx=1024, heading_deg=118.0, length_px=14, width_px=4, intensity_db=0.0)
        # Supply vessel at (920, 1200)
        add_ship_reflector(vv_db, vh_db, cy=920, cx=1200, heading_deg=310.0, length_px=6, width_px=2, intensity_db=-3.0)
    elif "clean" in scene_id:
        # Merchant ship transit in clean sea (proving non-spill ship detection)
        add_ship_reflector(vv_db, vh_db, cy=750, cx=1250, heading_deg=80.0, length_px=10, width_px=3, intensity_db=0.0)

    # Physical clipping strictly adhering to Sentinel-1 calibrated dB ranges [-30.0, 0.0] for VV and [-35.0, -5.0] for VH
    vv_db = np.clip(vv_db, -30.0, 0.0).astype(np.float32)
    vh_db = np.clip(vh_db, -35.0, -5.0).astype(np.float32)

    return vv_db, vh_db


def render_radar_db_rgba(vv_db: np.ndarray) -> Image.Image:
    """
    Render Mode 'Radar dB': Deep oceanic slate-navy clutter with wave swell crests,
    brilliant white specular vessels, and a stark, pitch-black petroleum damping void.
    """
    H, W = vv_db.shape
    # Normalize backscatter from -28.0 dB (void) to -8.0 dB (bright wave crests)
    norm = np.clip((vv_db - (-28.0)) / 20.0, 0.0, 1.0)

    # Tactical oceanic radar palette:
    # Oil slick (<0.12): stark jet-black / dark charcoal [2, 4, 8]
    # Ambient sea (0.6-0.8): textured deep slate navy [55, 80, 115] to [75, 110, 150]
    # Wave crests (>0.85): [95, 130, 180]
    # Ship specular returns (> -4.0 dB): pure white [255, 255, 255]
    r = np.where(norm < 0.12, 2 + norm * 20, 24 + norm * 62)
    g = np.where(norm < 0.12, 4 + norm * 30, 42 + norm * 88)
    b = np.where(norm < 0.12, 8 + norm * 45, 64 + norm * 118)

    # Specular vessel highlights
    ship_pixels = vv_db > -4.0
    r = np.where(ship_pixels, 255, r)
    g = np.where(ship_pixels, 255, g)
    b = np.where(ship_pixels, 255, b)

    # Subtle edge feathering for smooth seamless overlay transition
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    edge_d = np.minimum(np.minimum(x_idx, W - 1 - x_idx), np.minimum(y_idx, H - 1 - y_idx))
    alpha = np.clip(edge_d / 16.0, 0.0, 1.0) * 255.0

    rgba = np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
        alpha.astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def render_thermal_void_rgba(vv_db: np.ndarray) -> Image.Image:
    """
    Render Mode 'Thermal Void': Radiometric forensic thermal mapping.
    Cryogenic dark-purple petroleum damping void, warm amber/volcanic ocean clutter,
    and white-hot blazing yellow specular ship returns.
    """
    H, W = vv_db.shape
    norm = np.clip((vv_db - (-28.0)) / 20.0, 0.0, 1.0)

    # Damping void (<0.15): Deep midnight purple / obsidian abyss [10, 4, 28]
    # Ambient ocean: Radiant warm amber/copper [195, 105, 30] to gold [245, 175, 50]
    # Specular returns: White-hot gold [255, 255, 220]
    r = np.where(norm < 0.15, 6 + norm * 55, 50 + norm * 195)
    g = np.where(norm < 0.15, 3 + norm * 20, 20 + norm * 155)
    b = np.where(norm < 0.15, 18 + norm * 70, 45 - norm * 30)

    ship_pixels = vv_db > -4.0
    r = np.where(ship_pixels, 255, r)
    g = np.where(ship_pixels, 255, g)
    b = np.where(ship_pixels, 220, b)

    y_idx, x_idx = np.mgrid[0:H, 0:W]
    edge_d = np.minimum(np.minimum(x_idx, W - 1 - x_idx), np.minimum(y_idx, H - 1 - y_idx))
    alpha = np.clip(edge_d / 16.0, 0.0, 1.0) * 255.0

    rgba = np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
        alpha.astype(np.uint8)
    ], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def render_false_color_rgba(vv_db: np.ndarray, vh_db: np.ndarray) -> Image.Image:
    """
    Render Mode 'Dual-Pol False Color': Authentic Sentinel-1 Polarimetric Composite.
    Red: VV Co-pol intensity + cross-polarization ratio
    Green: VH Cross-pol volume scattering (rough water chop)
    Blue: Depolarization ratio (surface roughness index)
    Ocean: Vibrant emerald / turquoise sea clutter.
    Oil spill: Stark, jet-black void (both VV and VH suppressed).
    Ships: Blazing neon magenta / coral specular reflector.
    """
    H, W = vv_db.shape
    norm_vv = np.clip((vv_db - (-28.0)) / 20.0, 0.0, 1.0)
    norm_vh = np.clip((vh_db - (-33.0)) / 20.0, 0.0, 1.0)
    ratio = np.clip(((vv_db - vh_db) - 4.0) / 10.0, 0.0, 1.0)

    # Ocean clutter: Vibrant emerald / turquoise [20-40, 150-200, 120-165]
    # Oil spill: Jet-black void [2, 5, 8]
    # Specular returns: Neon magenta [255, 45, 180]
    is_ocean = norm_vv >= 0.12
    r = np.where(is_ocean, 15 + norm_vv * 40.0 + (1.0 - ratio) * 25.0, 2 + norm_vv * 15.0)
    g = np.where(is_ocean, 40 + norm_vh * 165.0 + norm_vv * 35.0, 4 + norm_vv * 20.0)
    b = np.where(is_ocean, 30 + ratio * 125.0 + norm_vv * 40.0, 7 + norm_vv * 30.0)

    ship_pixels = vv_db > -4.0
    r = np.where(ship_pixels, 255, r)
    g = np.where(ship_pixels, 55, g)
    b = np.where(ship_pixels, 205, b)

    y_idx, x_idx = np.mgrid[0:H, 0:W]
    edge_d = np.minimum(np.minimum(x_idx, W - 1 - x_idx), np.minimum(y_idx, H - 1 - y_idx))
    alpha = np.clip(edge_d / 16.0, 0.0, 1.0) * 255.0

    rgba = np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
        alpha.astype(np.uint8)
    ], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def generate_all_realistic_scenes():
    """Overhaul all demo scenes with realistic radar physics and save multi-palette rasters."""
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    scenes = ["scene_mumbai_001", "scene_chennai_002", "scene_gom_003", "scene_clean_004"]

    logger.info("=" * 70)
    logger.info(" GENERATING REALISTIC HIGH-RESOLUTION SENTINEL-1 SAR RASTERS")
    logger.info("=" * 70)

    for scene_id in scenes:
        mask_path = SCENES_DIR / f"{scene_id}_mask.npy"
        if not mask_path.exists():
            logger.warning(f"Mask {mask_path} not found, creating dummy 2048x2048 mask.")
            mask = np.zeros((2048, 2048), dtype=np.uint8)
        else:
            mask = np.load(mask_path)

        seed = 42 if "mumbai" in scene_id else (43 if "chennai" in scene_id else (44 if "gom" in scene_id else 45))
        logger.info(f"Synthesizing SAR backscatter channels for {scene_id} (seed={seed})...")
        vv_db, vh_db = synthesize_realistic_sar_channels(scene_id, mask, seed=seed)

        # 1. Save Dual-Channel NumPy Array (shape: 2, 2048, 2048)
        npy_path = SCENES_DIR / f"{scene_id}.npy"
        dual_pol_array = np.stack([vv_db, vh_db], axis=0)
        np.save(npy_path, dual_pol_array)
        logger.info(f"Saved {npy_path} ({dual_pol_array.nbytes / (1024*1024):.1f} MB)")

        # 2. Save GeoTIFF
        tif_path = SCENES_DIR / f"{scene_id}.tif"
        try:
            import tifffile
            json_path = SCENES_DIR / f"{scene_id}.json"
            meta = {}
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            tifffile.imwrite(
                str(tif_path),
                dual_pol_array,
                photometric="minisblack",
                metadata={
                    "affine": meta.get("affine", [0.0001953125, 0.0, 71.95, 0.0, -0.0001953125, 19.15]),
                    "crs": meta.get("crs", "EPSG:4326"),
                    "bands": ["VV_dB", "VH_dB"],
                }
            )
            logger.info(f"Saved GeoTIFF {tif_path}")
        except Exception as e:
            logger.warning(f"GeoTIFF save error (fallback): {e}")

        # 3. Generate and save Multi-Palette RGBA Images at full 2048x2048 resolution
        img_radar = render_radar_db_rgba(vv_db)
        out_preview = SCENES_DIR / f"{scene_id}_preview.png"
        img_radar.save(out_preview, optimize=True)
        logger.info(f"Saved Radar dB overlay -> {out_preview} ({out_preview.stat().st_size / 1024:.1f} KB)")

        img_thermal = render_thermal_void_rgba(vv_db)
        out_thermal = SCENES_DIR / f"{scene_id}_thermal.png"
        img_thermal.save(out_thermal, optimize=True)
        logger.info(f"Saved Thermal Void overlay -> {out_thermal} ({out_thermal.stat().st_size / 1024:.1f} KB)")

        img_false_color = render_false_color_rgba(vv_db, vh_db)
        out_fc = SCENES_DIR / f"{scene_id}_false_color.png"
        img_false_color.save(out_fc, optimize=True)
        logger.info(f"Saved Dual-Pol False Color overlay -> {out_fc} ({out_fc.stat().st_size / 1024:.1f} KB)")

    logger.info("All realistic SAR rasters successfully generated!")


if __name__ == "__main__":
    generate_all_realistic_scenes()
