#!/usr/bin/env python3
"""
TideTrace Synthetic AIS Generation Utility
SIH 26143 / NTRO Naval Forensic Oil Spill Attribution Console

Populates SQLite database (data/ais/tidetrace_ais.db) with MarineCadastre 16-field schema:
- Planted Culprit: MT OCEAN VOYAGER (MMSI 419000001, Crude Oil Tanker, transit through origin zone
  with a 52-minute transponder blackout delta_t=52m, speed 12.5 kn, attribution score >90%).
- Distractor 1: FV SEA HARVEST (MMSI 419000002, Fishing Vessel, nearby d=0.8km, continuous reporting, score ~43%).
- Distractor 2: MV PACIFIC TRADER (MMSI 419000003, Container Cargo, distance d=8.5km, score ~20%).
- Background: 28+ realistic non-culprit merchant and fishing vessels operating across the sector.

Also generates full realistic AIS traffic for scene_chennai_002 and scene_gom_003.
Creates high-performance composite B-Tree indexes: (BaseDateTime, LAT, LON) & (MMSI, BaseDateTime).
"""

import os
import sys
import math
import sqlite3
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tidetrace.ais")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "ais" / "tidetrace_ais.db"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "ais" / "tidetrace_ais.csv"

# Earth radius in km for geodesic approximation
R_EARTH_KM = 6371.0


def parse_iso_time(iso_str: str) -> datetime:
    clean_str = iso_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean_str)


def format_iso_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def move_point(lat: float, lon: float, distance_km: float, bearing_deg: float) -> Tuple[float, float]:
    """Move a geographic coordinate by distance (km) and bearing (degrees)."""
    rad_lat = math.radians(lat)
    rad_lon = math.radians(lon)
    rad_bearing = math.radians(bearing_deg)
    ang_dist = distance_km / R_EARTH_KM

    new_lat = math.asin(
        math.sin(rad_lat) * math.cos(ang_dist) +
        math.cos(rad_lat) * math.sin(ang_dist) * math.cos(rad_bearing)
    )
    new_lon = rad_lon + math.atan2(
        math.sin(rad_bearing) * math.sin(ang_dist) * math.cos(rad_lat),
        math.cos(ang_dist) - math.sin(rad_lat) * math.sin(new_lat)
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in kilometers."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(dlon / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R_EARTH_KM * c


def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with MarineCadastre schema and composite indexes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS ais_records;")

    # MarineCadastre 16-field AIS schema + epoch index
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ais_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        MMSI INTEGER NOT NULL,
        BaseDateTime TEXT NOT NULL,
        timestamp_epoch REAL NOT NULL,
        LAT REAL NOT NULL,
        LON REAL NOT NULL,
        SOG REAL NOT NULL,
        COG REAL NOT NULL,
        Heading REAL NOT NULL,
        VesselName TEXT NOT NULL,
        IMO TEXT,
        CallSign TEXT,
        VesselType INTEGER NOT NULL,
        Status INTEGER DEFAULT 0,
        Length REAL,
        Width REAL,
        Draft REAL,
        Cargo INTEGER DEFAULT 0
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_time_lat_lon ON ais_records (BaseDateTime, LAT, LON);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_epoch_lat_lon ON ais_records (timestamp_epoch, LAT, LON);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_mmsi_epoch ON ais_records (MMSI, timestamp_epoch);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_mmsi_time ON ais_records (MMSI, BaseDateTime);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_vessel_type ON ais_records (VesselType);")

    conn.commit()
    return conn


def generate_scenario_mumbai() -> List[Dict[str, Any]]:
    """
    Generate deterministic synthetic AIS traffic for scene_mumbai_001.
    
    SAR Acquisition: 2026-09-02T12:00:00Z at (18.95, 72.15)
    Backtrack Origin: 2026-09-01T18:00:00Z at (18.754, 71.842)
    """
    records = []
    
    t_sat = parse_iso_time("2026-09-02T12:00:00Z")
    t_start = t_sat - timedelta(hours=48)
    t_end = t_sat + timedelta(hours=24)
    
    # Origin location at t_origin = 2026-09-01T18:00:00Z (aligned with Lagrangian backtrack centroid)
    t_origin = parse_iso_time("2026-09-01T18:00:00Z")
    orig_lat, orig_lon = 18.814, 71.842

    # =========================================================================
    # 1. PLANTED CULPRIT: MT OCEAN VOYAGER (MMSI 419000001)
    # Crude Oil Tanker (Type 80), SOG 12.5 kn, COG 65.0 deg
    # 52-minute transponder blackout between 17:34:00Z and 18:26:00Z
    # =========================================================================
    culprit_speed_kn = 12.5
    culprit_speed_kmh = culprit_speed_kn * 1.852  # ~23.15 km/h
    culprit_course = 65.0
    
    # Culprit transits from SW to NE across Mumbai shipping channel
    # Total track duration 14 hours centered around t_origin
    culprit_start_time = t_origin - timedelta(hours=7)
    culprit_end_time = t_origin + timedelta(hours=7)
    
    blackout_start = t_origin - timedelta(minutes=26)  # 17:34:00Z
    blackout_end = t_origin + timedelta(minutes=26)    # 18:26:00Z
    
    curr_time = culprit_start_time
    sample_interval_sec = 300  # 5-min normal reporting
    
    while curr_time <= culprit_end_time:
        # Check blackout gap: do NOT report if strictly between blackout_start and blackout_end
        in_blackout = (blackout_start < curr_time < blackout_end)
        
        # Exact position calculation along transit vector through origin at t_origin
        dt_hours = (curr_time - t_origin).total_seconds() / 3600.0
        dist_from_orig_km = dt_hours * culprit_speed_kmh
        bearing = culprit_course if dist_from_orig_km >= 0 else (culprit_course + 180.0) % 360.0
        pos_lat, pos_lon = move_point(orig_lat, orig_lon, abs(dist_from_orig_km), bearing)
        
        # Add slight 0.5 km offset so closest approach is ~0.58 km
        offset_lat, offset_lon = move_point(pos_lat, pos_lon, 0.58, (culprit_course + 90.0) % 360.0)
        
        if not in_blackout:
            records.append({
                "MMSI": 419000001,
                "BaseDateTime": format_iso_time(curr_time),
                "LAT": round(offset_lat, 5),
                "LON": round(offset_lon, 5),
                "SOG": 12.5,
                "COG": 65.0,
                "Heading": 65.0,
                "VesselName": "MT OCEAN VOYAGER",
                "IMO": "IMO9312014",
                "CallSign": "VTOL",
                "VesselType": 80,  # Tanker
                "Status": 0,       # Under way using engine
                "Length": 274.0,
                "Width": 48.0,
                "Draft": 16.5,
                "Cargo": 80
            })
            
        curr_time += timedelta(seconds=sample_interval_sec)

    # Ensure boundary blackout points exist exactly at 17:34:00Z and 18:26:00Z
    for exact_dt in [blackout_start, blackout_end]:
        dt_hours = (exact_dt - t_origin).total_seconds() / 3600.0
        dist_from_orig_km = dt_hours * culprit_speed_kmh
        bearing = culprit_course if dist_from_orig_km >= 0 else (culprit_course + 180.0) % 360.0
        pos_lat, pos_lon = move_point(orig_lat, orig_lon, abs(dist_from_orig_km), bearing)
        offset_lat, offset_lon = move_point(pos_lat, pos_lon, 0.58, (culprit_course + 90.0) % 360.0)
        iso_t = format_iso_time(exact_dt)
        if not any(r["MMSI"] == 419000001 and r["BaseDateTime"] == iso_t for r in records):
            records.append({
                "MMSI": 419000001,
                "BaseDateTime": iso_t,
                "LAT": round(offset_lat, 5),
                "LON": round(offset_lon, 5),
                "SOG": 12.5,
                "COG": 65.0,
                "Heading": 65.0,
                "VesselName": "MT OCEAN VOYAGER",
                "IMO": "IMO9312014",
                "CallSign": "VTOL",
                "VesselType": 80,
                "Status": 0,
                "Length": 274.0,
                "Width": 48.0,
                "Draft": 16.5,
                "Cargo": 80
            })

    # =========================================================================
    # 2. DISTRACTOR 1: FV SEA HARVEST (MMSI 419000002)
    # Fishing Vessel (Type 30), SOG 4.2 kn, distance d ~ 0.8 km, continuous reporting
    # =========================================================================
    distractor1_start = t_origin - timedelta(hours=6)
    distractor1_end = t_origin + timedelta(hours=6)
    curr_time = distractor1_start
    
    while curr_time <= distractor1_end:
        dt_hours = (curr_time - t_origin).total_seconds() / 3600.0
        # Trawling zig-zag pattern near origin
        d_km = 0.82 + 0.3 * math.sin(dt_hours * 1.5)
        bearing = (140.0 + 25.0 * dt_hours) % 360.0
        pos_lat, pos_lon = move_point(orig_lat, orig_lon, d_km, bearing)
        
        records.append({
            "MMSI": 419000002,
            "BaseDateTime": format_iso_time(curr_time),
            "LAT": round(pos_lat, 5),
            "LON": round(pos_lon, 5),
            "SOG": round(4.2 + 0.4 * math.sin(dt_hours), 1),
            "COG": round((bearing + 90.0) % 360.0, 1),
            "Heading": round((bearing + 90.0) % 360.0, 1),
            "VesselName": "FV SEA HARVEST",
            "IMO": "IMO8923411",
            "CallSign": "VUFH",
            "VesselType": 30,  # Fishing
            "Status": 7,       # Engaged in fishing
            "Length": 32.0,
            "Width": 8.2,
            "Draft": 3.4,
            "Cargo": 30
        })
        curr_time += timedelta(minutes=5)

    # =========================================================================
    # 3. DISTRACTOR 2: MV PACIFIC TRADER (MMSI 419000003)
    # Container Cargo (Type 70), SOG 18.2 kn, distance d ~ 8.5 km, continuous reporting
    # =========================================================================
    distractor2_start = t_origin - timedelta(hours=5)
    distractor2_end = t_origin + timedelta(hours=5)
    curr_time = distractor2_start
    
    while curr_time <= distractor2_end:
        dt_hours = (curr_time - t_origin).total_seconds() / 3600.0
        dist_parallel_km = dt_hours * (18.2 * 1.852)
        # Offset 8.5 km to the north-northwest
        base_lat, base_lon = move_point(orig_lat, orig_lon, 8.5, 335.0)
        pos_lat, pos_lon = move_point(base_lat, base_lon, abs(dist_parallel_km), 65.0 if dist_parallel_km >= 0 else 245.0)
        
        records.append({
            "MMSI": 419000003,
            "BaseDateTime": format_iso_time(curr_time),
            "LAT": round(pos_lat, 5),
            "LON": round(pos_lon, 5),
            "SOG": 18.2,
            "COG": 65.0,
            "Heading": 64.0,
            "VesselName": "MV PACIFIC TRADER",
            "IMO": "IMO9554321",
            "CallSign": "9V8821",
            "VesselType": 70,  # Cargo / Container
            "Status": 0,
            "Length": 294.0,
            "Width": 32.2,
            "Draft": 12.1,
            "Cargo": 70
        })
        curr_time += timedelta(minutes=5)

    # =========================================================================
    # 4. BACKGROUND FLEET (28 realistic non-culprit vessels)
    # Covering Arabian Sea shipping corridor, offshore supply, and local craft
    # =========================================================================
    bg_fleet = [
        {"mmsi": 419000010, "name": "MT BHARAT RATNA", "type": 80, "sog": 13.1, "cog": 245.0, "len": 245, "wid": 42, "draft": 14.8, "offset_km": 16.0, "bearing": 210, "t_offset_h": -4.0},
        {"mmsi": 419000011, "name": "MV MAERSK KARACHI", "type": 70, "sog": 19.5, "cog": 68.0, "len": 310, "wid": 44, "draft": 13.5, "offset_km": 22.0, "bearing": 180, "t_offset_h": 2.0},
        {"mmsi": 419000012, "name": "MT AL-KUWAIT STAR", "type": 81, "sog": 11.8, "cog": 72.0, "len": 280, "wid": 48, "draft": 17.0, "offset_km": 28.0, "bearing": 150, "t_offset_h": 12.0},
        {"mmsi": 419000013, "name": "FV SAGAR KANYA", "type": 30, "sog": 3.8, "cog": 110.0, "len": 28, "wid": 7, "draft": 2.8, "offset_km": 14.5, "bearing": 45, "t_offset_h": -1.0},
        {"mmsi": 419000014, "name": "TB OCEAN PRIDE", "type": 52, "sog": 8.4, "cog": 190.0, "len": 45, "wid": 12, "draft": 4.5, "offset_km": 18.0, "bearing": 300, "t_offset_h": -8.0},
        {"mmsi": 419000015, "name": "OSV SAMUDRA SEVAK", "type": 60, "sog": 9.2, "cog": 340.0, "len": 65, "wid": 15, "draft": 5.2, "offset_km": 12.0, "bearing": 10, "t_offset_h": 5.0},
        {"mmsi": 419000016, "name": "MV ARABIAN STAR", "type": 71, "sog": 14.6, "cog": 250.0, "len": 180, "wid": 28, "draft": 9.5, "offset_km": 25.0, "bearing": 90, "t_offset_h": -14.0},
        {"mmsi": 419000017, "name": "LPG GAS MONARCH", "type": 82, "sog": 15.0, "cog": 65.0, "len": 220, "wid": 36, "draft": 11.2, "offset_km": 30.0, "bearing": 260, "t_offset_h": -20.0},
        {"mmsi": 419000018, "name": "CMA CGM MUMBAI", "type": 70, "sog": 20.2, "cog": 242.0, "len": 366, "wid": 51, "draft": 15.2, "offset_km": 19.5, "bearing": 320, "t_offset_h": 8.0},
        {"mmsi": 419000019, "name": "INS TRISHUL", "type": 35, "sog": 16.5, "cog": 135.0, "len": 125, "wid": 15, "draft": 4.5, "offset_km": 24.0, "bearing": 40, "t_offset_h": 0.0},
        {"mmsi": 419000020, "name": "FV MATSYA VIGYAN", "type": 30, "sog": 4.0, "cog": 290.0, "len": 35, "wid": 8, "draft": 3.0, "offset_km": 17.0, "bearing": 130, "t_offset_h": 6.0},
        {"mmsi": 419000021, "name": "MT NORDIC EXPLORER", "type": 80, "sog": 12.8, "cog": 70.0, "len": 250, "wid": 44, "draft": 15.0, "offset_km": 35.0, "bearing": 200, "t_offset_h": 18.0},
        {"mmsi": 419000022, "name": "MV OLYMPIC CARRIER", "type": 70, "sog": 13.9, "cog": 248.0, "len": 225, "wid": 32, "draft": 12.0, "offset_km": 31.0, "bearing": 170, "t_offset_h": -10.0},
        {"mmsi": 419000023, "name": "OSV SINDHU KALYAN", "type": 60, "sog": 10.1, "cog": 80.0, "len": 72, "wid": 16, "draft": 5.8, "offset_km": 13.5, "bearing": 80, "t_offset_h": 14.0},
        {"mmsi": 419000024, "name": "FV MAHA SAGAR", "type": 30, "sog": 3.5, "cog": 200.0, "len": 25, "wid": 6, "draft": 2.2, "offset_km": 21.0, "bearing": 290, "t_offset_h": -16.0},
        {"mmsi": 419000025, "name": "MT GLOBAL EMERALD", "type": 80, "sog": 13.4, "cog": 66.0, "len": 270, "wid": 46, "draft": 16.0, "offset_km": 40.0, "bearing": 240, "t_offset_h": -24.0},
        {"mmsi": 419000026, "name": "MV DELTA PIONEER", "type": 70, "sog": 15.5, "cog": 252.0, "len": 200, "wid": 30, "draft": 10.5, "offset_km": 27.0, "bearing": 110, "t_offset_h": 22.0},
        {"mmsi": 419000027, "name": "TB TARANG", "type": 52, "sog": 7.5, "cog": 15.0, "len": 38, "wid": 10, "draft": 4.0, "offset_km": 15.0, "bearing": 60, "t_offset_h": 10.0},
        {"mmsi": 419000028, "name": "FV JAL DHARA", "type": 30, "sog": 4.5, "cog": 330.0, "len": 29, "wid": 7, "draft": 2.5, "offset_km": 26.0, "bearing": 340, "t_offset_h": -6.0},
        {"mmsi": 419000029, "name": "MT SEASPAN FALCON", "type": 80, "sog": 14.1, "cog": 63.0, "len": 260, "wid": 45, "draft": 15.5, "offset_km": 45.0, "bearing": 215, "t_offset_h": -30.0},
        {"mmsi": 419000030, "name": "MV INDUS PRIDE", "type": 70, "sog": 16.2, "cog": 240.0, "len": 280, "wid": 40, "draft": 13.0, "offset_km": 38.0, "bearing": 160, "t_offset_h": 16.0},
        {"mmsi": 419000031, "name": "OSV GANGA SAGAR", "type": 60, "sog": 8.8, "cog": 120.0, "len": 68, "wid": 14, "draft": 5.0, "offset_km": 19.0, "bearing": 270, "t_offset_h": -22.0},
        {"mmsi": 419000032, "name": "FV KONKAN QUEEN", "type": 30, "sog": 3.2, "cog": 45.0, "len": 31, "wid": 8, "draft": 2.9, "offset_km": 33.0, "bearing": 15, "t_offset_h": 4.0},
        {"mmsi": 419000033, "name": "MT VISHVA VIJAY", "type": 80, "sog": 12.0, "cog": 75.0, "len": 240, "wid": 40, "draft": 14.2, "offset_km": 50.0, "bearing": 230, "t_offset_h": -36.0},
        {"mmsi": 419000034, "name": "MV JINDAL MEENAKSHI", "type": 70, "sog": 14.0, "cog": 260.0, "len": 190, "wid": 28, "draft": 10.0, "offset_km": 42.0, "bearing": 100, "t_offset_h": 28.0},
        {"mmsi": 419000035, "name": "TB GARUDA", "type": 52, "sog": 6.8, "cog": 170.0, "len": 40, "wid": 11, "draft": 4.2, "offset_km": 29.0, "bearing": 310, "t_offset_h": -12.0},
        {"mmsi": 419000036, "name": "FV RATNAGIRI PEARL", "type": 30, "sog": 4.1, "cog": 215.0, "len": 26, "wid": 6.5, "draft": 2.4, "offset_km": 36.0, "bearing": 195, "t_offset_h": -18.0},
        {"mmsi": 419000037, "name": "MV PACIFIC VOYAGER", "type": 70, "sog": 17.0, "cog": 60.0, "len": 300, "wid": 42, "draft": 13.8, "offset_km": 48.0, "bearing": 250, "t_offset_h": 20.0}
    ]

    for vessel in bg_fleet:
        v_center_time = t_origin + timedelta(hours=vessel["t_offset_h"])
        v_start_time = v_center_time - timedelta(hours=4)
        v_end_time = v_center_time + timedelta(hours=4)
        
        # Base anchor location
        base_lat, base_lon = move_point(orig_lat, orig_lon, vessel["offset_km"], vessel["bearing"])
        speed_kmh = vessel["sog"] * 1.852
        cog = vessel["cog"]
        
        curr_t = v_start_time
        while curr_t <= v_end_time:
            dt_h = (curr_t - v_center_time).total_seconds() / 3600.0
            dist_km = dt_h * speed_kmh
            p_lat, p_lon = move_point(base_lat, base_lon, abs(dist_km), cog if dist_km >= 0 else (cog + 180.0) % 360.0)
            
            records.append({
                "MMSI": vessel["mmsi"],
                "BaseDateTime": format_iso_time(curr_t),
                "LAT": round(p_lat, 5),
                "LON": round(p_lon, 5),
                "SOG": vessel["sog"],
                "COG": cog,
                "Heading": cog,
                "VesselName": vessel["name"],
                "IMO": f"IMO9{vessel['mmsi'] % 1000000:06d}",
                "CallSign": f"CALL{vessel['mmsi'] % 10000}",
                "VesselType": vessel["type"],
                "Status": 0 if vessel["type"] != 30 else 7,
                "Length": vessel["len"],
                "Width": vessel["wid"],
                "Draft": vessel["draft"],
                "Cargo": vessel["type"]
            })
            curr_t += timedelta(minutes=10)

    return records


def generate_scenario_chennai() -> List[Dict[str, Any]]:
    """Synthetic AIS scenario for scene_chennai_002 (Bay of Bengal)."""
    records = []
    t_sat = parse_iso_time("2026-09-01T18:30:00Z")
    t_origin = t_sat - timedelta(hours=16)
    orig_lat, orig_lon = 12.999, 80.195
    
    # Culprit Tanker
    culprit_start = t_origin - timedelta(hours=5)
    culprit_end = t_origin + timedelta(hours=5)
    curr_t = culprit_start
    while curr_t <= culprit_end:
        dt_h = (curr_t - t_origin).total_seconds() / 3600.0
        # Transit northward along Coromandel coast
        dist_km = dt_h * (13.0 * 1.852)
        p_lat, p_lon = move_point(orig_lat, orig_lon, abs(dist_km), 15.0 if dist_km >= 0 else 195.0)
        
        # 45 min blackout
        if not (-0.35 < dt_h < 0.40):
            records.append({
                "MMSI": 419000101,
                "BaseDateTime": format_iso_time(curr_t),
                "LAT": round(p_lat, 5),
                "LON": round(p_lon, 5),
                "SOG": 13.0,
                "COG": 15.0,
                "Heading": 15.0,
                "VesselName": "MT COROMANDEL GLORY",
                "IMO": "IMO9412345",
                "CallSign": "VUCG",
                "VesselType": 80,
                "Status": 0,
                "Length": 260.0,
                "Width": 44.0,
                "Draft": 15.5,
                "Cargo": 80
            })
        curr_t += timedelta(minutes=5)
        
    # Background traffic
    for i in range(15):
        mmsi = 419000110 + i
        b_lat, b_lon = move_point(orig_lat, orig_lon, 12.0 + i * 2.5, (i * 24.0) % 360.0)
        curr_t = t_origin - timedelta(hours=3)
        while curr_t <= t_origin + timedelta(hours=3):
            records.append({
                "MMSI": mmsi,
                "BaseDateTime": format_iso_time(curr_t),
                "LAT": round(b_lat + 0.01 * math.sin(i), 5),
                "LON": round(b_lon + 0.01 * math.cos(i), 5),
                "SOG": 11.5,
                "COG": 20.0,
                "Heading": 20.0,
                "VesselName": f"MV CHENNAI STAR {i+1}",
                "IMO": f"IMO948{i:04d}",
                "CallSign": f"CALL{mmsi%1000}",
                "VesselType": 70 if i % 2 == 0 else 30,
                "Status": 0,
                "Length": 180.0,
                "Width": 28.0,
                "Draft": 9.0,
                "Cargo": 70
            })
            curr_t += timedelta(minutes=10)
            
    return records


def generate_scenario_gom() -> List[Dict[str, Any]]:
    """Synthetic AIS scenario for scene_gom_003 (Gulf of Mexico)."""
    records = []
    t_sat = parse_iso_time("2026-08-30T06:00:00Z")
    t_origin = t_sat - timedelta(hours=20)
    orig_lat, orig_lon = 28.780, -88.446
    
    # Culprit Tanker
    curr_t = t_origin - timedelta(hours=5)
    while curr_t <= t_origin + timedelta(hours=5):
        dt_h = (curr_t - t_origin).total_seconds() / 3600.0
        dist_km = dt_h * (12.8 * 1.852)
        p_lat, p_lon = move_point(orig_lat, orig_lon, abs(dist_km), 120.0 if dist_km >= 0 else 300.0)
        if not (-0.4 < dt_h < 0.45):
            records.append({
                "MMSI": 367000001,
                "BaseDateTime": format_iso_time(curr_t),
                "LAT": round(p_lat, 5),
                "LON": round(p_lon, 5),
                "SOG": 12.8,
                "COG": 120.0,
                "Heading": 120.0,
                "VesselName": "MT GULF DISCOVERY",
                "IMO": "IMO9388112",
                "CallSign": "WDC1",
                "VesselType": 80,
                "Status": 0,
                "Length": 280.0,
                "Width": 50.0,
                "Draft": 16.8,
                "Cargo": 80
            })
        curr_t += timedelta(minutes=5)
        
    # Background OSV and Gulf traffic
    for i in range(15):
        mmsi = 367000010 + i
        b_lat, b_lon = move_point(orig_lat, orig_lon, 10.0 + i * 2.0, (i * 24.0) % 360.0)
        curr_t = t_origin - timedelta(hours=3)
        while curr_t <= t_origin + timedelta(hours=3):
            records.append({
                "MMSI": mmsi,
                "BaseDateTime": format_iso_time(curr_t),
                "LAT": round(b_lat, 5),
                "LON": round(b_lon, 5),
                "SOG": 10.0,
                "COG": 115.0,
                "Heading": 115.0,
                "VesselName": f"OSV MISSISSIPPI PRIDE {i+1}",
                "IMO": f"IMO920{i:04d}",
                "CallSign": f"WDC{i+10}",
                "VesselType": 60,
                "Status": 0,
                "Length": 85.0,
                "Width": 18.0,
                "Draft": 6.0,
                "Cargo": 60
            })
            curr_t += timedelta(minutes=10)
            
    return records


def build_database(
    db_path: Path = DEFAULT_DB_PATH,
    csv_path: Optional[Path] = DEFAULT_CSV_PATH
) -> int:
    """Generate all scenarios and commit into SQLite database and CSV."""
    logger.info(f"Initializing SQLite database at {db_path}...")
    conn = init_sqlite_db(db_path)
    cursor = conn.cursor()

    all_records = []
    
    logger.info("Generating synthetic AIS for scene_mumbai_001 (Arabian Sea)...")
    rec_mumbai = generate_scenario_mumbai()
    all_records.extend(rec_mumbai)
    logger.info(f"  Generated {len(rec_mumbai)} records for scene_mumbai_001")

    logger.info("Generating synthetic AIS for scene_chennai_002 (Bay of Bengal)...")
    rec_chennai = generate_scenario_chennai()
    all_records.extend(rec_chennai)
    logger.info(f"  Generated {len(rec_chennai)} records for scene_chennai_002")

    logger.info("Generating synthetic AIS for scene_gom_003 (Gulf of Mexico)...")
    rec_gom = generate_scenario_gom()
    all_records.extend(rec_gom)
    logger.info(f"  Generated {len(rec_gom)} records for scene_gom_003")

    # Compute timestamp_epoch and sort deterministically by timestamp_epoch, MMSI
    for r in all_records:
        r["timestamp_epoch"] = parse_iso_time(r["BaseDateTime"]).timestamp()
    all_records.sort(key=lambda r: (r["timestamp_epoch"], r["MMSI"]))

    logger.info(f"Inserting {len(all_records)} total AIS records into SQLite database...")
    insert_sql = """
    INSERT INTO ais_records (
        MMSI, BaseDateTime, timestamp_epoch, LAT, LON, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo
    ) VALUES (
        :MMSI, :BaseDateTime, :timestamp_epoch, :LAT, :LON, :SOG, :COG, :Heading, :VesselName, :IMO, :CallSign, :VesselType, :Status, :Length, :Width, :Draft, :Cargo
    );
    """
    cursor.executemany(insert_sql, all_records)
    conn.commit()

    # Integrity verification
    cursor.execute("SELECT COUNT(*) FROM ais_records;")
    total_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT MMSI) FROM ais_records;")
    distinct_vessels = cursor.fetchone()[0]

    # Verify planted culprit
    cursor.execute("SELECT COUNT(*) FROM ais_records WHERE MMSI = 419000001;")
    culprit_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(BaseDateTime), MAX(BaseDateTime) FROM ais_records WHERE MMSI = 419000001;")
    culprit_tmin, culprit_tmax = cursor.fetchone()

    conn.close()

    logger.info(f"SQLite database populated successfully:")
    logger.info(f"  Total records:    {total_count}")
    logger.info(f"  Distinct vessels: {distinct_vessels}")
    logger.info(f"  MT OCEAN VOYAGER: {culprit_count} pings ({culprit_tmin} to {culprit_tmax})")

    # Export optional CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "MMSI", "BaseDateTime", "timestamp_epoch", "LAT", "LON", "SOG", "COG", "Heading",
                "VesselName", "IMO", "CallSign", "VesselType", "Status", "Length", "Width", "Draft", "Cargo"
            ])
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(f"Exported AIS CSV to {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

    return total_count


def main():
    parser = argparse.ArgumentParser(
        description="TideTrace Synthetic AIS Builder (MarineCadastre 16-field SQLite store)"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV export path (default: {DEFAULT_CSV_PATH})"
    )

    args = parser.parse_args()

    print("\n========================================================")
    print(" TideTrace Synthetic MarineCadastre AIS Builder")
    print("========================================================")

    count = build_database(db_path=args.db_path, csv_path=args.csv_path)

    print("\nDatabase generation complete:")
    print(f"  Database File: {args.db_path} ({args.db_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Total Pings:   {count}")
    print("  Planted Culprit: MT OCEAN VOYAGER (MMSI 419000001, Crude Oil Tanker, 52m blackout)")
    print("  Distractor 1:    FV SEA HARVEST (MMSI 419000002, Fishing, d=0.8km, continuous)")
    print("  Distractor 2:    MV PACIFIC TRADER (MMSI 419000003, Cargo, d=8.5km, continuous)")
    print("--------------------------------------------------------\n")


if __name__ == "__main__":
    main()
