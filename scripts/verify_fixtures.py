#!/usr/bin/env python3
"""
TideTrace Demo Fixtures and Verification Suite
SIH 26143 / NTRO Naval Forensic Oil Spill Attribution Console

Comprehensive test and validation runner for all Worker M5 deliverables:
1. SQLite MarineCadastre AIS Database (tidetrace_ais.db)
2. Pre-cached 4D Metocean Grids (data/metocean/{scene_id}.json)
3. PyTorch Segmentation Model Checkpoint (data/models/oil_unet_best.pt)
4. SAR Demo Scenes and Catalog (data/scenes/*)
5. Hugging Face Hub Sync Script (scripts/sync_hf.py)
6. Kaggle Training Notebook (scripts/train_kaggle.ipynb)
"""

import os
import sys
import json
import sqlite3
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import io

# Ensure UTF-8 output encoding on Windows terminals
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_sqlite_ais():
    print("\n[TEST 1/6] Verifying MarineCadastre SQLite AIS Database...")
    db_path = PROJECT_ROOT / "data" / "ais" / "tidetrace_ais.db"
    assert db_path.exists(), f"Database not found at {db_path}"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 1. Verify schema
    cursor.execute("PRAGMA table_info(ais_records);")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    required_cols = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselName", "IMO", "CallSign", "VesselType", "Status", "Length", "Width", "Draft", "Cargo"]
    for col in required_cols:
        assert col in columns, f"Missing required column {col} in ais_records"
    print(f"  ✓ Schema verification passed ({len(columns)} fields matching MarineCadastre standard)")
    
    # 2. Verify indexes
    cursor.execute("PRAGMA index_list(ais_records);")
    indexes = [row[1] for row in cursor.fetchall()]
    assert any("idx_ais_time_lat_lon" in idx for idx in indexes), "Missing composite index idx_ais_time_lat_lon"
    assert any("idx_ais_mmsi_time" in idx for idx in indexes), "Missing composite index idx_ais_mmsi_time"
    print(f"  ✓ Composite B-Tree indexes verified: {indexes}")
    
    # 3. Total pings & vessels
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT MMSI) FROM ais_records;")
    total_pings, total_vessels = cursor.fetchone()
    assert total_pings >= 1000, f"Expected >= 1000 pings, found {total_pings}"
    assert total_vessels >= 25, f"Expected >= 25 vessels, found {total_vessels}"
    print(f"  ✓ AIS traffic verified: {total_pings} pings across {total_vessels} distinct vessels")
    
    # 4. Verify Planted Culprit: MT OCEAN VOYAGER (MMSI 419000001)
    cursor.execute("SELECT MMSI, VesselName, VesselType, SOG, COG, Length, Draft FROM ais_records WHERE MMSI = 419000001 LIMIT 1;")
    culprit = cursor.fetchone()
    assert culprit is not None, "Planted culprit MT OCEAN VOYAGER not found in database!"
    assert culprit[1] == "MT OCEAN VOYAGER"
    assert culprit[2] == 80, f"Expected VesselType 80 (Tanker), got {culprit[2]}"
    assert abs(culprit[3] - 12.5) < 0.1, f"Expected SOG ~12.5 kn, got {culprit[3]}"
    print(f"  ✓ Culprit metadata verified: {culprit[1]} (MMSI: {culprit[0]}, Type: {culprit[2]}, SOG: {culprit[3]} kn)")
    
    # 5. Verify 52-minute transponder blackout gap
    cursor.execute("""
    SELECT BaseDateTime, LAT, LON, SOG FROM ais_records 
    WHERE MMSI = 419000001 AND BaseDateTime >= '2026-09-01T17:00:00Z' AND BaseDateTime <= '2026-09-01T19:00:00Z'
    ORDER BY BaseDateTime;
    """)
    gap_pings = cursor.fetchall()
    times = [datetime.fromisoformat(p[0].replace("Z", "+00:00")) for p in gap_pings]
    max_gap_minutes = 0.0
    gap_pair = None
    for i in range(len(times) - 1):
        dt_min = (times[i+1] - times[i]).total_seconds() / 60.0
        if dt_min > max_gap_minutes:
            max_gap_minutes = dt_min
            gap_pair = (gap_pings[i][0], gap_pings[i+1][0])
            
    assert abs(max_gap_minutes - 52.0) < 1.0, f"Expected 52m blackout gap, found {max_gap_minutes:.1f}m between {gap_pair}"
    print(f"  ✓ Planted AIS blackout verified: {max_gap_minutes:.1f} min transponder gap between {gap_pair[0]} and {gap_pair[1]}")
    
    # 6. Verify Distractors
    cursor.execute("SELECT MMSI, VesselName, VesselType, SOG FROM ais_records WHERE MMSI = 419000002 LIMIT 1;")
    d1 = cursor.fetchone()
    assert d1 and d1[1] == "FV SEA HARVEST" and d1[2] == 30, "Distractor 1 FV SEA HARVEST missing or incorrect"
    print(f"  ✓ Distractor 1 verified: {d1[1]} (Fishing, Type {d1[2]}, SOG {d1[3]} kn)")
    
    cursor.execute("SELECT MMSI, VesselName, VesselType, SOG FROM ais_records WHERE MMSI = 419000003 LIMIT 1;")
    d2 = cursor.fetchone()
    assert d2 and d2[1] == "MV PACIFIC TRADER" and d2[2] == 70, "Distractor 2 MV PACIFIC TRADER missing or incorrect"
    print(f"  ✓ Distractor 2 verified: {d2[1]} (Cargo, Type {d2[2]}, SOG {d2[3]} kn)")
    
    conn.close()
    return True


def test_metocean_cache():
    print("\n[TEST 2/6] Verifying Pre-cached Metocean Grids...")
    metocean_dir = PROJECT_ROOT / "data" / "metocean"
    expected_scenes = ["scene_mumbai_001", "scene_chennai_002", "scene_gom_003"]
    
    for s_id in expected_scenes:
        cache_file = metocean_dir / f"{s_id}.json"
        assert cache_file.exists(), f"Metocean cache missing for {s_id}"
        
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert data["scene_id"] == s_id
        assert "u_current" in data and "v_current" in data and "u_wind" in data and "v_wind" in data
        assert "timestamps" in data and len(data["timestamps"]) == 121, f"Expected 121 hourly steps, got {len(data['timestamps'])}"
        
        u_curr = np.array(data["u_current"])
        v_curr = np.array(data["v_current"])
        u_wind = np.array(data["u_wind"])
        v_wind = np.array(data["v_wind"])
        
        assert u_curr.shape == (121, 21, 21), f"Unexpected shape {u_curr.shape}"
        assert -3.0 < u_curr.mean() < 3.0, "Current magnitude out of realistic bounds"
        assert -30.0 < u_wind.mean() < 30.0, "Wind magnitude out of realistic bounds"
        
        size_kb = cache_file.stat().st_size / 1024
        print(f"  ✓ {s_id}: verified 4D grid shape {data['shape']} ({size_kb:.1f} KB, {len(data['timestamps'])} hours)")
    return True


def test_pytorch_model():
    print("\n[TEST 3/6] Verifying PyTorch Model Checkpoint...")
    model_path = PROJECT_ROOT / "data" / "models" / "oil_unet_best.pt"
    assert model_path.exists(), f"Model checkpoint missing at {model_path}"
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    assert size_mb < 80.0, f"Model size {size_mb:.2f} MB violates <80MB constraint"
    print(f"  ✓ Checkpoint size verified: {size_mb:.2f} MB (< 80 MB constraint passed)")
    
    # Load checkpoint
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt or "state_dict" in ckpt
    assert ckpt.get("num_classes", 3) == 3
    assert ckpt.get("in_channels", 2) == 2
    
    # Import model from builder
    from app.ml.model import TideTraceUNet
    model = TideTraceUNet.load_checkpoint(str(model_path), device="cpu")
    model.eval()
    
    # Run test inference
    dummy_input = torch.randn(1, 2, 512, 512)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
        
    assert logits.shape == (1, 3, 512, 512), f"Expected shape (1, 3, 512, 512), got {logits.shape}"
    assert probs.shape == (1, 3, 512, 512)
    print(f"  ✓ PyTorch inference verified: input (1, 2, 512, 512) -> output logits (1, 3, 512, 512)")
    print(f"  ✓ Reported Validation IoU: Oil={ckpt.get('val_iou_oil')}, Look-alike={ckpt.get('val_iou_lookalike')}, mIoU={ckpt.get('val_miou')}")
    return True


def test_demo_scenes():
    print("\n[TEST 4/6] Verifying SAR Demo Scenes and Catalog...")
    scenes_dir = PROJECT_ROOT / "data" / "scenes"
    catalog_path = scenes_dir / "catalog.json"
    assert catalog_path.exists(), "Master catalog.json missing"
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    assert len(catalog["scenes"]) >= 4, f"Expected at least 4 demo scenes, got {len(catalog['scenes'])}"
    
    for sc in catalog["scenes"]:
        s_id = sc["id"]
        tif_path = PROJECT_ROOT / sc["path"]
        npy_path = scenes_dir / f"{s_id}.npy"
        json_path = scenes_dir / f"{s_id}.json"
        
        assert tif_path.exists(), f"GeoTIFF missing for {s_id}"
        assert npy_path.exists(), f"NumPy array missing for {s_id}"
        assert json_path.exists(), f"JSON metadata missing for {s_id}"
        
        # Check numpy array shape & values
        arr = np.load(npy_path)
        assert arr.shape == (2, 2048, 2048), f"Expected shape (2, 2048, 2048), got {arr.shape}"
        assert -30.0 <= arr[0].min() and arr[0].max() <= 0.0, f"VV out of bounds for {s_id}: [{arr[0].min()}, {arr[0].max()}]"
        assert -35.0 <= arr[1].min() and arr[1].max() <= -5.0, f"VH out of bounds for {s_id}: [{arr[1].min()}, {arr[1].max()}]"
        
        print(f"  ✓ {s_id:<18}: {sc['title']} (VV: [{arr[0].min():.1f}, {arr[0].max():.1f}] dB)")
        
    return True


def test_sync_hf():
    print("\n[TEST 5/6] Verifying Hugging Face Hub Sync Utility...")
    from scripts.sync_hf import check_status
    status = check_status()
    assert status["all_required_present"] is True, "Sync script reported missing required files"
    print("  ✓ HF sync script verified: status check confirms all required local files present")
    return True


def test_kaggle_notebook():
    print("\n[TEST 6/6] Verifying Kaggle Training Notebook (scripts/train_kaggle.ipynb)...")
    nb_path = PROJECT_ROOT / "scripts" / "train_kaggle.ipynb"
    assert nb_path.exists(), "Notebook scripts/train_kaggle.ipynb missing"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    assert "cells" in nb and len(nb["cells"]) >= 10, f"Notebook has only {len(nb.get('cells', []))} cells"
    
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6, f"Expected >= 6 code cells, found {len(code_cells)}"
    
    # Check syntax of code cells by compiling them
    for idx, c in enumerate(code_cells):
        code_src = "".join(c["source"])
        try:
            compile(code_src, f"cell_{idx}", "exec")
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in notebook code cell {idx}: {e}")
            
    print(f"  ✓ Notebook structure verified: {len(nb['cells'])} cells ({len(code_cells)} executable code cells, all syntax-valid)")
    return True


def main():
    print("=" * 70)
    print(" TideTrace Worker M5 Fixture Verification Suite")
    print("=" * 70)
    
    results = [
        test_sqlite_ais(),
        test_metocean_cache(),
        test_pytorch_model(),
        test_demo_scenes(),
        test_sync_hf(),
        test_kaggle_notebook()
    ]
    
    if all(results):
        print("\n" + "=" * 70)
        print(" ALL WORKER M5 VERIFICATIONS PASSED SUCCESSFULLY (6/6)")
        print(" Ready for 100% offline air-gapped demo execution.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\nVerification failures encountered.")
        sys.exit(1)


if __name__ == "__main__":
    main()
