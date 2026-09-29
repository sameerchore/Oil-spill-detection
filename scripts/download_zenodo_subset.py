"""
TideTrace: Zenodo Sentinel-1 SAR Dataset Downloader (SIH 26143 / NTRO)
Downloads real Sentinel-1 SAR dual-pol (VV/VH dB) GeoTIFFs and binary ground-truth masks
from Zenodo Part I (8346860), Part II (8253899), and Part III (13761290).

Per p2 (1).pdf:
- Part I images are ~40.7 GB. Do not download all of it to the local laptop.
- Minimum real subset: 4 oil chips + 1 look-alike true-negative from Part III.
- Supports optional Zenodo Personal Access Token (--token) to bypass rate limits.
"""

import os
import sys
import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path

ZENODO_RECORDS = {
    "part_1_oil": "8346860",       # Part I: Oil Spills (40.7 GB total dump)
    "part_2_lookalike": "8253899", # Part II: Look-alikes and Clean Sea
    "part_3_test": "13761290",      # Part III: Standardized 150/class Benchmark
}

def download_zenodo_metadata(record_id: str, token: str = None) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "TideTrace-Zenodo-Downloader/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"[ERROR] Zenodo returned HTTP 403 (Cloudflare Bot Protection / Rate Limit).")
            print(f"        To authenticate, provide your Zenodo Personal Access Token via:")
            print(f"        python scripts/download_zenodo_subset.py --token <YOUR_ZENODO_TOKEN>")
        elif e.code == 404:
            print(f"[ERROR] Zenodo record {record_id} not found.")
        else:
            print(f"[ERROR] HTTP Error {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch record {record_id}: {e}")
        return None

def download_file(file_url: str, output_path: Path, token: str = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(file_url)
    req.add_header("User-Agent", "TideTrace-Zenodo-Downloader/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    
    print(f"[INFO] Downloading: {output_path.name}...")
    with urllib.request.urlopen(req, timeout=60) as resp, open(output_path, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                sys.stdout.write(f"\r[INFO] Progress: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                sys.stdout.flush()
    print(f"\n[SUCCESS] Downloaded to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="TideTrace Zenodo Sentinel-1 SAR Dataset Downloader")
    parser.add_argument("--token", type=str, default=os.environ.get("ZENODO_TOKEN", ""), help="Zenodo Personal Access Token")
    parser.add_argument("--part", type=str, default="part_1_oil", choices=["part_1_oil", "part_2_lookalike", "part_3_test", "all"], help="Zenodo record part to download")
    parser.add_argument("--outdir", type=str, default="data/zenodo", help="Target download directory")
    parser.add_argument("--masks_only", action="store_true", default=True, help="Download ground-truth masks only (<10 MB total) to protect disk")
    parser.add_argument("--all_files", action="store_true", help="Download full multi-gigabyte archives (Caution: requires 40+ GB)")
    parser.add_argument("--max_files", type=int, default=5, help="Maximum number of files/chips to download")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts_to_fetch = [args.part] if args.part != "all" else list(ZENODO_RECORDS.keys())

    for part in parts_to_fetch:
        rec_id = ZENODO_RECORDS[part]
        print(f"\n========================================================")
        print(f" Querying Zenodo Record: {part} (DOI: 10.5281/zenodo.{rec_id})")
        print(f"========================================================")
        meta = download_zenodo_metadata(rec_id, token=args.token)
        if not meta:
            continue

        files = meta.get("files", [])
        print(f"[INFO] Found {len(files)} files in record {rec_id}.")
        for f in files:
            fname = f.get("key") or f.get("name")
            fsize_mb = f.get("size", 0) / (1024 * 1024)
            furl = f.get("links", {}).get("self")
            is_mask = "mask" in fname.lower()
            
            if args.masks_only and not args.all_files and not is_mask:
                print(f"[SKIP] {fname} ({fsize_mb:.2f} MB) - Skipped heavy archive (use --all_files to override).")
                continue

            print(f" - {fname} ({fsize_mb:.2f} MB)")
            if furl:
                try:
                    download_file(furl, out_dir / fname, token=args.token)
                except Exception as e:
                    print(f"[WARN] Could not download {fname}: {e}")

if __name__ == "__main__":
    main()
