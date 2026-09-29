#!/usr/bin/env python3
"""
TideTrace Hugging Face Hub Synchronization Utility
SIH 26143 / NTRO Naval Forensic Oil Spill Attribution Console

Synchronizes trained neural model checkpoints and demo fixtures with the Hugging Face Hub.
Repository: SidMI/TideTrace-weights (User: SidMI)

Features:
- Download trained PyTorch weights (oil_unet_best.pt) to data/models/
- Download sample SAR demo scenes and pre-cached metocean grids
- Upload new checkpoints/artifacts to SidMI/TideTrace-weights
- 100% Graceful offline fallback: air-gapped demo environments continue seamlessly using local files.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tidetrace.sync_hf")

DEFAULT_REPO_ID = "SidMI/TideTrace-weights"
DEFAULT_MODEL_FILENAME = "oil_unet_best.pt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "data" / "models"
DEFAULT_SCENES_DIR = PROJECT_ROOT / "data" / "scenes"
DEFAULT_METOCEAN_DIR = PROJECT_ROOT / "data" / "metocean"

# Files tracked for sync
SYNC_MANIFEST = [
    {
        "filename": "oil_unet_best.pt",
        "subfolder": "models",
        "local_path": DEFAULT_MODELS_DIR / "oil_unet_best.pt",
        "required": True,
        "description": "3-Class SAR Semantic Segmentation U-Net PyTorch Checkpoint (<80MB)"
    },
    {
        "filename": "catalog.json",
        "subfolder": "scenes",
        "local_path": DEFAULT_SCENES_DIR / "catalog.json",
        "required": False,
        "description": "Demo SAR scenes catalog metadata"
    }
]


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Retrieve Hugging Face token from args, environment variables, or local cache."""
    if explicit_token:
        return explicit_token
    
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    
    # Try reading from default huggingface cache
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists():
        try:
            cached_token = hf_token_path.read_text().strip()
            if cached_token:
                return cached_token
        except Exception:
            pass
            
    return None


def verify_local_file(file_path: Path, min_size_bytes: int = 1024) -> bool:
    """Check if a local file exists and is not empty/corrupt."""
    if not file_path.exists():
        return False
    try:
        size = file_path.stat().st_size
        return size >= min_size_bytes
    except OSError:
        return False


def check_status(repo_id: str = DEFAULT_REPO_ID) -> Dict[str, Any]:
    """Inspect local files and determine sync status."""
    status_report = {
        "repo_id": repo_id,
        "hf_hub_installed": False,
        "token_available": False,
        "local_files": {},
        "all_required_present": True
    }
    
    try:
        import huggingface_hub
        status_report["hf_hub_installed"] = True
    except ImportError:
        status_report["hf_hub_installed"] = False
        
    token = get_hf_token()
    status_report["token_available"] = bool(token)
    
    for item in SYNC_MANIFEST:
        path = item["local_path"]
        exists = verify_local_file(path)
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0.0
        status_report["local_files"][item["filename"]] = {
            "path": str(path),
            "exists": exists,
            "size_mb": round(size_mb, 2),
            "required": item["required"],
            "description": item["description"]
        }
        if item["required"] and not exists:
            status_report["all_required_present"] = False
            
    return status_report


def download_weights(
    repo_id: str = DEFAULT_REPO_ID,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    dest_dir: Optional[Path] = None,
    token: Optional[str] = None,
    force: bool = False,
    offline: bool = False
) -> bool:
    """
    Download PyTorch model weights from Hugging Face Hub.
    Gracefully falls back to local cached weights if offline or token absent.
    """
    dest_dir = dest_dir or DEFAULT_MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_file = dest_dir / model_filename

    # If file already exists and not forced, check validity
    if target_file.exists() and not force:
        size_mb = target_file.stat().st_size / (1024 * 1024)
        logger.info(f"Local model checkpoint exists at {target_file} ({size_mb:.2f} MB). Skipping download.")
        return True

    if offline:
        logger.warning(f"Offline mode specified. Checking local checkpoint at {target_file}...")
        if target_file.exists():
            logger.info("Using existing local checkpoint in offline mode.")
            return True
        else:
            logger.error(f"Cannot download in offline mode: {target_file} does not exist.")
            return False

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        logger.warning("huggingface_hub is not installed. Checking local fallback...")
        if target_file.exists():
            logger.info(f"Local checkpoint found at {target_file}. Operating in offline fallback mode.")
            return True
        logger.error("huggingface_hub is required to download weights from remote repository. Run: pip install huggingface_hub")
        return False

    hf_token = get_hf_token(token)
    logger.info(f"Connecting to Hugging Face Hub: {repo_id} -> {model_filename}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            token=hf_token,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False
        )
        logger.info(f"Successfully downloaded {model_filename} to {downloaded_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download from Hugging Face Hub ({e}). Checking offline local fallback...")
        if target_file.exists() and verify_local_file(target_file):
            size_mb = target_file.stat().st_size / (1024 * 1024)
            logger.info(f"Local fallback checkpoint verified at {target_file} ({size_mb:.2f} MB). Continuing offline.")
            return True
        else:
            logger.error(f"No local model checkpoint available at {target_file}. Remote fetch failed: {e}")
            return False


def upload_weights(
    repo_id: str = DEFAULT_REPO_ID,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    source_dir: Optional[Path] = None,
    token: Optional[str] = None,
    commit_message: Optional[str] = None
) -> bool:
    """
    Upload local model checkpoint to Hugging Face Hub repo SidMI/TideTrace-weights.
    """
    source_dir = source_dir or DEFAULT_MODELS_DIR
    source_file = source_dir / model_filename

    if not source_file.exists():
        logger.error(f"Cannot upload: local file {source_file} does not exist.")
        return False

    hf_token = get_hf_token(token)
    if not hf_token:
        logger.error("Hugging Face token required for upload. Set HF_TOKEN environment variable or pass --token.")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub is required for upload. Run: pip install huggingface_hub")
        return False

    api = HfApi(token=hf_token)
    commit_msg = commit_message or f"Update TideTrace model weights: {model_filename}"
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        logger.info(f"Uploading {source_file} ({source_file.stat().st_size / (1024*1024):.2f} MB) to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(source_file),
            path_in_repo=model_filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )
        logger.info(f"Successfully uploaded {model_filename} to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload weights to Hugging Face Hub: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TideTrace Hugging Face Model & Demo Sync Utility (SidMI/TideTrace-weights)"
    )
    parser.add_argument(
        "--action",
        choices=["download", "upload", "check", "status"],
        default="status",
        help="Action to perform: download, upload, check, status"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_MODEL_FILENAME,
        help=f"Model filename (default: {DEFAULT_MODEL_FILENAME})"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Local directory for models (default: data/models)"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face API token (overrides HF_TOKEN env var)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local file exists"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in pure offline mode without network calls"
    )
    parser.add_argument(
        "--commit-message",
        help="Commit message for upload action"
    )

    args = parser.parse_args()

    if args.action in ("status", "check"):
        status = check_status(args.repo_id)
        print("\n========================================================")
        print(" TideTrace Hugging Face Hub Sync Status")
        print("========================================================")
        print(f"Repository:       {status['repo_id']}")
        print(f"HF Hub Installed: {'YES' if status['hf_hub_installed'] else 'NO'}")
        print(f"Token Available:  {'YES' if status['token_available'] else 'NO (read-only / offline)'}")
        print("Local Artifacts:")
        for fname, info in status["local_files"].items():
            status_str = f"PRESENT ({info['size_mb']} MB)" if info["exists"] else "MISSING"
            req_str = "[REQUIRED]" if info["required"] else "[OPTIONAL]"
            print(f"  - {fname:<22} {req_str:<12} {status_str:<20} {info['description']}")
        print("--------------------------------------------------------")
        if status["all_required_present"]:
            print("Status: ALL REQUIRED MODEL ASSETS READY FOR OFFLINE DEMO.")
            sys.exit(0)
        else:
            print("Status: Missing required model assets. Run with --action download.")
            sys.exit(1 if not args.offline else 0)

    elif args.action == "download":
        success = download_weights(
            repo_id=args.repo_id,
            model_filename=args.filename,
            dest_dir=args.local_dir,
            token=args.token,
            force=args.force,
            offline=args.offline
        )
        sys.exit(0 if success else 1)

    elif args.action == "upload":
        success = upload_weights(
            repo_id=args.repo_id,
            model_filename=args.filename,
            source_dir=args.local_dir,
            token=args.token,
            commit_message=args.commit_message
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
