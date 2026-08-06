"""
dataset_exploration.py - Dataset Exploration Script for BigEarthNet-S1.

This script programmatically performs the exploration tasks specified in TASK 1:
- Scans the dataset directory.
- Counts the total number of image patches.
- Detects the file types/extensions and band suffix patterns.
- Samples patches to print physical image properties and dimensions.
- Checks VV and VH band presence and reads a subset of pixels to show raw statistics.
- Generates a structured markdown report under outputs/dataset_exploration_report.md.

Designed for team integration: clean logging, type hints, PEP8 compliance, and a runnable demo.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import numpy as np

# We try to import rasterio for reading GeoTIFF properties. If not available, we fall back to PIL
HAS_RASTERIO = False
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    try:
        from PIL import Image
        HAS_RASTERIO = False
    except ImportError:
        pass

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import DATA_DIR, BASE_DIR, VV_BAND_SUFFIX, VH_BAND_SUFFIX, setup_logging

logger = setup_logging("DatasetExploration")


def load_band_array(file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads band image data as a numpy array and returns its metadata.

    Args:
        file_path: Path to the GeoTIFF file.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Data array and metadata dict.
    """
    if HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                meta = {
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                    "dtype": str(src.dtypes[0]),
                    "crs": str(src.crs) if src.crs else "None",
                }
                return data, meta
        except Exception as e:
            logger.error(f"Failed to read image with rasterio: {e}")
            raise e
    else:
        try:
            with Image.open(file_path) as img:
                data = np.array(img)
                meta = {
                    "width": img.width,
                    "height": img.height,
                    "count": 1,
                    "dtype": str(data.dtype),
                    "crs": "None (PIL fallback)",
                }
                return data, meta
        except Exception as e:
            logger.error(f"Failed to read image with PIL: {e}")
            raise e


def explore_dataset(data_path: Path) -> Dict[str, Any]:
    """
    Reads the dataset folder and gathers high-level directory and band statistics.

    Args:
        data_path: Path to the BigEarthNet-S1 dataset root.

    Returns:
        Dict[str, Any]: Gathered statistics.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    # Gather group subdirectories
    group_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    logger.info(f"Discovered {len(group_dirs)} subdirectories under dataset root.")

    # Collect patch folders
    patch_folders: List[Path] = []
    for gd in group_dirs:
        for patch in gd.iterdir():
            if patch.is_dir():
                patch_folders.append(patch)

    total_patches = len(patch_folders)
    logger.info(f"Total patch folders detected: {total_patches}")

    if total_patches == 0:
        raise ValueError(f"No patch folders detected inside subdirectories of: {data_path}")

    # Check file extensions and sample patch files
    file_extensions: Set[str] = set()
    sample_patch = patch_folders[0]
    sample_files = list(sample_patch.iterdir())
    
    for f in sample_files:
        if f.is_file():
            file_extensions.add(f.suffix.lower())

    # Detect bands in the sample patch
    vv_file = next((f for f in sample_files if f.name.endswith(VV_BAND_SUFFIX)), None)
    vh_file = next((f for f in sample_files if f.name.endswith(VH_BAND_SUFFIX)), None)

    # Read image metrics
    vv_data, vv_meta = load_band_array(vv_file) if vv_file else (None, None)
    vh_data, vh_meta = load_band_array(vh_file) if vh_file else (None, None)

    # Sample statistics over a small batch (e.g. 50 random patches)
    num_samples = min(50, total_patches)
    random_samples = random.sample(patch_folders, num_samples)
    
    vv_mins, vv_maxs, vv_means = [], [], []
    vh_mins, vh_maxs, vh_means = [], [], []

    logger.info(f"Calculating pixel statistics over a sample of {num_samples} patches...")
    for patch in random_samples:
        p_vv = next(patch.glob(f"*{VV_BAND_SUFFIX}"), None)
        p_vh = next(patch.glob(f"*{VH_BAND_SUFFIX}"), None)
        
        if p_vv:
            data, _ = load_band_array(p_vv)
            vv_mins.append(data.min())
            vv_maxs.append(data.max())
            vv_means.append(data.mean())
        if p_vh:
            data, _ = load_band_array(p_vh)
            vh_mins.append(data.min())
            vh_maxs.append(data.max())
            vh_means.append(data.mean())

    stats = {
        "dataset_root": str(data_path),
        "total_group_dirs": len(group_dirs),
        "total_patches": total_patches,
        "extensions": list(file_extensions),
        "sample_patch_name": sample_patch.name,
        "sample_vv_file": vv_file.name if vv_file else "Not Found",
        "sample_vh_file": vh_file.name if vh_file else "Not Found",
        "image_dims": f"{vv_meta['width']}x{vv_meta['height']}" if vv_meta else "Unknown",
        "image_dtype": vv_meta["dtype"] if vv_meta else "Unknown",
        "vv_stats": {
            "min_est": float(np.min(vv_mins)) if vv_mins else 0.0,
            "max_est": float(np.max(vv_maxs)) if vv_maxs else 0.0,
            "mean_est": float(np.mean(vv_means)) if vv_means else 0.0,
        },
        "vh_stats": {
            "min_est": float(np.min(vh_mins)) if vh_mins else 0.0,
            "max_est": float(np.max(vh_maxs)) if vh_maxs else 0.0,
            "mean_est": float(np.mean(vh_means)) if vh_means else 0.0,
        }
    }

    return stats


def generate_exploration_report(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Saves the exploration findings as a structured Markdown report.

    Args:
        stats: Dictionary of exploration results.
        output_path: Target path to write the Markdown report.
    """
    os.makedirs(output_path.parent, exist_ok=True)
    
    report_content = f"""# BigEarthNet-S1 Exploration & Statistics Report

This report summarizes the dataset structure and statistics discovered during the initial exploration phase.

## 1. Directory Structure & Patches
- **Dataset Path**: `{stats["dataset_root"]}`
- **Sub-group Directories**: `{stats["total_group_dirs"]}`
- **Total Image Patches Detected**: `{stats["total_patches"]:,}`
- **File Extensions found inside Patch Folders**: `{", ".join(stats["extensions"])}`

## 2. Band Information
- **Sentinel-1 Polarizations**: VV and VH
- **Sample Patch Name**: `{stats["sample_patch_name"]}`
- **VV Suffix Match**: `{stats["sample_vv_file"]}`
- **VH Suffix Match**: `{stats["sample_vh_file"]}`

## 3. Image Properties
- **Dimensions**: `{stats["image_dims"]} (pixels)`
- **Data Type**: `{stats["image_dtype"]}`

## 4. Band Pixel Ranges & Statistics (Sampled over 50 patches)
### VV Polarization Band
- **Estimated Min (dB)**: `{stats["vv_stats"]["min_est"]:.4f}`
- **Estimated Max (dB)**: `{stats["vv_stats"]["max_est"]:.4f}`
- **Estimated Mean (dB)**: `{stats["vv_stats"]["mean_est"]:.4f}`

### VH Polarization Band
- **Estimated Min (dB)**: `{stats["vh_stats"]["min_est"]:.4f}`
- **Estimated Max (dB)**: `{stats["vh_stats"]["max_est"]:.4f}`
- **Estimated Mean (dB)**: `{stats["vh_stats"]["mean_est"]:.4f}`

---
*Note: S1 values are stored in Float32 format representing radar backscatter intensity in Decibel (dB) scale. Scaling constants in `config/settings.py` can be set using these ranges to clip outliers before normalizing.*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Exploration report written successfully to {output_path}")


if __name__ == "__main__":
    output_report = BASE_DIR / "outputs" / "dataset_exploration_report.md"
    logger.info("Starting programmatic dataset exploration...")
    
    try:
        exploration_stats = explore_dataset(DATA_DIR)
        generate_exploration_report(exploration_stats, output_report)
        print("\n=== Dataset Exploration Successful ===")
        print(f"Total Patches Found: {exploration_stats['total_patches']:,}")
        print(f"Image Dimensions: {exploration_stats['image_dims']}")
        print(f"Report saved to: {output_report}")
    except Exception as e:
        logger.exception("Exploration failed due to an error:")
