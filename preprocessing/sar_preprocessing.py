"""
sar_preprocessing.py - Sentinel-1 (SAR) Preprocessing Pipeline.

This module implements modular, reusable preprocessing utilities for Sentinel-1
SAR polarization bands (VV and VH). Steps include:
- Loading raw Float32 backscatter intensity (in Decibels - dB) from GeoTIFF.
- Outlier clipping using configurable thresholds.
- Dynamic resizing (defaulting to preserving original resolution).
- Three normalization modes: None, Min-Max (to [0, 1]), and Z-Score standardization.
- Conversions to PyTorch tensors of shape (2, H, W).

Includes clean exception handling, typing, logging, and an independent runnable demo.
"""

import os
import json
import functools
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import PIL.Image as Image

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import (
    IMAGE_SIZE,
    PRESERVE_RESOLUTION,
    NORMALIZATION_MODE,
    S1_STATS_PATH,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    VV_CLIP_RANGE,
    VH_CLIP_RANGE,
    VV_ZSCORE_STATS,
    VH_ZSCORE_STATS,
    setup_logging,
)

logger = setup_logging("SAR_Preprocessing")

# Check if rasterio is available for optimal GeoTIFF reading
HAS_RASTERIO = False
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    logger.warning("rasterio is not installed. Falling back to PIL for image reading.")


@functools.lru_cache(maxsize=8)
def _load_s1_stats_cached(path_str: str, mtime: float) -> Dict[str, Tuple[float, float]]:
    """
    Internal cached helper to parse and validate s1_stats.json.
    """
    file_path = Path(path_str)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse Sentinel-1 statistics file at '{file_path}': {e}")

    extracted_stats: Dict[str, Tuple[float, float]] = {}
    for band in ("VV", "VH"):
        if band not in stats or not isinstance(stats[band], dict):
            raise ValueError(f"Sentinel-1 statistics file '{file_path}' is missing dictionary for '{band}' band.")
        b_dict = stats[band]
        if "mean" not in b_dict or "std" not in b_dict:
            raise ValueError(f"Sentinel-1 statistics file '{file_path}' band '{band}' is missing 'mean' or 'std'.")
        try:
            mean = float(b_dict["mean"])
            std = float(b_dict["std"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Non-numeric statistics for '{band}' in '{file_path}': {e}")

        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"Non-finite statistics for '{band}' in '{file_path}': mean={mean}, std={std}")
        if std <= 0.0:
            raise ValueError(f"Invalid standard deviation for '{band}' in '{file_path}': {std} (must be > 0).")

        extracted_stats[band] = (mean, std)

    return extracted_stats


def load_s1_stats(stats_path: Optional[Union[str, Path]] = None) -> Dict[str, Tuple[float, float]]:
    """
    Dynamically loads and validates Sentinel-1 normalization statistics from JSON.

    Args:
        stats_path: Path to the s1_stats.json file. If None, defaults to S1_STATS_PATH.

    Returns:
        Dict[str, Tuple[float, float]]: Mapping of "VV" and "VH" to (mean, std) tuples.

    Raises:
        FileNotFoundError: If the statistics file does not exist.
        ValueError: If statistics are missing, invalid, non-finite, or std <= 0.
    """
    resolved_path = Path(stats_path) if stats_path is not None else S1_STATS_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Sentinel-1 statistics file not found at: {resolved_path}. "
            "Verify outputs/s1_stats.json exists before running preprocessing."
        )
    mtime = resolved_path.stat().st_mtime
    return _load_s1_stats_cached(str(resolved_path.resolve()), mtime)


def load_sar_band(file_path: Path) -> np.ndarray:
    """
    Loads a single SAR band (GeoTIFF) as a Float32 numpy array.

    Args:
        file_path: Path to the GeoTIFF file.

    Returns:
        np.ndarray: Float32 array containing the backscatter values.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If reading the file fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SAR band file not found: {file_path}")

    try:
        if HAS_RASTERIO:
            with rasterio.open(file_path) as src:
                # Read first channel (Sentinel-1 bands are single band rasters)
                data = src.read(1).astype(np.float32)
                return data
        else:
            with Image.open(file_path) as img:
                data = np.array(img).astype(np.float32)
                return data
    except Exception as e:
        logger.error(f"Error reading SAR band file at {file_path}: {e}")
        raise IOError(f"Could not read SAR band: {file_path}. Error: {e}")


def clip_sar_band(band_data: np.ndarray, clip_range: Tuple[float, float]) -> np.ndarray:
    """
    Clips extreme outliers in backscatter values to a specified dB range.

    Args:
        band_data: Input backscatter array in decibel (dB) scale.
        clip_range: Tuple containing (min_db, max_db) values.

    Returns:
        np.ndarray: Clipped backscatter array.
    """
    min_db, max_db = clip_range
    if min_db >= max_db:
        raise ValueError(f"Min clip boundary ({min_db}) must be less than max boundary ({max_db}).")
    return np.clip(band_data, min_db, max_db)


def normalize_sar_band(
    band_data: np.ndarray,
    clip_range: Tuple[float, float],
    norm_mode: str,
    zscore_stats: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Applies normalization to backscatter data.

    Args:
        band_data: Input (clipped) backscatter array.
        clip_range: Normalization min/max clip bounds (used for Min-Max mode).
        norm_mode: Normalization strategy, one of: "none", "min-max", "z-score".
        zscore_stats: Optional Tuple of (mean, std) used for Z-score normalization.

    Returns:
        np.ndarray: Normalized backscatter array of float32.
    """
    mode = norm_mode.strip().lower()
    if mode == "none":
        return band_data.astype(np.float32)

    elif mode == "min-max":
        min_val, max_val = clip_range
        denom = max_val - min_val
        if abs(denom) < 1e-6:
            return np.zeros_like(band_data, dtype=np.float32)
        return ((band_data - min_val) / denom).astype(np.float32)

    elif mode == "z-score":
        if zscore_stats is None:
            raise ValueError("Z-score normalization requires zscore_stats (mean, std) to be provided.")
        mean, std = zscore_stats
        if not np.isfinite(std) or std <= 1e-6:
            raise ValueError(f"Standard deviation for Z-score normalization must be > 0, got {std}.")
        return ((band_data - mean) / std).astype(np.float32)

    else:
        raise ValueError(
            f"Unsupported normalization mode: '{norm_mode}'. "
            "Supported modes are: 'none', 'min-max', 'z-score'."
        )


def resize_sar_band(band_data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resizes backscatter array to target dimensions using bilinear interpolation.

    Args:
        band_data: Input numpy array of shape (H, W).
        target_size: Target dimensions (height, width).

    Returns:
        np.ndarray: Resized numpy array of shape (target_H, target_W) with dtype float32.
    """
    h, w = band_data.shape
    target_h, target_w = target_size
    if (h, w) == (target_h, target_w):
        return band_data.astype(np.float32)

    # PIL Image from float32 array uses mode 'F'
    img = Image.fromarray(band_data.astype(np.float32), mode="F")
    # PIL resize expects (width, height)
    img_resized = img.resize((target_w, target_h), resample=Image.BILINEAR)
    return np.array(img_resized).astype(np.float32)


def preprocess_sar_patch(
    vv_path: Path,
    vh_path: Path,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    preserve_res: bool = PRESERVE_RESOLUTION,
    norm_mode: str = NORMALIZATION_MODE,
    vv_clip: Tuple[float, float] = VV_CLIP_RANGE,
    vh_clip: Tuple[float, float] = VH_CLIP_RANGE,
    vv_zstats: Optional[Tuple[float, float]] = None,
    vh_zstats: Optional[Tuple[float, float]] = None,
    stats_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Main pipeline entrypoint to load, preprocess, and stack VV and VH bands into a PyTorch tensor.

    Pipeline Order:
        1. Load raw VV and VH as float32 from GeoTIFF.
        2. Clip VV to [-25.0, 0.0] dB and VH to [-25.0, 0.0] dB.
        3. Dynamically load normalization statistics from outputs/s1_stats.json (if z-score).
        4. Normalize each band independently: (clipped - mean) / std.
        5. Resize each normalized band from 120x120 to 224x224 (if preserve_res=False).
        6. Stack channels in exact order: channel 0 = VV, channel 1 = VH.
        7. Return float32 PyTorch tensor of shape (2, 224, 224).

    Args:
        vv_path: Path to the VV polarization band GeoTIFF.
        vh_path: Path to the VH polarization band GeoTIFF.
        target_size: Target (height, width) dimensions if resizing is applied (default (224, 224)).
        preserve_res: If True, bypasses resizing and preserves raw dimensions (default False).
        norm_mode: Normalization strategy ("none", "min-max", "z-score", default "z-score").
        vv_clip: Clipping boundaries for VV band (default (-25.0, 0.0)).
        vh_clip: Clipping boundaries for VH band (default (-25.0, 0.0)).
        vv_zstats: Optional (mean, std) for VV Z-score standardization. If None and norm_mode is "z-score",
                   dynamically loaded from outputs/s1_stats.json.
        vh_zstats: Optional (mean, std) for VH Z-score standardization. If None and norm_mode is "z-score",
                   dynamically loaded from outputs/s1_stats.json.
        stats_path: Optional path to s1_stats.json. If None, defaults to S1_STATS_PATH.

    Returns:
        torch.Tensor: PyTorch FloatTensor of shape (2, 224, 224) containing stacked channels (0: VV, 1: VH).
    """
    # 1. Load raw data as float32
    vv_raw = load_sar_band(vv_path)
    vh_raw = load_sar_band(vh_path)

    # 2. Outlier Clipping to [-25.0, 0.0] dB
    vv_clipped = clip_sar_band(vv_raw, vv_clip)
    vh_clipped = clip_sar_band(vh_raw, vh_clip)

    # 3. Dynamic Stats Resolution (if Z-score normalization requested)
    if norm_mode.strip().lower() == "z-score":
        if vv_zstats is None or vh_zstats is None:
            loaded_stats = load_s1_stats(stats_path)
            if vv_zstats is None:
                vv_zstats = loaded_stats["VV"]
            if vh_zstats is None:
                vh_zstats = loaded_stats["VH"]

    # 4. Normalization: (clipped - mean) / std
    vv_norm = normalize_sar_band(vv_clipped, vv_clip, norm_mode, vv_zstats)
    vh_norm = normalize_sar_band(vh_clipped, vh_clip, norm_mode, vh_zstats)

    # 5. Resizing to 224x224
    if not preserve_res:
        vv_final = resize_sar_band(vv_norm, target_size)
        vh_final = resize_sar_band(vh_norm, target_size)
    else:
        vv_final = vv_norm
        vh_final = vh_norm

    # 6. PyTorch Tensor Conversion
    # Stack channels in exact order: channel 0 = VV, channel 1 = VH
    stacked_np = np.stack([vv_final, vh_final], axis=0).astype(np.float32)
    tensor = torch.from_numpy(stacked_np)

    return tensor


if __name__ == "__main__":
    # Standard self-contained demo block
    import random
    from config.settings import DATA_DIR
    
    logger.info("Running SAR Preprocessing demo...")
    
    # Try to find a random Sentinel-1 patch in the dataset directory
    try:
        # Scan data root directory using os.scandir for instant random directory selection
        subdirs = []
        with os.scandir(DATA_DIR) as it:
            for entry in it:
                if entry.is_dir():
                    subdirs.append(entry.path)
                    
        if not subdirs:
            raise FileNotFoundError("No subdirectories found under DATA_DIR.")
            
        # Select a random group subdirectory
        random_group = random.choice(subdirs)
        
        # Select a random patch from this group
        patch_folders = []
        with os.scandir(random_group) as it:
            for entry in it:
                if entry.is_dir():
                    patch_folders.append(Path(entry.path))
                    
        if not patch_folders:
            raise FileNotFoundError(f"No patch directories found in group: {random_group}")

        sample_patch = random.choice(patch_folders)
        
        # Scan patch directory for bands
        vv_tif = None
        vh_tif = None
        with os.scandir(sample_patch) as it:
            for entry in it:
                if entry.is_file():
                    if entry.name.endswith(VV_BAND_SUFFIX):
                        vv_tif = Path(entry.path)
                    elif entry.name.endswith(VH_BAND_SUFFIX):
                        vh_tif = Path(entry.path)

        if not vv_tif or not vh_tif:
            raise FileNotFoundError(f"Missing VV or VH bands in patch folder: {sample_patch}")

        logger.info(f"Sampled Patch: {sample_patch.name}")
        logger.info(f"VV Path: {vv_tif}")
        logger.info(f"VH Path: {vh_tif}")
        # Run final pipeline with default parameters (dynamic s1_stats.json, 224x224, z-score)
        tensor_final = preprocess_sar_patch(
            vv_path=vv_tif,
            vh_path=vh_tif,
        )
        logger.info("--- FINAL SAR Pipeline Preprocessing Successful ---")
        logger.info(f"Tensor Shape: {tensor_final.shape}")
        logger.info(f"Tensor Dtype: {tensor_final.dtype}")
        logger.info(f"VV Band Stats: Mean={tensor_final[0].mean():.4f}, Std={tensor_final[0].std():.4f}, Min={tensor_final[0].min():.4f}, Max={tensor_final[0].max():.4f}")
        logger.info(f"VH Band Stats: Mean={tensor_final[1].mean():.4f}, Std={tensor_final[1].std():.4f}, Min={tensor_final[1].min():.4f}, Max={tensor_final[1].max():.4f}")

        # Run preprocessing with Min-Max normalization for comparison
        tensor_minmax = preprocess_sar_patch(
            vv_path=vv_tif,
            vh_path=vh_tif,
            norm_mode="min-max",
            preserve_res=True,
        )
        logger.info("--- MIN-MAX Preprocessing (120x120 preserved) ---")
        logger.info(f"Tensor Shape: {tensor_minmax.shape}")
        logger.info(f"VV Band Range: Min={tensor_minmax[0].min():.4f}, Max={tensor_minmax[0].max():.4f}")
        logger.info(f"VH Band Range: Min={tensor_minmax[1].min():.4f}, Max={tensor_minmax[1].max():.4f}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
