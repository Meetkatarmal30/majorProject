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
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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
        np.ndarray: Normalized backscatter array.
    """
    mode = norm_mode.strip().lower()
    if mode == "none":
        return band_data

    elif mode == "min-max":
        min_val, max_val = clip_range
        # Avoid division by zero
        denom = max_val - min_val
        if abs(denom) < 1e-6:
            return np.zeros_like(band_data)
        return (band_data - min_val) / denom

    elif mode == "z-score":
        if zscore_stats is None:
            raise ValueError("Z-score normalization requires zscore_stats (mean, std) to be provided.")
        mean, std = zscore_stats
        if abs(std) < 1e-6:
            raise ValueError("Standard deviation for Z-score normalization cannot be zero.")
        return (band_data - mean) / std

    else:
        raise ValueError(
            f"Unsupported normalization mode: '{norm_mode}'. "
            "Supported modes are: 'none', 'min-max', 'z-score'."
        )


def resize_sar_band(band_data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resizes backscatter array to target dimensions using bilinear interpolation.

    Args:
        band_data: Input numpy array.
        target_size: Tuple containing target (width, height).

    Returns:
        np.ndarray: Resized numpy array.
    """
    # If the size is already identical, skip resizing
    h, w = band_data.shape
    if (w, h) == target_size:
        return band_data

    # PIL Image from float32 array uses mode 'F'
    img = Image.fromarray(band_data)
    # Resize expects (width, height)
    img_resized = img.resize(target_size, resample=Image.BILINEAR)
    return np.array(img_resized).astype(np.float32)


def preprocess_sar_patch(
    vv_path: Path,
    vh_path: Path,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    preserve_res: bool = PRESERVE_RESOLUTION,
    norm_mode: str = NORMALIZATION_MODE,
    vv_clip: Tuple[float, float] = VV_CLIP_RANGE,
    vh_clip: Tuple[float, float] = VH_CLIP_RANGE,
    vv_zstats: Tuple[float, float] = VV_ZSCORE_STATS,
    vh_zstats: Tuple[float, float] = VH_ZSCORE_STATS,
) -> torch.Tensor:
    """
    Main pipeline entrypoint to load, preprocess, and stack VV and VH bands into a PyTorch tensor.

    Args:
        vv_path: Path to the VV polarization band GeoTIFF.
        vh_path: Path to the VH polarization band GeoTIFF.
        target_size: Target (width, height) dimensions if resizing is applied.
        preserve_res: If True, bypasses resizing and preserves raw dimensions.
        norm_mode: Normalization strategy ("none", "min-max", "z-score").
        vv_clip: Clipping boundaries for VV band.
        vh_clip: Clipping boundaries for VH band.
        vv_zstats: Tuple of (mean, std) for VV Z-score standardization.
        vh_zstats: Tuple of (mean, std) for VH Z-score standardization.

    Returns:
        torch.Tensor: PyTorch FloatTensor of shape (2, H, W) containing stacked channels.
    """
    # 1. Load raw data
    vv_raw = load_sar_band(vv_path)
    vh_raw = load_sar_band(vh_path)

    # 2. Outlier Clipping
    vv_clipped = clip_sar_band(vv_raw, vv_clip)
    vh_clipped = clip_sar_band(vh_raw, vh_clip)

    # 3. Normalization
    vv_norm = normalize_sar_band(vv_clipped, vv_clip, norm_mode, vv_zstats)
    vh_norm = normalize_sar_band(vh_clipped, vh_clip, norm_mode, vh_zstats)

    # 4. Resizing
    if not preserve_res:
        vv_final = resize_sar_band(vv_norm, target_size)
        vh_final = resize_sar_band(vh_norm, target_size)
    else:
        vv_final = vv_norm
        vh_final = vh_norm

    # 5. PyTorch Tensor Conversion
    # Stack along channel axis (axis=0) to construct shape (2, H, W)
    stacked_np = np.stack([vv_final, vh_final], axis=0)
    tensor = torch.from_numpy(stacked_np).type(torch.FloatTensor)

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

        # Run preprocessing with Min-Max normalization
        tensor_minmax = preprocess_sar_patch(
            vv_path=vv_tif,
            vh_path=vh_tif,
            preserve_res=True,
            norm_mode="min-max"
        )
        logger.info("--- MIN-MAX Preprocessing Successful ---")
        logger.info(f"Tensor Shape: {tensor_minmax.shape}")
        logger.info(f"Tensor Dtype: {tensor_minmax.dtype}")
        logger.info(f"VV Band Range: Min={tensor_minmax[0].min():.4f}, Max={tensor_minmax[0].max():.4f}")
        logger.info(f"VH Band Range: Min={tensor_minmax[1].min():.4f}, Max={tensor_minmax[1].max():.4f}")

        # Run preprocessing with Z-score standardization
        tensor_zscore = preprocess_sar_patch(
            vv_path=vv_tif,
            vh_path=vh_tif,
            preserve_res=True,
            norm_mode="z-score"
        )
        logger.info("--- Z-SCORE Preprocessing Successful ---")
        logger.info(f"Tensor Shape: {tensor_zscore.shape}")
        logger.info(f"VV Band Stats: Mean={tensor_zscore[0].mean():.4f}, Std={tensor_zscore[0].std():.4f}")
        logger.info(f"VH Band Stats: Mean={tensor_zscore[1].mean():.4f}, Std={tensor_zscore[1].std():.4f}")

        # Run preprocessing with resize forced (e.g. 64x64)
        tensor_resized = preprocess_sar_patch(
            vv_path=vv_tif,
            vh_path=vh_tif,
            target_size=(64, 64),
            preserve_res=False,
            norm_mode="min-max"
        )
        logger.info("--- RESIZED (64x64) Preprocessing Successful ---")
        logger.info(f"Tensor Shape: {tensor_resized.shape}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
