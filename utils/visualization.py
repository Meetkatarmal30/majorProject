"""
visualization.py - Sentinel-1 (SAR) Visualization Module.

This module provides reusable plotting utilities for Sentinel-1 imagery:
1. Raw VV vs. VH polarization band visualization.
2. Raw vs. Preprocessed comparison (illustrates clipping and normalization).
3. Histogram plotting for VV and VH bands (visualizes backscatter distributions).

Plots are automatically saved to the outputs/ directory.
Includes type hints, docstrings, clean error handling, and a runnable self-demo.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import (
    DATA_DIR,
    BASE_DIR,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    VV_CLIP_RANGE,
    VH_CLIP_RANGE,
    setup_logging,
)
from preprocessing.sar_preprocessing import load_sar_band, preprocess_sar_patch

logger = setup_logging("Visualization")


def plot_raw_vv_vh(
    vv_path: Path, vh_path: Path, patch_name: str, save_path: Optional[Path] = None
) -> None:
    """
    Plots raw VV and VH bands side-by-side.
    Applies standard percentile clipping (1% and 99%) locally to improve visualization contrast.

    Args:
        vv_path: Path to the VV GeoTIFF file.
        vh_path: Path to the VH GeoTIFF file.
        patch_name: Name of the image patch.
        save_path: Optional path to save the generated plot.
    """
    try:
        vv_arr = load_sar_band(vv_path)
        vh_arr = load_sar_band(vh_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Helper to clip percentiles locally for visualization contrast
        def clip_for_viz(arr):
            p1, p99 = np.percentile(arr, [1, 99])
            return np.clip(arr, p1, p99)

        # Plot VV
        im0 = axes[0].imshow(clip_for_viz(vv_arr), cmap="gray")
        axes[0].set_title(f"VV Channel (dB)\nRange: [{vv_arr.min():.1f}, {vv_arr.max():.1f}]", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.05, label="Intensity (dB)")

        # Plot VH
        im1 = axes[1].imshow(clip_for_viz(vh_arr), cmap="gray")
        axes[1].set_title(f"VH Channel (dB)\nRange: [{vh_arr.min():.1f}, {vh_arr.max():.1f}]", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.05, label="Intensity (dB)")

        plt.suptitle(f"Sentinel-1 SAR Raw Channels: {patch_name}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved raw channels plot to: {save_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot raw channels for {patch_name}: {e}")
        raise


def plot_raw_vs_preprocessed(
    vv_path: Path,
    vh_path: Path,
    preprocessed_tensor: torch.Tensor,
    patch_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plots raw backscatter bands (clipped for viz contrast) alongside preprocessed/normalized channels.
    This shows exactly how clipping, normalization, and reshaping affect the data.

    Args:
        vv_path: Path to the raw VV band.
        vh_path: Path to the raw VH band.
        preprocessed_tensor: Preprocessed tensor output of shape (2, H, W).
        patch_name: Name of the image patch.
        save_path: Optional path to save the generated plot.
    """
    try:
        # Load raw data
        vv_raw = load_sar_band(vv_path)
        vh_raw = load_sar_band(vh_path)

        # Convert torch tensor back to numpy for plotting
        if isinstance(preprocessed_tensor, torch.Tensor):
            prep_np = preprocessed_tensor.detach().cpu().numpy()
        else:
            prep_np = preprocessed_tensor

        vv_prep = prep_np[0]
        vh_prep = prep_np[1]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Helper to clip percentiles locally for visualization contrast
        def clip_for_viz(arr):
            p1, p99 = np.percentile(arr, [1, 99])
            return np.clip(arr, p1, p99)

        # Row 1: VV comparison
        im00 = axes[0, 0].imshow(clip_for_viz(vv_raw), cmap="gray")
        axes[0, 0].set_title("1. Raw VV (Percentile Clipped)", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")
        fig.colorbar(im00, ax=axes[0, 0], orientation="vertical", label="dB")

        im01 = axes[0, 1].imshow(vv_prep, cmap="viridis")
        axes[0, 1].set_title(f"2. Preprocessed VV\nRange: [{vv_prep.min():.2f}, {vv_prep.max():.2f}]", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")
        fig.colorbar(im01, ax=axes[0, 1], orientation="vertical", label="Normalized Value")

        # Row 2: VH comparison
        im10 = axes[1, 0].imshow(clip_for_viz(vh_raw), cmap="gray")
        axes[1, 0].set_title("3. Raw VH (Percentile Clipped)", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")
        fig.colorbar(im10, ax=axes[1, 0], orientation="vertical", label="dB")

        im11 = axes[1, 1].imshow(vh_prep, cmap="viridis")
        axes[1, 1].set_title(f"4. Preprocessed VH\nRange: [{vh_prep.min():.2f}, {vh_prep.max():.2f}]", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")
        fig.colorbar(im11, ax=axes[1, 1], orientation="vertical", label="Normalized Value")

        plt.suptitle(f"Sentinel-1 SAR Preprocessing Pipeline Comparison: {patch_name}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved preprocessing comparison plot to: {save_path}")

        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot comparison for {patch_name}: {e}")
        raise


def plot_histograms(stats: Dict[str, Any], save_dir: Path) -> None:
    """
    Plots the backscatter intensity distribution histograms for VV and VH channels.

    Args:
        stats: Dictionary containing histogram bins and counts.
        save_dir: Folder to save the output PNG files.
    """
    try:
        from utils.stats_generator import save_histograms
        save_histograms(stats, save_dir)
    except Exception as e:
        logger.error(f"Failed to plot histograms: {e}")
        raise


if __name__ == "__main__":
    # Settings and path resolution
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running visualization module demo...")
    try:
        # 1. Locate a random Sentinel-1 patch in the dataset directory
        subdirs = []
        with os.scandir(DATA_DIR) as it:
            for entry in it:
                if entry.is_dir():
                    subdirs.append(entry.path)
                    
        if not subdirs:
            raise FileNotFoundError("No subdirectories found under DATA_DIR.")
            
        random_group = random.choice(subdirs)
        
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

        logger.info(f"Sampled patch: {sample_patch.name}")

        # 2. Run raw VV vs. VH channels plotting
        raw_plot_path = outputs_dir / "sample_raw_channels.png"
        plot_raw_vv_vh(vv_tif, vh_tif, sample_patch.name, save_plot_path := raw_plot_path)

        # 3. Generate preprocessed tensor and run raw vs. preprocessed comparison plotting
        # Let's run with Z-Score normalization mode (using the newly computed constants)
        preprocessed_tensor = preprocess_sar_patch(
            vv_path=vv_tif,
            vh_path=vh_tif,
            preserve_res=True,
            norm_mode="z-score"
        )
        
        comparison_plot_path = outputs_dir / "sample_pipeline_comparison.png"
        plot_raw_vs_preprocessed(
            vv_path=vv_tif,
            vh_path=vh_tif,
            preprocessed_tensor=preprocessed_tensor,
            patch_name=sample_patch.name,
            save_path=comparison_plot_path
        )

        print("\n========================================")
        print("Visualization Demo Successful!")
        print("========================================")
        print(f"Generated raw channels plot: outputs/sample_raw_channels.png")
        print(f"Generated comparison plot: outputs/sample_pipeline_comparison.png")
        print("========================================\n")

    except Exception as e:
        logger.exception(f"Visualization demo failed: {e}")
