"""
stats_generator.py - Sentinel-1 (SAR) Global Dataset Statistics Generator.

This module processes the BigEarthNet-S1 dataset to compute true global statistics:
- Number of patches/images.
- Image dimensions and channel information.
- Global minimum, maximum, mean, and standard deviation for VV and VH bands.
- Running histograms of pixel values for both polarizations.

The statistics are calculated over a random sample of 20,000 patches.
Results are saved to JSON/CSV, and histograms are plotted as PNG files.
Upon successful completion, config/settings.py is updated automatically.

Includes clean type hints, logging, multiprocessing support, and a runnable main demo.
"""

import os
import re
import json
import logging
import csv
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import DATA_DIR, BASE_DIR, VV_BAND_SUFFIX, VH_BAND_SUFFIX, setup_logging
from preprocessing.sar_preprocessing import load_sar_band

logger = setup_logging("StatsGenerator")


def process_patch_stats(patch_path: Path) -> Optional[Dict[str, Any]]:
    """
    Worker function to load a single patch's VV and VH bands and compute local stats.
    Designed to run in parallel processes.

    Args:
        patch_path: Path to the patch folder.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing local stats for VV and VH, or None.
    """
    try:
        # Resolve band paths
        vv_path = None
        vh_path = None
        with os.scandir(patch_path) as it:
            for entry in it:
                if entry.is_file():
                    if entry.name.endswith(VV_BAND_SUFFIX):
                        vv_path = Path(entry.path)
                    elif entry.name.endswith(VH_BAND_SUFFIX):
                        vh_path = Path(entry.path)

        if not vv_path or not vh_path:
            return None

        # Load bands
        vv_arr = load_sar_band(vv_path)
        vh_arr = load_sar_band(vh_path)

        # Compute stats for VV
        vv_sum = float(np.sum(vv_arr))
        vv_sq_sum = float(np.sum(vv_arr ** 2))
        vv_min = float(np.min(vv_arr))
        vv_max = float(np.max(vv_arr))

        # Compute stats for VH
        vh_sum = float(np.sum(vh_arr))
        vh_sq_sum = float(np.sum(vh_arr ** 2))
        vh_min = float(np.min(vh_arr))
        vh_max = float(np.max(vh_arr))

        # Compute histogram locally to avoid massive IPC transfer overhead
        hist_bins = np.linspace(-50.0, 20.0, 101)
        vv_counts, _ = np.histogram(vv_arr, bins=hist_bins)
        vh_counts, _ = np.histogram(vh_arr, bins=hist_bins)

        return {
            "num_pixels": vv_arr.size,
            "height": vv_arr.shape[0],
            "width": vv_arr.shape[1],
            "vv": {"sum": vv_sum, "sq_sum": vv_sq_sum, "min": vv_min, "max": vv_max, "hist": vv_counts.tolist()},
            "vh": {"sum": vh_sum, "sq_sum": vh_sq_sum, "min": vh_min, "max": vh_max, "hist": vh_counts.tolist()},
        }
    except Exception:
        # Return None to indicate corrupted/missing file
        return None


def accumulate_statistics(
    dataset_path: Path, sample_size: int = 20000, num_workers: int = 4
) -> Dict[str, Any]:
    """
    Iterates over a random sample of the dataset to calculate global mean, std, min, max, and histograms.

    Args:
        dataset_path: Root folder of the dataset.
        sample_size: Number of patches to process.
        num_workers: Number of parallel worker processes.

    Returns:
        Dict[str, Any]: Accumulated global statistics.
    """
    logger.info(f"Scanning for patch directories in: {dataset_path}")
    patch_paths = []
    with os.scandir(dataset_path) as it:
        for entry in it:
            if entry.is_dir():
                with os.scandir(entry.path) as it2:
                    for entry2 in it2:
                        if entry2.is_dir():
                            patch_paths.append(Path(entry2.path))

    total_patches_found = len(patch_paths)
    logger.info(f"Found {total_patches_found} patch directories.")

    # Randomly sample patches
    actual_sample_size = min(sample_size, total_patches_found)
    logger.info(f"Randomly sampling {actual_sample_size} patches for statistics...")
    sampled_paths = random.sample(patch_paths, actual_sample_size)

    # Accumulators
    total_pixels = 0
    skipped_count = 0
    vv_global_sum = 0.0
    vv_global_sq_sum = 0.0
    vv_global_min = float("inf")
    vv_global_max = float("-inf")

    vh_global_sum = 0.0
    vh_global_sq_sum = 0.0
    vh_global_min = float("inf")
    vh_global_max = float("-inf")

    img_height = 0
    img_width = 0
    valid_patches = 0

    # Histogram binning parameters
    hist_bins = np.linspace(-50.0, 20.0, 101)
    vv_hist_counts = np.zeros(100, dtype=np.int64)
    vh_hist_counts = np.zeros(100, dtype=np.int64)

    # Process patches in parallel
    logger.info(f"Accumulating stats using {num_workers} worker processes...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_patch_stats, p): p for p in sampled_paths}
        
        for future in tqdm(as_completed(futures), total=len(sampled_paths), desc="Processing patches"):
            res = future.result()
            if res is None:
                skipped_count += 1
                continue

            valid_patches += 1
            total_pixels += res["num_pixels"]
            img_height = res["height"]
            img_width = res["width"]

            # Accumulate VV
            vv_global_sum += res["vv"]["sum"]
            vv_global_sq_sum += res["vv"]["sq_sum"]
            if res["vv"]["min"] < vv_global_min:
                vv_global_min = res["vv"]["min"]
            if res["vv"]["max"] > vv_global_max:
                vv_global_max = res["vv"]["max"]

            # Accumulate VH
            vh_global_sum += res["vh"]["sum"]
            vh_global_sq_sum += res["vh"]["sq_sum"]
            if res["vh"]["min"] < vh_global_min:
                vh_global_min = res["vh"]["min"]
            if res["vh"]["max"] > vh_global_max:
                vh_global_max = res["vh"]["max"]

            # Aggregate histograms from worker processes
            vv_hist_counts += np.array(res["vv"]["hist"])
            vh_hist_counts += np.array(res["vh"]["hist"])

    if valid_patches == 0:
        raise ValueError("No valid patches were processed successfully.")

    # Calculate global mean and standard deviation
    vv_mean = vv_global_sum / total_pixels
    vh_mean = vh_global_sum / total_pixels

    vv_var = (vv_global_sq_sum / total_pixels) - (vv_mean ** 2)
    vh_var = (vh_global_sq_sum / total_pixels) - (vh_mean ** 2)

    vv_std = np.sqrt(max(0.0, vv_var))
    vh_std = np.sqrt(max(0.0, vh_var))

    results = {
        "total_patches_scanned": total_patches_found,
        "valid_patches_processed": valid_patches,
        "skipped_corrupted_patches": skipped_count,
        "image_dimensions": f"{img_width}x{img_height}",
        "channels": 2,
        "total_pixels_processed": total_pixels,
        "vv": {
            "min": vv_global_min,
            "max": vv_global_max,
            "mean": vv_mean,
            "std": vv_std,
            "histogram": {
                "bins": hist_bins.tolist(),
                "counts": vv_hist_counts.tolist(),
            }
        },
        "vh": {
            "min": vh_global_min,
            "max": vh_global_max,
            "mean": vh_mean,
            "std": vh_std,
            "histogram": {
                "bins": hist_bins.tolist(),
                "counts": vh_hist_counts.tolist(),
            }
        }
    }

    return results


def save_statistics(stats: Dict[str, Any], json_path: Path, csv_path: Path) -> None:
    """
    Saves the computed statistics to JSON and CSV files.

    Args:
        stats: Accumulated statistics dictionary.
        json_path: Path to write JSON output.
        csv_path: Path to write CSV output.
    """
    # Create output directories if missing
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(stats, jf, indent=4)
    logger.info(f"Statistics written to JSON: {json_path}")

    # Save to CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Metric", "VV_Band", "VH_Band"])
        writer.writerow(["Min", stats["vv"]["min"], stats["vh"]["min"]])
        writer.writerow(["Max", stats["vv"]["max"], stats["vh"]["max"]])
        writer.writerow(["Mean", stats["vv"]["mean"], stats["vh"]["mean"]])
        writer.writerow(["StdDev", stats["vv"]["std"], stats["vh"]["std"]])
        writer.writerow(["Processed Patches", stats["valid_patches_processed"], stats["valid_patches_processed"]])
        writer.writerow(["Skipped Patches", stats["skipped_corrupted_patches"], stats["skipped_corrupted_patches"]])
        writer.writerow(["Image Dimensions", stats["image_dimensions"], stats["image_dimensions"]])
    logger.info(f"Statistics written to CSV: {csv_path}")


def save_histograms(stats: Dict[str, Any], output_dir: Path) -> None:
    """
    Generates and saves pixel backscatter histograms as PNG files.

    Args:
        stats: Accumulated statistics containing histogram bins and counts.
        output_dir: Folder to save the PNG plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot VV
    plt.figure(figsize=(8, 4))
    bins = np.array(stats["vv"]["histogram"]["bins"])[:-1]
    counts = np.array(stats["vv"]["histogram"]["counts"])
    plt.bar(bins, counts, width=(bins[1]-bins[0]), color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.2)
    plt.title("Sentinel-1 VV Polarization Backscatter Histogram (dB)", fontsize=11, fontweight='bold')
    plt.xlabel("Backscatter (dB)", fontsize=10)
    plt.ylabel("Pixel Count", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    vv_path = output_dir / "vv_histogram.png"
    plt.savefig(vv_path, dpi=150)
    plt.close()
    logger.info(f"Saved VV histogram plot to: {vv_path}")

    # Plot VH
    plt.figure(figsize=(8, 4))
    bins = np.array(stats["vh"]["histogram"]["bins"])[:-1]
    counts = np.array(stats["vh"]["histogram"]["counts"])
    plt.bar(bins, counts, width=(bins[1]-bins[0]), color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.2)
    plt.title("Sentinel-1 VH Polarization Backscatter Histogram (dB)", fontsize=11, fontweight='bold')
    plt.xlabel("Backscatter (dB)", fontsize=10)
    plt.ylabel("Pixel Count", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    vh_path = output_dir / "vh_histogram.png"
    plt.savefig(vh_path, dpi=150)
    plt.close()
    logger.info(f"Saved VH histogram plot to: {vh_path}")


def update_settings_config(vv_mean: float, vv_std: float, vh_mean: float, vh_std: float) -> None:
    """
    Rewrites config/settings.py to update Z-score statistics with actual computed values.

    Args:
        vv_mean: Computed VV mean.
        vv_std: Computed VV standard deviation.
        vh_mean: Computed VH mean.
        vh_std: Computed VH standard deviation.
    """
    settings_path = BASE_DIR / "config" / "settings.py"
    if not settings_path.exists():
        logger.warning(f"settings.py not found at {settings_path}. Skip auto config update.")
        return

    with open(settings_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update VV_ZSCORE_STATS
    content = re.sub(
        r"VV_ZSCORE_STATS:\s*Tuple\[float,\s*float\]\s*=\s*\([^)]*\)",
        f"VV_ZSCORE_STATS: Tuple[float, float] = ({vv_mean:.6f}, {vv_std:.6f})",
        content
    )

    # Update VH_ZSCORE_STATS
    content = re.sub(
        r"VH_ZSCORE_STATS:\s*Tuple\[float,\s*float\]\s*=\s*\([^)]*\)",
        f"VH_ZSCORE_STATS: Tuple[float, float] = ({vh_mean:.6f}, {vh_std:.6f})",
        content
    )

    with open(settings_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Successfully updated Z-score normalization constants in config/settings.py.")


if __name__ == "__main__":
    json_out = BASE_DIR / "outputs" / "statistics.json"
    csv_out = BASE_DIR / "outputs" / "statistics.csv"
    plot_dir = BASE_DIR / "outputs"
    
    logger.info("Starting global statistics accumulator demo (20,000 random patches)...")
    try:
        global_stats = accumulate_statistics(DATA_DIR, sample_size=20000, num_workers=4)
        
        # Save outputs
        save_statistics(global_stats, json_out, csv_out)
        save_histograms(global_stats, plot_dir)
        
        # Update config file
        update_settings_config(
            vv_mean=global_stats["vv"]["mean"],
            vv_std=global_stats["vv"]["std"],
            vh_mean=global_stats["vh"]["mean"],
            vh_std=global_stats["vh"]["std"]
        )
        
        # Print precise Verification Output as required
        print("\n========================================")
        print("Dataset Statistics")
        print("========================================")
        print(f"Total Sampled Patches: {global_stats['valid_patches_processed']}")
        print(f"Total Pixels Processed: {global_stats['total_pixels_processed']}")
        print(f"Skipped Files: {global_stats['skipped_corrupted_patches']}")
        print("\nVV")
        print(f"Mean: {global_stats['vv']['mean']:.4f}")
        print(f"Std: {global_stats['vv']['std']:.4f}")
        print(f"Min: {global_stats['vv']['min']:.4f}")
        print(f"Max: {global_stats['vv']['max']:.4f}")
        print("\nVH")
        print(f"Mean: {global_stats['vh']['mean']:.4f}")
        print(f"Std: {global_stats['vh']['std']:.4f}")
        print(f"Min: {global_stats['vh']['min']:.4f}")
        print(f"Max: {global_stats['vh']['max']:.4f}")
        print("========================================\n")
        
    except Exception as e:
        logger.exception(f"Statistics generation failed: {e}")
