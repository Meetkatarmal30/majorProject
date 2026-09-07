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

# Import torch first on Windows to avoid OpenMP C-runtime DLL conflicts
try:
    import torch
except ImportError:
    pass

import os
import re
import json
import logging
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import DATA_DIR, BASE_DIR, VV_BAND_SUFFIX, VH_BAND_SUFFIX, setup_logging
from preprocessing.sar_preprocessing import load_sar_band, clip_sar_band

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
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning(f"matplotlib import failed ({e}); skipping histogram generation.")
        return

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


def compute_s1_training_stats(
    val_split_csv: Path = BASE_DIR / "teammate_inputs" / "val_split.csv",
    cache_path: Path = BASE_DIR / ".s1_patch_cache.json",
    output_json_path: Path = BASE_DIR / "outputs" / "s1_stats.json",
    sample_size: int = 21000,
    seed: int = 42,
    clip_range: Tuple[float, float] = (-25.0, 0.0),
    sanity_test_only: bool = False,
    sanity_sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Computes global Sentinel-1 (SAR) statistics strictly over training patches.
    
    Guarantees:
    - 4,500 validation patches from val_split.csv are explicitly excluded (zero data leakage).
    - Exactly sample_size (21,000) patches are deterministically selected using seed 42.
    - Zero file copies, moves, or subset duplicates; reads TIFFs directly in-place.
    - Outlier clipping applied: VV and VH both clipped to [-25.0, 0.0] dB.
    - Sequential, memory-efficient streaming accumulation in float64.
    - No NaNs or Infs allowed.
    - Saves output in exact primary structure required:
        {"VV": {"mean": ..., "std": ...}, "VH": {"mean": ..., "std": ...}}
    
    Args:
        val_split_csv: Path to teammate validation CSV.
        cache_path: Path to .s1_patch_cache.json.
        output_json_path: Destination path for s1_stats.json.
        sample_size: Number of training patches to sample (21,000).
        seed: Random seed for deterministic selection (42).
        clip_range: Outlier clipping boundaries (-25.0, 0.0).
        sanity_test_only: If True, only processes sanity_sample_size patches.
        sanity_sample_size: Number of patches for sanity testing (default 10).
        
    Returns:
        Dict[str, Any]: Computed statistics dictionary.
    """
    logger.info("=" * 60)
    logger.info("Computing Sentinel-1 Training Statistics (Week 3)")
    logger.info("=" * 60)

    # 1. Load validation patch names to strictly isolate them
    if not val_split_csv.exists():
        raise FileNotFoundError(f"Validation split CSV not found: {val_split_csv}")
        
    logger.info(f"Loading validation patches from: {val_split_csv}")
    val_s1_names = set()
    with open(val_split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_s1_names.add(row["s1_name"])

    logger.info(f"Loaded {len(val_s1_names)} validation patch names to exclude.")
    if len(val_s1_names) != 4500:
        logger.warning(f"Expected 4,500 validation patches, found {len(val_s1_names)}.")

    # 2. Load dataset patch cache
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {cache_path}. "
            "Please initialize dataset first to generate cache."
        )

    logger.info(f"Loading patch directory cache from: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    total_cached = len(cache)
    logger.info(f"Total patches indexed in cache: {total_cached}")

    # 3. Form candidate training pool (strictly excluding validation patches)
    candidate_training_patches = [k for k in sorted(cache.keys()) if k not in val_s1_names]
    logger.info(
        f"Candidate training patches available after excluding validation: "
        f"{len(candidate_training_patches)} (Excluded: {total_cached - len(candidate_training_patches)})"
    )

    if len(candidate_training_patches) < sample_size:
        raise ValueError(
            f"Not enough candidate training patches ({len(candidate_training_patches)}) "
            f"to sample {sample_size} patches."
        )

    # 4. Deterministically sample training patches using random.Random(seed)
    rng = random.Random(seed)
    selected_training_patches = rng.sample(candidate_training_patches, sample_size)

    # 5. Validation / Safety assertions
    assert len(selected_training_patches) == sample_size, (
        f"Selected count mismatch: expected {sample_size}, got {len(selected_training_patches)}"
    )
    assert len(set(selected_training_patches)) == sample_size, (
        "Duplicate patches detected in selected training sample."
    )
    overlap = set(selected_training_patches).intersection(val_s1_names)
    assert len(overlap) == 0, (
        f"CRITICAL DATA LEAKAGE: {len(overlap)} validation patches found in training sample!"
    )

    logger.info(f"Selected {sample_size} unique training patches with seed={seed}.")
    logger.info(f"Overlap with validation set: {len(overlap)} (ZERO data leakage confirmed).")

    # Determine processing set
    if sanity_test_only:
        patches_to_process = selected_training_patches[:sanity_sample_size]
        logger.info(f"[SANITY TEST MODE] Processing first {len(patches_to_process)} patches only...")
    else:
        patches_to_process = selected_training_patches
        logger.info(f"Starting sequential accumulation over all {len(patches_to_process)} training patches...")

    # 6. Sequential streaming accumulation in float64
    vv_count = 0
    vv_sum = 0.0
    vv_sq_sum = 0.0
    vv_min = float("inf")
    vv_max = float("-inf")

    vh_count = 0
    vh_sum = 0.0
    vh_sq_sum = 0.0
    vh_min = float("inf")
    vh_max = float("-inf")

    has_nan = False
    has_inf = False
    processed_count = 0

    progress_bar = tqdm(patches_to_process, desc="Computing S1 Stats", unit="patch")

    for s1_name in progress_bar:
        patch_dir = Path(cache[s1_name])
        vv_file = patch_dir / f"{s1_name}{VV_BAND_SUFFIX}"
        vh_file = patch_dir / f"{s1_name}{VH_BAND_SUFFIX}"

        if not vv_file.exists():
            raise FileNotFoundError(f"Missing VV band: {vv_file}")
        if not vh_file.exists():
            raise FileNotFoundError(f"Missing VH band: {vh_file}")

        # Load raw Float32 data
        vv_raw = load_sar_band(vv_file)
        vh_raw = load_sar_band(vh_file)

        # Check for NaNs or Infs in raw data
        if np.isnan(vv_raw).any() or np.isnan(vh_raw).any():
            has_nan = True
            raise ValueError(f"NaN values encountered in patch: {s1_name}")
        if np.isinf(vv_raw).any() or np.isinf(vh_raw).any():
            has_inf = True
            raise ValueError(f"Inf values encountered in patch: {s1_name}")

        # Outlier clipping to [-25.0, 0.0] dB
        vv_clipped = clip_sar_band(vv_raw, clip_range)
        vh_clipped = clip_sar_band(vh_raw, clip_range)

        # Validate clipping boundaries
        if vv_clipped.min() < clip_range[0] or vv_clipped.max() > clip_range[1]:
            raise ValueError(f"VV clipping bounds violated in {s1_name}: min={vv_clipped.min()}, max={vv_clipped.max()}")
        if vh_clipped.min() < clip_range[0] or vh_clipped.max() > clip_range[1]:
            raise ValueError(f"VH clipping bounds violated in {s1_name}: min={vh_clipped.min()}, max={vh_clipped.max()}")

        # Accumulate VV in float64
        vv_count += vv_clipped.size
        vv_sum += float(np.sum(vv_clipped, dtype=np.float64))
        vv_sq_sum += float(np.sum(vv_clipped.astype(np.float64) ** 2))
        curr_vv_min = float(np.min(vv_clipped))
        curr_vv_max = float(np.max(vv_clipped))
        if curr_vv_min < vv_min:
            vv_min = curr_vv_min
        if curr_vv_max > vv_max:
            vv_max = curr_vv_max

        # Accumulate VH in float64
        vh_count += vh_clipped.size
        vh_sum += float(np.sum(vh_clipped, dtype=np.float64))
        vh_sq_sum += float(np.sum(vh_clipped.astype(np.float64) ** 2))
        curr_vh_min = float(np.min(vh_clipped))
        curr_vh_max = float(np.max(vh_clipped))
        if curr_vh_min < vh_min:
            vh_min = curr_vh_min
        if curr_vh_max > vh_max:
            vh_max = curr_vh_max

        processed_count += 1

    # 7. Compute population statistics
    vv_mean = vv_sum / vv_count
    vv_var = (vv_sq_sum / vv_count) - (vv_mean ** 2)
    vv_std = float(np.sqrt(max(0.0, vv_var)))

    vh_mean = vh_sum / vh_count
    vh_var = (vh_sq_sum / vh_count) - (vh_mean ** 2)
    vh_std = float(np.sqrt(max(0.0, vh_var)))

    # 8. Post-computation verification assertions
    assert processed_count == len(patches_to_process), "Processed patch count mismatch"
    assert vv_count > 0 and vh_count > 0, "Pixel count must be positive"
    assert vv_std > 0.0 and vh_std > 0.0, "Standard deviation must be positive and non-zero"
    assert not np.isnan(vv_mean) and not np.isnan(vh_mean), "Mean contains NaN"
    assert not np.isnan(vv_std) and not np.isnan(vh_std), "Std contains NaN"
    assert vv_min >= clip_range[0] and vv_max <= clip_range[1], "VV min/max out of clip bounds"
    assert vh_min >= clip_range[0] and vh_max <= clip_range[1], "VH min/max out of clip bounds"

    stats_result = {
        "VV": {
            "mean": vv_mean,
            "std": vv_std,
            "min": vv_min,
            "max": vv_max,
            "pixel_count": vv_count,
        },
        "VH": {
            "mean": vh_mean,
            "std": vh_std,
            "min": vh_min,
            "max": vh_max,
            "pixel_count": vh_count,
        },
        "metadata": {
            "dataset": "BigEarthNet-S1",
            "split": "train",
            "num_training_patches": processed_count,
            "excluded_validation_patches": len(val_s1_names),
            "seed": seed,
            "clip_range_db": list(clip_range),
            "pixels_per_patch": vv_count // processed_count if processed_count > 0 else 0,
            "has_nan": has_nan,
            "has_inf": has_inf,
        },
    }

    logger.info("Statistics Computation Complete!")
    logger.info(f"VV: Mean={vv_mean:.6f}, Std={vv_std:.6f}, Min={vv_min:.4f}, Max={vv_max:.4f}, Pixels={vv_count}")
    logger.info(f"VH: Mean={vh_mean:.6f}, Std={vh_std:.6f}, Min={vh_min:.4f}, Max={vh_max:.4f}, Pixels={vh_count}")

    # 9. Save to outputs/s1_stats.json if not in sanity test mode
    if not sanity_test_only:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(stats_result, f, indent=4)
        logger.info(f"Saved final training S1 statistics to: {output_json_path}")

    return stats_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel-1 SAR Statistics Generator")
    parser.add_argument(
        "--mode",
        choices=["training", "sanity", "demo"],
        default="training",
        help="Statistics mode: 'training' (21,000 training patches), 'sanity' (10-patch verification), 'demo' (legacy 20,000 random sample)",
    )
    args = parser.parse_args()

    if args.mode == "sanity":
        logger.info("Running SANITY TEST on 10 training patches...")
        sanity_res = compute_s1_training_stats(sanity_test_only=True, sanity_sample_size=10)
        print("\n--- SANITY TEST RESULT ---")
        print(json.dumps(sanity_res, indent=4))
        print("--- SANITY TEST PASSED ---\n")

    elif args.mode == "training":
        logger.info("Running FULL TRAINING STATISTICS over 21,000 patches...")
        stats_out = BASE_DIR / "outputs" / "s1_stats.json"
        res = compute_s1_training_stats(output_json_path=stats_out, sample_size=21000)
        print("\n========================================")
        print("Final S1 Training Statistics (21,000 patches)")
        print("========================================")
        print(f"Output File: {stats_out}")
        print(f"Patches Processed: {res['metadata']['num_training_patches']}")
        print(f"Excluded Validation Patches: {res['metadata']['excluded_validation_patches']}")
        print(f"VV Mean: {res['VV']['mean']:.6f}, VV Std: {res['VV']['std']:.6f}")
        print(f"VH Mean: {res['VH']['mean']:.6f}, VH Std: {res['VH']['std']:.6f}")
        print("========================================\n")

    elif args.mode == "demo":
        json_out = BASE_DIR / "outputs" / "statistics.json"
        csv_out = BASE_DIR / "outputs" / "statistics.csv"
        plot_dir = BASE_DIR / "outputs"
        
        logger.info("Starting legacy global statistics accumulator demo (20,000 random patches)...")
        try:
            global_stats = accumulate_statistics(DATA_DIR, sample_size=20000, num_workers=4)
            save_statistics(global_stats, json_out, csv_out)
            save_histograms(global_stats, plot_dir)
            update_settings_config(
                vv_mean=global_stats["vv"]["mean"],
                vv_std=global_stats["vv"]["std"],
                vh_mean=global_stats["vh"]["mean"],
                vh_std=global_stats["vh"]["std"],
            )
        except Exception as e:
            logger.exception(f"Statistics generation failed: {e}")
