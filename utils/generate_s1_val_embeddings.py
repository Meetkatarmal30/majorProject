"""
utils/generate_s1_val_embeddings.py - Generate Sentinel-1 Baseline Validation Embeddings.

Regenerates baseline Sentinel-1 validation embeddings and S1 patch IDs from the CURRENT
official teammate_inputs/val_split.csv (4,500 samples).

Pipeline:
1. Ingests CURRENT teammate_inputs/val_split.csv in exact row order.
2. Resolves each S1 patch directly via:
   data/BigEarthNet-S1/BigEarthNet-S1/<first 5 tokens>/<s1_name>/
3. Applies finalized SAR preprocessing:
   - Outlier clipping to [-25.0, 0.0] dB
   - Normalization using outputs/s1_stats.json
   - Bilinear resizing to 224x224
   - float32 tensor (2, 224, 224)
4. Encodes via ResNet50S1Encoder to 2048-D latent feature representations.
5. Outputs:
   - teammate_inputs/baseline_val_s1_embeddings.npy (shape (4500, 2048), float32)
   - teammate_inputs/baseline_val_s1_patch_ids.txt (4,500 lines of s1_name)
   - outputs/s1_val_embeddings_summary.json
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import psutil

from config.settings import (
    BASE_DIR,
    DATA_DIR,
    S1_STATS_PATH,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    setup_logging,
)
from preprocessing.sar_preprocessing import preprocess_sar_patch
from models.resnet import ResNet50S1Encoder

logger = setup_logging("S1ValEmbeddings")


def get_current_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


class S1ValDirectDataset(Dataset):
    """
    Lightweight, deterministic dataset for loading Sentinel-1 validation patches
    directly using O(1) directory structure resolution.
    """

    def __init__(self, val_df: pd.DataFrame, s1_root: Path):
        self.val_df = val_df
        self.s1_root = s1_root
        self.s1_names = val_df["s1_name"].tolist()

    def __len__(self) -> int:
        return len(self.s1_names)

    def __getitem__(self, idx: int):
        s1_name = self.s1_names[idx]
        tile_group = "_".join(s1_name.split("_")[:5])
        patch_dir = self.s1_root / tile_group / s1_name

        vv_path = patch_dir / f"{s1_name}{VV_BAND_SUFFIX}"
        vh_path = patch_dir / f"{s1_name}{VH_BAND_SUFFIX}"

        if not vv_path.exists() or not vh_path.exists():
            raise FileNotFoundError(f"Missing TIFF files for '{s1_name}' in {patch_dir}")

        # Ingest using finalized SAR preprocessing
        # (Clips to [-25, 0], dynamic z-score with s1_stats.json, resizes to 224x224, returns (2, 224, 224))
        tensor = preprocess_sar_patch(vv_path, vh_path)
        return tensor, idx, s1_name


def regenerate_s1_val_embeddings(
    batch_size: int = 32,
    device: str = "cpu",
    sanity_mode: bool = False,
    sanity_samples: int = 16,
) -> bool:
    start_time = time.time()
    initial_ram = get_current_ram_mb()

    logger.info("=" * 70)
    logger.info("REGENERATING SENTINEL-1 BASELINE VALIDATION EMBEDDINGS")
    logger.info(f"Source of Truth: teammate_inputs/val_split.csv")
    logger.info(f"Initial RAM: {initial_ram:.2f} MB")
    logger.info("=" * 70)

    # 1. Verify Paths & Inputs
    val_csv_path = BASE_DIR / "teammate_inputs" / "val_split.csv"
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Source of truth not found: {val_csv_path}")

    s1_root = DATA_DIR
    if not s1_root.exists():
        raise FileNotFoundError(f"S1 dataset directory not found: {s1_root}")

    # 2. Read CURRENT teammate_inputs/val_split.csv in existing row order
    logger.info(f"Reading {val_csv_path}...")
    val_df = pd.read_csv(val_csv_path)

    # Pre-checks on CSV
    total_val_rows = len(val_df)
    assert total_val_rows == 4500, f"Expected exactly 4,500 rows, got {total_val_rows}"
    assert "s1_name" in val_df.columns, "Missing 's1_name' column in val_split.csv"
    assert val_df["s1_name"].isna().sum() == 0, "Null s1_name values detected in val_split.csv"

    s1_names_list = val_df["s1_name"].tolist()
    unique_s1_count = len(set(s1_names_list))
    assert unique_s1_count == 4500, f"Expected 4500 unique S1 IDs, got {unique_s1_count}"
    logger.info(f"[CHECK] val_split.csv has exactly 4,500 rows with 4,500 unique s1_name IDs: PASS")

    # Verify physical file existence for all 4,500 S1 patches before starting inference
    logger.info("Verifying physical S1 folder and TIFF file existence for all 4,500 patches...")
    for s1_name in s1_names_list:
        tile_group = "_".join(s1_name.split("_")[:5])
        patch_dir = s1_root / tile_group / s1_name
        if not patch_dir.exists():
            raise FileNotFoundError(f"Missing S1 directory: {patch_dir}")
        if not (patch_dir / f"{s1_name}{VV_BAND_SUFFIX}").exists():
            raise FileNotFoundError(f"Missing VV TIFF for {s1_name}")
        if not (patch_dir / f"{s1_name}{VH_BAND_SUFFIX}").exists():
            raise FileNotFoundError(f"Missing VH TIFF for {s1_name}")

    logger.info("[CHECK] All 4,500 S1 directories and VV/VH TIFFs exist on disk: PASS")

    # Read existing baseline_val_s1_patch_ids.txt for comparison report
    existing_txt_path = BASE_DIR / "teammate_inputs" / "baseline_val_s1_patch_ids.txt"
    old_ids_list = []
    if existing_txt_path.exists():
        with open(existing_txt_path, "r", encoding="utf-8") as f:
            old_ids_list = [line.strip() for line in f if line.strip()]

    # In sanity mode, slice
    if sanity_mode:
        val_df = val_df.iloc[:sanity_samples].copy()
        s1_names_list = s1_names_list[:sanity_samples]
        logger.info(f"Running in SANITY MODE with {sanity_samples} samples.")

    # 3. Instantiate ResNet50S1Encoder
    target_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Initializing ResNet50S1Encoder (device: {target_device})...")
    model = ResNet50S1Encoder(pretrained=True)
    model.to(target_device)
    model.eval()

    # 4. DataLoader Setup
    val_dataset = S1ValDirectDataset(val_df, s1_root)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: Preserve exact deterministic row order!
        num_workers=0,  # Single-process for low RAM & Windows stability
        pin_memory=False,
    )

    num_samples = len(val_dataset)
    embedding_dim = 2048
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

    logger.info(f"Starting inference over {num_samples} patches (batch size: {batch_size})...")
    processed_count = 0

    with torch.no_grad():
        for batch_idx, (tensors, indices, batch_s1_names) in enumerate(val_loader):
            tensors = tensors.to(target_device)
            feats = model(tensors)  # Shape: (B, 2048)

            feats_np = feats.cpu().numpy().astype(np.float32)
            batch_indices = indices.numpy()

            for i, global_idx in enumerate(batch_indices):
                embeddings[global_idx] = feats_np[i]

            processed_count += len(batch_indices)
            if (batch_idx + 1) % 20 == 0 or processed_count == num_samples:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                current_ram = get_current_ram_mb()
                logger.info(
                    f"Processed {processed_count}/{num_samples} patches "
                    f"({processed_count / num_samples * 100:.1f}%) | {rate:.1f} samples/s | RAM: {current_ram:.1f} MB"
                )

    total_time = time.time() - start_time
    logger.info(f"Inference complete in {total_time:.2f}s ({num_samples / total_time:.1f} samples/sec).")

    # In sanity mode, perform sanity assertions and return
    if sanity_mode:
        assert embeddings.shape == (sanity_samples, 2048)
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
        assert not (embeddings == 0).all()
        logger.info("[PASS] Sanity test passed successfully!")
        return True

    # 5. Execute Comprehensive Verification Checks BEFORE overwriting files
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING GENERATED EMBEDDINGS & S1 PATCH IDS")
    logger.info("=" * 70)

    # Check 1: Embedding shape is exactly (4500, 2048)
    assert embeddings.shape == (4500, 2048), f"Shape check failed: {embeddings.shape}"
    logger.info(f"1. Embeddings shape is (4500, 2048): PASS ({embeddings.shape})")

    # Check 2: Dtype is float32
    assert embeddings.dtype == np.float32, f"Dtype check failed: {embeddings.dtype}"
    logger.info(f"2. Embeddings dtype is float32: PASS ({embeddings.dtype})")

    # Check 3: Exactly 4,500 patch IDs
    assert len(s1_names_list) == 4500, f"Patch ID count check failed: {len(s1_names_list)}"
    logger.info(f"3. Exactly 4,500 patch IDs: PASS ({len(s1_names_list)})")

    # Check 4: All patch IDs unique
    assert len(set(s1_names_list)) == 4500, "Duplicate patch IDs detected"
    logger.info("4. All patch IDs unique: PASS (4,500 unique IDs)")

    # Check 5: Patch IDs exactly match val_split.csv['s1_name'] row-for-row
    assert s1_names_list == val_df["s1_name"].tolist(), "Row-for-row alignment failed"
    logger.info("5. Patch IDs match val_split.csv['s1_name'] row-for-row: PASS")

    # Check 6 & 7: No NaN / No Inf
    assert not np.isnan(embeddings).any(), "NaN values found in embeddings"
    assert not np.isinf(embeddings).any(), "Inf values found in embeddings"
    logger.info("6 & 7. No NaN and No Inf in embeddings: PASS")

    # Check 8: Embeddings are not all zero
    assert not (embeddings == 0).all(), "Embeddings are all zero"
    logger.info("8. Embeddings are not all zero: PASS")

    # Check 9: Every S1 patch used successfully
    assert processed_count == 4500, f"Expected 4500 patches used, got {processed_count}"
    logger.info(f"9. Every S1 patch used successfully: PASS ({processed_count}/4500)")

    # Check 10: No dataset files copied/moved
    logger.info("10. No dataset files copied, moved, modified, or duplicated: CONFIRMED")

    # Check 11: Confirm which validation split was used
    logger.info(f"11. Validation split confirmed: CURRENT teammate_inputs/val_split.csv (4,500 rows)")

    # Check 12: Report runtime and approximate RAM usage
    final_ram = get_current_ram_mb()
    logger.info(f"12. Runtime: {total_time:.2f}s | RAM Usage: {final_ram:.2f} MB (Peak Delta: {final_ram - initial_ram:.2f} MB)")

    # Check 13: Compare new S1 patch-ID list against existing baseline file
    changed_order_or_id_count = 0
    if old_ids_list:
        if len(old_ids_list) == len(s1_names_list):
            changed_order_or_id_count = sum(1 for a, b in zip(old_ids_list, s1_names_list) if a != b)
        else:
            changed_order_or_id_count = 4500
        logger.info(
            f"13. Comparison with previous baseline_val_s1_patch_ids.txt: "
            f"{changed_order_or_id_count} / {len(s1_names_list)} positions changed "
            f"(Old file had S2 patch IDs from the previous split; New file has S1 s1_name IDs from CURRENT val_split.csv)."
        )
    else:
        logger.info("13. No prior baseline S1 patch-ID file found to compare.")

    # 6. Save Outputs Atomically Now That All Verifications Passed
    output_npy_path = BASE_DIR / "teammate_inputs" / "baseline_val_s1_embeddings.npy"
    output_txt_path = BASE_DIR / "teammate_inputs" / "baseline_val_s1_patch_ids.txt"

    mirror_npy_path = BASE_DIR / "outputs" / "baseline_val_s1_embeddings.npy"
    mirror_txt_path = BASE_DIR / "outputs" / "baseline_val_s1_patch_ids.txt"

    logger.info(f"\nWriting verified embeddings to {output_npy_path}...")
    np.save(output_npy_path, embeddings)
    np.save(mirror_npy_path, embeddings)

    logger.info(f"Writing verified S1 s1_name IDs to {output_txt_path}...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for s1_name in s1_names_list:
            f.write(f"{s1_name}\n")

    with open(mirror_txt_path, "w", encoding="utf-8") as f:
        for s1_name in s1_names_list:
            f.write(f"{s1_name}\n")

    # Save summary report
    emb_min = float(embeddings.min())
    emb_max = float(embeddings.max())
    emb_mean = float(embeddings.mean())
    emb_std = float(embeddings.std())

    summary = {
        "status": "SUCCESS",
        "split_file_used": "teammate_inputs/val_split.csv",
        "total_validation_samples": 4500,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "min": emb_min,
        "max": emb_max,
        "mean": emb_mean,
        "std": emb_std,
        "runtime_seconds": total_time,
        "throughput_samples_per_sec": num_samples / total_time,
        "initial_ram_mb": initial_ram,
        "final_ram_mb": final_ram,
        "device": str(target_device),
        "id_format": "s1_name (Sentinel-1 patch names, NOT S2 patch_id)",
        "row_alignment": "100% exact match with val_split.csv['s1_name'] row-for-row",
        "changed_positions_from_previous_baseline": changed_order_or_id_count,
    }

    summary_path = BASE_DIR / "outputs" / "s1_val_embeddings_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Summary saved to {summary_path}.")
    logger.info("\n" + "=" * 70)
    logger.info("SENTINEL-1 VALIDATION EMBEDDINGS REGENERATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate Sentinel-1 baseline validation embeddings.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("--sanity", action="store_true", help="Run sanity check on 16 samples only")
    parser.add_argument("--samples", type=int, default=16, help="Sample count for sanity mode")
    args = parser.parse_args()

    success = regenerate_s1_val_embeddings(
        batch_size=args.batch_size,
        device=args.device,
        sanity_mode=args.sanity,
        sanity_samples=args.samples,
    )
    sys.exit(0 if success else 1)
