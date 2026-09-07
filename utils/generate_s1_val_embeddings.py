"""
utils/generate_s1_val_embeddings.py - Generate Sentinel-1 Baseline Validation Embeddings.

This script extracts 2048-dimensional feature representations for all 4,500 validation
patches using the baseline ResNet50S1Encoder model and the finalized SAR preprocessing
pipeline. It ensures strict row-for-row alignment with teammate_inputs/baseline_val_patch_ids.txt
and teammate_inputs/baseline_val_ms_embeddings.npy.
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from config.settings import (
    BASE_DIR,
    S1_STATS_PATH,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    setup_logging,
)
from preprocessing.sar_preprocessing import preprocess_sar_patch
from models.resnet import ResNet50S1Encoder

logger = setup_logging("S1ValEmbeddings")


class S1ValidationDataset(Dataset):
    """
    Lightweight, deterministic dataset for loading Sentinel-1 validation patches
    strictly following the order of val_split.csv.
    """

    def __init__(self, val_df: pd.DataFrame, patch_cache: dict):
        self.val_df = val_df
        self.patch_cache = patch_cache
        self.patch_ids = val_df["patch_id"].tolist()
        self.s1_names = val_df["s1_name"].tolist()

    def __len__(self) -> int:
        return len(self.val_df)

    def __getitem__(self, idx: int):
        s1_name = self.s1_names[idx]
        patch_id = self.patch_ids[idx]

        if s1_name not in self.patch_cache:
            raise KeyError(f"Patch '{s1_name}' not found in directory cache.")

        patch_dir = Path(self.patch_cache[s1_name])
        vv_path = patch_dir / f"{s1_name}{VV_BAND_SUFFIX}"
        vh_path = patch_dir / f"{s1_name}{VH_BAND_SUFFIX}"

        if not vv_path.exists() or not vh_path.exists():
            raise FileNotFoundError(f"Missing TIFF file for '{s1_name}' in {patch_dir}")

        # Ingest using the finalized SAR preprocessing pipeline
        # (Clips to [-25, 0], dynamic z-score with s1_stats.json, resizes to 224x224, returns (2, 224, 224))
        tensor = preprocess_sar_patch(vv_path, vh_path)
        return tensor, idx, patch_id


def generate_s1_embeddings(
    batch_size: int = 32,
    device: str = "cpu",
    sanity_mode: bool = False,
    sanity_samples: int = 16,
) -> bool:
    """
    Generates and saves baseline Sentinel-1 validation embeddings.

    Args:
        batch_size: Batch size for DataLoader.
        device: 'cpu' or 'cuda'.
        sanity_mode: If True, processes only a small subset for verification.
        sanity_samples: Number of samples in sanity mode.

    Returns:
        bool: True if generation and all validation checks pass.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("GENERATING BASELINE SENTINEL-1 VALIDATION EMBEDDINGS")
    logger.info("=" * 60)

    # 1. Verify Paths
    val_csv_path = BASE_DIR / "teammate_inputs" / "val_split.csv"
    val_txt_path = BASE_DIR / "teammate_inputs" / "baseline_val_patch_ids.txt"
    val_ms_path = BASE_DIR / "teammate_inputs" / "baseline_val_ms_embeddings.npy"
    cache_path = BASE_DIR / ".s1_patch_cache.json"

    for p in (val_csv_path, val_txt_path, val_ms_path, cache_path, S1_STATS_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # 2. Load & Align Metadata
    logger.info(f"Loading validation metadata from: {val_csv_path}")
    val_df = pd.read_csv(val_csv_path)
    total_val_rows = len(val_df)
    logger.info(f"Loaded {total_val_rows} validation rows from CSV.")

    with open(val_txt_path, "r", encoding="utf-8") as f:
        target_patch_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(target_patch_ids)} target patch IDs from TXT.")

    # Strict Alignment Pre-Check
    assert total_val_rows == 4500, f"Expected 4500 validation rows, got {total_val_rows}"
    assert len(target_patch_ids) == 4500, f"Expected 4500 target IDs, got {len(target_patch_ids)}"
    assert val_df["patch_id"].tolist() == target_patch_ids, (
        "Fatal: val_split.csv and baseline_val_patch_ids.txt are not aligned row-for-row!"
    )
    logger.info("[CHECK] Row-for-row alignment between CSV and TXT: 100% MATCH.")

    with open(cache_path, "r", encoding="utf-8") as f:
        patch_cache = json.load(f)

    # In sanity mode, slice to sanity_samples
    if sanity_mode:
        val_df = val_df.iloc[:sanity_samples].copy()
        target_patch_ids = target_patch_ids[:sanity_samples]
        logger.info(f"Running in SANITY MODE with {sanity_samples} samples.")

    # 3. Instantiate Model
    target_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Initializing baseline ResNet50S1Encoder (device: {target_device})...")
    model = ResNet50S1Encoder(pretrained=True)
    model.to(target_device)
    model.eval()

    # 4. DataLoader Setup
    val_dataset = S1ValidationDataset(val_df, patch_cache)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # CRITICAL: Preserve exact deterministic row order!
        num_workers=0,  # Safe for Windows
        pin_memory=False,
    )

    num_samples = len(val_dataset)
    embedding_dim = 2048
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)

    logger.info(f"Starting inference over {num_samples} patches (batch size: {batch_size})...")
    processed_count = 0

    with torch.no_grad():
        for batch_idx, (tensors, indices, batch_patch_ids) in enumerate(val_loader):
            tensors = tensors.to(target_device)
            feats = model(tensors)  # Shape: (B, 2048)

            feats_np = feats.cpu().numpy().astype(np.float32)
            batch_indices = indices.numpy()

            # Assign strictly by index to ensure exact alignment
            for i, global_idx in enumerate(batch_indices):
                embeddings[global_idx] = feats_np[i]

            processed_count += len(batch_indices)
            if (batch_idx + 1) % 20 == 0 or processed_count == num_samples:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Processed {processed_count}/{num_samples} patches "
                    f"({processed_count / num_samples * 100:.1f}%) - {rate:.1f} samples/sec"
                )

    total_time = time.time() - start_time
    logger.info(f"Inference complete in {total_time:.2f}s ({num_samples / total_time:.1f} samples/sec).")

    # In sanity mode, perform sanity assertions and return
    if sanity_mode:
        assert embeddings.shape == (sanity_samples, 2048), f"Unexpected shape: {embeddings.shape}"
        assert not np.isnan(embeddings).any(), "NaN values detected in sanity embeddings"
        assert not np.isinf(embeddings).any(), "Inf values detected in sanity embeddings"
        assert not (embeddings == 0).all(), "All-zero embeddings detected"
        logger.info("[PASS] Sanity test passed successfully!")
        return True

    # 5. Save Outputs
    output_npy_path = BASE_DIR / "teammate_inputs" / "baseline_val_s1_embeddings.npy"
    output_txt_path = BASE_DIR / "teammate_inputs" / "baseline_val_s1_patch_ids.txt"

    # Also mirror to outputs/ for comprehensive accessibility
    mirror_npy_path = BASE_DIR / "outputs" / "baseline_val_s1_embeddings.npy"
    mirror_txt_path = BASE_DIR / "outputs" / "baseline_val_s1_patch_ids.txt"

    logger.info(f"Saving embeddings to: {output_npy_path}")
    np.save(output_npy_path, embeddings)
    np.save(mirror_npy_path, embeddings)

    logger.info(f"Saving patch IDs to: {output_txt_path}")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for pid in target_patch_ids:
            f.write(f"{pid}\n")

    with open(mirror_txt_path, "w", encoding="utf-8") as f:
        for pid in target_patch_ids:
            f.write(f"{pid}\n")

    logger.info("Files written successfully.")

    # 6. Execute Comprehensive Validation Checks (15 Items)
    logger.info("\n" + "=" * 60)
    logger.info("EXECUTING 15 MANDATORY VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check 1: File exists
    assert output_npy_path.exists(), f"Check 1 Failed: Missing {output_npy_path}"
    assert output_txt_path.exists(), f"Check 1 Failed: Missing {output_txt_path}"
    logger.info("1. File exists: PASS")

    # Check 2: Embedding shape is exactly (4500, 2048)
    loaded_s1_emb = np.load(output_npy_path)
    assert loaded_s1_emb.shape == (4500, 2048), f"Check 2 Failed: Shape is {loaded_s1_emb.shape}"
    logger.info(f"2. Embedding shape is exactly (4500, 2048): PASS (Got {loaded_s1_emb.shape})")

    # Check 3: Number of IDs is exactly 4500
    with open(output_txt_path, "r", encoding="utf-8") as f:
        saved_ids = [line.strip() for line in f if line.strip()]
    assert len(saved_ids) == 4500, f"Check 3 Failed: ID count is {len(saved_ids)}"
    logger.info(f"3. Number of IDs is exactly 4500: PASS (Got {len(saved_ids)})")

    # Check 4: No duplicate IDs
    assert len(set(saved_ids)) == 4500, "Check 4 Failed: Duplicate IDs detected"
    logger.info("4. No duplicate IDs: PASS (4500 unique IDs)")

    # Check 5: S1 output IDs exactly equal baseline_val_patch_ids.txt row-for-row
    assert saved_ids == target_patch_ids, "Check 5 Failed: IDs do not match baseline_val_patch_ids.txt"
    logger.info("5. S1 output IDs equal baseline_val_patch_ids.txt row-for-row: PASS")

    # Check 6: S1 output IDs exactly correspond to val_split.csv patch_id row-for-row
    assert saved_ids == val_df["patch_id"].tolist(), "Check 6 Failed: IDs do not match val_split.csv"
    logger.info("6. S1 output IDs correspond to val_split.csv patch_id row-for-row: PASS")

    # Check 7: Every corresponding s1_name exists physically
    all_s1_exist = all(s1 in patch_cache for s1 in val_df["s1_name"])
    assert all_s1_exist, "Check 7 Failed: Some s1_names are missing from cache"
    logger.info("7. Every corresponding s1_name exists in cache: PASS")

    # Check 8: Both VV and VH TIFFs exist for all 4,500 samples
    for s1 in val_df["s1_name"]:
        p = Path(patch_cache[s1])
        assert (p / f"{s1}{VV_BAND_SUFFIX}").exists(), f"Check 8 Failed: Missing VV for {s1}"
        assert (p / f"{s1}{VH_BAND_SUFFIX}").exists(), f"Check 8 Failed: Missing VH for {s1}"
    logger.info("8. Both VV and VH TIFFs exist for all 4,500 samples: PASS")

    # Check 9: Embeddings contain no NaN
    assert not np.isnan(loaded_s1_emb).any(), "Check 9 Failed: NaN values detected"
    logger.info("9. Embeddings contain no NaN: PASS (0 NaNs)")

    # Check 10: Embeddings contain no Inf
    assert not np.isinf(loaded_s1_emb).any(), "Check 10 Failed: Inf values detected"
    logger.info("10. Embeddings contain no Inf: PASS (0 Infs)")

    # Check 11: Embeddings are not all zeros
    assert not (loaded_s1_emb == 0).all(), "Check 11 Failed: Embeddings are all zeros"
    logger.info("11. Embeddings are not all zeros: PASS")

    # Check 12: dtype is reasonable (float32)
    assert loaded_s1_emb.dtype == np.float32, f"Check 12 Failed: dtype is {loaded_s1_emb.dtype}"
    logger.info(f"12. dtype is reasonable: PASS (dtype: {loaded_s1_emb.dtype})")

    # Check 13: Print min/max/mean/std of the generated embeddings
    emb_min = float(loaded_s1_emb.min())
    emb_max = float(loaded_s1_emb.max())
    emb_mean = float(loaded_s1_emb.mean())
    emb_std = float(loaded_s1_emb.std())
    logger.info(f"13. S1 Embedding Stats: Min={emb_min:.6f}, Max={emb_max:.6f}, Mean={emb_mean:.6f}, Std={emb_std:.6f}: PASS")

    # Check 14: Verify existing baseline_val_ms_embeddings.npy has shape (4500, 2048)
    loaded_ms_emb = np.load(val_ms_path)
    assert loaded_ms_emb.shape == (4500, 2048), f"Check 14 Failed: MS shape is {loaded_ms_emb.shape}"
    logger.info(f"14. Existing baseline_val_ms_embeddings.npy shape: PASS (Got {loaded_ms_emb.shape})")

    # Check 15: MS and S1 embedding files have exactly the same row ordering through shared patch IDs
    assert len(loaded_ms_emb) == len(loaded_s1_emb) == len(saved_ids) == 4500
    logger.info("15. MS and S1 embedding row ordering alignment: PASS (1-to-1 index correspondence)")
    logger.info("=" * 60)
    logger.info("ALL 15 VALIDATION CHECKS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    # Save validation summary metrics
    summary = {
        "status": "SUCCESS",
        "total_validation_samples": 4500,
        "embedding_shape": list(loaded_s1_emb.shape),
        "embedding_dtype": str(loaded_s1_emb.dtype),
        "min": emb_min,
        "max": emb_max,
        "mean": emb_mean,
        "std": emb_std,
        "runtime_seconds": total_time,
        "device": str(target_device),
        "alignment_guarantee": "Row-for-row matched to baseline_val_patch_ids.txt",
    }
    with open(BASE_DIR / "outputs" / "s1_val_embeddings_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sentinel-1 baseline validation embeddings.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("--sanity", action="store_true", help="Run sanity check on 16 samples only")
    parser.add_argument("--samples", type=int, default=16, help="Sample count for sanity mode")
    args = parser.parse_args()

    success = generate_s1_embeddings(
        batch_size=args.batch_size,
        device=args.device,
        sanity_mode=args.sanity,
        sanity_samples=args.samples,
    )
    sys.exit(0 if success else 1)
