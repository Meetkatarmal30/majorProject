"""
dataset.py - Custom PyTorch Dataset for BigEarthNet-S1.

This module implements the BigEarthNetS1Dataset class, which:
- Reads the BigEarthNet-S1 dataset and discovers patch directories.
- Automates directory scanning using a JSON cache (`.s1_patch_cache.json`).
- Handles train, validation, and test splits (loading from files or splitting deterministically).
- Gracefully handles missing or corrupted GeoTIFF files via a retry loop.
- Returns preprocessed stacked 2-channel Float32 tensors and patch names: (tensor, patch_name).

Reuses the preprocessing module implemented in Week 1.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import (
    DATA_DIR,
    BASE_DIR,
    SPLITS_DIR,
    NORMALIZATION_MODE,
    PRESERVE_RESOLUTION,
    IMAGE_SIZE,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    setup_logging,
)
from preprocessing.sar_preprocessing import preprocess_sar_patch

logger = setup_logging("S1Dataset")


class BigEarthNetS1Dataset(Dataset):
    """
    Custom PyTorch Dataset for Sentinel-1 SAR patches in BigEarthNet-S1.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        split: str = "train",
        splits_dir: Optional[Path] = None,
        norm_mode: str = NORMALIZATION_MODE,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        preserve_res: bool = PRESERVE_RESOLUTION,
        cache_name: str = ".s1_patch_cache.json",
    ) -> None:
        """
        Initializes the dataset.

        Args:
            data_dir: Path to the BigEarthNet-S1 directory containing group folders.
            split: Split type ('train', 'validation', 'test').
            splits_dir: Optional path to the directory containing train.txt, val.txt, test.txt.
            norm_mode: Normalization mode ('none', 'min-max', 'z-score').
            target_size: Image size dimensions (H, W).
            preserve_res: If True, preserves original dimensions.
            cache_name: Name of the cache file.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.splits_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
        self.norm_mode = norm_mode
        self.target_size = target_size
        self.preserve_res = preserve_res
        self.cache_path = BASE_DIR / cache_name

        if self.split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split name '{self.split}'. Must be 'train', 'validation', or 'test'.")

        # Step 1: Scan / Load Directory Cache
        self.patch_dict = self._load_or_build_cache()

        # Step 2: Extract & Apply Splits
        self.patch_names = self._manage_splits()
        logger.info(f"Initialized BigEarthNetS1Dataset for split '{self.split}' with {len(self.patch_names)} patches.")

    def _load_or_build_cache(self) -> Dict[str, str]:
        """
        Loads the patch directory mapping from cache, or scans the disk if cache is missing.

        Returns:
            Dict[str, str]: Mapping of patch name to its absolute directory path string.
        """
        if self.cache_path.exists():
            logger.info(f"Loading patch directory cache from: {self.cache_path}")
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                # Verify that cache is not empty
                if cache_data:
                    return cache_data
                logger.warning("Cache file is empty. Forcing rescan.")
            except Exception as e:
                logger.warning(f"Failed to read cache file ({e}). Forcing rescan.")

        # Re-build cache
        logger.info(f"Scanning dataset folders under: {self.data_dir} (this may take a moment)...")
        patch_dict = {}
        try:
            with os.scandir(self.data_dir) as it:
                for entry in it:
                    if entry.is_dir():
                        with os.scandir(entry.path) as it2:
                            for entry2 in it2:
                                if entry2.is_dir():
                                    patch_dict[entry2.name] = entry2.path
        except Exception as e:
            logger.error(f"Failed to scan directory {self.data_dir}: {e}")
            raise FileNotFoundError(f"Could not read dataset root: {self.data_dir}") from e

        if not patch_dict:
            raise FileNotFoundError(f"No patch directories discovered under {self.data_dir}.")

        # Save to cache
        logger.info(f"Writing {len(patch_dict)} patches to cache file: {self.cache_path}")
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(patch_dict, f, indent=4)
        except Exception as e:
            logger.warning(f"Could not write cache file to disk: {e}")

        return patch_dict

    def _manage_splits(self) -> List[str]:
        """
        Loads splits lists from splits_dir or partitions deterministically.

        Returns:
            List[str]: List of patch names corresponding to this split.
        """
        # Convert split to filename (e.g. validation split is val.txt in standard lists)
        split_file_map = {
            "train": "train.txt",
            "validation": "val.txt",
            "test": "test.txt"
        }
        split_filename = split_file_map[self.split]
        split_filepath = self.splits_dir / split_filename

        if split_filepath.exists():
            logger.info(f"Loading split list from split file: {split_filepath}")
            try:
                with open(split_filepath, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                # Filter split to ensure they exist on disk/cache
                valid_names = [name for name in names if name in self.patch_dict]
                if valid_names:
                    return valid_names
                logger.warning(f"No patches from split file {split_filepath.name} exist in local dataset.")
            except Exception as e:
                logger.warning(f"Failed to read split file ({e}). Falling back to dynamic split.")

        # Fallback: Deterministic dynamic partition
        logger.warning(f"Split file not found at {split_filepath}. Generating deterministic 70/15/15 split...")
        all_patches = sorted(list(self.patch_dict.keys()))
        
        # Deterministically shuffle names
        rng = random.Random(42)
        rng.shuffle(all_patches)

        n = len(all_patches)
        train_idx = int(0.70 * n)
        val_idx = train_idx + int(0.15 * n)

        if self.split == "train":
            return all_patches[:train_idx]
        elif self.split == "validation":
            return all_patches[train_idx:val_idx]
        else:  # test
            return all_patches[val_idx:]

    def __len__(self) -> int:
        """
        Returns the total number of patches in this split.
        """
        return len(self.patch_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves a patch from the dataset.

        Args:
            idx: Index of the item.

        Returns:
            Tuple[torch.Tensor, str]: Stacked preprocessed tensor of shape (2, H, W) and the patch name.
        """
        retry_count = 0
        current_idx = idx

        while retry_count < 10:
            patch_name = self.patch_names[current_idx]
            patch_dir = Path(self.patch_dict[patch_name])

            # Resolve polarization band files
            vv_path = None
            vh_path = None
            try:
                with os.scandir(patch_dir) as it:
                    for entry in it:
                        if entry.is_file():
                            if entry.name.endswith(VV_BAND_SUFFIX):
                                vv_path = Path(entry.path)
                            elif entry.name.endswith(VH_BAND_SUFFIX):
                                vh_path = Path(entry.path)

                if not vv_path or not vh_path:
                    raise FileNotFoundError(f"Missing VV or VH bands in: {patch_dir}")

                # Load and preprocess
                tensor = preprocess_sar_patch(
                    vv_path=vv_path,
                    vh_path=vh_path,
                    target_size=self.target_size,
                    preserve_res=self.preserve_res,
                    norm_mode=self.norm_mode,
                )
                return tensor, patch_name

            except Exception as e:
                logger.warning(
                    f"Failed to load patch {patch_name} (Index: {current_idx}): {e}. "
                    f"Retrying with random index..."
                )
                retry_count += 1
                current_idx = random.randint(0, len(self.patch_names) - 1)

        raise RuntimeError(f"Failed to load a valid patch after 10 consecutive retries.")


if __name__ == "__main__":
    # Self-contained verification demo
    logger.info("Starting BigEarthNetS1Dataset demo...")
    
    # 1. Initialize dataset (first run will build cache; second run will load cache instantly)
    import time
    start_time = time.time()
    dataset = BigEarthNetS1Dataset(split="train")
    init_time = time.time() - start_time
    logger.info(f"Dataset initialization took: {init_time:.4f} seconds.")

    # 2. Get and print dataset length
    logger.info(f"Total patches in train split: {len(dataset)}")

    # 3. Retrieve a sample
    if len(dataset) > 0:
        tensor, name = dataset[0]
        logger.info(f"--- Sample 0 Loaded Successfully ---")
        logger.info(f"Patch Name: {name}")
        logger.info(f"Tensor Shape: {tensor.shape}")
        logger.info(f"Tensor Dtype: {tensor.dtype}")
        logger.info(f"VV Channel Range: Min={tensor[0].min().item():.4f}, Max={tensor[0].max().item():.4f}")
        logger.info(f"VH Channel Range: Min={tensor[1].min().item():.4f}, Max={tensor[1].max().item():.4f}")
        
        # 4. Trigger error recovery simulation
        logger.info("Simulating corrupted file load (error recovery)...")
        # Artificially insert a non-existent patch name to trigger warning and retry
        dataset.patch_names[0] = "Corrupted_Patch_Mock_Name"
        dataset.patch_dict["Corrupted_Patch_Mock_Name"] = "invalid/path/to/patch"
        
        # This call should log a warning, retry, and return a valid patch from another random index
        recovered_tensor, recovered_name = dataset[0]
        logger.info(f"--- Recovery Simulation Successful ---")
        logger.info(f"Recovered Patch Name: {recovered_name}")
        logger.info(f"Recovered Tensor Shape: {recovered_tensor.shape}")
    else:
        logger.warning("No samples found to test.")
