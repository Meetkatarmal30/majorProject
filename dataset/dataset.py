"""
dataset.py - Custom PyTorch Datasets for BigEarthNet.

This module implements:
1. BigEarthNetLabelEncoder: Deterministic 19-class multi-hot encoder.
2. BigEarthNetS1Dataset: Dedicated single-modality Sentinel-1 (SAR) dataset.
3. PairedBigEarthNetDataset: Final paired Sentinel-1 (SAR) and Sentinel-2 (Multispectral)
   dataset operating on the official train/validation split mappings.
4. paired_collate_fn: DataLoader collation handling both multimodal and S1-verification batches.
"""

import os
import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

import torch
from torch.utils.data import Dataset
import pandas as pd

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
from preprocessing.preprocess_s2 import preprocess_s2_patch

logger = setup_logging("Dataset")

# Official BigEarthNet 19-class land-cover taxonomy (alphabetically sorted, deterministic)
BIGEARTHNET_19_CLASSES: List[str] = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]


class BigEarthNetLabelEncoder:
    """
    Deterministic 19-class multi-hot label encoder for BigEarthNet.
    Parses label strings or iterables into a multi-hot FloatTensor of shape (19,).
    """

    def __init__(self, classes: Optional[List[str]] = None) -> None:
        self.classes = list(classes) if classes else list(BIGEARTHNET_19_CLASSES)
        self.class_to_idx: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.num_classes: int = len(self.classes)

    def parse_labels(self, raw_label: Any) -> List[str]:
        """
        Parses raw string representation of labels or list into a list of clean class strings.
        Handles formatting such as:
        - "['Arable land' 'Pastures']"
        - "['Arable land', 'Pastures']"
        - Python lists / sets
        """
        if isinstance(raw_label, (list, tuple, set)):
            return [str(x).strip() for x in raw_label if str(x).strip()]

        val_str = str(raw_label).strip()
        if val_str.startswith("[") and val_str.endswith("]"):
            inner = val_str[1:-1].strip()
            # Match quoted strings (single or double quotes)
            items = re.findall(r"['\"]([^'\"]+)['\"]", inner)
            if not items and inner:
                items = [x.strip() for x in inner.split(",") if x.strip()]
            return items
        elif val_str and val_str.lower() != "nan":
            return [x.strip() for x in val_str.split(",") if x.strip()]
        return []

    def encode(self, raw_label: Any) -> torch.Tensor:
        """
        Encodes the raw label into a (19,) float32 multi-hot tensor.
        """
        labels_list = self.parse_labels(raw_label)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label_name in labels_list:
            if label_name in self.class_to_idx:
                target[self.class_to_idx[label_name]] = 1.0
        return target


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
        if self.cache_path.exists():
            logger.info(f"Loading patch directory cache from: {self.cache_path}")
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data:
                    return cache_data
                logger.warning("Cache file is empty. Forcing rescan.")
            except Exception as e:
                logger.warning(f"Failed to read cache file ({e}). Forcing rescan.")

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

        logger.info(f"Writing {len(patch_dict)} patches to cache file: {self.cache_path}")
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(patch_dict, f, indent=4)
        except Exception as e:
            logger.warning(f"Could not write cache file to disk: {e}")

        return patch_dict

    def _manage_splits(self) -> List[str]:
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
                valid_names = [name for name in names if name in self.patch_dict]
                if valid_names:
                    return valid_names
                logger.warning(f"No patches from split file {split_filepath.name} exist in local dataset.")
            except Exception as e:
                logger.warning(f"Failed to read split file ({e}). Falling back to dynamic split.")

        logger.warning(f"Split file not found at {split_filepath}. Generating deterministic 70/15/15 split...")
        all_patches = sorted(list(self.patch_dict.keys()))
        rng = random.Random(42)
        rng.shuffle(all_patches)

        n = len(all_patches)
        train_idx = int(0.70 * n)
        val_idx = train_idx + int(0.15 * n)

        if self.split == "train":
            return all_patches[:train_idx]
        elif self.split == "validation":
            return all_patches[train_idx:val_idx]
        else:
            return all_patches[val_idx:]

    def __len__(self) -> int:
        return len(self.patch_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        retry_count = 0
        current_idx = idx

        while retry_count < 10:
            patch_name = self.patch_names[current_idx]
            patch_dir = Path(self.patch_dict[patch_name])

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

        raise RuntimeError("Failed to load a valid patch after 10 consecutive retries.")


class PairedBigEarthNetDataset(Dataset):
    """
    Official Paired Sentinel-1 (SAR) and Sentinel-2 (Multispectral) Dataset for BigEarthNet.

    Associates Sentinel-1 SAR patches, Sentinel-2 RGB patches, and 19-class
    multi-hot labels based on the official training (train_split.csv) or validation (val_split.csv)
    mapping files.

    Conceptual Interface:
        PairedBigEarthNetDataset(
            csv_path,
            s1_root,
            s2_root,
            s1_transform=None,
            s2_transform=None,
            require_s2=False,
            s1_cache_name=".s1_patch_cache.json"
        )

    Item Return:
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, str]:
        - sar_tensor: (2, 224, 224), float32
        - ms_tensor: (3, 224, 224), float32 (or None if require_s2=False and s2_root is unavailable)
        - label_tensor: (19,), float32
        - patch_id: str
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        s1_root: Union[str, Path] = DATA_DIR,
        s2_root: Optional[Union[str, Path]] = None,
        s1_transform: Optional[Callable] = None,
        s2_transform: Optional[Callable] = None,
        require_s2: bool = False,
        s1_cache_name: str = ".s1_patch_cache.json",
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Mapping CSV file not found: {self.csv_path}")

        self.s1_root = Path(s1_root)
        self.s2_root = Path(s2_root) if s2_root else None
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.require_s2 = require_s2
        self.s1_cache_path = BASE_DIR / s1_cache_name

        # Validate S2 availability
        self.s2_available: bool = (self.s2_root is not None and self.s2_root.exists() and self.s2_root.is_dir())
        if self.require_s2 and not self.s2_available:
            raise FileNotFoundError(
                f"Sentinel-2 dataset root directory is unavailable or not found at: '{s2_root}'. "
                f"BigEarthNet-S2 raw imagery is required on the teammate's machine when require_s2=True. "
                f"(Synthetic or placeholder S2 data is strictly prohibited)."
            )

        # Load CSV metadata
        logger.info(f"Loading paired split metadata from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # Validate required columns
        required_cols = ["patch_id", "labels", "split", "s1_name", "s2v1_name"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Mapping CSV {self.csv_path} is missing required columns: {missing_cols}")

        self.patch_ids: List[str] = self.df["patch_id"].tolist()
        self.s1_names: List[str] = self.df["s1_name"].tolist()
        self.s2v1_names: List[str] = self.df["s2v1_name"].tolist()
        self.raw_labels: List[Any] = self.df["labels"].tolist()

        # Multi-hot Label Encoder
        self.label_encoder = BigEarthNetLabelEncoder()

        # Load S1 patch directory cache
        self.s1_cache = self._load_s1_cache()

        logger.info(
            f"Initialized PairedBigEarthNetDataset with {len(self.df)} samples "
            f"(S2 Available: {self.s2_available}, Require S2: {self.require_s2})."
        )

    def _load_s1_cache(self) -> Dict[str, str]:
        """
        Loads the Sentinel-1 patch directory cache.
        """
        if self.s1_cache_path.exists():
            try:
                with open(self.s1_cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if cache:
                    return cache
            except Exception as e:
                logger.warning(f"Could not load cache {self.s1_cache_path}: {e}")

        # Fallback to scanning S1 root
        logger.info(f"Scanning S1 root for patch directories: {self.s1_root}...")
        cache = {}
        with os.scandir(self.s1_root) as it:
            for entry in it:
                if entry.is_dir():
                    with os.scandir(entry.path) as it2:
                        for entry2 in it2:
                            if entry2.is_dir():
                                cache[entry2.name] = entry2.path

        if not cache:
            raise FileNotFoundError(f"No Sentinel-1 patch directories found under {self.s1_root}")

        try:
            with open(self.s1_cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4)
        except Exception:
            pass

        return cache

    def _resolve_s2_folder(self, patch_id: str, s2v1_name: str) -> Path:
        """
        Resolves the Sentinel-2 patch directory on the teammate's machine.
        Supports standard BigEarthNet nesting schemes.
        """
        if not self.s2_root:
            raise FileNotFoundError("s2_root is not configured.")

        # Candidate 1: Direct subfolder named patch_id
        candidate1 = self.s2_root / patch_id
        if candidate1.exists() and candidate1.is_dir():
            return candidate1

        # Candidate 2: Direct subfolder named s2v1_name
        candidate2 = self.s2_root / s2v1_name
        if candidate2.exists() and candidate2.is_dir():
            return candidate2

        # Candidate 3: Nested under granule tile folder
        # e.g. S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_29_55 -> parent tile is ..._T33UUP
        parts = patch_id.rsplit("_", 2)
        if len(parts) >= 2:
            tile_folder = parts[0]
            candidate3 = self.s2_root / tile_folder / patch_id
            if candidate3.exists() and candidate3.is_dir():
                return candidate3
            candidate4 = self.s2_root / tile_folder / s2v1_name
            if candidate4.exists() and candidate4.is_dir():
                return candidate4

        raise FileNotFoundError(
            f"Could not locate Sentinel-2 patch directory for '{patch_id}' / '{s2v1_name}' "
            f"under root: {self.s2_root}"
        )

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, str]:
        """
        Retrieves a paired sample from the dataset.

        Returns:
            Tuple:
            - sar_tensor: (2, 224, 224), float32
            - ms_tensor: (3, 224, 224), float32 (or None if require_s2=False and S2 is unavailable)
            - label_tensor: (19,), float32
            - patch_id: str
        """
        patch_id = self.patch_ids[idx]
        s1_name = self.s1_names[idx]
        s2v1_name = self.s2v1_names[idx]
        raw_label = self.raw_labels[idx]

        # 1. Sentinel-1 Processing
        if s1_name not in self.s1_cache:
            raise KeyError(f"Sentinel-1 patch '{s1_name}' not found in directory cache.")

        s1_dir = Path(self.s1_cache[s1_name])
        vv_path = s1_dir / f"{s1_name}{VV_BAND_SUFFIX}"
        vh_path = s1_dir / f"{s1_name}{VH_BAND_SUFFIX}"

        if not vv_path.exists() or not vh_path.exists():
            raise FileNotFoundError(f"Missing VV/VH TIFF files for S1 patch: {s1_name} in {s1_dir}")

        # On-demand SAR preprocessing -> (2, 224, 224)
        sar_tensor = preprocess_sar_patch(vv_path=vv_path, vh_path=vh_path)
        if self.s1_transform is not None:
            sar_tensor = self.s1_transform(sar_tensor)

        # 2. Sentinel-2 Processing
        ms_tensor: Optional[torch.Tensor] = None
        if self.s2_available:
            s2_folder = self._resolve_s2_folder(patch_id, s2v1_name)
            ms_tensor = preprocess_s2_patch(s2_folder)
            if self.s2_transform is not None:
                ms_tensor = self.s2_transform(ms_tensor)
        else:
            if self.require_s2:
                raise FileNotFoundError(
                    f"Sentinel-2 dataset root is unavailable. Real BigEarthNet-S2 data is required "
                    f"when require_s2=True. (Fake or placeholder S2 tensors are strictly disabled)."
                )
            # Safe S1-only verification mode
            ms_tensor = None

        # 3. 19-Class Multi-Hot Label
        label_tensor = self.label_encoder.encode(raw_label)

        return sar_tensor, ms_tensor, label_tensor, patch_id


def paired_collate_fn(
    batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, str]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[str]]:
    """
    PyTorch DataLoader collation function for PairedBigEarthNetDataset.

    Handles:
    - Full multimodal batches: returns (sar_batch, ms_batch, label_batch, patch_ids)
    - Safe S1-verification batches: returns (sar_batch, None, label_batch, patch_ids)
    """
    sar_tensors = torch.stack([item[0] for item in batch], dim=0)

    if batch[0][1] is not None:
        ms_tensors = torch.stack([item[1] for item in batch], dim=0)
    else:
        ms_tensors = None

    label_tensors = torch.stack([item[2] for item in batch], dim=0)
    patch_ids = [item[3] for item in batch]

    return sar_tensors, ms_tensors, label_tensors, patch_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing BigEarthNet Label Encoder...")
    encoder = BigEarthNetLabelEncoder()
    sample_raw = "['Arable land' 'Complex cultivation patterns']"
    encoded = encoder.encode(sample_raw)
    print(f"Sample raw: {sample_raw}")
    print(f"Encoded shape: {encoded.shape}, dtype: {encoded.dtype}, active: {torch.nonzero(encoded).squeeze().tolist()}")
    assert encoded.shape == (19,)
    assert encoded.sum() == 2.0
    print("[PASS] BigEarthNetLabelEncoder test passed successfully.")
