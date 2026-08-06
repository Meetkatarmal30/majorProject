"""
dataloader.py - PyTorch DataLoader Integration for BigEarthNet-S1.

This module provides the create_dataloaders() function to instantiate PyTorch
DataLoaders for the train, validation, and test splits of BigEarthNet-S1.
It automatically sets up shuffling, batching, GPU pinned memory (if CUDA is available),
multiprocess workers, and integrates the custom BigEarthNetS1Dataset.

Includes type hints, logging, proper exception handling, and a runnable self-test.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import (
    DATA_DIR,
    SPLITS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    NORMALIZATION_MODE,
    PRESERVE_RESOLUTION,
    IMAGE_SIZE,
    setup_logging,
)
from dataset.dataset import BigEarthNetS1Dataset

logger = setup_logging("S1DataLoader")


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    splits_dir: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    norm_mode: str = NORMALIZATION_MODE,
    preserve_resolution: bool = PRESERVE_RESOLUTION,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    pin_memory: Optional[bool] = None,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test PyTorch DataLoaders for BigEarthNet-S1.

    Args:
        data_dir: Root directory of the dataset.
        splits_dir: Directory containing split list files (train.txt, val.txt, test.txt).
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        norm_mode: Normalization mode ('none', 'min-max', 'z-score').
        preserve_resolution: If True, keeps original dimensions.
        target_size: Target dimensions (H, W) if resizing.
        pin_memory: If True, copies Tensors to CUDA pinned memory. If None, checks CUDA availability.
        drop_last_train: If True, drops the last incomplete batch in the training set.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir_path}")

    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer. Got: {batch_size}")

    if num_workers < 0:
        raise ValueError(f"Number of workers cannot be negative. Got: {num_workers}")

    # Automatically set pin_memory based on GPU availability if not explicitly specified
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
        logger.info(f"Auto-configured pin_memory to {pin_memory} (CUDA available: {torch.cuda.is_available()})")

    logger.info("Initializing datasets for train, validation, and test splits...")
    
    try:
        # Create Train Dataset
        train_dataset = BigEarthNetS1Dataset(
            data_dir=data_dir_path,
            split="train",
            splits_dir=splits_dir,
            norm_mode=norm_mode,
            target_size=target_size,
            preserve_res=preserve_resolution,
        )
        
        # Create Validation Dataset
        val_dataset = BigEarthNetS1Dataset(
            data_dir=data_dir_path,
            split="validation",
            splits_dir=splits_dir,
            norm_mode=norm_mode,
            target_size=target_size,
            preserve_res=preserve_resolution,
        )
        
        # Create Test Dataset
        test_dataset = BigEarthNetS1Dataset(
            data_dir=data_dir_path,
            split="test",
            splits_dir=splits_dir,
            norm_mode=norm_mode,
            target_size=target_size,
            preserve_res=preserve_resolution,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate datasets: {e}")
        raise

    # Verify datasets are not empty
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("One or more splits yielded an empty dataset. Check dataset paths and split lists.")

    logger.info("Creating PyTorch DataLoaders...")

    # Train DataLoader: shuffles and drops last incomplete batch by default
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )

    # Validation DataLoader: no shuffle, do not drop last
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Test DataLoader: no shuffle, do not drop last
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Self-contained integration demo
    logger.info("Running DataLoader integration self-test...")
    try:
        # Create dataloaders using default batch size and 4 workers
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=64,
            num_workers=4,
        )

        print("\n========================================")
        print("DataLoaders Successfully Created!")
        print("========================================")
        print(f"Train Samples:       {len(train_loader.dataset)}")
        print(f"Validation Samples:  {len(val_loader.dataset)}")
        print(f"Test Samples:        {len(test_loader.dataset)}")
        print(f"Batch Size:          {train_loader.batch_size}")
        print(f"Number of Workers:   {train_loader.num_workers}")
        print("========================================\n")

        # Load one batch from the training loader
        logger.info("Loading one batch from train_loader...")
        
        # Pull the first batch
        batch_iter = iter(train_loader)
        images, patch_names = next(batch_iter)
        
        print("\n========================================")
        print("Train Batch Verification")
        print("========================================")
        print(f"Images Shape:        {images.shape}")
        print(f"Images Dtype:        {images.dtype}")
        print("First 5 Patch Names in Batch:")
        for i, name in enumerate(patch_names[:5]):
            print(f"  {i+1}. {name}")
        print("========================================\n")

    except Exception as e:
        logger.exception(f"DataLoader self-test failed: {e}")
