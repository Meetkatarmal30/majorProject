"""
test_pipeline.py - Automated Unit & Integration Tests for Week 2 SAR Data Pipeline.

This test suite validates:
1. Dataset initialization (caching, length, splits).
2. __getitem__ outputs (tensor shapes, dtypes, ranges, no NaN/Inf).
3. Cache load performance speedups.
4. Deterministic split management (train + val + test total size).
5. Corruption/missing file recovery (logs warning, retries, loads valid sample).
6. DataLoader batch loading and shape correctness.
7. Normalization modes application.

Run using:
    python -m unittest tests/test_pipeline.py
"""

import os
import json
import time
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import DATA_DIR, BASE_DIR
from dataset.dataset import BigEarthNetS1Dataset
from dataset.dataloader import create_dataloaders


class TestSARPipeline(unittest.TestCase):
    """
    Automated test suite for Sentinel-1 dataset and dataloader components.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup cache file path and check dataset directory.
        """
        cls.cache_path = BASE_DIR / ".s1_patch_cache.json"
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Test aborted: Dataset directory not found at {DATA_DIR}")

    def test_01_dataset_initialization(self) -> None:
        """
        Test 1: Verify S1 dataset initializes successfully and parameters match.
        """
        try:
            dataset = BigEarthNetS1Dataset(split="train")
            self.assertIsNotNone(dataset)
            self.assertTrue(len(dataset) > 0, "Dataset length should be greater than 0")
            self.assertEqual(dataset.split, "train")
            self.assertTrue(self.cache_path.exists(), "Cache file should exist after initialization")
        except Exception as e:
            self.fail(f"Dataset initialization raised an exception: {e}")

    def test_02_dataset_getitem(self) -> None:
        """
        Test 2: Verify that __getitem__ returns valid tensors and patch names.
        """
        dataset = BigEarthNetS1Dataset(split="train")
        tensor, patch_name = dataset[0]

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape, (2, 224, 224))
        self.assertIsInstance(patch_name, str)

        # Check for NaN and Inf values
        self.assertFalse(torch.isnan(tensor).any(), "Tensor should not contain NaN values")
        self.assertFalse(torch.isinf(tensor).any(), "Tensor should not contain Inf values")

    def test_03_cache_loading(self) -> None:
        """
        Test 3: Verify cache creation and check instantiation speedup.
        """
        # Delete cache if it exists to test cold instantiation
        if self.cache_path.exists():
            os.remove(self.cache_path)

        # First run (Cold)
        start_cold = time.time()
        dataset_cold = BigEarthNetS1Dataset(split="train")
        time_cold = time.time() - start_cold

        self.assertTrue(self.cache_path.exists(), "Cache file was not created on first run")

        # Second run (Warm)
        start_warm = time.time()
        dataset_warm = BigEarthNetS1Dataset(split="train")
        time_warm = time.time() - start_warm

        # Warm loading should be significantly faster
        self.assertTrue(
            time_warm < time_cold,
            f"Warm initialization ({time_warm:.4f}s) should be faster than cold ({time_cold:.4f}s)"
        )

    def test_04_split_generation(self) -> None:
        """
        Test 4: Verify train, validation, and test splits sum to total patches.
        """
        train_ds = BigEarthNetS1Dataset(split="train")
        val_ds = BigEarthNetS1Dataset(split="validation")
        test_ds = BigEarthNetS1Dataset(split="test")

        train_len = len(train_ds)
        val_len = len(val_ds)
        test_len = len(test_ds)

        # Total count from cache
        with open(self.cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        total_cache_size = len(cache_data)

        self.assertEqual(
            train_len + val_len + test_len,
            total_cache_size,
            "Sum of splits should equal total patches in directory cache."
        )

    def test_05_corrupted_file_handling(self) -> None:
        """
        Test 5: Verify dataset recovery from missing or corrupted patch folders.
        """
        dataset = BigEarthNetS1Dataset(split="train")
        
        # Inject corrupted entry at index 0
        original_name = dataset.patch_names[0]
        original_dir = dataset.patch_dict[original_name]

        # Corrupt path details
        dataset.patch_names[0] = "Corrupted_Patch_Mock_Name"
        dataset.patch_dict["Corrupted_Patch_Mock_Name"] = str(BASE_DIR / "non_existent_folder_xyz")

        try:
            # Should log a warning, catch FileNotFoundError, retry, and return a valid patch from another index
            tensor, patch_name = dataset[0]
            self.assertNotEqual(patch_name, "Corrupted_Patch_Mock_Name")
            self.assertEqual(tensor.shape, (2, 224, 224))
        finally:
            # Restore original values to prevent side effects in other tests
            dataset.patch_names[0] = original_name
            dataset.patch_dict[original_name] = original_dir
            if "Corrupted_Patch_Mock_Name" in dataset.patch_dict:
                del dataset.patch_dict["Corrupted_Patch_Mock_Name"]

    def test_06_dataloader(self) -> None:
        """
        Test 6: Verify PyTorch DataLoader configurations and batch dimensions.
        """
        batch_size = 16
        train_loader, _, _ = create_dataloaders(batch_size=batch_size, num_workers=2)

        self.assertIsInstance(train_loader, DataLoader)
        self.assertEqual(train_loader.batch_size, batch_size)

        # Load one batch
        batch_iter = iter(train_loader)
        images, patch_names = next(batch_iter)

        self.assertEqual(images.shape, (batch_size, 2, 224, 224))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(len(patch_names), batch_size)

    def test_07_normalization_verification(self) -> None:
        """
        Test 7: Verify that preprocessing is applied and normalized output is valid.
        """
        # Test Min-Max normalization mode
        dataset = BigEarthNetS1Dataset(split="train", norm_mode="min-max")
        tensor, _ = dataset[0]

        # Verify values range matches [0, 1] approximately (subject to floating point clipping bounds)
        self.assertTrue(tensor.min() >= 0.0, f"Min-Max normalized min value should be >= 0. Got: {tensor.min()}")
        self.assertTrue(tensor.max() <= 1.0, f"Min-Max normalized max value should be <= 1. Got: {tensor.max()}")
        self.assertFalse(torch.isnan(tensor).any(), "Normalized tensor contains NaN")
        self.assertFalse(torch.isinf(tensor).any(), "Normalized tensor contains Inf")


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("Running Unit Tests")
    print("" + "=" * 40)
    
    # Run tests using unittest loader
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSARPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 40)
    if result.wasSuccessful():
        print("[PASS] Dataset Initialization")
        print("[PASS] Dataset Length")
        print("[PASS] Dataset getitem()")
        print("[PASS] Cache")
        print("[PASS] Split")
        print("[PASS] Corrupted File Recovery")
        print("[PASS] DataLoader")
        print("[PASS] Normalization")
        print("\nAll Tests Passed")
    else:
        print("[FAIL] Some Tests Failed")
    print("=" * 40 + "\n")
