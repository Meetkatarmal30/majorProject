"""
test_pipeline.py - Automated Unit & Integration Tests for SAR and Paired Data Pipeline.

This test suite validates:
1. Dataset initialization (caching, length, splits).
2. __getitem__ outputs (tensor shapes, dtypes, ranges, no NaN/Inf).
3. Cache load performance speedups.
4. Deterministic split management (train + val + test total size).
5. Corruption/missing file recovery (logs warning, retries, loads valid sample).
6. DataLoader batch loading and shape correctness.
7. Normalization modes application.
8. BigEarthNetLabelEncoder (deterministic 19-class multi-hot encoding).
9. PairedBigEarthNetDataset error handling when S2 is unavailable and required.
10. PairedBigEarthNetDataset safe S1 loading (returns valid SAR and 19-class labels).
11. Paired DataLoader batch generation for train and validation splits.

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
from dataset.dataset import (
    BigEarthNetS1Dataset,
    PairedBigEarthNetDataset,
    BigEarthNetLabelEncoder,
    BIGEARTHNET_19_CLASSES,
)
from dataset.dataloader import (
    create_dataloaders,
    create_paired_dataloaders,
)


class TestSARPipeline(unittest.TestCase):
    """
    Automated test suite for Sentinel-1 dataset and paired dataloader components.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup cache file path and check dataset directory.
        """
        cls.cache_path = BASE_DIR / ".s1_patch_cache.json"
        cls.train_csv = BASE_DIR / "teammate_inputs" / "train_split.csv"
        cls.val_csv = BASE_DIR / "teammate_inputs" / "val_split.csv"

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
        Test 3: Verify cache file exists and loads quickly.
        """
        self.assertTrue(self.cache_path.exists(), "Cache file should exist")
        start_time = time.time()
        dataset_warm = BigEarthNetS1Dataset(split="train")
        time_warm = time.time() - start_time
        self.assertTrue(len(dataset_warm) > 0)
        self.assertTrue(time_warm < 10.0, f"Warm cache initialization should be fast: {time_warm:.4f}s")

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

        original_name = dataset.patch_names[0]
        original_dir = dataset.patch_dict[original_name]

        dataset.patch_names[0] = "Corrupted_Patch_Mock_Name"
        dataset.patch_dict["Corrupted_Patch_Mock_Name"] = str(BASE_DIR / "non_existent_folder_xyz")

        try:
            tensor, patch_name = dataset[0]
            self.assertNotEqual(patch_name, "Corrupted_Patch_Mock_Name")
            self.assertEqual(tensor.shape, (2, 224, 224))
        finally:
            dataset.patch_names[0] = original_name
            dataset.patch_dict[original_name] = original_dir
            if "Corrupted_Patch_Mock_Name" in dataset.patch_dict:
                del dataset.patch_dict["Corrupted_Patch_Mock_Name"]

    def test_06_dataloader(self) -> None:
        """
        Test 6: Verify PyTorch DataLoader configurations and batch dimensions.
        """
        batch_size = 16
        train_loader, _, _ = create_dataloaders(batch_size=batch_size, num_workers=0)

        self.assertIsInstance(train_loader, DataLoader)
        self.assertEqual(train_loader.batch_size, batch_size)

        batch_iter = iter(train_loader)
        images, patch_names = next(batch_iter)

        self.assertEqual(images.shape, (batch_size, 2, 224, 224))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(len(patch_names), batch_size)

    def test_07_normalization_verification(self) -> None:
        """
        Test 7: Verify that preprocessing is applied and normalized output is valid.
        """
        dataset = BigEarthNetS1Dataset(split="train", norm_mode="min-max")
        tensor, _ = dataset[0]

        self.assertTrue(tensor.min() >= 0.0, f"Min-Max normalized min value should be >= 0. Got: {tensor.min()}")
        self.assertTrue(tensor.max() <= 1.0, f"Min-Max normalized max value should be <= 1. Got: {tensor.max()}")
        self.assertFalse(torch.isnan(tensor).any(), "Normalized tensor contains NaN")
        self.assertFalse(torch.isinf(tensor).any(), "Normalized tensor contains Inf")

    def test_08_label_encoder(self) -> None:
        """
        Test 8: Verify BigEarthNetLabelEncoder deterministic 19-class multi-hot encoding.
        """
        encoder = BigEarthNetLabelEncoder()
        self.assertEqual(encoder.num_classes, 19)
        self.assertEqual(len(encoder.classes), 19)

        # Single class string representation
        target1 = encoder.encode("['Arable land']")
        self.assertEqual(target1.shape, (19,))
        self.assertEqual(target1.dtype, torch.float32)
        self.assertEqual(target1.sum().item(), 1.0)
        self.assertEqual(target1[encoder.class_to_idx["Arable land"]].item(), 1.0)

        # Multi-class numpy-style string
        target2 = encoder.encode("['Arable land' 'Complex cultivation patterns' 'Pastures']")
        self.assertEqual(target2.shape, (19,))
        self.assertEqual(target2.sum().item(), 3.0)
        self.assertEqual(target2[encoder.class_to_idx["Arable land"]].item(), 1.0)
        self.assertEqual(target2[encoder.class_to_idx["Complex cultivation patterns"]].item(), 1.0)
        self.assertEqual(target2[encoder.class_to_idx["Pastures"]].item(), 1.0)

        # Empty string / list
        target3 = encoder.encode("[]")
        self.assertEqual(target3.sum().item(), 0.0)

    def test_09_paired_dataset_s2_unavailable_error(self) -> None:
        """
        Test 9: Verify that PairedBigEarthNetDataset raises FileNotFoundError
        when require_s2=True and s2_root is unavailable, preventing fake tensors.
        """
        # Should raise when s2_root is None and require_s2=True
        with self.assertRaises(FileNotFoundError):
            PairedBigEarthNetDataset(
                csv_path=self.train_csv,
                s2_root=None,
                require_s2=True,
            )

        # Should raise when s2_root is non-existent directory and require_s2=True
        with self.assertRaises(FileNotFoundError):
            PairedBigEarthNetDataset(
                csv_path=self.train_csv,
                s2_root=BASE_DIR / "non_existent_s2_folder",
                require_s2=True,
            )

    def test_10_paired_dataset_s1_mode(self) -> None:
        """
        Test 10: Verify PairedBigEarthNetDataset in safe S1 verification mode (require_s2=False).
        """
        dataset = PairedBigEarthNetDataset(
            csv_path=self.train_csv,
            require_s2=False,
        )

        self.assertEqual(len(dataset), 21000)

        # Retrieve first sample
        sar_tensor, ms_tensor, label_tensor, patch_id = dataset[0]

        # SAR Checks
        self.assertIsInstance(sar_tensor, torch.Tensor)
        self.assertEqual(sar_tensor.shape, (2, 224, 224))
        self.assertEqual(sar_tensor.dtype, torch.float32)
        self.assertFalse(torch.isnan(sar_tensor).any(), "SAR tensor contains NaN")
        self.assertFalse(torch.isinf(sar_tensor).any(), "SAR tensor contains Inf")
        self.assertFalse((sar_tensor == 0).all(), "SAR tensor is all zeros")

        # S2 is None (no fake data produced)
        self.assertIsNone(ms_tensor, "ms_tensor should be None when S2 is unavailable and require_s2=False")

        # Label Checks
        self.assertIsInstance(label_tensor, torch.Tensor)
        self.assertEqual(label_tensor.shape, (19,))
        self.assertEqual(label_tensor.dtype, torch.float32)
        self.assertTrue(label_tensor.sum().item() >= 1.0, "Sample should have at least one active class")

        # Patch ID Check
        self.assertIsInstance(patch_id, str)
        self.assertTrue(len(patch_id) > 0)

    def test_11_paired_dataloader_batch(self) -> None:
        """
        Test 11: Verify create_paired_dataloaders batch loading and collation.
        """
        train_loader, val_loader = create_paired_dataloaders(
            batch_size=16,
            num_workers=0,
            require_s2=False,
        )

        self.assertEqual(len(train_loader.dataset), 21000)
        self.assertEqual(len(val_loader.dataset), 4500)

        # Test 1 training batch
        train_iter = iter(train_loader)
        sar_b, ms_b, label_b, p_ids = next(train_iter)

        self.assertEqual(sar_b.shape, (16, 2, 224, 224))
        self.assertEqual(sar_b.dtype, torch.float32)
        self.assertIsNone(ms_b)
        self.assertEqual(label_b.shape, (16, 19))
        self.assertEqual(label_b.dtype, torch.float32)
        self.assertEqual(len(p_ids), 16)
        self.assertFalse(torch.isnan(sar_b).any())
        self.assertFalse(torch.isinf(sar_b).any())

        # Test 1 validation batch
        val_iter = iter(val_loader)
        v_sar_b, v_ms_b, v_label_b, v_p_ids = next(val_iter)

        self.assertEqual(v_sar_b.shape, (16, 2, 224, 224))
        self.assertEqual(v_sar_b.dtype, torch.float32)
        self.assertIsNone(v_ms_b)
        self.assertEqual(v_label_b.shape, (16, 19))
        self.assertEqual(v_label_b.dtype, torch.float32)
        self.assertEqual(len(v_p_ids), 16)
        self.assertFalse(torch.isnan(v_sar_b).any())
        self.assertFalse(torch.isinf(v_sar_b).any())


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running Full Automated Test Suite")
    print("=" * 50)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestSARPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print(f"[PASS] All {result.testsRun} Tests Passed Successfully!")
    else:
        print(f"[FAIL] {len(result.failures)} Failures, {len(result.errors)} Errors")
    print("=" * 50 + "\n")
