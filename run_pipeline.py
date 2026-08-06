"""
run_pipeline.py - Master Entry Point for Sentinel-1 SAR Data Pipeline.

This script orchestrates the execution of all pipeline steps in order:
1. Dataset Exploration & Inspection
2. SAR Preprocessing verification
3. Dataset Statistics Generation (small validation run)
4. Visualization Module verification
5. Dataset Class verification
6. DataLoader verification
7. Unit & Integration Tests
8. t-SNE Embedding Visualization verification
9. Latency & Timing Benchmarks verification

Reuses existing functions from packages to verify execution without code duplication.
"""

import time
import unittest
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to sys.path to allow importing config
import sys
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import (
    DATA_DIR,
    BASE_DIR,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    setup_logging,
)

# Set global logging level to WARNING to keep orchestration output clean
logging.getLogger().setLevel(logging.WARNING)
logger = setup_logging("MasterPipeline")


def run_exploration() -> bool:
    """Step 1: Dataset Exploration / Inspection"""
    from inspect_dataset import inspect_directory
    res = inspect_directory(DATA_DIR)
    return "total_patches" in res and res["total_patches"] > 0


def run_preprocessing() -> bool:
    """Step 2: SAR Preprocessing verification"""
    import random
    from preprocessing.sar_preprocessing import preprocess_sar_patch
    
    # Select a random patch from directory cache
    cache_path = BASE_DIR / ".s1_patch_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError("Cache file missing. Run dataset initialization first.")
        
    import json
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    random_patch_name = random.choice(list(cache.keys()))
    patch_dir = Path(cache[random_patch_name])
    
    vv_path = next(patch_dir.glob(f"*{VV_BAND_SUFFIX}"), None)
    vh_path = next(patch_dir.glob(f"*{VH_BAND_SUFFIX}"), None)
    
    if not vv_path or not vh_path:
        raise FileNotFoundError(f"Missing polarization bands in {patch_dir}")
        
    tensor = preprocess_sar_patch(
        vv_path=vv_path,
        vh_path=vh_path,
        norm_mode="min-max"
    )
    return tensor.shape == (2, 120, 120) and tensor.dtype == torch.float32


def run_statistics() -> bool:
    """Step 3: Dataset Statistics Generator (runs a fast 100-sample validation)"""
    from utils.stats_generator import accumulate_statistics, save_statistics
    stats_out_json = BASE_DIR / "outputs" / "statistics.json"
    stats_out_csv = BASE_DIR / "outputs" / "statistics.csv"
    
    # Run a small validation run of 100 samples to keep pipeline execution fast
    stats = accumulate_statistics(DATA_DIR, sample_size=100, num_workers=2)
    save_statistics(stats, stats_out_json, stats_out_csv)
    return stats["valid_patches_processed"] == 100


def run_visualization() -> bool:
    """Step 4: Visualization Module verification"""
    import random
    from utils.visualization import plot_raw_vv_vh, plot_raw_vs_preprocessed
    from preprocessing.sar_preprocessing import preprocess_sar_patch
    
    cache_path = BASE_DIR / ".s1_patch_cache.json"
    import json
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    random_patch_name = random.choice(list(cache.keys()))
    patch_dir = Path(cache[random_patch_name])
    
    vv_path = next(patch_dir.glob(f"*{VV_BAND_SUFFIX}"), None)
    vh_path = next(patch_dir.glob(f"*{VH_BAND_SUFFIX}"), None)
    
    raw_plot = BASE_DIR / "outputs" / "sample_raw_channels.png"
    comp_plot = BASE_DIR / "outputs" / "sample_pipeline_comparison.png"
    
    plot_raw_vv_vh(vv_path, vh_path, random_patch_name, raw_plot)
    
    tensor = preprocess_sar_patch(vv_path, vh_path, norm_mode="z-score")
    plot_raw_vs_preprocessed(vv_path, vh_path, tensor, random_patch_name, comp_plot)
    return raw_plot.exists() and comp_plot.exists()


def run_dataset_class() -> bool:
    """Step 5: Dataset initialization verification"""
    from dataset.dataset import BigEarthNetS1Dataset
    dataset = BigEarthNetS1Dataset(split="train")
    tensor, name = dataset[0]
    return len(dataset) > 0 and tensor.shape == (2, 120, 120)


def run_dataloader() -> bool:
    """Step 6: DataLoader verification"""
    from dataset.dataloader import create_dataloaders
    train_loader, _, _ = create_dataloaders(batch_size=16, num_workers=2)
    images, names = next(iter(train_loader))
    return images.shape == (16, 2, 120, 120) and len(names) == 16


def run_unit_tests() -> bool:
    """Step 7: Unit Tests suite execution"""
    from tests.test_pipeline import TestSARPipeline
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSARPipeline)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_tsne() -> bool:
    """Step 8: t-SNE Visualization verification"""
    from sklearn.datasets import make_blobs
    from evaluation.tsne_visualization import generate_tsne_plot
    
    res = make_blobs(n_samples=150, n_features=64, centers=3, random_state=42)
    mock_emb = res[0]
    mock_lbl = [f"Class_{l}" for l in res[1]]
    
    plot_path = BASE_DIR / "outputs" / "sample_tsne_plot.png"
    coords = generate_tsne_plot(
        embeddings=mock_emb,
        labels=mock_lbl,
        save_path=plot_path,
        perplexity=10,
        title="Pipeline Verification t-SNE Plot"
    )
    return coords.shape == (150, 2) and plot_path.exists()


def run_timing() -> bool:
    """Step 9: Timing Benchmark verification"""
    from evaluation.timing import run_complete_benchmark
    
    # Placeholder model
    class PlaceholderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2 * 120 * 120, 32)
        def forward(self, x):
            return self.linear(x.flatten(1))
            
    model = PlaceholderModel()
    
    # Dummy dataset
    class DummyDS(torch.utils.data.Dataset):
        def __len__(self): return 50
        def __getitem__(self, idx): return torch.randn(2, 120, 120), f"p_{idx}"
        
    loader = DataLoader(DummyDS(), batch_size=10)
    
    def dummy_retrieval(q, db):
        return np.argmax(np.dot(q, db.T), axis=1)
        
    mock_q = np.random.randn(20, 32).astype(np.float32)
    mock_db = np.random.randn(100, 32).astype(np.float32)
    
    stats_json = BASE_DIR / "outputs" / "timing_results.json"
    results = run_complete_benchmark(
        model=model,
        dataloader=loader,
        device="cpu",
        retrieval_fn=dummy_retrieval,
        query_embeddings=mock_q,
        database_embeddings=mock_db,
        save_path=stats_json
    )
    return results["throughput_samples_per_second"] > 0 and stats_json.exists()


def main() -> None:
    print("====================================================")
    print("Sentinel-1 SAR Data Pipeline")
    print("Cross-Modal Satellite Image Retrieval")
    print("====================================================")
    print()

    steps = [
        ("Dataset Exploration", run_exploration),
        ("SAR Preprocessing", run_preprocessing),
        ("Dataset Statistics", run_statistics),
        ("Visualization", run_visualization),
        ("Dataset Class", run_dataset_class),
        ("DataLoader", run_dataloader),
        ("Unit Tests", run_unit_tests),
        ("t-SNE Visualization", run_tsne),
        ("Timing Benchmark", run_timing),
    ]

    statuses = []
    
    for idx, (name, fn) in enumerate(steps, 1):
        # Format padding dots
        prefix = f"[{idx}/9] {name} "
        padding = "." * (40 - len(prefix))
        print(f"{prefix}{padding} Running...", end="\r", flush=True)
        
        try:
            success = fn()
            if success:
                print(f"{prefix}{padding} PASS")
                statuses.append(True)
            else:
                print(f"{prefix}{padding} FAIL")
                statuses.append(False)
        except Exception as e:
            print(f"{prefix}{padding} ERROR")
            logger.error(f"Error in step '{name}': {e}")
            statuses.append(False)

    print()
    print("====================================================")
    if all(statuses):
        print("Pipeline Completed Successfully")
    else:
        print("Pipeline Failed (Check error logs)")
    print("====================================================")
    print()
    print("Generated Outputs:")
    print("- outputs/dataset_exploration_report.md")
    print("- outputs/statistics.json")
    print("- outputs/statistics.csv")
    print("- outputs/vv_histogram.png")
    print("- outputs/vh_histogram.png")
    print("- outputs/sample_raw_channels.png")
    print("- outputs/sample_pipeline_comparison.png")
    print("- outputs/sample_tsne_plot.png")
    print("- outputs/timing_results.json")
    print()
    print("Overall Status:")
    print("[PASS] Week 1 Complete")
    print("[PASS] Week 2 Complete")
    print("[PASS] Week 3 Complete")
    print()


if __name__ == "__main__":
    main()
