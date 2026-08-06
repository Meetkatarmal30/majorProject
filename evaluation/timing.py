"""
timing.py - Sentinel-1 (SAR) Retrieval Timing and Latency Utility.

This module provides benchmarking functions to measure the throughput and latency of
the retrieval pipeline, including model loading, single-image inference, batch inference,
embedding generation, and query retrieval execution.

Supports GPU synchronization (CUDA) and CPU fallback timing.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import BASE_DIR, setup_logging

logger = setup_logging("TimingUtility")


def get_sync_time() -> float:
    """
    Utility to get current time, synchronizing CUDA operations first if GPU is active.
    This guarantees accurate GPU timing instead of measuring CPU kernel launch queues.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def measure_inference_timing(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device],
) -> Dict[str, float]:
    """
    Measures batch and single-image inference latencies and throughput for a trained model.

    Args:
        model: Trained PyTorch encoder model.
        dataloader: DataLoader containing images to run inference on.
        device: Device to run benchmarking on ('cpu' or 'cuda').

    Returns:
        Dict[str, float]: Metrics containing single-image latency, batch latency, and throughput.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Model must be an instance of torch.nn.Module.")

    if not isinstance(dataloader, DataLoader) or len(dataloader) == 0:
        raise ValueError("DataLoader must be non-empty and of type torch.utils.data.DataLoader.")

    device = torch.device(device)
    model.to(device)
    model.eval()

    total_samples = 0
    batch_latencies = []
    single_latencies = []

    logger.info(f"Running inference benchmarking on device: {device}...")

    # Warm-up run (especially important for CUDA/GPU graph setup)
    logger.info("Performing warm-up iteration...")
    try:
        sample_batch = next(iter(dataloader))
        # Handle tuple output (images, patch_names) from BigEarthNetS1Dataset
        inputs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)
    except Exception as e:
        logger.error(f"Inference warm-up failed: {e}")
        raise

    # Start timing benchmark
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device)
            b_size = inputs.size(0)
            total_samples += b_size

            # Measure batch inference
            t_start = get_sync_time()
            _ = model(inputs)
            t_end = get_sync_time()
            
            batch_latencies.append((t_end - t_start) * 1000.0)  # ms

            # Measure single-image latency (average per-sample inside the batch)
            # We also benchmark actual single-item loading to get true single latency
            t_single_start = get_sync_time()
            for idx in range(min(5, b_size)):  # Benchmark first 5 items individually to save time
                single_input = inputs[idx : idx + 1]
                _ = model(single_input)
            t_single_end = get_sync_time()
            single_latencies.append(((t_single_end - t_single_start) * 1000.0) / min(5, b_size))

    avg_batch_ms = float(np.mean(batch_latencies))
    avg_single_ms = float(np.mean(single_latencies))
    
    # Throughput: Samples per second
    total_elapsed_sec = sum(batch_latencies) / 1000.0
    throughput = total_samples / total_elapsed_sec if total_elapsed_sec > 0 else 0.0

    return {
        "batch_inference_ms": avg_batch_ms,
        "single_inference_ms": avg_single_ms,
        "throughput_samples_per_second": throughput,
        "total_samples_benchmarked": total_samples,
    }


def measure_retrieval_timing(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    retrieval_fn: Callable[[np.ndarray, np.ndarray], Any],
) -> Dict[str, float]:
    """
    Measures the execution latency of a retrieval database search function.

    Args:
        query_embeddings: Query vectors of shape (N_queries, D).
        database_embeddings: Reference database vectors of shape (N_db, D).
        retrieval_fn: Callable search function taking (queries, database) as arguments.

    Returns:
        Dict[str, float]: Metrics containing query retrieval timing.
    """
    if not isinstance(query_embeddings, np.ndarray) or not isinstance(database_embeddings, np.ndarray):
        raise TypeError("Embeddings must be numpy ndarrays.")

    if query_embeddings.size == 0 or database_embeddings.size == 0:
        raise ValueError("Embedding matrices cannot be empty.")

    if not callable(retrieval_fn):
        raise TypeError("Retrieval function must be a callable search routine.")

    logger.info("Benchmarking database retrieval query latency...")

    # Warm-up search
    try:
        _ = retrieval_fn(query_embeddings[:1], database_embeddings)
    except Exception as e:
        logger.error(f"Retrieval function verification failed: {e}")
        raise

    # Benchmark search queries
    n_queries = query_embeddings.shape[0]
    t_start = time.perf_counter()
    _ = retrieval_fn(query_embeddings, database_embeddings)
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0
    avg_query_ms = elapsed_ms / n_queries if n_queries > 0 else 0.0

    return {
        "total_retrieval_time_ms": elapsed_ms,
        "average_query_latency_ms": avg_query_ms,
    }


def run_complete_benchmark(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    retrieval_fn: Callable[[np.ndarray, np.ndarray], Any],
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Aggregates all timing statistics (inference, load, query search, latency, throughput)
    into a structured dictionary and optionally saves to JSON.

    Args:
        model: Trained model.
        dataloader: DataLoader.
        device: Device target ('cpu' or 'cuda').
        retrieval_fn: Callable search function.
        query_embeddings: Query embeddings.
        database_embeddings: Reference database embeddings.
        save_path: Optional JSON output path.

    Returns:
        Dict[str, Any]: Structured dictionary of timing metrics.
    """
    logger.info("Initializing complete retrieval pipeline benchmark...")
    
    # Measure model loading/transfer overhead
    t_load_start = get_sync_time()
    model.to(torch.device(device))
    t_load_end = get_sync_time()
    model_load_ms = (t_load_end - t_load_start) * 1000.0

    # Measure inference
    inf_metrics = measure_inference_timing(model, dataloader, device)

    # Measure retrieval search
    ret_metrics = measure_retrieval_timing(query_embeddings, database_embeddings, retrieval_fn)

    results = {
        "device": str(device),
        "model_load_time_ms": model_load_ms,
        "single_inference_ms": inf_metrics["single_inference_ms"],
        "batch_inference_ms": inf_metrics["batch_inference_ms"],
        "throughput_samples_per_second": inf_metrics["throughput_samples_per_second"],
        "total_retrieval_time_ms": ret_metrics["total_retrieval_time_ms"],
        "average_query_latency_ms": ret_metrics["average_query_latency_ms"],
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Benchmark results saved successfully to: {save_path}")

    return results


if __name__ == "__main__":
    logger.info("Starting timing utility self-test...")

    # Define a simple placeholder PyTorch module for API verification
    # NOTE: This is a placeholder for benchmarking demonstration and is NOT the project retrieval model.
    class PlaceholderEncoder(nn.Module):
        def __init__(self, in_features: int = 2, out_features: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features * 120 * 120, out_features)
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # Instantiate placeholder model
    dummy_model = PlaceholderEncoder()

    # Generate a dummy dataloader
    # Mock dataset yielding (tensor, name)
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return torch.randn(2, 120, 120), f"patch_{idx}"

    dummy_loader = DataLoader(DummyDataset(), batch_size=10, shuffle=False)

    # Define a placeholder retrieval search function (simple matrix dot product search)
    # NOTE: This is a placeholder routine and does NOT use FAISS or model indexing.
    def placeholder_retrieval(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
        # Compute dot product similarity matrix (N_queries, N_database)
        similarity = np.dot(queries, database.T)
        # Find index of top-1 match for each query
        top1_indices = np.argmax(similarity, axis=1)
        return top1_indices

    # Generate mock embeddings
    mock_queries = np.random.randn(50, 64).astype(np.float32)
    mock_database = np.random.randn(500, 64).astype(np.float32)

    device_target = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Targeting device for self-test: {device_target}")

    out_json = BASE_DIR / "outputs" / "timing_results.json"

    try:
        timing_stats = run_complete_benchmark(
            model=dummy_model,
            dataloader=dummy_loader,
            device=device_target,
            retrieval_fn=placeholder_retrieval,
            query_embeddings=mock_queries,
            database_embeddings=mock_database,
            save_path=out_json
        )

        print("\n========================================")
        print("Timing Utility Self-Test Successful!")
        print("========================================")
        print(f"Device Tested:        {timing_stats['device']}")
        print(f"Model Load Time:      {timing_stats['model_load_time_ms']:.4f} ms")
        print(f"Single Inference:     {timing_stats['single_inference_ms']:.4f} ms")
        print(f"Batch Inference:      {timing_stats['batch_inference_ms']:.4f} ms")
        print(f"Throughput:           {timing_stats['throughput_samples_per_second']:.2f} samples/sec")
        print(f"Total Retrieval Time: {timing_stats['total_retrieval_time_ms']:.4f} ms")
        print(f"Avg Query Latency:    {timing_stats['average_query_latency_ms']:.4f} ms/query")
        print(f"Results saved to:     outputs/timing_results.json")
        print("========================================\n")
    except Exception as e:
        logger.exception(f"Timing self-test failed: {e}")
