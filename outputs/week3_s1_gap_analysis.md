# Sentinel-1 (SAR) Pipeline: Week 3 Gap Analysis Report

This report provides a detailed inspection of the current Sentinel-1 pipeline implementation against the latest Week 3 project requirements. No code changes have been made to the repository.

---

## A. Existing S1 Components Found
The following Sentinel-1 pipeline components were identified in the project workspace:
1. **`config/settings.py`**: Centralized configuration parameters including default paths, band suffixes, clipping ranges, default batch sizes, and logging setups.
2. **`preprocessing/sar_preprocessing.py`**: Preprocessing functions including band loading (using Rasterio or PIL), outlier clipping, Z-score and Min-Max normalization, and stacking into PyTorch tensors.
3. **`dataset/dataset.py`**: Implements `BigEarthNetS1Dataset` with support for scanning the filesystem, JSON directory caching (`.s1_patch_cache.json`), and dynamic train/validation/test split generation.
4. **`dataset/dataloader.py`**: Contains `create_dataloaders` to instantiate train, validation, and test PyTorch `DataLoader` instances with pinned memory and multi-worker loading.
5. **`utils/stats_generator.py`**: Statistics generator that computes min, max, mean, standard deviation, and histograms of radar backscatter over a subset of patches using parallel worker processes.
6. **`evaluation/tsne_visualization.py`**: Utility to reduce high-dimensional embeddings to 2D using t-SNE and generate scatter plots.
7. **`evaluation/timing.py`**: Latency and throughput benchmarking tool that profiles model loading, single-image inference, batch inference, and similarity retrieval speeds.
8. **`tests/test_pipeline.py`**: Suite of 7 automated unit and integration tests verifying dataset caching, shapes, recovery from corrupted files, and normalization.
9. **`run_pipeline.py`**: Master orchestration script that verifies each step of the pipeline.
10. **`teammate_inputs/`**: Contains baseline validation inputs from teammates:
    * `baseline_val_ms_embeddings.npy` (4,500 validation multispectral embeddings)
    * `baseline_val_patch_ids.txt` (4,500 validation patch IDs)
    * `val_split.csv` (label metadata and mappings for the 4,500 validation patches)

---

## B. What is Already COMPLETE
* **Single-Modal S1 Data Ingestion**: The `BigEarthNetS1Dataset` class is fully functional for loading and caching Sentinel-1 patches.
* **Basic Preprocessing Operations**: The math for loading TIFF files, clipping backscatter values, normalizations, and stacking into a 2-channel `(2, H, W)` tensor is complete.
* **DataLoader Infrastructure**: Multi-process loading, memory pinning, and batching are implemented for S1.
* **Downstream Evaluation Utilities**: Both the `tsne_visualization` and `timing` modules are fully complete and validated using structured mock data.
* **Test Coverage**: Basic S1 dataset integration and preprocessing unit tests are in place.

---

## C. What is PARTIALLY COMPLETE
* **S1/SAR Statistics Generation**: The script `utils/stats_generator.py` exists but was only run for a demo of **100 validation patches** (saved in `outputs/statistics.json`). It has **not** been run on the training patches.
* **SAR Preprocessing Parameters**: Preprocessing functions support clipping and resizing, but the default settings (e.g., resizing to `(120, 120)` or preserving resolution) do not match the required target shape `(2, 224, 224)`.
* **DataLoaders**: PyTorch dataloaders can be instantiated at batch size 64, but they only output single-modal S1 batches `(sar_tensor, patch_name)` instead of paired multimodal batches.

---

## D. What is MISSING
* **`s1_stats.json`**: The statistics file calculated over the training set does not exist. The current statistics are stored in `outputs/statistics.json` and are incorrect (based on only 100 patches).
* **`PairedBigEarthNetDataset`**: This required class does **not** exist. The dataset package only contains `BigEarthNetS1Dataset` (S1-only).
* **S2/MS Training Data**: Raw S2/MS images and pre-computed training embeddings (`baseline_train_ms_embeddings.npy`) are missing from the workspace. Only validation embeddings are present.
* **Validation Split Isolation**: The dataset script lacks logic to read `teammate_inputs/baseline_val_patch_ids.txt` and exclude these 4,500 validation patches from the training loader. Currently, using the dynamic 70% train split results in **severe data leakage** because the validation patches are mixed into the training set.
* **S1 Encoder Architecture**: No S1 PyTorch model definition file (e.g., `models/resnet.py`) exists in the codebase.
* **Training Script and Epoch Test**: No script exists to run a training epoch or verify training-epoch pipeline readiness.
* **Training Epoch Timing**: No utility is implemented to profile training loop bottlenecks (e.g., data loading lag vs. backpropagation time).

---

## E. Exact Differences: Current Code vs. Week 3 Requirements

| Dimension | Current Code | Week 3 Requirements | Gap / Required Action |
| :--- | :--- | :--- | :--- |
| **Statistics File** | `outputs/statistics.json` (100 patches scanned) | `s1_stats.json` (21,000 training patches scanned) | **Stats are incorrect**. Must compute new stats on 21,000 training patches and write to `s1_stats.json`. |
| **VV Clip Range** | `[-25.0, 0.0]` dB | `[-25.0, 0.0]` dB | Matches requirement. |
| **VH Clip Range** | `[-32.0, -5.0]` dB | `[-25.0, 0.0]` dB | **Mismatch**. Must update VH clipping bounds to match VV. |
| **Output Shape** | `(2, 120, 120)` (preserves resolution) | `(2, 224, 224)` | **Mismatch**. Must force resize to `224x224` during preprocessing. |
| **Normalization Stats** | Hardcoded stats in `settings.py` | Loaded from `s1_stats.json` | **Mismatch**. Must load dynamic stats from `s1_stats.json` into preprocessing. |
| **Dataset Class** | `BigEarthNetS1Dataset` | `PairedBigEarthNetDataset` | **Missing**. Must implement paired dataset returning `(sar_tensor, ms_tensor, label_tensor, patch_id)`. |
| **Multimodal Pairing** | Missing (S1-only) | S1 and S2 paired using CSV split | **Missing**. Must pair S1 and S2 using `val_split.csv` / teammate split files. |
| **Validation Isolation** | None (dynamic splits lead to data leakage) | Validation patches isolated using `baseline_val_patch_ids.txt` | **Missing**. Must filter out validation patches from the training set. |
| **DataLoaders** | Yields `(sar_tensor, patch_name)` | Yields `(sar_tensor, ms_tensor, label_tensor, patch_id)` | **Missing**. Must update dataloaders to support the paired dataset structure. |

---

## F. Whether `s1_stats.json` is Correct and How it Was Generated
* **Status**: **NOT correct**.
* **Reasoning**: The file `s1_stats.json` does not exist. The existing `outputs/statistics.json` was generated by `run_pipeline.py` scanning a tiny subset of **100 random validation patches** to run quick verification tests. Its stats do not represent the training set.

---

## G. Whether SAR Preprocessing Produces `(2, 224, 224)`
* **Status**: **NO**.
* **Reasoning**: It currently produces `(2, 120, 120)` tensors. `config/settings.py` sets `PRESERVE_RESOLUTION = True`, which bypasses the resizing step and outputs the raw Sentinel-1 patch dimensions (120x120 pixels).

---

## H. Whether `PairedBigEarthNetDataset` Satisfies the Required Interface
* **Status**: **NO**.
* **Reasoning**: The class is entirely **missing** from the workspace. It needs to be implemented to ingest both S1 imagery and S2 data (or S2 embeddings), load labels from the CSV split, and return the four-element tuple `(sar_tensor, ms_tensor, label_tensor, patch_id)`.

---

## I. Whether DataLoaders Satisfy the Batch-Size-64 Requirement
* **Status**: **NO**.
* **Reasoning**: While `settings.py` lists `DEFAULT_BATCH_SIZE = 64`, the dataloader function `create_dataloaders` is hardcoded to load `BigEarthNetS1Dataset` (S1-only) and yields `(sar_tensor, patch_name)` instead of the paired batches.

---

## J. Whether the Pipeline is Ready for Fateh's Two-Tower Training
* **Status**: **NOT READY**.
* **Critical Issues**:
  1. The paired dataset class `PairedBigEarthNetDataset` is missing.
  2. The S2/MS training data or embeddings are not present in the workspace, meaning joint training cannot run with real training data.
  3. Validation patch IDs are not filtered out of the training split, which would leak validation data into training.
  4. There is no S1 model encoder architecture or training script in the codebase.

---

## K. Exact Files That Would Need Modification
To resolve the gaps without breaking existing single-modal components, the following modifications are required:
1. **`config/settings.py`**:
   * Update `VH_CLIP_RANGE` to `(-25.0, 0.0)` to match the VV clip requirement.
   * Add configuration variables pointing to `outputs/s1_stats.json` and teammate inputs.
2. **`preprocessing/sar_preprocessing.py`**:
   * Update `preprocess_sar_patch` to dynamically load mean and std values from `s1_stats.json` if available.
   * Force default resizing to `(224, 224)` and set `preserve_res=False` for model integration.
3. **`dataset/dataset.py`**:
   * Add `PairedBigEarthNetDataset` inheriting from `Dataset`.
   * Add logic in `_manage_splits` to load `teammate_inputs/baseline_val_patch_ids.txt` and exclude those patches from the training set.
   * Add labels parser to extract label vectors from `val_split.csv` (using multi-hot encoding).
4. **`dataset/dataloader.py`**:
   * Update `create_dataloaders` (or add `create_paired_dataloaders`) to instantiate the paired dataset and return paired loaders with batch size 64.
5. **[NEW] `outputs/s1_stats.json`**:
   * Generate this statistics file by running the stats generator on the 21,000 training patches.
6. **[NEW] `models/resnet.py`**:
   * Create the model architecture file defining the 2-channel ResNet50 encoder.
7. **[NEW] `train_epoch_test.py`**:
   * Implement a test script to perform 1 training epoch (using dummy S2 training embeddings to bypass missing S2 data) and output timing measurements.

---

## L. Exact Commands That Should Be Run Later
After approval to execute changes, these are the commands that must be run to complete the requirements:

1. **Calculate S1 Statistics over 21,000 Training Patches**:
   ```bash
   python -c "from utils.stats_generator import accumulate_statistics, save_statistics; from config.settings import DATA_DIR, BASE_DIR; stats = accumulate_statistics(DATA_DIR, sample_size=21000, num_workers=4); save_statistics(stats, BASE_DIR / 'outputs' / 's1_stats.json', BASE_DIR / 'outputs' / 's1_stats.csv')"
   ```
   *(Note: Logic will be integrated to exclude the 4,500 validation patches from this sample)*.

2. **Execute Unit Tests to Verify Preprocessing and Data Loading**:
   ```bash
   python -m unittest tests/test_pipeline.py
   ```

3. **Run the Training Epoch Readiness Test**:
   ```bash
   python train_epoch_test.py
   ```

---

## M. Estimated Computational/Disk Impact of Pending Operations

### 1. S1 Statistics Calculation over 21,000 patches
* **Files to access**: 42,000 GeoTIFF files (21,000 patches × 2 polarizations: VV and VH).
* **Disk read quantity**: ~2.5 GB of raw data.
* **Disk I/O impact**: High disk seek rate due to reading thousands of small files. On a standard HDD, this can cause Windows Explorer to hang or lag. On an SSD, it will read efficiently.
* **CPU usage**: High (100% of 4 cores) for the duration of the calculation.
* **RAM usage**: Low (< 200 MB) as statistics are accumulated in running counts and not kept in memory.
* **Execution time**: ~1 to 2 minutes on an SSD; ~10 to 15 minutes on an HDD.
* **Safety assessment**: **Safe** to run on the user's PC because it uses read-only access and can be run with fewer CPU workers (e.g., `num_workers=2`) to prevent PC freezing/lagging.

### 2. Training-Epoch Readiness Test (1 Epoch over 21,000 patches)
* **Files to access**: 42,000 GeoTIFF files.
* **Disk read quantity**: ~2.5 GB of raw data.
* **Disk I/O impact**: Continuous high disk read activity.
* **CPU/GPU usage**: Very high. CPU will handle dataloader workers and resizing; GPU (if CUDA is used) will run the encoder forward pass.
* **RAM usage**: Moderate (~2 to 4 GB depending on dataloader worker processes and batch buffer sizes).
* **Execution time**: ~2 to 5 minutes on GPU; ~15 to 30 minutes on CPU.
* **Safety assessment**: Running a full epoch of 21,000 patches just for a test is computationally expensive. **Recommendation**: We should first run a mock epoch test over only **5 to 10 batches** (approx. 320-640 patches) to verify pipeline integration (shapes, loss gradients, timing) before running the full 21,000-patch epoch. This reduces I/O and CPU/GPU load to seconds.

---

## N. Final Week 3 S1 Readiness Status
**PARTIALLY READY**

* **Why**: The retrieval evaluation utilities (t-SNE and timing profiling) are complete and correct. However, the data ingestion pipeline is **NOT READY** for model training due to:
  1. Mismatch in clipping bounds (VH clipped to -32/-5 instead of -25/0 dB).
  2. Mismatch in output resolution (producing 120x120 instead of 224x224).
  3. Missing `PairedBigEarthNetDataset` and paired loaders.
  4. Mismatch in statistics size (generated from 100 patches instead of 21,000).
  5. Missing validation isolation list, leading to critical data leakage.
