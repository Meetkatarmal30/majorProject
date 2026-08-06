# Sentinel-1 (SAR) Data Pipeline & Loading Utilities
## Cross-Modal Satellite Image Retrieval System

This repository contains the complete Sentinel-1 (SAR) data pipeline and loading utilities developed for the **Cross-Modal Satellite Image Retrieval Using Multi-Sensor Remote Sensing Data** project.

This module is designed to ingest raw, nested BigEarthNet-S1 SAR patches, apply outlier clipping and backscatter normalization, and output batched PyTorch tensors of shape `(B, 2, H, W)` corresponding to the dual polarization bands (VV and VH). 

---

## 1. Project Overview

The objective of this module is to build a robust, production-quality, and highly optimized data loading and preprocessing pipeline for the Sentinel-1 (SAR) portion of the BigEarthNet dataset. 

* **Week 1 Achievements**: Data directory inspection, statistical analysis, outlier clipping, Z-score normalization scaling, and visualization plots.
* **Week 2 Achievements**: High-speed directory caching, PyTorch `Dataset` integration with retry-recovery error handling, split management, `DataLoader` setups, and automated unit testing.

---

## 2. Folder Structure

The project layout and the purpose of each directory are detailed below:

```text
Major-project/
├── config/              # Centralized configuration parameters
│   ├── __init__.py
│   └── settings.py      # Paths, normalization bounds, and default hyperparameters
├── dataset/             # Core PyTorch dataset and dataloader implementation
│   ├── __init__.py
│   ├── dataset.py       # BigEarthNetS1Dataset class with cache & split loading
│   └── dataloader.py    # create_dataloaders() builder function
├── preprocessing/       # Geotiff loading, clipping, and normalization
│   ├── __init__.py
│   └── sar_preprocessing.py
├── utils/               # Dataset statistics generator and visualization utilities
│   ├── __init__.py
│   ├── stats_generator.py
│   └── visualization.py
├── tests/               # Automated unit and integration testing suite
│   └── test_pipeline.py
├── outputs/             # Generated stats (CSV/JSON) and verification plots
├── docs/                # Project documentation and integration guides
│   └── README.md
├── requirements.txt     # Python package dependencies
└── .s1_patch_cache.json # Speedup directory listing mapping cache (generated)
```

---

## 3. Installation

This module is compatible with Python 3.11+. Follow these steps to set up:

1. **Activate Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   Install PyTorch and standard satellite image processing libraries from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Dataset Setup

1. **Dataset Location**: Place the unzipped `BigEarthNet-S1` dataset inside the `data/` folder.
2. **Expected Directory Hierarchy**:
   ```text
   data/
   └── BigEarthNet-S1/
       └── BigEarthNet-S1/
           ├── S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39/
           │   ├── S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VV.tif
           │   └── S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VH.tif
           ├── S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_40/
           │   ...
   ```
3. **Caching**: On the first run, the dataset class traverses all 549,488 directories and writes `.s1_patch_cache.json` to the root folder. Subsequent runs load the JSON directly, skipping slow recursive scanning.

---

## 5. Configuration (`config/settings.py`)

All pipeline behaviors are controlled in `config/settings.py`. Key properties include:

* `DATA_DIR`: Path to the BigEarthNet-S1 dataset root.
* `NORMALIZATION_MODE`: `"none"`, `"min-max"`, or `"z-score"` (default).
* `VV_CLIP_RANGE` / `VH_CLIP_RANGE`: Clipped decibel (dB) boundaries to remove outlier backscatter coefficients.
* `VV_ZSCORE_STATS` / `VH_ZSCORE_STATS`: Global mean and standard deviation calculated over the dataset used for Z-score normalization.

---

## 6. Dataset Usage

To load individual preprocessed Sentinel-1 patches, import the dataset:

```python
from pathlib import Path
from dataset.dataset import BigEarthNetS1Dataset

# Initialize training dataset
dataset = BigEarthNetS1Dataset(
    split="train",
    norm_mode="z-score"
)

# Access a sample
tensor, patch_name = dataset[0]

print(f"Patch Name: {patch_name}")
print(f"Tensor Shape: {tensor.shape}")  # torch.Size([2, 120, 120])
print(f"Tensor Dtype: {tensor.dtype}")  # torch.float32
```

* **Returned Format**:
  - `tensor`: Stacked 2-channel Float32 PyTorch tensor containing VV (channel 0) and VH (channel 1) bands.
  - `patch_name`: The directory name string of the patch, used for cross-modal matching with Sentinel-2 datasets.

---

## 7. DataLoader Usage

To instantiate loaders for training loops, use the helper function:

```python
from dataset.dataloader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=64,
    num_workers=4,
    norm_mode="z-score"
)

# Loop through batches
for batch_idx, (images, patch_names) in enumerate(train_loader):
    # images shape: torch.Size([64, 2, 120, 120])
    # patch_names: tuple of 64 strings matching the batch images
    
    # Pass images to S1 retrieval encoder
    pass
```

* **Configuration Rules**:
  - **Train Loader**: `shuffle=True`, `drop_last=True`.
  - **Validation & Test Loaders**: `shuffle=False`, `drop_last=False`.
  - **Pin Memory**: Automatically enabled if CUDA (GPU) is available.

---

## 8. Preprocessing Pipeline

The preprocessing pipeline is automatically executed when drawing samples from the dataset:

1. **GeoTIFF Loading**: Reads separate Float32 polarization band GeoTIFF files.
2. **Clipping**: Clips values outside VV `[-25.0, 0.0]` dB and VH `[-32.0, -5.0]` dB ranges.
3. **Normalization**:
   - `min-max`: Scales clipped values linearly into `[0.0, 1.0]`.
   - `z-score`: Standardizes values using computed dataset stats: VV mean `-12.4967` (std `5.0475`), VH mean `-19.1207` (std `5.3495`).
4. **Channel Stacking**: Stacks VV and VH bands into a `(2, H, W)` shape.

---

## 9. Unit Tests

Run the automated test suite to verify pipeline integrity:

```bash
python tests/test_pipeline.py
```

The suite tests:
* Dataset initialization, caching, and lengths.
* Valid shapes `(2, 120, 120)` and dtypes (`float32`).
* Deterministic split partition allocations.
* Error recovery from corrupted/missing TIFF files.
* Dataloader batch configuration sizes.

---

## 10. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError` | `DATA_DIR` path is incorrect. | Update `DATA_DIR` in `config/settings.py` to match your local path. |
| Slow Startup (30s) | First cold scan. | Expected behavior. Startup will drop to `<2s` once `.s1_patch_cache.json` is generated. |
| Corrupted TIFF | Downloader interrupted. | The dataset logs a warning and automatically skips the sample, drawing a valid random index. |
| CUDA Warnings | No GPU found. | Ignored. Dataloader automatically disables pinned memory on CPU-only machines. |

---

## 11. Performance Notes

* **Scanning Overhead**: Traversal of 549,488 directories takes **~27.1 seconds** on a standard Windows machine (Cold run).
* **Caching Speedup**: Subsequent warm loadings read directly from the `.s1_patch_cache.json` file in **`1.95` seconds** (a **14x speedup**).
* **Multiprocessing**: Recommends `num_workers = 4` to parallelize image loading and Z-score calculations across CPU cores.

---

## 12. Teammate Integration Guide

For the teammate developing the contrastive dual-encoder model:

1. Copy the `Major-project` directories (`config`, `dataset`, `preprocessing`) into your training workspace.
2. Import `create_dataloaders` from `dataset.dataloader` inside your model trainer module.
3. Load S1 batches: S1 images are passed through the SAR encoder, and the corresponding `patch_names` tuple is used to pair S1 features with S2 features matching the same geographical patch name.
