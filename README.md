# ATLAS: Cross-Modal Satellite Image Retrieval System & Remote Sensing Intelligence Portal

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%2FGPU-00599C?style=flat)](https://github.com/facebookresearch/faiss)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic-blue.svg)](LICENSE)

An end-to-end deep learning and high-speed vector retrieval platform designed for multi-sensor Earth Observation (EO). ATLAS maps dual-polarization Synthetic Aperture Radar (SAR, Sentinel-1 VV/VH) and 3-channel Optical Multispectral (MS, Sentinel-2 B04/B03/B02 RGB) imagery into a shared 512-dimensional metric space, enabling real-time cross-sensor intelligence and situational awareness.

---

## 1. System Architecture

```
                  +-------------------------------------------------------------+
                  |                   QUERY SATELLITE IMAGE                     |
                  |     (Sentinel-1 SAR GeoTIFF or Sentinel-2 MS GeoTIFF/PNG)   |
                  +-------------------------------------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |               SENSOR-SPECIFIC PREPROCESSING                 |
                  |  - SAR (VV, VH): dB Outlier Clipping + Z-score Normalization|
                  |  - MS (B04, B03, B02): Standardized via s2_stats.json       |
                  |  - Bilinear Resampling to (224, 224)                        |
                  +-------------------------------------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |            TWO-TOWER RESNET-50 FEATURE ENCODERS             |
                  |  Tower A: Conv2d(2, 64) -> ResNet-50 Trunk -> Linear(512)  |
                  |  Tower B: Conv2d(3, 64) -> ResNet-50 Trunk -> Linear(512)  |
                  |  L2 Normalization: ||z||_2 = 1.0 (Unit Hypersphere)         |
                  +-------------------------------------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |          FAISS VECTOR DATABASE (IndexFlatIP, 512-D)         |
                  |  4,500 Geographically Paired Sentinel Test Scenes           |
                  |  Query Latency: 0.03 - 0.05 ms per retrieval                |
                  +-------------------------------------------------------------+
                                                 |
                                                 v
                  +-------------------------------------------------------------+
                  |                 ATLAS WEB INTELLIGENCE PORTAL               |
                  |  - Domain Context (Disaster Response, Environment, Urban/Ag)|
                  |  - Modality Recommendation Engine (SAR vs MS vs Hybrid)     |
                  |  - Metric Representation Alignment vs Event Classification  |
                  |  - Side-by-Side Visual Comparison & Confidence Ranking      |
                  +-------------------------------------------------------------+
```

---

## 2. Key Highlights & Performance

ATLAS was evaluated on the official 4,500 geographically paired test scenes from the BigEarthNet multi-sensor benchmark:

### Quantitative Retrieval Benchmarks

| Retrieval Task | Modality Direction | Top-1 Recall | Top-5 Recall | Top-10 Recall | mAP@10 | Mean Search Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cross-Modal Retrieval** | **SAR $\rightarrow$ MS** | **28.31%** | **58.71%** | **75.04%** | **40.09%** | **0.032 ms** |
| **Cross-Modal Retrieval** | **MS $\rightarrow$ SAR** | **27.67%** | **57.40%** | **74.00%** | **39.36%** | **0.046 ms** |
| **Same-Modal Retrieval** | **SAR $\rightarrow$ SAR** | **45.47%** | **79.93%** | **92.49%** | **61.45%** | **0.045 ms** |
| **Same-Modal Retrieval** | **MS $\rightarrow$ MS** | **47.64%** | **81.93%** | **93.47%** | **64.08%** | **0.038 ms** |

* Latency measured using FAISS CPU `IndexFlatIP` across 4,500 gallery vectors (512-D float32) on standard hardware.

---

## 3. Mission Domains & Operational Use Cases

The web portal organizes multi-sensor intelligence into three operational domains and six specialized use-cases (configured in `config/disaster_modality_config.json`):

```
ATLAS Remote Sensing Portal
│
├── 01. Disaster Response and Crisis Mapping
│   ├── Flood (SAR-recommended: Cloud & weather penetrating radar backscatter)
│   ├── Wildfire (Multispectral-recommended: Visual/spectral burn scar vegetation stress)
│   └── Earthquake Damage (SAR-recommended: Structural backscatter & rubble coherence)
│
├── 02. Environmental Monitoring
│   ├── Deforestation (Hybrid-recommended: Optical canopy color + SAR structural roughness)
│   └── Drought (Multispectral-recommended: Direct chlorophyll & vegetation drying reflection)
│
└── 03. Urban and Agricultural Intelligence
    └── Crop Mapping (Hybrid-recommended: Optical vegetation indices + radar surface roughness)
```

> **Important Conceptual Note**: Subcategories represent **Operational Modality Recommendations** based on the physical properties of SAR and Optical sensors. Similarity scores measure **representation proximity in 512-D metric embedding space**. Retrieved BigEarthNet Corine land-cover labels describe surface terrain (e.g. Coniferous forest, Arable land, Pastures) and do not represent automated disaster classifications or event detectors.

---

## 4. Repository Structure

```text
Major-project/
├── app.py                          # Flask application server & routing
├── config/
│   └── settings.py                 # Central configurations, normalization parameters & paths
├── dataset/
│   ├── dataset.py                  # PyTorch BigEarthNet-S1 dataset with directory caching
│   └── dataloader.py               # DataLoader factories with pin_memory & num_workers
├── preprocessing/
│   └── sar_preprocessing.py        # Robust GeoTIFF loader, dB clipping & Z-score normalization
├── src/
│   ├── models/
│   │   └── two_tower.py            # Two-Tower PyTorch Lightning model (ResNet-50 encoders)
│   └── inference/
│       └── retriever.py            # FAISS cross-modal retriever & GeoTIFF thumbnail rendering
├── scripts/
│   ├── evaluate_trained_cross_modal.py # Full 4,500 test set cross-modal benchmark (SAR<->MS)
│   ├── evaluate_trained_same_modal.py  # Full 4,500 test set same-modal benchmark (SAR<->SAR, MS<->MS)
│   ├── test_all_subcategories.py       # Comprehensive regression test suite across all 6 domains
│   └── extract_test_embeddings.py      # Feature extraction utility for new dataset partitions
├── static/
│   └── style.css                   # Custom responsive dark-mode styling & animations
├── templates/
│   ├── base.html                   # Master layout template
│   ├── domains.html                # High-level domain selection grid
│   ├── subcategories.html          # Operational use-case selection
│   ├── search.html                 # Interactive upload, preview, sample scenes & comparison
│   └── results.html                # Gallery results view
├── teammate_inputs/                # Precomputed test embeddings & benchmark sample assets
│   ├── test_split.csv              # 4,500 test scene patch IDs & metadata
│   ├── trained_test_sar_embeddings.npy # 4,500 x 512 SAR L2-normalized embeddings
│   ├── trained_test_ms_embeddings.npy  # 4,500 x 512 MS L2-normalized embeddings
│   ├── trained_test_sar_patch_ids.txt  # Ordered SAR patch identifiers
│   └── trained_test_ms_patch_ids.txt   # Ordered MS patch identifiers
├── outputs/
│   ├── indices/                    # Prebuilt FAISS galleries (sar_gallery.index, ms_gallery.index)
│   └── reports/                    # Generated benchmark evaluation CSV reports
└── requirements.txt                # Production Python dependencies
```

---

## 5. Quickstart & Installation

### Step 1: Environment Setup
Ensure Python 3.10+ is installed. Create and activate a clean virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```


### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `pytorch-lightning`, `faiss-cpu`, `rasterio`, `flask`, `pillow`, `scikit-learn`, `numpy`.

### Step 3: Run Validation & Evaluation Scripts

To verify cross-modal retrieval across all 4,500 test scenes:
```bash
python scripts/evaluate_trained_cross_modal.py
```
Output results will be printed and exported to `outputs/reports/trained_cross_modal_results.csv`.

To verify same-modal baseline retrieval:
```bash
python scripts/evaluate_trained_same_modal.py
```
Output results will be printed and exported to `outputs/reports/trained_same_modal_results.csv`.

To verify end-to-end integration across all 6 subcategories:
```bash
python scripts/test_all_subcategories.py
```

### Step 4: Launch the Web Portal
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 6. Web Portal Features

1. **One-Click Benchmark Sample Testing**:
   - Test with verified satellite scenes (e.g. Sentinel-1 Flood Inundation, Sentinel-2 Wildfire Burn Scar, Agricultural Cropland) directly from the UI without needing local GeoTIFF files.
2. **Interactive File Selection**:
   - Real-time client-side thumbnail rendering for standard image formats and file size metadata for high-resolution GeoTIFF files.
3. **Side-by-Side Visual Comparison**:
   - Compare the input query scene directly with the top-10 retrieved matching scenes from the multi-sensor test gallery.
4. **Dynamic Similarity Metering**:
   - Visual cosine similarity bar (`0.0000` to `1.0000`), match quality categorization, and sensor provenance metadata.
5. **Robust Error & Storage Management**:
   - Automated pruning of temporary uploaded queries (retention bounded to 1 hour / max 50 items) and graceful 413 file-size error handling (25 MB max request limit).

---

## 7. Model Checkpoint & FAISS Index Specifications

- **Model Checkpoint**: `models/best_model-v1.ckpt` (Two-Tower ResNet-50, ~628 MB)
  - *SAR Tower*: `in_channels=2` (VV, VH polarizations) $\rightarrow$ ResNet-50 Trunk $\rightarrow$ `ProjectionHead(2048 -> 1024 -> 512)` $\rightarrow$ `L2-norm`
  - *MS Tower*: `in_channels=3` (B04 Red, B03 Green, B02 Blue RGB) $\rightarrow$ ResNet-50 Trunk $\rightarrow$ `ProjectionHead(2048 -> 1024 -> 512)` $\rightarrow$ `L2-norm`
- **FAISS Galleries**:
  - `outputs/indices/sar_gallery.index`: 4,500 SAR test vectors (512-D float32, `IndexFlatIP`, 9.2 MB)
  - `outputs/indices/ms_gallery.index`: 4,500 MS test vectors (512-D float32, `IndexFlatIP`, 9.2 MB)
  - Both indexes utilize exact Inner Product search on unit L2-normalized representations, mathematically identical to Cosine Similarity.

---

## 8. Required Artifacts for Fresh-Clone Reproduction

For a fresh clone of ATLAS to run evaluations and the web portal out of the box, verify that the following essential artifacts are present:

| Artifact Path | Expected Size / Format | Purpose |
| :--- | :--- | :--- |
| `models/best_model-v1.ckpt` | ~628 MB PyTorch Checkpoint | Learned ResNet-50 Two-Tower weights and projection heads. |
| `outputs/indices/sar_gallery.index` | 9.2 MB FAISS Index | 4,500 512-D L2-normalized SAR test vectors (`IndexFlatIP`). |
| `outputs/indices/ms_gallery.index` | 9.2 MB FAISS Index | 4,500 512-D L2-normalized MS test vectors (`IndexFlatIP`). |
| `teammate_inputs/test_split.csv` | ~580 KB CSV | Canonical test split containing 4,500 patch IDs, land-cover labels, country, and paired filenames. |
| `teammate_inputs/trained_test_sar_embeddings.npy` | 9.2 MB NumPy Array | Precomputed 4,500 x 512 float32 L2-normalized test SAR embeddings. |
| `teammate_inputs/trained_test_ms_embeddings.npy` | 9.2 MB NumPy Array | Precomputed 4,500 x 512 float32 L2-normalized test MS embeddings. |
| `teammate_inputs/trained_test_sar_patch_ids.txt` | ~240 KB Text | Ordered patch ID list matching `sar_gallery.index` rows. |
| `teammate_inputs/trained_test_ms_patch_ids.txt` | ~240 KB Text | Ordered patch ID list matching `ms_gallery.index` rows. |
| `teammate_inputs/s1_stats.json` | JSON Dictionary | Normalization statistics (mean, std) for SAR VV and VH bands. |
| `teammate_inputs/s2_stats.json` | JSON Dictionary | Standardization statistics (mean, std) for MS B04, B03, B02 bands. |
| `config/disaster_modality_config.json` | JSON Dictionary | Authoritative configuration of the 3 domains, 6 subcategories, and recommended modalities. |
| `teammate_inputs/BigEarthNet-S1-Required/` | Local Directory | Verified Sentinel-1 GeoTIFF scenes for 1-click sample queries and thumbnail rendering. |
| `teammate_inputs/BigEarthNet-S2-Required/` | Local Directory | Verified Sentinel-2 GeoTIFF scenes for 1-click sample queries and thumbnail rendering. |
