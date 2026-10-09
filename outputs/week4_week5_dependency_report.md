# Week 4 and Week 5 Dependency & Integration Report
**Project:** Cross-Modal Satellite Image Retrieval System with Remote Sensing Intelligence Portal  
**Author:** Meet (Person 3)  
**Date:** 2026-09-22  
**Status:** Inspection & Verification Complete  

---

## 1. Executive Summary

| Phase | Status | Summary |
|---|---|---|
| **Week 4 Readiness** | **YES** | All required dependencies for Week 4 t-SNE generation (same 2,000 points, same 5 classes), trained validation embeddings `(4,500, 512)`, trained test embeddings `(4,500, 512)`, and patch-ID alignment are verified and ready. |
| **Week 5 Readiness** | **PARTIAL (90% Ready)** | All metadata, subcategory mappings, test embeddings, index creation scripts, and benchmark results are ready. The physical checkpoint file (`best_model-v1.ckpt`, ~350 MB) was gitignored by teammates and must be shared directly to support live query encoding. |

---

## 2. Upstream Teammate Sources

| Teammate | Repository URL | Branch | Commit Hash | Key Responsibilities |
|---|---|---|---|---|
| **Jainam** | [cross-modal-remote-sensing-retrieval](https://github.com/Jainamnj2131/cross-modal-remote-sensing-retrieval) | `main` | `f4c57d94612033fc5ddf5a0f61b6102c162d6702` | Model training, Lightning wrapper, FAISS builder, evaluation scripts |
| **Adarsh** | [deep-cross-final-yr-project](https://github.com/AdarshKanaujiya/deep-cross-final-yr-project) | `main` | `eb98481fb0ca0797e08baa934a93c65b52b54f6a` | Trained validation & test embeddings, baseline t-SNE 2,000-point subset, retrieval analysis |
| **Fateh** | [majorproject_week3](https://github.com/fatteali7/majorproject_week3) | `main` | `81399b9bf6466c1971a502d2a73c378f84c32c35` | Subcategory configuration (6 subcategories), Flask web UI prototype, templates & styling |

---

## 3. Critical Data Alignment Verification

All artifacts were rigorously cross-checked against local splits and ground truth:

1. **Trained Embedding Dimension:**
   - Expected: `512`
   - Verified: Exactly `512` for all SAR and MS trained arrays (`float32`, no NaNs).
2. **Trained Validation Embeddings:**
   - SAR: `(4500, 512)`
   - MS: `(4500, 512)`
   - Patch IDs: Exactly 4,500 lines matching `teammate_inputs/val_split.csv` row-by-row (`100% exact match`).
3. **Trained Test Embeddings:**
   - SAR: `(4500, 512)`
   - MS: `(4500, 512)`
   - Patch IDs: Exactly 4,500 lines matching `teammate_inputs/test_split.csv` row-by-row (`100% exact match`).
4. **Baseline t-SNE Subset (The 2,000 Points & 5 Classes):**
   - Source: `Adarsh/outputs/visualizations/baseline_tsne_patch_ids.txt`
   - Structure: Exactly 1,000 Multispectral points + 1,000 SAR points.
   - Ground truth check: 100% of the 2,000 IDs exist in `val_split.csv`.
   - The exact 5 classes:
     1. `Arable land`
     2. `Coniferous forest`
     3. `Broad-leaved forest`
     4. `Marine waters`
     5. `Agro-forestry areas`

---

## 4. Comprehensive Artifact Inventory

### A. Available Locally in Meet's Project
| Artifact | Location | Size | Role / Description |
|---|---|---|---|
| `train_split.csv` | `teammate_inputs/train_split.csv` | 4.92 MB | 21,000 training pairs (patch_id, labels, split, country, s1_name) |
| `val_split.csv` | `teammate_inputs/val_split.csv` | 1.10 MB | 4,500 validation pairs |
| `test_split.csv` | `teammate_inputs/test_split.csv` | 1.06 MB | 4,500 test pairs |
| `baseline_val_s1_embeddings.npy` | `outputs/baseline_val_s1_embeddings.npy` | 36.86 MB | Pre-projection baseline validation SAR features `(4500, 2048)` |
| `baseline_val_s1_patch_ids.txt` | `outputs/baseline_val_s1_patch_ids.txt` | 207 KB | 4,500 validation patch IDs |
| `baseline_val_ms_embeddings.npy` | `teammate_inputs/baseline_val_ms_embeddings.npy` | 36.86 MB | Pre-projection baseline validation MS features `(4500, 2048)` |
| `baseline_val_patch_ids.txt` | `teammate_inputs/baseline_val_patch_ids.txt` | 234 KB | 4,500 validation patch IDs |
| `s1_stats.json` | `outputs/s1_stats.json` | 738 B | S1 normalization constants (mean/std) |
| `s2_stats.json` | `teammate_inputs/s2_stats.json` | 325 B | S2 normalization constants (mean/std) |
| `tsne_visualization.py` | `evaluation/tsne_visualization.py` | 7.17 KB | Meet's publication-quality t-SNE generator |
| `paired_bigearthnet_dataset.py` | `./paired_bigearthnet_dataset.py` | 15.45 KB | Meet's paired data loader |

### B. Required from Teammates (Collected in Scratch, Ready to Copy)
| Artifact | Upstream Source | Target Path | Size | Req W4 | Req W5 |
|---|---|---|---|---|---|
| `baseline_tsne_patch_ids.txt` | Adarsh: `outputs/visualizations/` | `teammate_inputs/baseline_tsne_patch_ids.txt` | 98 KB | **YES** | NO |
| `trained_val_ms_embeddings.npy` | Adarsh: `outputs/embeddings/` | `teammate_inputs/trained_val_ms_embeddings.npy` | 9.22 MB | **YES** | NO |
| `trained_val_sar_embeddings.npy` | Adarsh: `outputs/embeddings/` | `teammate_inputs/trained_val_sar_embeddings.npy` | 9.22 MB | **YES** | NO |
| `trained_val_patch_ids.txt` | Adarsh: `outputs/embeddings/` | `teammate_inputs/trained_val_patch_ids.txt` | 234 KB | **YES** | NO |
| `trained_test_ms_embeddings.npy` | Adarsh: `week5/` | `teammate_inputs/trained_test_ms_embeddings.npy` | 9.22 MB | **YES** | **YES** |
| `trained_test_sar_embeddings.npy` | Adarsh: `week5/` | `teammate_inputs/trained_test_sar_embeddings.npy` | 9.22 MB | **YES** | **YES** |
| `trained_test_ms_patch_ids.txt` | Adarsh: `week5/` | `teammate_inputs/trained_test_ms_patch_ids.txt` | 234 KB | **YES** | **YES** |
| `trained_test_sar_patch_ids.txt` | Adarsh: `week5/` | `teammate_inputs/trained_test_sar_patch_ids.txt` | 234 KB | **YES** | **YES** |
| `two_tower.py` | Jainam: `src/models/` | `src/models/two_tower.py` | 4.01 KB | **YES** | **YES** |
| `build_faiss_indices.py` | Jainam: `scripts/` | `scripts/build_faiss_indices.py` | 5.78 KB | NO | **YES** |
| `extract_test_embeddings.py` | Jainam: `scripts/` | `scripts/extract_test_embeddings.py` | 9.70 KB | **YES** | **YES** |
| `evaluate_trained_cross_modal.py` | Jainam: `scripts/` | `scripts/evaluate_trained_cross_modal.py` | 9.02 KB | NO | **YES** |
| `evaluate_trained_same_modal.py` | Jainam: `scripts/` | `scripts/evaluate_trained_same_modal.py` | 13.08 KB | NO | **YES** |
| `disaster_modality_config.json` | Fateh: root | `config/disaster_modality_config.json` | 3.80 KB | NO | **YES** |
| `fateh_app.py` | Fateh: `app.py` | `teammate_inputs/fateh_app.py` | 3.62 KB | NO | **YES** |
| `style.css` | Fateh: `static/` | `teammate_inputs/static/style.css` | 9.92 KB | NO | **YES** |
| UI Templates (`domain`, `subcategory`, `search`) | Fateh: `templates/` | `teammate_inputs/templates/` | ~6.2 KB | NO | **YES** |
| `Week_4_training_record.txt` | Jainam: `outputs/reports/` | `teammate_inputs/Week_4_training_record.txt` | 4.93 KB | **YES** | NO |
| `Week_5_final_results.txt` | Jainam: `outputs/reports/` | `teammate_inputs/Week_5_final_results.txt` | 2.09 KB | NO | **YES** |

### C. Missing & Blocked Dependencies
| Artifact | Expected Location | Purpose | Size | Blocker Status | Mitigation |
|---|---|---|---|---|---|
| `best_model-v1.ckpt` | `checkpoints/best_model-v1.ckpt` | Loading model weights for live single image query inference | ~350 MB | **BLOCKED for Live Query Inference** | Jainam intentionally gitignored checkpoints (`*.ckpt`) to prevent bloat. Teammate must share via Google Drive/OneDrive. |
| `sar_gallery.index` | `indices/sar_gallery.index` | FAISS IndexFlatIP for SAR test gallery | 9.22 MB | **UNBLOCKED** | Gitignored by teammates, but can be built in <1 second locally using `scripts/build_faiss_indices.py`. |
| `ms_gallery.index` | `indices/ms_gallery.index` | FAISS IndexFlatIP for MS test gallery | 9.22 MB | **UNBLOCKED** | Gitignored by teammates, but can be built in <1 second locally using `scripts/build_faiss_indices.py`. |

---

## 5. Application Subcategories & Routing Rules (Week 5)

The 6 subcategories across 3 domains are defined in `disaster_modality_config.json`:

1. **Disaster Response and Crisis Mapping:**
   - **Flood Mapping** &rarr; `SAR` (all-weather, day/night radar water penetrability)
   - **Wildfire Scar** &rarr; `Multispectral` (spectral vegetation burn index analysis)
   - **Earthquake Damage / Landslide** &rarr; `SAR` (structural surface displacement & texture)
2. **Environmental Monitoring:**
   - **Deforestation / Forest Change** &rarr; `Hybrid` (combined optical canopy & radar structure)
   - **Drought / Water & Wetland Monitoring** &rarr; `Multispectral` (NIR/visible water & vegetation separation)
3. **Urban and Agricultural Intelligence:**
   - **Crop Mapping / Monitoring** &rarr; `Hybrid` / `Multispectral` (vegetation health + structural texture)

---

## 6. Recommendations & Next Steps

1. **Proceed with Week 4 t-SNE Execution (When instructed):**
   - Use `teammate_inputs/baseline_tsne_patch_ids.txt` to select the exact 2,000 points (1,000 SAR, 1,000 MS).
   - Use `teammate_inputs/trained_val_sar_embeddings.npy` and `teammate_inputs/trained_val_ms_embeddings.npy` as the high-dimensional vectors.
   - Run `evaluation/tsne_visualization.py` with `perplexity=30`, `max_iter=1000`, `random_state=42`.
2. **Build FAISS Indices:**
   - Run `scripts/build_faiss_indices.py` using `trained_test_sar_embeddings.npy` and `trained_test_ms_embeddings.npy` to create `sar_gallery.index` and `ms_gallery.index`.
3. **Request `best_model-v1.ckpt` from Jainam:**
   - Reach out to Jainam to download the checkpoint file for `checkpoints/best_model-v1.ckpt` to enable single-image encoding in `src/inference/retriever.py`.
