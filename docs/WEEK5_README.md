# Week 5 — Cross-Modal Satellite Image Retrieval & Intelligence Portal
**Project:** Cross-Modal Satellite Image Retrieval System with Remote Sensing Intelligence Portal  
**Person 3:** Meet Katarmal  
**Module:** Live Retrieval Inference, FAISS Galleries, Domain-Aware Modality Routing & Flask Integration

---

## 1. Overview
Week 5 delivers the complete end-to-end operational retrieval system:
1. **Model Deployment:** Loads the trained `TwoTowerNetwork` (ResNet-50 SAR and MS towers + 512-D projection heads) from `models/best_model-v1.ckpt`.
2. **FAISS Indexing:** Inner-product search (`IndexFlatIP`) over 4,500 L2-normalized 512-D test embeddings for SAR (`outputs/indices/sar_gallery.index`) and MS (`outputs/indices/ms_gallery.index`).
3. **Domain-Aware Routing:** Literature-backed modality recommendation across 3 domains and 6 subcategories (`config/disaster_modality_config.json`).
4. **Hybrid Retrieval:** Multi-modal dual-gallery candidate retrieval with deterministic score-based ranking and provenance tracking.
5. **Interactive Web Application:** Production Flask portal (`app.py`) with real-time inference, latency metrics, and satellite thumbnail streaming.

---

## 2. Directory Structure

```text
Major-project/
├── app.py                             # Production Flask intelligence portal
├── config/
│   ├── disaster_modality_config.json  # 3 domains and 6 authoritative subcategories
│   └── settings.py                    # Pipeline configuration
├── outputs/
│   ├── indices/
│   │   ├── sar_gallery.index          # 4,500-vector SAR FAISS gallery
│   │   └── ms_gallery.index           # 4,500-vector MS FAISS gallery
│   ├── week5_integration_report.md    # Comprehensive integration verification report
│   └── week5_subcategory_results.json # Structured benchmark records for all 6 subcategories
├── scripts/
│   ├── build_faiss_indices.py         # FAISS index builder script
│   ├── test_retriever.py              # Retriever unit test suite (14 checks)
│   └── test_all_subcategories.py      # End-to-end subcategory & Flask integration tests
├── src/
│   ├── inference/
│   │   └── retriever.py               # CrossModalRetriever with hybrid and thumbnail support
│   └── models/
│       └── two_tower.py               # TwoTowerNetwork & InfoNCELoss
├── static/
│   └── style.css                      # Portal styling with thumbnail badges and latency pill
├── templates/
│   ├── domain.html                    # Domain selection view
│   ├── subcategory.html               # Subcategory exploration view
│   └── search.html                    # Image upload and Top-10 results view
└── requirements.txt                   # Deployment dependencies
```

---

## 3. Domain & Subcategory Modality Routing

| Domain | Subcategory | Recommended Modality | Retrieval Path |
| :--- | :--- | :---: | :--- |
| **Disaster Response & Crisis Mapping** | Flood | **SAR** | SAR Query $\to$ MS Gallery |
| | Wildfire | **Multispectral** | MS Query $\to$ SAR Gallery |
| | Earthquake Damage | **SAR** | SAR Query $\to$ MS Gallery |
| **Environmental Monitoring** | Deforestation | **Hybrid** | Query $\to$ MS + SAR Galleries (Score Merge) |
| | Drought | **Multispectral** | MS Query $\to$ SAR Gallery |
| **Urban & Agricultural Intelligence** | Crop Mapping | **Hybrid** | Query $\to$ MS + SAR Galleries (Score Merge) |

---

## 4. How to Run

### Run the Flask Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### Run Unit and Integration Tests
```bash
# Test retriever components (14 unit checks)
python scripts/test_retriever.py

# Test all 6 subcategories and Flask endpoints end-to-end
python scripts/test_all_subcategories.py
```

---

## 5. Web Portal Endpoints

* `GET /`: Research domain selection view.
* `GET /<domain_slug>`: Subcategory exploration and modality guidance.
* `GET /search?domain=<d>&subcategory=<s>`: Query scene upload form.
* `POST /search?domain=<d>&subcategory=<s>`: Real TwoTower forward pass + FAISS search $\to$ Top-10 results.
* `GET /patch_image/<patch_id>`: Live streaming of rendered GeoTIFF thumbnail.
* `GET /health`: Portal operational status (checkpoint, vector count, galleries).
