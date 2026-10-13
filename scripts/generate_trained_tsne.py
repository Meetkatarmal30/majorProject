"""
generate_trained_tsne.py - Week 4 Trained Embedding t-SNE Generator

Generates a publication-quality t-SNE 2D projection using the trained
Two-Tower model validation embeddings (512-D), evaluated on the EXACT
same 2,000 patch IDs (1,000 MS + 1,000 SAR) and 5 classes as the baseline t-SNE.

Methodology adheres strictly to Adarsh's baseline reference
(teammate_inputs/adarsh_generate_tsne.py):
- Perplexity: 30
- Random State / Seed: 42
- Iterations: 1000
- Concatenation order: MS (0..999) then SAR (1000..1999)
- Color convention: MS=#2196F3 (Blue), SAR=#F44336 (Red)
- Markers: 'o', 's', '^', 'D', 'P' across the 5 top land-cover classes
"""

import os
import re
import json
import inspect
import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from sklearn.manifold import TSNE

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

MS_EMBEDDINGS_PATH = BASE_DIR / "teammate_inputs" / "trained_val_ms_embeddings.npy"
SAR_EMBEDDINGS_PATH = BASE_DIR / "teammate_inputs" / "trained_val_sar_embeddings.npy"
TSNE_IDS_PATH = BASE_DIR / "teammate_inputs" / "baseline_tsne_patch_ids.txt"
TRAINED_VAL_IDS_PATH = BASE_DIR / "teammate_inputs" / "trained_val_patch_ids.txt"
VAL_SPLIT_PATH = BASE_DIR / "teammate_inputs" / "val_split.csv"

OUTPUT_PLOT = BASE_DIR / "outputs" / "trained_tsne.png"
OUTPUT_METADATA = BASE_DIR / "outputs" / "trained_tsne_metadata.json"

RANDOM_SEED = 42
PERPLEXITY = 30
N_ITER = 1000

TARGET_CLASSES = [
    "Arable land",
    "Coniferous forest",
    "Broad-leaved forest",
    "Marine waters",
    "Agro-forestry areas",
]

CLASS_MARKERS = ["o", "s", "^", "D", "P"]
CLASS_TO_MARKER = dict(zip(TARGET_CLASSES, CLASS_MARKERS))

MS_COLOR = "#2196F3"   # Material Blue
SAR_COLOR = "#F44336"  # Material Red


def parse_primary_label(label_str: str) -> str:
    """Extract primary (first) land-cover class from string representation."""
    if not isinstance(label_str, str):
        return str(label_str)
    inner = label_str.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()
    tokens = re.findall(r"['\"]([^'\"]+)['\"]", inner)
    if tokens:
        return tokens[0]
    return inner


def main():
    print("=" * 70)
    print("WEEK 4: TRAINED EMBEDDING t-SNE GENERATION")
    print("=" * 70)

    # 1. Load trained embeddings
    print(f"Loading trained validation embeddings...")
    ms_embeddings = np.load(MS_EMBEDDINGS_PATH)
    sar_embeddings = np.load(SAR_EMBEDDINGS_PATH)

    print(f"  MS embeddings shape : {ms_embeddings.shape}, dtype: {ms_embeddings.dtype}")
    print(f"  SAR embeddings shape: {sar_embeddings.shape}, dtype: {sar_embeddings.dtype}")

    assert ms_embeddings.shape == (4500, 512), f"Unexpected MS shape: {ms_embeddings.shape}"
    assert sar_embeddings.shape == (4500, 512), f"Unexpected SAR shape: {sar_embeddings.shape}"
    assert not np.isnan(ms_embeddings).any(), "NaN found in MS embeddings"
    assert not np.isnan(sar_embeddings).any(), "NaN found in SAR embeddings"
    assert not np.isinf(ms_embeddings).any(), "Inf found in MS embeddings"
    assert not np.isinf(sar_embeddings).any(), "Inf found in SAR embeddings"

    # 2. Load split metadata and trained val patch IDs
    val_split = pd.read_csv(VAL_SPLIT_PATH)
    with open(TRAINED_VAL_IDS_PATH, "r", encoding="utf-8") as f:
        trained_val_ids = [line.strip() for line in f if line.strip()]

    assert len(trained_val_ids) == 4500
    assert trained_val_ids == val_split["patch_id"].tolist(), "trained_val_ids mismatch with val_split"

    # 3. Load baseline t-SNE patch IDs
    print(f"Loading baseline t-SNE IDs from: {TSNE_IDS_PATH}")
    with open(TSNE_IDS_PATH, "r", encoding="utf-8") as f:
        tsne_lines = [line.strip() for line in f if line.strip()]

    assert tsne_lines[0] == "MULTISPECTRAL", f"Expected MULTISPECTRAL header, got {tsne_lines[0]}"
    assert tsne_lines[1001] == "SAR", f"Expected SAR header, got {tsne_lines[1001]}"

    ms_tsne_ids = tsne_lines[1:1001]
    sar_tsne_ids = tsne_lines[1002:2002]

    assert len(ms_tsne_ids) == 1000, f"Expected 1,000 MS IDs, got {len(ms_tsne_ids)}"
    assert len(sar_tsne_ids) == 1000, f"Expected 1,000 SAR IDs, got {len(sar_tsne_ids)}"

    # 4. Map IDs to exact rows
    ms_to_row = {pid: i for i, pid in enumerate(val_split["patch_id"])}
    sar_to_row = {s1: i for i, s1 in enumerate(val_split["s1_name"])}

    ms_sample_idx = [ms_to_row[pid] for pid in ms_tsne_ids]
    sar_sample_idx = [sar_to_row[s1] for s1 in sar_tsne_ids]

    ms_sample_emb = ms_embeddings[ms_sample_idx]
    sar_sample_emb = sar_embeddings[sar_sample_idx]

    assert ms_sample_emb.shape == (1000, 512)
    assert sar_sample_emb.shape == (1000, 512)

    # 5. Extract and verify labels
    ms_sample_labels = [parse_primary_label(val_split.iloc[i]["labels"]) for i in ms_sample_idx]
    sar_sample_labels = [parse_primary_label(val_split.iloc[i]["labels"]) for i in sar_sample_idx]

    ms_counts = Counter(ms_sample_labels)
    sar_counts = Counter(sar_sample_labels)
    total_counts = ms_counts + sar_counts

    print("\nClass breakdown:")
    for cls in TARGET_CLASSES:
        print(f"  {cls:<22}: MS={ms_counts[cls]:>3}, SAR={sar_counts[cls]:>3}, Total={total_counts[cls]:>4}")

    assert set(ms_sample_labels) == set(TARGET_CLASSES), f"MS labels mismatch: {set(ms_sample_labels)}"
    assert set(sar_sample_labels) == set(TARGET_CLASSES), f"SAR labels mismatch: {set(sar_sample_labels)}"
    print("[OK] All 2,000 selected points strictly belong to the 5 target classes.")

    # 6. Concatenate embeddings following Adarsh's baseline convention
    combined_emb = np.vstack([ms_sample_emb, sar_sample_emb])
    combined_labels = ms_sample_labels + sar_sample_labels
    assert combined_emb.shape == (2000, 512)

    # 7. Execute t-SNE
    print(f"\nRunning t-SNE on {len(combined_emb)} embeddings (perplexity={PERPLEXITY}, seed={RANDOM_SEED})...")
    tsne_params = {
        "n_components": 2,
        "perplexity": PERPLEXITY,
        "random_state": RANDOM_SEED,
        "verbose": 1,
        "init": "pca"
    }
    sig = inspect.signature(TSNE.__init__)
    if "n_iter" in sig.parameters:
        tsne_params["n_iter"] = N_ITER
    elif "max_iter" in sig.parameters:
        tsne_params["max_iter"] = N_ITER

    tsne = TSNE(**tsne_params)
    reduced = tsne.fit_transform(combined_emb)
    print(f"[OK] t-SNE projection completed: {reduced.shape}")

    # 8. Publication-Quality Plotting
    print("\nRendering visualization...")
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=150)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")

    n_ms = len(ms_sample_idx)

    # Plot points by class and modality
    for cls in TARGET_CLASSES:
        marker = CLASS_TO_MARKER[cls]
        ms_mask = [j for j, lbl in enumerate(ms_sample_labels) if lbl == cls]
        sar_mask = [j for j, lbl in enumerate(sar_sample_labels) if lbl == cls]

        if ms_mask:
            ax.scatter(
                reduced[ms_mask, 0],
                reduced[ms_mask, 1],
                c=MS_COLOR,
                marker=marker,
                alpha=0.65,
                s=30,
                edgecolors="none",
                label=f"MS — {cls}"
            )
        if sar_mask:
            sar_global = [j + n_ms for j in sar_mask]
            ax.scatter(
                reduced[sar_global, 0],
                reduced[sar_global, 1],
                c=SAR_COLOR,
                marker=marker,
                alpha=0.65,
                s=30,
                edgecolors="none",
                label=f"SAR — {cls}"
            )

    # Legends: Modality + Class Markers
    modality_handles = [
        mpatches.Patch(color=MS_COLOR, label="Multispectral (Sentinel-2)"),
        mpatches.Patch(color=SAR_COLOR, label="SAR (Sentinel-1)"),
    ]
    class_handles = [
        mlines.Line2D([], [], color="#444444", marker=CLASS_TO_MARKER[cls], linestyle="None",
                      markersize=7, label=f"{cls} ({total_counts[cls]})")
        for cls in TARGET_CLASSES
    ]

    leg1 = ax.legend(
        handles=modality_handles,
        loc="upper right",
        title="Modality",
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.9
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=class_handles,
        loc="lower left",
        title="Land-Cover Class (Count)",
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.9
    )

    ax.set_title(
        "t-SNE Projection: Trained Cross-Modal Embeddings (Two-Tower InfoNCE)\n"
        "Same 2,000 Points & 5 Classes: Shared Latent Space Across SAR and Multispectral",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, labelpad=8)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    caption = (
        "Each point represents one image patch from the validation set (1,000 MS, 1,000 SAR). "
        "Red = SAR (Sentinel-1), Blue = Multispectral (Sentinel-2).\n"
        "Contrastive InfoNCE alignment pulls semantically matching cross-modal pairs into shared neighborhood clusters."
    )
    fig.text(0.5, 0.015, caption, ha="center", fontsize=9, color="#555555", style="italic")

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved trained t-SNE plot to: {OUTPUT_PLOT}")

    # 9. Save Metadata JSON
    metadata = {
        "title": "Trained Two-Tower Validation Embeddings t-SNE Projection",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "engineer": "Meet (Person 3)",
        "week": "Week 4",
        "sample_selection": {
            "total_points": 2000,
            "multispectral_points": 1000,
            "sar_points": 1000,
            "source_id_file": "teammate_inputs/baseline_tsne_patch_ids.txt",
            "identical_to_baseline_subset": True
        },
        "class_breakdown": {
            cls: {
                "multispectral": int(ms_counts[cls]),
                "sar": int(sar_counts[cls]),
                "total": int(total_counts[cls])
            }
            for cls in TARGET_CLASSES
        },
        "embedding_properties": {
            "dimension": 512,
            "dtype": "float32",
            "normalization": "L2-normalized projection",
            "ms_embedding_file": "teammate_inputs/trained_val_ms_embeddings.npy",
            "sar_embedding_file": "teammate_inputs/trained_val_sar_embeddings.npy"
        },
        "tsne_parameters": {
            "n_components": 2,
            "perplexity": PERPLEXITY,
            "random_state": RANDOM_SEED,
            "iterations": N_ITER,
            "init": "pca",
            "metric": "euclidean"
        },
        "output_files": {
            "plot": str(OUTPUT_PLOT.relative_to(BASE_DIR)),
            "metadata": str(OUTPUT_METADATA.relative_to(BASE_DIR)),
            "baseline_comparison_reference": "teammate_inputs/adarsh_baseline_tsne.png"
        }
    }

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata to: {OUTPUT_METADATA}")

    print("\n" + "=" * 70)
    print("WEEK 4 TRAINED t-SNE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
