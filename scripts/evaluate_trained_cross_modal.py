"""
Week 5 - Trained Cross-Modal Retrieval Evaluation.

Evaluates:
    1. SAR -> MS
    2. MS -> SAR

Metrics:
    Recall@1
    Recall@5
    Recall@10
    Average retrieval time per query

Ground truth:
    Paired patch IDs from test_split.csv
"""

from __future__ import annotations

from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TEST_CSV = Path("outputs/metadata/test_split.csv")

EMBEDDINGS_DIR = Path("outputs/embeddings")
INDEX_DIR = Path("outputs/indices")
REPORT_DIR = Path("outputs/reports")

SAR_EMBEDDINGS = (
    EMBEDDINGS_DIR / "trained_test_sar_embeddings.npy"
)

MS_EMBEDDINGS = (
    EMBEDDINGS_DIR / "trained_test_ms_embeddings.npy"
)

SAR_INDEX_PATH = INDEX_DIR / "sar_gallery.index"
MS_INDEX_PATH = INDEX_DIR / "ms_gallery.index"

OUTPUT_CSV = (
    REPORT_DIR / "trained_cross_modal_results.csv"
)

TOP_K = 10
EXPECTED_SAMPLES = 4500


# ============================================================
# LOAD TEST PATCH IDS
# ============================================================

def load_test_patch_ids() -> list[str]:
    """Load test patch IDs in the same order as the embeddings."""

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    df = pd.read_csv(TEST_CSV)

    if "patch_id" not in df.columns:
        raise ValueError(
            "test_split.csv does not contain "
            "'patch_id' column."
        )

    patch_ids = (
        df["patch_id"]
        .astype(str)
        .tolist()
    )

    if len(patch_ids) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} test patch IDs, "
            f"found {len(patch_ids)}."
        )

    if len(set(patch_ids)) != EXPECTED_SAMPLES:
        raise ValueError(
            "Test patch IDs are not unique."
        )

    return patch_ids


# ============================================================
# VALIDATION
# ============================================================

def validate_embeddings(
    embeddings: np.ndarray,
    name: str,
) -> None:

    if embeddings.shape != (
        EXPECTED_SAMPLES,
        512,
    ):
        raise ValueError(
            f"{name} shape is {embeddings.shape}; "
            f"expected ({EXPECTED_SAMPLES}, 512)."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


# ============================================================
# CROSS-MODAL EVALUATION
# ============================================================

def evaluate_direction(
    query_embeddings: np.ndarray,
    gallery_index: faiss.Index,
    query_patch_ids: list[str],
    gallery_patch_ids: list[str],
    direction_name: str,
) -> dict[str, float]:

    print()
    print("=" * 70)
    print(direction_name)
    print("=" * 70)

    start_time = time.perf_counter()

    distances, indices = gallery_index.search(
        query_embeddings,
        TOP_K,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    total_queries = len(query_patch_ids)

    recall_counts = {
        1: 0,
        5: 0,
        10: 0,
    }

    for query_index in range(total_queries):

        query_patch_id = query_patch_ids[
            query_index
        ]

        retrieved_indices = indices[
            query_index
        ]

        retrieved_patch_ids = [
            gallery_patch_ids[index]
            for index in retrieved_indices
            if index >= 0
        ]

        for k in recall_counts:

            top_k_ids = retrieved_patch_ids[:k]

            if query_patch_id in top_k_ids:
                recall_counts[k] += 1

    results = {
        "Mode": direction_name,
        "Recall@1": (
            recall_counts[1] / total_queries
        ) * 100.0,
        "Recall@5": (
            recall_counts[5] / total_queries
        ) * 100.0,
        "Recall@10": (
            recall_counts[10] / total_queries
        ) * 100.0,
        "Avg Retrieval Time (ms)": (
            elapsed_seconds / total_queries
        ) * 1000.0,
    }

    print(
        f"Queries             : {total_queries:,}"
    )

    print(
        f"Recall@1            : "
        f"{results['Recall@1']:.4f}%"
    )

    print(
        f"Recall@5            : "
        f"{results['Recall@5']:.4f}%"
    )

    print(
        f"Recall@10           : "
        f"{results['Recall@10']:.4f}%"
    )

    print(
        f"Avg retrieval time  : "
        f"{results['Avg Retrieval Time (ms)']:.4f} ms/query"
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("WEEK 5 — TRAINED CROSS-MODAL RETRIEVAL")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    print("Loading trained test embeddings...")

    sar_embeddings = np.load(
        SAR_EMBEDDINGS
    ).astype(np.float32)

    ms_embeddings = np.load(
        MS_EMBEDDINGS
    ).astype(np.float32)

    validate_embeddings(
        sar_embeddings,
        "SAR embeddings",
    )

    validate_embeddings(
        ms_embeddings,
        "MS embeddings",
    )

    print(
        f"[OK] SAR embeddings: "
        f"{sar_embeddings.shape}"
    )

    print(
        f"[OK] MS embeddings: "
        f"{ms_embeddings.shape}"
    )

    # --------------------------------------------------------
    # LOAD PATCH IDS
    # --------------------------------------------------------

    patch_ids = load_test_patch_ids()

    print(
        f"[OK] Test patch IDs: "
        f"{len(patch_ids):,}"
    )

    # --------------------------------------------------------
    # LOAD FAISS INDICES
    # --------------------------------------------------------

    if not SAR_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"SAR FAISS index not found:\n"
            f"{SAR_INDEX_PATH}"
        )

    if not MS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"MS FAISS index not found:\n"
            f"{MS_INDEX_PATH}"
        )

    sar_index = faiss.read_index(
        str(SAR_INDEX_PATH)
    )

    ms_index = faiss.read_index(
        str(MS_INDEX_PATH)
    )

    if sar_index.ntotal != EXPECTED_SAMPLES:
        raise ValueError(
            "SAR index does not contain "
            "4,500 vectors."
        )

    if ms_index.ntotal != EXPECTED_SAMPLES:
        raise ValueError(
            "MS index does not contain "
            "4,500 vectors."
        )

    print(
        f"[OK] SAR gallery index: "
        f"{sar_index.ntotal:,} vectors"
    )

    print(
        f"[OK] MS gallery index: "
        f"{ms_index.ntotal:,} vectors"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Test CSV order must match embedding order.
    #
    # The extraction script used shuffle=False, so the
    # embedding order follows test_split.csv.
    # --------------------------------------------------------

    # SAR -> MS
    sar_to_ms = evaluate_direction(
        query_embeddings=sar_embeddings,
        gallery_index=ms_index,
        query_patch_ids=patch_ids,
        gallery_patch_ids=patch_ids,
        direction_name="SAR -> MS",
    )

    # MS -> SAR
    ms_to_sar = evaluate_direction(
        query_embeddings=ms_embeddings,
        gallery_index=sar_index,
        query_patch_ids=patch_ids,
        gallery_patch_ids=patch_ids,
        direction_name="MS -> SAR",
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            sar_to_ms,
            ms_to_sar,
        ]
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINED CROSS-MODAL EVALUATION COMPLETE")
    print("=" * 70)
    print()

    print(results_df.to_string(index=False))

    print()
    print(
        f"[OK] Results saved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()