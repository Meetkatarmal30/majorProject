"""
Week 5 - Trained Same-Modal Retrieval Evaluation.

Evaluates:
    1. SAR -> SAR
    2. MS -> MS

Metrics:
    Recall@1
    Recall@5
    Recall@10
    mAP
    Average retrieval time per query

Same-modal relevance:
    A gallery item is relevant when it shares at least one
    land-cover label with the query.

The query item itself is excluded from retrieval.
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
    REPORT_DIR / "trained_same_modal_results.csv"
)

TOP_K = 10
EXPECTED_SAMPLES = 4500
EXPECTED_DIMENSION = 512


# ============================================================
# LOAD TEST METADATA
# ============================================================

def load_test_metadata() -> pd.DataFrame:
    """Load test metadata and validate required columns."""

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{TEST_CSV}"
        )

    df = pd.read_csv(TEST_CSV)

    required_columns = {
        "patch_id",
        "labels",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} test samples, "
            f"found {len(df)}."
        )

    df["patch_id"] = (
        df["patch_id"]
        .astype(str)
    )

    return df


# ============================================================
# LABEL PARSING
# ============================================================

def parse_labels(value) -> set[str]:
    """
    Convert the labels column into a set of labels.

    Supports common representations such as:
        "a;b;c"
        "['a', 'b', 'c']"
        "a, b, c"
    """

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    # Handle Python-list-like representation.
    if (
        text.startswith("[")
        and text.endswith("]")
    ):
        text = text[1:-1]

    text = text.replace(
        "'",
        "",
    ).replace(
        '"',
        "",
    )

    if ";" in text:
        parts = text.split(";")
    elif "," in text:
        parts = text.split(",")
    else:
        parts = text.split()

    return {
        part.strip()
        for part in parts
        if part.strip()
    }


# ============================================================
# RELEVANCE
# ============================================================

def build_label_sets(
    metadata: pd.DataFrame,
) -> list[set[str]]:
    """Build a label set for every test patch."""

    return [
        parse_labels(value)
        for value in metadata["labels"]
    ]


def is_relevant(
    query_labels: set[str],
    gallery_labels: set[str],
) -> bool:
    """Return True when query and gallery share >= 1 label."""

    return bool(
        query_labels.intersection(
            gallery_labels
        )
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(
    path: Path,
    name: str,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} embeddings not found:\n{path}"
        )

    embeddings = np.load(path)

    if embeddings.shape != (
        EXPECTED_SAMPLES,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            f"{name} embeddings have shape "
            f"{embeddings.shape}; expected "
            f"({EXPECTED_SAMPLES}, {EXPECTED_DIMENSION})."
        )

    embeddings = embeddings.astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name} embeddings contain NaN or Inf."
        )

    return embeddings


# ============================================================
# SAME-MODAL EVALUATION
# ============================================================

def evaluate_same_modal(
    embeddings: np.ndarray,
    index: faiss.Index,
    label_sets: list[set[str]],
    mode_name: str,
) -> dict[str, float]:

    print()
    print("=" * 70)
    print(mode_name)
    print("=" * 70)

    total_queries = len(embeddings)

    # We need one extra result because the query itself
    # occupies one gallery position and must be removed.
    search_k = TOP_K + 1

    start_time = time.perf_counter()

    distances, indices = index.search(
        embeddings,
        search_k,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    recall_counts = {
        1: 0,
        5: 0,
        10: 0,
    }

    average_precisions = []

    for query_index in range(total_queries):

        query_labels = label_sets[
            query_index
        ]

        retrieved_indices = []

        for gallery_index in indices[
            query_index
        ]:

            if gallery_index < 0:
                continue

            # Exclude the query itself.
            if gallery_index == query_index:
                continue

            retrieved_indices.append(
                gallery_index
            )

            if len(retrieved_indices) >= TOP_K:
                break

        # ----------------------------------------------------
        # Build relevance list
        # ----------------------------------------------------

        relevance = []

        for gallery_index in retrieved_indices:

            relevant = is_relevant(
                query_labels,
                label_sets[gallery_index],
            )

            relevance.append(
                1 if relevant else 0
            )

        # ----------------------------------------------------
        # Recall@K
        # ----------------------------------------------------

        for k in recall_counts:

            top_k_relevance = relevance[:k]

            if any(top_k_relevance):
                recall_counts[k] += 1

        # ----------------------------------------------------
        # Average Precision @ 10
        # ----------------------------------------------------

        total_relevant_in_top_k = sum(
            relevance
        )

        if total_relevant_in_top_k == 0:
            average_precisions.append(0.0)
            continue

        precision_sum = 0.0
        relevant_seen = 0

        for rank, relevant in enumerate(
            relevance,
            start=1,
        ):

            if relevant:
                relevant_seen += 1

                precision_at_rank = (
                    relevant_seen / rank
                )

                precision_sum += (
                    precision_at_rank
                )

        ap_at_10 = (
            precision_sum
            / total_relevant_in_top_k
        )

        average_precisions.append(
            ap_at_10
        )

    map_at_10 = (
        float(np.mean(average_precisions))
        if average_precisions
        else 0.0
    )

    results = {
        "Mode": mode_name,
        "Recall@1": (
            recall_counts[1]
            / total_queries
            * 100.0
        ),
        "Recall@5": (
            recall_counts[5]
            / total_queries
            * 100.0
        ),
        "Recall@10": (
            recall_counts[10]
            / total_queries
            * 100.0
        ),
        "mAP": map_at_10 * 100.0,
        "Avg Retrieval Time (ms)": (
            elapsed_seconds
            / total_queries
            * 1000.0
        ),
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
        f"mAP@10              : "
        f"{results['mAP']:.4f}%"
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
    print("WEEK 5 — TRAINED SAME-MODAL RETRIEVAL")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata = load_test_metadata()

    patch_ids = (
        metadata["patch_id"]
        .tolist()
    )

    label_sets = build_label_sets(
        metadata
    )

    print(
        f"[OK] Test samples: "
        f"{len(metadata):,}"
    )

    print(
        f"[OK] Patch IDs: "
        f"{len(patch_ids):,}"
    )

    # --------------------------------------------------------
    # LOAD EMBEDDINGS
    # --------------------------------------------------------

    sar_embeddings = load_embeddings(
        SAR_EMBEDDINGS,
        "SAR",
    )

    ms_embeddings = load_embeddings(
        MS_EMBEDDINGS,
        "MS",
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
    # LOAD FAISS INDICES
    # --------------------------------------------------------

    if not SAR_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"SAR index not found:\n"
            f"{SAR_INDEX_PATH}"
        )

    if not MS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"MS index not found:\n"
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
        f"[OK] SAR gallery: "
        f"{sar_index.ntotal:,} vectors"
    )

    print(
        f"[OK] MS gallery: "
        f"{ms_index.ntotal:,} vectors"
    )

    # --------------------------------------------------------
    # SAR -> SAR
    # --------------------------------------------------------

    sar_to_sar = evaluate_same_modal(
        embeddings=sar_embeddings,
        index=sar_index,
        label_sets=label_sets,
        mode_name="SAR -> SAR",
    )

    # --------------------------------------------------------
    # MS -> MS
    # --------------------------------------------------------

    ms_to_ms = evaluate_same_modal(
        embeddings=ms_embeddings,
        index=ms_index,
        label_sets=label_sets,
        mode_name="MS -> MS",
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            sar_to_sar,
            ms_to_ms,
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
    print("TRAINED SAME-MODAL EVALUATION COMPLETE")
    print("=" * 70)
    print()

    print(
        results_df.to_string(index=False)
    )

    print()
    print(
        f"[OK] Results saved: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()