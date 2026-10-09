"""
Week 5 - Build FAISS gallery indices.

Builds two IndexFlatIP indices from the trained test embeddings:
    - SAR gallery
    - Multispectral gallery

The embeddings are already L2-normalized, so inner product
is equivalent to cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if (BASE_DIR / "outputs" / "embeddings" / "trained_test_sar_embeddings.npy").exists():
    EMBEDDINGS_DIR = BASE_DIR / "outputs" / "embeddings"
else:
    EMBEDDINGS_DIR = BASE_DIR / "teammate_inputs"

INDEX_DIR = BASE_DIR / "outputs" / "indices"

SAR_EMBEDDINGS = (
    EMBEDDINGS_DIR / "trained_test_sar_embeddings.npy"
)

MS_EMBEDDINGS = (
    EMBEDDINGS_DIR / "trained_test_ms_embeddings.npy"
)

SAR_INDEX = INDEX_DIR / "sar_gallery.index"
MS_INDEX = INDEX_DIR / "ms_gallery.index"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_SAMPLES = 4500
EXPECTED_DIMENSION = 512


# ============================================================
# LOAD AND VALIDATE EMBEDDINGS
# ============================================================

def load_embeddings(
    path: Path,
    name: str,
) -> np.ndarray:
    """Load and validate one embedding matrix."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} embeddings not found:\n{path}"
        )

    embeddings = np.load(path)

    print(
        f"[OK] Loaded {name} embeddings: "
        f"{embeddings.shape}"
    )

    if embeddings.shape != (
        EXPECTED_SAMPLES,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            f"Unexpected {name} embedding shape: "
            f"{embeddings.shape}. "
            f"Expected "
            f"({EXPECTED_SAMPLES}, {EXPECTED_DIMENSION})."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(
            np.float32
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{name} embeddings contain NaN or Inf."
        )

    return embeddings


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    embeddings: np.ndarray,
    output_path: Path,
    name: str,
) -> None:
    """Build and save an IndexFlatIP FAISS index."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    print(
        f"Building {name} FAISS index..."
    )

    index.add(embeddings)

    if index.ntotal != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"{name} index contains "
            f"{index.ntotal} vectors instead of "
            f"{EXPECTED_SAMPLES}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(output_path),
    )

    print(
        f"[OK] {name} index saved: "
        f"{output_path}"
    )

    print(
        f"[OK] Number of vectors: "
        f"{index.ntotal:,}"
    )

    print(
        f"[OK] Dimension: "
        f"{index.d}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("WEEK 5 — BUILD FAISS GALLERY INDICES")
    print("=" * 70)
    print()

    print(f"FAISS version: {faiss.__version__}")
    print()

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

    print()

    # --------------------------------------------------------
    # BUILD INDICES
    # --------------------------------------------------------

    build_index(
        embeddings=sar_embeddings,
        output_path=SAR_INDEX,
        name="SAR",
    )

    print()

    build_index(
        embeddings=ms_embeddings,
        output_path=MS_INDEX,
        name="MS",
    )

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    print()
    print("Verifying saved indices...")

    sar_index = faiss.read_index(
        str(SAR_INDEX)
    )

    ms_index = faiss.read_index(
        str(MS_INDEX)
    )

    if sar_index.ntotal != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Saved SAR index verification failed."
        )

    if ms_index.ntotal != EXPECTED_SAMPLES:
        raise RuntimeError(
            "Saved MS index verification failed."
        )

    if sar_index.d != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Saved SAR index dimension is incorrect."
        )

    if ms_index.d != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Saved MS index dimension is incorrect."
        )

    print(
        f"[OK] SAR index: "
        f"{sar_index.ntotal:,} vectors × "
        f"{sar_index.d} dimensions"
    )

    print(
        f"[OK] MS index: "
        f"{ms_index.ntotal:,} vectors × "
        f"{ms_index.d} dimensions"
    )

    print()
    print("=" * 70)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 70)
    print()

    print(
        f"SAR gallery: {SAR_INDEX}"
    )

    print(
        f"MS gallery : {MS_INDEX}"
    )


if __name__ == "__main__":
    main()