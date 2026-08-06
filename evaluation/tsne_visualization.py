"""
tsne_visualization.py - Sentinel-1 (SAR) Embedding Visualization Utility.

This module provides tools to project high-dimensional embeddings (e.g., from the S1 image encoder)
into 2D space using t-Distributed Stochastic Neighbor Embedding (t-SNE) and generate
publication-quality scatter plots.

Designed to be integrated directly with the trained model's embedding outputs.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# Add parent directory to sys.path to allow importing from root-level config package
import sys
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import BASE_DIR, setup_logging

logger = setup_logging("TSNE_Visualization")


def generate_tsne_plot(
    embeddings: np.ndarray,
    labels: Optional[Union[List[str], np.ndarray]] = None,
    save_path: Optional[Path] = None,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    random_state: int = 42,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "t-SNE Projection of Image Embeddings",
) -> np.ndarray:
    """
    Computes t-SNE dimensionality reduction and generates a publication-quality 2D scatter plot.

    Args:
        embeddings: High-dimensional embeddings matrix of shape (N, D).
        labels: Optional label names or categories of shape (N,) to color-code points.
        save_path: Optional path to save the generated PNG plot.
        perplexity: t-SNE perplexity parameter (relates to number of nearest neighbors).
        learning_rate: t-SNE learning rate (step size), either float or 'auto'.
        random_state: Random seed for deterministic reproducibility.
        figsize: Size of the figure (width, height) in inches.
        title: Title of the generated plot.

    Returns:
        np.ndarray: The 2D projected coordinates of shape (N, 2).
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Embeddings must be a numpy ndarray. Got: {type(embeddings)}")

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D matrix of shape (N, D). Got shape: {embeddings.shape}")

    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        raise ValueError(f"Embeddings must contain at least 2 samples to run t-SNE. Got: {n_samples}")

    # Clip perplexity to be less than the number of samples (t-SNE requirement)
    actual_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
    if actual_perplexity != perplexity:
        logger.warning(f"Perplexity was adjusted from {perplexity} to {actual_perplexity} to match sample size {n_samples}.")

    logger.info(f"Running t-SNE on {n_samples} embeddings of dimension {embeddings.shape[1]}...")
    
    # Initialize t-SNE (dynamically handle sklearn parameter rename from n_iter to max_iter)
    import inspect
    tsne_params = {
        "n_components": 2,
        "perplexity": actual_perplexity,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "init": "pca"
    }
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        tsne_params["max_iter"] = 1000
    elif "n_iter" in sig.parameters:
        tsne_params["n_iter"] = 1000

    tsne = TSNE(**tsne_params)
    
    try:
        # Fit and transform
        projected = tsne.fit_transform(embeddings)
    except Exception as e:
        logger.error(f"t-SNE fit failed: {e}")
        raise

    logger.info("t-SNE projection completed successfully. Generating plot...")

    # Create figure
    plt.figure(figsize=figsize, dpi=150)
    fig, ax = plt.subplots(figsize=figsize)

    # Color mapping logic
    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        
        # Use premium discrete color palette (tab10/tab20 style)
        cmap = plt.get_cmap("tab10") if len(unique_labels) <= 10 else plt.get_cmap("tab20")
        
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            color = cmap(i % 20)
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                label=str(label),
                color=color,
                alpha=0.8,
                edgecolors="none",
                s=25,
            )
        # Position legend outside the main plot area
        ax.legend(
            title="Classes",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
            shadow=False,
            framealpha=0.9
        )
    else:
        # Single-color plot if no labels provided
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            color="#1f77b4",
            alpha=0.7,
            edgecolors="none",
            s=25,
        )

    # Aesthetic styling (Premium Publication Style)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Hide top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved t-SNE plot to: {save_path}")
        
    plt.close("all")
    return projected


if __name__ == "__main__":
    # Demonstration of the visualization module
    logger.info("Initializing t-SNE visualization demo...")
    

    # Generate structured mock embeddings representing 3 clusters (classes)
    # This demonstrates the clustering visualization capabilities without random shapes.
    res = make_blobs(
        n_samples=300,
        n_features=128,  # Typical embedding dimension
        centers=3,
        cluster_std=1.5,
        random_state=42
    )
    mock_embeddings = res[0]
    mock_labels = res[1]
    
    # Convert labels to string categories
    label_names = [f"LandCover_Class_{lbl}" for lbl in mock_labels]
    
    out_plot = BASE_DIR / "outputs" / "sample_tsne_plot.png"
    
    try:
        generate_tsne_plot(
            embeddings=mock_embeddings,
            labels=label_names,
            save_path=out_plot,
            perplexity=30,
            learning_rate="auto",
            random_state=42,
            title="t-SNE Projection of Mock Land Cover Embeddings"
        )
        print("\n========================================")
        print("t-SNE Visualization Demo Successful!")
        print("========================================")
        print(f"Generated plot saved to: outputs/sample_tsne_plot.png")
        print("========================================\n")
    except Exception as e:
        logger.exception(f"t-SNE demo failed: {e}")
