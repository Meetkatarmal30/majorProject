"""
Week 5 - Extract trained test embeddings.

Loads the best Week 4 checkpoint and generates 512-D embeddings
for all 4,500 test SAR/MS pairs.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.paired_dataset import PairedBigEarthNetDataset
from src.models.two_tower import TwoTowerNetwork


# ============================================================
# PATHS
# ============================================================

TEST_CSV = Path("outputs/metadata/test_split.csv")

S1_ROOT = Path(
    r"C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required"
)

S2_ROOT = Path(
    r"C:\BigEarthNet-S2-Required"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/best_model-v1.ckpt"
)

OUTPUT_DIR = Path("outputs/embeddings")

SAR_OUTPUT = OUTPUT_DIR / "trained_test_sar_embeddings.npy"
MS_OUTPUT = OUTPUT_DIR / "trained_test_ms_embeddings.npy"


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 4


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model() -> TwoTowerNetwork:
    """
    Load the trained TwoTowerNetwork from the Lightning checkpoint.
    """

    print("Loading trained checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    network = TwoTowerNetwork(pretrained=False)

    checkpoint_state_dict = checkpoint["state_dict"]

    network_state_dict = {}

    for key, value in checkpoint_state_dict.items():
        if key.startswith("network."):
            network_state_dict[key[len("network."):]] = value

    missing_keys, unexpected_keys = network.load_state_dict(
        network_state_dict,
        strict=True,
    )

    if missing_keys:
        raise RuntimeError(
            f"Missing model keys: {missing_keys}"
        )

    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected model keys: {unexpected_keys}"
        )

    print("[OK] Trained model weights loaded.")

    return network


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_embeddings(
    model: TwoTowerNetwork,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract SAR and MS embeddings from the trained model.
    """

    model.eval()
    model.to(device)

    sar_embeddings = []
    ms_embeddings = []
    patch_ids = []

    print()
    print("Starting test embedding extraction...")
    print(f"Device       : {device}")
    print(f"Test samples : {len(dataloader.dataset):,}")
    print(f"Batch size   : {BATCH_SIZE}")
    print()

    with torch.inference_mode():

        for batch_index, batch in enumerate(dataloader, start=1):

            sar_batch, ms_batch, _, batch_patch_ids = batch

            sar_batch = sar_batch.to(
                device,
                non_blocking=True,
            )

            ms_batch = ms_batch.to(
                device,
                non_blocking=True,
            )

            sar_output, ms_output = model(
                sar_batch,
                ms_batch,
            )

            # Normalize embeddings for cosine similarity.
            sar_output = torch.nn.functional.normalize(
                sar_output,
                dim=1,
            )

            ms_output = torch.nn.functional.normalize(
                ms_output,
                dim=1,
            )

            sar_embeddings.append(
                sar_output.cpu().numpy()
            )

            ms_embeddings.append(
                ms_output.cpu().numpy()
            )

            patch_ids.extend(batch_patch_ids)

            if batch_index % 50 == 0:
                processed = min(
                    batch_index * BATCH_SIZE,
                    len(dataloader.dataset),
                )

                print(
                    f"Processed {processed:,} / "
                    f"{len(dataloader.dataset):,} test pairs"
                )

    sar_embeddings = np.concatenate(
        sar_embeddings,
        axis=0,
    )

    ms_embeddings = np.concatenate(
        ms_embeddings,
        axis=0,
    )

    return (
        sar_embeddings,
        ms_embeddings,
        patch_ids,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("WEEK 5 — TRAINED TEST EMBEDDING EXTRACTION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # CHECK PATHS
    # --------------------------------------------------------

    required_paths = {
        "Test CSV": TEST_CSV,
        "S1 root": S1_ROOT,
        "S2 root": S2_ROOT,
        "Checkpoint": CHECKPOINT_PATH,
    }

    for name, path in required_paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        print(f"[OK] {name}: {path}")

    print()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[OK] Device: {device}")

    if device.type == "cuda":
        print(
            f"[OK] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("Loading test dataset...")

    test_dataset = PairedBigEarthNetDataset(
        csv_path=TEST_CSV,
        s1_root=S1_ROOT,
        s2_root=S2_ROOT,
    )

    print(
        f"[OK] Test dataset: "
        f"{len(test_dataset):,} samples"
    )

    if len(test_dataset) != 4500:
        raise ValueError(
            f"Expected 4,500 test samples, "
            f"found {len(test_dataset)}."
        )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("[OK] Test DataLoader created.")
    print()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_trained_model()

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    (
        sar_embeddings,
        ms_embeddings,
        patch_ids,
    ) = extract_embeddings(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print()
    print("Validating embeddings...")

    print(
        f"SAR embedding shape: "
        f"{sar_embeddings.shape}"
    )

    print(
        f"MS embedding shape : "
        f"{ms_embeddings.shape}"
    )

    if sar_embeddings.shape != (4500, 512):
        raise ValueError(
            "Unexpected SAR embedding shape: "
            f"{sar_embeddings.shape}"
        )

    if ms_embeddings.shape != (4500, 512):
        raise ValueError(
            "Unexpected MS embedding shape: "
            f"{ms_embeddings.shape}"
        )

    if not np.isfinite(sar_embeddings).all():
        raise ValueError(
            "SAR embeddings contain NaN or Inf."
        )

    if not np.isfinite(ms_embeddings).all():
        raise ValueError(
            "MS embeddings contain NaN or Inf."
        )

    if len(patch_ids) != 4500:
        raise ValueError(
            f"Expected 4,500 patch IDs, "
            f"found {len(patch_ids)}."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        SAR_OUTPUT,
        sar_embeddings.astype(np.float32),
    )

    np.save(
        MS_OUTPUT,
        ms_embeddings.astype(np.float32),
    )

    print()
    print("=" * 70)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"SAR embeddings : {SAR_OUTPUT}"
    )

    print(
        f"MS embeddings  : {MS_OUTPUT}"
    )

    print()
    print(
        f"SAR shape      : "
        f"{sar_embeddings.shape}"
    )

    print(
        f"MS shape       : "
        f"{ms_embeddings.shape}"
    )

    print(
        f"Patch IDs      : "
        f"{len(patch_ids):,}"
    )

    print()
    print("SUCCESS: trained test embeddings generated.")


if __name__ == "__main__":
    main()