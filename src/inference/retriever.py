"""
retriever.py - Cross-Modal Satellite Image Retriever with Hybrid Support.

Core Week 5 inference module integrating:
- Trained TwoTowerNetwork (ResNet-50 towers + 512-D projections)
- Model checkpoint (models/best_model-v1.ckpt)
- Sentinel-1 (SAR: VV+VH) and Sentinel-2 (MS: B04,B03,B02) preprocessing
- FAISS L2-normalized IndexFlatIP search (inner product = cosine similarity)
  * SAR query -> MS gallery (outputs/indices/ms_gallery.index)
  * MS query -> SAR gallery (outputs/indices/sar_gallery.index)
  * Hybrid query -> Dual gallery search (MS + SAR) with deterministic score merge
- Patch ID mapping and metadata resolution from teammate_inputs/test_split.csv
- Real-time thumbnail rendering from local raw patch directories
- Domain and subcategory configuration routing
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rasterio.errors import NotGeoreferencedWarning
except ImportError:
    NotGeoreferencedWarning = UserWarning

import torch
import torch.nn.functional as F
import faiss
import numpy as np
import pandas as pd
from PIL import Image

# Import existing preprocessing and model
from preprocessing.sar_preprocessing import (
    clip_sar_band,
    load_s1_stats,
    load_sar_band,
    normalize_sar_band,
    preprocess_sar_patch,
    resize_sar_band,
)
from preprocessing.preprocess_s2 import (
    load_s2_stats,
    normalize_s2_image,
    preprocess_s2_patch,
    read_rgb_patch,
    resize_s2_image,
)
from src.models.two_tower import TwoTowerNetwork

logger = logging.getLogger("CrossModalRetriever")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ============================================================
# DEFAULT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_CHECKPOINT_PATH = BASE_DIR / "models" / "best_model-v1.ckpt"
DEFAULT_SAR_INDEX_PATH = BASE_DIR / "outputs" / "indices" / "sar_gallery.index"
DEFAULT_MS_INDEX_PATH = BASE_DIR / "outputs" / "indices" / "ms_gallery.index"

DEFAULT_SAR_PATCH_IDS_PATH = BASE_DIR / "teammate_inputs" / "trained_test_sar_patch_ids.txt"
DEFAULT_MS_PATCH_IDS_PATH = BASE_DIR / "teammate_inputs" / "trained_test_ms_patch_ids.txt"
DEFAULT_TEST_SPLIT_PATH = BASE_DIR / "teammate_inputs" / "test_split.csv"

DEFAULT_S1_STATS_PATH = BASE_DIR / "outputs" / "s1_stats.json"
DEFAULT_S2_STATS_PATH = BASE_DIR / "teammate_inputs" / "s2_stats.json"
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "disaster_modality_config.json"


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def load_modality_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads authoritative disaster modality routing configuration."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration not found at: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_subcategory_info(
    subcategory_name_or_slug: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolves domain and subcategory details given a subcategory slug or name.
    Matches case-insensitively across configured domains.
    """
    if config is None:
        config = load_modality_config()

    target = subcategory_name_or_slug.strip().lower().replace("_", "-")

    # Common aliases
    alias_map = {
        "flood-mapping": "flood",
        "wildfire-scar": "wildfire",
        "forest-change": "deforestation",
        "water-quality": "drought",
        "crop-monitoring": "crop-mapping",
    }
    target = alias_map.get(target, target)

    for domain_slug, domain_info in config.get("domains", {}).items():
        for sub_slug, sub_info in domain_info.get("subcategories", {}).items():
            if (
                target == sub_slug.lower()
                or target == sub_info.get("name", "").strip().lower()
                or target == sub_slug.replace("-", " ")
            ):
                result = dict(sub_info)
                result["domain_slug"] = domain_slug
                result["domain_name"] = domain_info.get("name", "")
                result["subcategory_slug"] = sub_slug
                return result

    return None


# ============================================================
# RETRIEVER CLASS
# ============================================================

class CrossModalRetriever:
    """
    Production-ready cross-modal retriever for satellite imagery.
    Loads the trained TwoTowerNetwork and FAISS galleries to perform:
    - SAR -> MS retrieval (cross-modal)
    - MS -> SAR retrieval (cross-modal)
    - Hybrid retrieval (dual gallery search + deterministic score merge)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        sar_index_path: Optional[Union[str, Path]] = None,
        ms_index_path: Optional[Union[str, Path]] = None,
        sar_patch_ids_path: Optional[Union[str, Path]] = None,
        ms_patch_ids_path: Optional[Union[str, Path]] = None,
        test_split_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.sar_index_path = Path(sar_index_path) if sar_index_path else DEFAULT_SAR_INDEX_PATH
        self.ms_index_path = Path(ms_index_path) if ms_index_path else DEFAULT_MS_INDEX_PATH
        self.sar_patch_ids_path = Path(sar_patch_ids_path) if sar_patch_ids_path else DEFAULT_SAR_PATCH_IDS_PATH
        self.ms_patch_ids_path = Path(ms_patch_ids_path) if ms_patch_ids_path else DEFAULT_MS_PATCH_IDS_PATH
        self.test_split_path = Path(test_split_path) if test_split_path else DEFAULT_TEST_SPLIT_PATH

        # 1. Load model
        self.model = self._load_model()

        # 2. Load FAISS indices
        self.sar_index, self.ms_index = self._load_indices()

        # 3. Load patch IDs
        self.sar_patch_ids, self.ms_patch_ids = self._load_patch_ids()

        # 4. Load metadata dictionary from test_split.csv
        self.s1_to_patch_id: Dict[str, str] = {}
        self.s2v1_to_patch_id: Dict[str, str] = {}
        self.metadata_lookup = self._load_metadata()

        # 5. Pre-load normalization stats
        self.s1_stats = load_s1_stats(DEFAULT_S1_STATS_PATH)
        self.s2_mean, self.s2_std = load_s2_stats(DEFAULT_S2_STATS_PATH)

        # 6. Lazy patch cache for image thumbnails
        self._local_s1_patches: Optional[Dict[str, Path]] = None
        self._local_s2_patches: Optional[Dict[str, Path]] = None
        self._patch_cache_initialized: bool = False

        # Load authoritative modality config
        self.config = load_modality_config()

        logger.info(
            "CrossModalRetriever initialized successfully. "
            f"SAR Gallery: {self.sar_index.ntotal} vectors, MS Gallery: {self.ms_index.ntotal} vectors."
        )

    # --------------------------------------------------------
    # INITIALIZATION HELPERS
    # --------------------------------------------------------

    def _load_model(self) -> TwoTowerNetwork:
        """Load trained TwoTowerNetwork checkpoint in eval mode."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)

        if "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint at {self.checkpoint_path} missing 'state_dict' key.")

        raw_state_dict = checkpoint["state_dict"]
        # Strip 'network.' prefix from PyTorch Lightning checkpoint keys
        clean_state_dict = {
            k.replace("network.", ""): v
            for k, v in raw_state_dict.items()
            if k.startswith("network.")
        }

        model = TwoTowerNetwork(pretrained=False)
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=True)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"State dict mismatch! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
            )

        model.to(self.device)
        model.eval()
        return model

    def _load_indices(self) -> Tuple[faiss.Index, faiss.Index]:
        """Load FAISS gallery indices."""
        if not self.sar_index_path.exists():
            raise FileNotFoundError(f"SAR gallery index not found at: {self.sar_index_path}")
        if not self.ms_index_path.exists():
            raise FileNotFoundError(f"MS gallery index not found at: {self.ms_index_path}")

        sar_index = faiss.read_index(str(self.sar_index_path))
        ms_index = faiss.read_index(str(self.ms_index_path))

        if sar_index.d != 512 or ms_index.d != 512:
            raise ValueError(f"Expected index dimension 512, got SAR={sar_index.d}, MS={ms_index.d}")

        return sar_index, ms_index

    def _load_patch_ids(self) -> Tuple[List[str], List[str]]:
        """Load ordered test patch IDs for SAR and MS galleries."""
        if not self.sar_patch_ids_path.exists():
            raise FileNotFoundError(f"SAR patch IDs file not found: {self.sar_patch_ids_path}")
        if not self.ms_patch_ids_path.exists():
            raise FileNotFoundError(f"MS patch IDs file not found: {self.ms_patch_ids_path}")

        with open(self.sar_patch_ids_path, "r", encoding="utf-8") as f:
            sar_ids = [line.strip() for line in f if line.strip()]

        with open(self.ms_patch_ids_path, "r", encoding="utf-8") as f:
            ms_ids = [line.strip() for line in f if line.strip()]

        if len(sar_ids) != self.sar_index.ntotal:
            raise ValueError(
                f"SAR patch IDs count ({len(sar_ids)}) != SAR index count ({self.sar_index.ntotal})"
            )
        if len(ms_ids) != self.ms_index.ntotal:
            raise ValueError(
                f"MS patch IDs count ({len(ms_ids)}) != MS index count ({self.ms_index.ntotal})"
            )

        return sar_ids, ms_ids

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load and index test_split.csv metadata by patch_id, plus reverse maps."""
        if not self.test_split_path.exists():
            raise FileNotFoundError(f"test_split.csv not found at: {self.test_split_path}")

        df = pd.read_csv(self.test_split_path)
        lookup: Dict[str, Dict[str, Any]] = {}
        self.s1_to_patch_id = {}
        self.s2v1_to_patch_id = {}

        for _, row in df.iterrows():
            pid = str(row["patch_id"]).strip()
            raw_labels = str(row.get("labels", ""))
            extracted_labels = re.findall(r"'([^']+)'", raw_labels)
            if not extracted_labels and raw_labels:
                extracted_labels = [l.strip(" []'\"") for l in raw_labels.split(",") if l.strip(" []'\"")]

            primary_label = extracted_labels[0] if extracted_labels else "Unlabeled"
            s1_name = str(row.get("s1_name", "")).strip()
            s2v1_name = str(row.get("s2v1_name", "")).strip()

            if s1_name and s1_name.lower() not in ("nan", "none", ""):
                self.s1_to_patch_id[s1_name] = pid
            if s2v1_name and s2v1_name.lower() not in ("nan", "none", ""):
                self.s2v1_to_patch_id[s2v1_name] = pid

            lookup[pid] = {
                "patch_id": pid,
                "label": primary_label,
                "labels": extracted_labels,
                "country": str(row.get("country", "")),
                "s1_name": s1_name,
                "s2v1_name": s2v1_name,
                "split": str(row.get("split", "test")),
            }

        return lookup

    def get_metadata(self, identifier: Optional[str]) -> Dict[str, Any]:
        """
        Deterministically resolve metadata record by patch_id, s1_name, or s2v1_name.
        Safely handles None, empty string, or missing records.
        """
        if not identifier:
            return {}
        clean_id = str(identifier).strip()
        # 1. Primary lookup by patch_id
        if clean_id in self.metadata_lookup:
            return self.metadata_lookup[clean_id]
        # 2. Reverse lookup by s1_name
        if hasattr(self, "s1_to_patch_id") and clean_id in self.s1_to_patch_id:
            pid = self.s1_to_patch_id[clean_id]
            return self.metadata_lookup.get(pid, {})
        # 3. Reverse lookup by s2v1_name
        if hasattr(self, "s2v1_to_patch_id") and clean_id in self.s2v1_to_patch_id:
            pid = self.s2v1_to_patch_id[clean_id]
            return self.metadata_lookup.get(pid, {})
        return {}

    # --------------------------------------------------------
    # LOCAL THUMBNAIL CACHE & RENDERER
    # --------------------------------------------------------

    def _ensure_patch_cache(self) -> None:
        """
        Lazily builds lookup table for local raw patch directories.
        Guaranteed to run at most once to avoid scanning the filesystem on every request.
        Maps both v2 patch_id and v1 s2v1_name for Sentinel-2,
        and s1_name and patch_id for Sentinel-1.
        """
        if getattr(self, "_patch_cache_initialized", False):
            return

        self._local_s1_patches = {}
        self._local_s2_patches = {}

        s1_dir = BASE_DIR / "teammate_inputs" / "BigEarthNet-S1-Required"
        s2_dir = BASE_DIR / "teammate_inputs" / "BigEarthNet-S2-Required"

        if s1_dir.exists():
            for scene_dir in s1_dir.iterdir():
                if scene_dir.is_dir():
                    for patch_dir in scene_dir.iterdir():
                        if patch_dir.is_dir():
                            name = patch_dir.name
                            self._local_s1_patches[name] = patch_dir
                            if hasattr(self, "s1_to_patch_id") and name in self.s1_to_patch_id:
                                pid = self.s1_to_patch_id[name]
                                self._local_s1_patches[pid] = patch_dir

        if s2_dir.exists():
            for scene_dir in s2_dir.iterdir():
                if scene_dir.is_dir():
                    for patch_dir in scene_dir.iterdir():
                        if patch_dir.is_dir():
                            name = patch_dir.name
                            self._local_s2_patches[name] = patch_dir
                            if name in self.metadata_lookup:
                                s2v1 = self.metadata_lookup[name].get("s2v1_name")
                                if s2v1:
                                    self._local_s2_patches[s2v1] = patch_dir
                            elif hasattr(self, "s2v1_to_patch_id") and name in self.s2v1_to_patch_id:
                                pid = self.s2v1_to_patch_id[name]
                                self._local_s2_patches[pid] = patch_dir

        self._patch_cache_initialized = True
        logger.info(
            f"Patch cache built: {len(self._local_s1_patches)} S1 entries, "
            f"{len(self._local_s2_patches)} S2 entries."
        )

    def _render_s2_thumbnail(self, patch_id: str, meta: Dict[str, Any]) -> Optional[Image.Image]:
        """Render RGB thumbnail for Sentinel-2 using B04/B03/B02 with NaN/Inf/nodata/constant safety."""
        folder = None
        for key in (patch_id, meta.get("s2v1_name"), meta.get("patch_id")):
            if key and key in self._local_s2_patches:
                folder = self._local_s2_patches[key]
                break

        if folder is None or not folder.is_dir():
            return None

        try:
            b04_files = sorted(folder.glob("*_B04.tif"))
            b03_files = sorted(folder.glob("*_B03.tif"))
            b02_files = sorted(folder.glob("*_B02.tif"))
            if not (b04_files and b03_files and b02_files):
                return None

            import rasterio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(b04_files[0]) as s4, \
                     rasterio.open(b03_files[0]) as s3, \
                     rasterio.open(b02_files[0]) as s2:
                    nodata4 = s4.nodata
                    nodata3 = s3.nodata
                    nodata2 = s2.nodata
                    b04 = s4.read(1).astype(np.float32)
                    b03 = s3.read(1).astype(np.float32)
                    b02 = s2.read(1).astype(np.float32)

            if b04.shape != b03.shape or b04.shape != b02.shape or b04.ndim != 2:
                return None

            # Correct RGB ordering: B04 (Red), B03 (Green), B02 (Blue)
            rgb = np.stack([b04, b03, b02], axis=-1)  # (H, W, 3)
            mask = np.isfinite(rgb)
            if nodata4 is not None:
                mask[..., 0] &= (rgb[..., 0] != nodata4)
            if nodata3 is not None:
                mask[..., 1] &= (rgb[..., 1] != nodata3)
            if nodata2 is not None:
                mask[..., 2] &= (rgb[..., 2] != nodata2)

            valid_mask = mask.all(axis=-1)
            if not np.any(valid_mask):
                return Image.fromarray(np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8))

            valid_pixels = rgb[valid_mask]
            p2, p98 = np.percentile(valid_pixels, (2, 98))
            denom = p98 - p2
            if denom <= 1e-5:
                scaled = np.full(rgb.shape, 128, dtype=np.uint8)
                scaled = np.where(valid_mask[..., None], scaled, 0).astype(np.uint8)
            else:
                clean_rgb = np.clip(rgb, p2, p98)
                scaled_calc = np.clip((clean_rgb - p2) / denom * 255.0, 0, 255).astype(np.uint8)
                scaled = np.where(valid_mask[..., None], scaled_calc, 0).astype(np.uint8)

            return Image.fromarray(scaled)
        except Exception as e:
            logger.warning(f"Error rendering S2 thumbnail for {patch_id}: {e}")
            return None

    def _render_s1_thumbnail(self, patch_id: str, meta: Dict[str, Any]) -> Optional[Image.Image]:
        """Render grayscale-to-RGB thumbnail for Sentinel-1 with NaN/Inf/nodata/constant safety and percentile stretch."""
        folder = None
        for key in (meta.get("s1_name"), patch_id, meta.get("patch_id")):
            if key and key in self._local_s1_patches:
                folder = self._local_s1_patches[key]
                break

        if folder is None or not folder.is_dir():
            return None

        try:
            vv_files = sorted(folder.glob("*_VV.tif"))
            if not vv_files:
                return None

            import rasterio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(vv_files[0]) as src:
                    nodata = src.nodata
                    vv = src.read(1).astype(np.float32)

            if vv.ndim != 2:
                return None

            mask = np.isfinite(vv)
            if nodata is not None:
                mask &= (vv != nodata)

            if not np.any(mask):
                return Image.fromarray(np.zeros((vv.shape[0], vv.shape[1], 3), dtype=np.uint8))

            valid_vv = vv[mask]
            p2, p98 = np.percentile(valid_vv, (2, 98))
            denom = p98 - p2
            if denom <= 1e-5:
                scaled_vv = np.where(mask, 128, 0).astype(np.uint8)
            else:
                clean_vv = np.clip(vv, p2, p98)
                scaled_calc = np.clip((clean_vv - p2) / denom * 255.0, 0, 255).astype(np.uint8)
                scaled_vv = np.where(mask, scaled_calc, 0).astype(np.uint8)

            return Image.fromarray(scaled_vv).convert("RGB")
        except Exception as e:
            logger.warning(f"Error rendering S1 thumbnail for {patch_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error rendering S1 thumbnail for {patch_id}: {e}")
            return None

    def render_patch_thumbnail(self, patch_id: str, modality: Optional[str] = None) -> Optional[Image.Image]:
        """
        Renders a PIL RGB thumbnail for a given patch_id if available locally.
        Strictly respects the requested modality:
        - If requested modality is MS/Multispectral/S2: only renders Sentinel-2.
        - If requested modality is SAR/S1: only renders Sentinel-1.
        - If requested modality is Hybrid: renders dual composite if both available, else None.
        - If modality is None: permits fallback (tries S2 first, then S1).
        - If modality is unsupported: returns None gracefully.
        - Never silently substitutes one modality for another.
        """
        self._ensure_patch_cache()
        meta = self.get_metadata(patch_id)

        norm_mod = modality.strip().upper() if modality else None

        # Check for unsupported modality
        if norm_mod is not None and norm_mod not in (
            "MS", "MULTISPECTRAL", "S2", "SENTINEL-2",
            "SAR", "S1", "SENTINEL-1",
            "HYBRID", "DUAL",
        ):
            logger.warning(f"Unsupported modality requested for thumbnail: {modality}")
            return None

        # 1. Multispectral (Sentinel-2)
        if norm_mod in ("MS", "MULTISPECTRAL", "S2", "SENTINEL-2") or norm_mod is None:
            s2_img = self._render_s2_thumbnail(patch_id, meta)
            if s2_img is not None:
                return s2_img
            if norm_mod is not None:
                # Strictly DO NOT fall back to SAR if MS was explicitly requested
                return None

        # 2. SAR (Sentinel-1)
        if norm_mod in ("SAR", "S1", "SENTINEL-1") or norm_mod is None:
            s1_img = self._render_s1_thumbnail(patch_id, meta)
            if s1_img is not None:
                return s1_img
            if norm_mod is not None:
                # Strictly DO NOT fall back to MS if SAR was explicitly requested
                return None

        # 3. Hybrid
        if norm_mod in ("HYBRID", "DUAL"):
            s2_img = self._render_s2_thumbnail(patch_id, meta)
            s1_img = self._render_s1_thumbnail(patch_id, meta)
            if s2_img is not None and s1_img is not None:
                w1, h1 = s2_img.size
                w2, h2 = s1_img.size
                h = max(h1, h2)
                composite = Image.new("RGB", (w1 + w2, h), (0, 0, 0))
                composite.paste(s2_img, (0, (h - h1) // 2))
                composite.paste(s1_img, (w1, (h - h2) // 2))
                return composite
            return None

        return None

    # --------------------------------------------------------
    # PREPROCESSING METHODS
    # --------------------------------------------------------

    def preprocess_sar(self, query_input: Union[str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Preprocess SAR query input to a (1, 2, 224, 224) Float32 PyTorch tensor.
        Handles:
        - Path to folder containing VV and VH GeoTIFFs
        - Path to single GeoTIFF / image file (with auto-partner band resolution or multi-channel read)
        - Direct torch.Tensor or numpy.ndarray
        """
        if isinstance(query_input, torch.Tensor):
            t = query_input.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != 2 or t.shape[2:] != (224, 224):
                raise ValueError(f"SAR tensor must have shape (1, 2, 224, 224), got {t.shape}")
            return t.to(self.device)

        if isinstance(query_input, np.ndarray):
            arr = query_input.astype(np.float32)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, 0)
            t = torch.from_numpy(arr)
            return self.preprocess_sar(t)

        input_path = Path(query_input)
        if not input_path.exists():
            raise FileNotFoundError(f"SAR query input path does not exist: {input_path}")

        # Directory containing bands
        if input_path.is_dir():
            vv_files = list(input_path.glob("*_VV.tif")) + list(input_path.glob("*VV*.tif"))
            vh_files = list(input_path.glob("*_VH.tif")) + list(input_path.glob("*VH*.tif"))
            if not vv_files or not vh_files:
                raise FileNotFoundError(f"Directory {input_path} missing VV or VH GeoTIFFs.")
            tensor = preprocess_sar_patch(vv_files[0], vh_files[0])
            return tensor.unsqueeze(0).to(self.device)

        # Single file
        if input_path.is_file():
            name = input_path.name
            parent = input_path.parent
            if "_VV" in name:
                partner_name = name.replace("_VV", "_VH")
                if (parent / partner_name).exists():
                    tensor = preprocess_sar_patch(input_path, parent / partner_name)
                    return tensor.unsqueeze(0).to(self.device)
            elif "_VH" in name:
                partner_name = name.replace("_VH", "_VV")
                if (parent / partner_name).exists():
                    tensor = preprocess_sar_patch(parent / partner_name, input_path)
                    return tensor.unsqueeze(0).to(self.device)

            # Fallback loading
            try:
                import rasterio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(input_path) as src:
                        if src.count >= 2:
                            vv_raw = src.read(1).astype(np.float32)
                            vh_raw = src.read(2).astype(np.float32)
                        else:
                            vv_raw = src.read(1).astype(np.float32)
                            vh_raw = vv_raw.copy()
            except Exception:
                img = Image.open(input_path).convert("L")
                vv_raw = np.array(img, dtype=np.float32)
                vh_raw = vv_raw.copy()

            vv_clipped = clip_sar_band(vv_raw, (-25.0, 0.0))
            vh_clipped = clip_sar_band(vh_raw, (-25.0, 0.0))

            vv_norm = normalize_sar_band(vv_clipped, (-25.0, 0.0), "z-score", self.s1_stats["VV"])
            vh_norm = normalize_sar_band(vh_clipped, (-25.0, 0.0), "z-score", self.s1_stats["VH"])

            vv_final = resize_sar_band(vv_norm, (224, 224))
            vh_final = resize_sar_band(vh_norm, (224, 224))

            stacked = np.stack([vv_final, vh_final], axis=0).astype(np.float32)
            tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
            return tensor

        raise ValueError(f"Unsupported SAR input: {query_input}")

    def preprocess_ms(self, query_input: Union[str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Preprocess Multispectral query input to a (1, 3, 224, 224) Float32 PyTorch tensor.
        Handles:
        - Path to patch folder containing B04, B03, B02 GeoTIFFs
        - Path to single RGB image file (TIFF, PNG, JPG)
        - Direct torch.Tensor or numpy.ndarray
        """
        if isinstance(query_input, torch.Tensor):
            t = query_input.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != 3 or t.shape[2:] != (224, 224):
                raise ValueError(f"MS tensor must have shape (1, 3, 224, 224), got {t.shape}")
            return t.to(self.device)

        if isinstance(query_input, np.ndarray):
            arr = query_input.astype(np.float32)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, 0)
            t = torch.from_numpy(arr)
            return self.preprocess_ms(t)

        input_path = Path(query_input)
        if not input_path.exists():
            raise FileNotFoundError(f"MS query input path does not exist: {input_path}")

        # Directory containing S2 bands
        if input_path.is_dir():
            tensor = preprocess_s2_patch(input_path, target_size=(224, 224))
            return tensor.unsqueeze(0).to(self.device)

        # Single file
        if input_path.is_file():
            name = input_path.name
            parent = input_path.parent
            b04_file, b03_file, b02_file = None, None, None
            for band in ("_B04", "_B03", "_B02"):
                if band in name:
                    b4 = parent / name.replace(band, "_B04")
                    b3 = parent / name.replace(band, "_B03")
                    b2 = parent / name.replace(band, "_B02")
                    if b4.exists() and b3.exists() and b2.exists():
                        b04_file, b03_file, b02_file = b4, b3, b2
                        break

            if b04_file and b03_file and b02_file:
                import rasterio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(b04_file) as s4, rasterio.open(b03_file) as s3, rasterio.open(b02_file) as s2:
                        b4_arr = s4.read(1).astype(np.float32)
                        b3_arr = s3.read(1).astype(np.float32)
                        b2_arr = s2.read(1).astype(np.float32)
                rgb = np.stack([b4_arr, b3_arr, b2_arr], axis=0)
                resized = resize_s2_image(rgb, target_size=(224, 224))
                normalized = normalize_s2_image(resized, mean=self.s2_mean, std=self.s2_std)
                tensor = torch.from_numpy(normalized).type(torch.float32).unsqueeze(0).to(self.device)
                return tensor

            try:
                import rasterio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    with rasterio.open(input_path) as src:
                        if src.count >= 3:
                            rgb = src.read([1, 2, 3]).astype(np.float32)
                        else:
                            single = src.read(1).astype(np.float32)
                            rgb = np.stack([single, single, single], axis=0)
            except Exception:
                img = Image.open(input_path).convert("RGB")
                arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)
                rgb = arr

            resized = resize_s2_image(rgb, target_size=(224, 224))
            normalized = normalize_s2_image(resized, mean=self.s2_mean, std=self.s2_std)
            tensor = torch.from_numpy(normalized).type(torch.float32).unsqueeze(0).to(self.device)
            return tensor

        raise ValueError(f"Unsupported MS input: {query_input}")

    # --------------------------------------------------------
    # EMBEDDING INFERENCE
    # --------------------------------------------------------

    def encode_sar(self, sar_tensor: torch.Tensor) -> np.ndarray:
        """
        Runs SAR ResNet-50 tower + projection head -> 512-D L2-normalized embedding.
        Input: (B, 2, 224, 224)
        Output: (B, 512) float32 numpy array
        """
        sar_tensor = sar_tensor.to(self.device)
        with torch.no_grad():
            features = self.model.sar_encoder(sar_tensor)
            proj = self.model.sar_projection(features)
            normalized = F.normalize(proj, p=2, dim=1)
            emb = normalized.cpu().numpy().astype(np.float32)
        return emb

    def encode_ms(self, ms_tensor: torch.Tensor) -> np.ndarray:
        """
        Runs MS ResNet-50 tower + projection head -> 512-D L2-normalized embedding.
        Input: (B, 3, 224, 224)
        Output: (B, 512) float32 numpy array
        """
        ms_tensor = ms_tensor.to(self.device)
        with torch.no_grad():
            features = self.model.ms_encoder(ms_tensor)
            proj = self.model.ms_projection(features)
            normalized = F.normalize(proj, p=2, dim=1)
            emb = normalized.cpu().numpy().astype(np.float32)
        return emb

    # --------------------------------------------------------
    # FAISS GALLERY SEARCH
    # --------------------------------------------------------

    def search_gallery(
        self,
        query_embedding: np.ndarray,
        target_gallery: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search target FAISS gallery and map indices to patch IDs and metadata.

        Args:
            query_embedding: L2-normalized 512-D vector of shape (1, 512) or (512,)
            target_gallery: "MS" or "SAR"
            k: number of nearest neighbors to retrieve (default: 10)

        Returns:
            List of structured result dicts sorted descending by similarity score.
        """
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, 0)

        if query_embedding.shape != (1, 512):
            raise ValueError(f"Query embedding shape must be (1, 512), got {query_embedding.shape}")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains NaN or Inf.")

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        target = target_gallery.strip().upper()
        if target == "MS":
            index = self.ms_index
            patch_ids = self.ms_patch_ids
            result_modality = "Multispectral"
        elif target == "SAR":
            index = self.sar_index
            patch_ids = self.sar_patch_ids
            result_modality = "SAR"
        else:
            raise ValueError(f"Unsupported target gallery: {target_gallery}. Must be 'MS' or 'SAR'.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        actual_k = min(k, index.ntotal)
        if actual_k == 0:
            return []

        scores, indices = index.search(query_embedding, actual_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(patch_ids):
                continue

            patch_id = patch_ids[idx]
            meta = self.get_metadata(patch_id)
            sim_score = float(score)

            result_entry: Dict[str, Any] = {
                "rank": rank,
                "patch_id": patch_id,
                "similarity_score": sim_score,
                "score": f"{sim_score:.4f}",
                "modality": result_modality,
                "label": meta.get("label", "Unlabeled"),
                "labels": meta.get("labels", []),
                "country": meta.get("country", ""),
                "s1_name": meta.get("s1_name", ""),
                "s2v1_name": meta.get("s2v1_name", ""),
                "split": meta.get("split", "test"),
                "tone": ((rank - 1) % 5) + 1,
                "provenance": f"gallery:{target}",
            }
            results.append(result_entry)

        return results

    # --------------------------------------------------------
    # HYBRID RETRIEVAL IMPLEMENTATION
    # --------------------------------------------------------

    def retrieve_hybrid(
        self,
        query_image_path: Union[str, Path, torch.Tensor, np.ndarray],
        query_modality: str = "auto",
        subcategory: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic retrieval-level Hybrid search:
        1. Encodes the uploaded query image using its native tower (SAR or MS).
        2. Queries BOTH FAISS galleries:
           - Cross-modal gallery (e.g. SAR -> MS or MS -> SAR)
           - Unimodal gallery (e.g. SAR -> SAR or MS -> MS)
        3. Tags provenance clearly on every retrieved result candidate.
        4. Deterministically merges the 2k candidate list sorted descending by similarity score.
        5. Deduplicates patch IDs to ensure unique scenes in top results.
        6. Returns top-k structured results preserving modality, score, rank, and metadata.
        """
        # Determine actual query modality
        modality = query_modality.strip().upper()
        if modality in ("AUTO", "HYBRID", "DUAL"):
            if isinstance(query_input := query_image_path, (torch.Tensor, np.ndarray)):
                channels = query_input.shape[1] if query_input.ndim >= 3 else query_input.shape[0]
                modality = "SAR" if channels == 2 else "MS"
            else:
                p_str = str(query_image_path).upper()
                if "_VV" in p_str or "_VH" in p_str or "S1" in p_str:
                    modality = "SAR"
                else:
                    modality = "MS"

        if modality in ("SAR", "S1", "SENTINEL-1"):
            sar_tensor = self.preprocess_sar(query_image_path)
            query_emb = self.encode_sar(sar_tensor)
            # Retrieve k candidates from MS gallery (cross-modal)
            res_cross = self.search_gallery(query_emb, target_gallery="MS", k=k)
            for r in res_cross:
                r["provenance"] = "cross-modal (SAR -> MS)"
            # Retrieve k candidates from SAR gallery (unimodal)
            res_uni = self.search_gallery(query_emb, target_gallery="SAR", k=k)
            for r in res_uni:
                r["provenance"] = "unimodal (SAR -> SAR)"

        elif modality in ("MS", "S2", "MULTISPECTRAL", "SENTINEL-2"):
            ms_tensor = self.preprocess_ms(query_image_path)
            query_emb = self.encode_ms(ms_tensor)
            # Retrieve k candidates from SAR gallery (cross-modal)
            res_cross = self.search_gallery(query_emb, target_gallery="SAR", k=k)
            for r in res_cross:
                r["provenance"] = "cross-modal (MS -> SAR)"
            # Retrieve k candidates from MS gallery (unimodal)
            res_uni = self.search_gallery(query_emb, target_gallery="MS", k=k)
            for r in res_uni:
                r["provenance"] = "unimodal (MS -> MS)"
        else:
            raise ValueError(f"Cannot resolve query modality '{query_modality}' for Hybrid retrieval.")

        # Deterministic merge sorted descending by similarity score
        combined = res_cross + res_uni
        combined.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Deduplicate while preserving highest score
        seen_pids = set()
        merged: List[Dict[str, Any]] = []
        for item in combined:
            pid = item["patch_id"]
            if pid not in seen_pids:
                seen_pids.add(pid)
                merged.append(item)
            if len(merged) == k:
                break

        # Re-assign ranks 1..k
        for rank, item in enumerate(merged, start=1):
            item["rank"] = rank
            item["tone"] = ((rank - 1) % 5) + 1
            if subcategory:
                item["subcategory"] = subcategory

        return merged

    # --------------------------------------------------------
    # MAIN RETRIEVE INTERFACE
    # --------------------------------------------------------

    def retrieve(
        self,
        query_image_path: Union[str, Path, torch.Tensor, np.ndarray],
        query_modality: str,
        subcategory: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method fulfilling Week 5 interface:
            retrieve(query_image_path, query_modality, subcategory=None, k=10)

        Flow:
            1. Resolve subcategory and modality routing from config.
            2. If subcategory or query_modality is 'Hybrid', execute retrieve_hybrid.
            3. Otherwise run standard cross-modal retrieval:
               - SAR query -> MS gallery
               - MS query -> SAR gallery
            4. Return top-k structured results.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        # Check subcategory configuration if provided
        configured_modality = None
        if subcategory:
            sub_info = resolve_subcategory_info(subcategory, self.config)
            if sub_info:
                configured_modality = sub_info.get("modality", "")

        is_hybrid = (
            query_modality.strip().upper() in ("HYBRID", "DUAL")
            or (configured_modality and configured_modality.strip().upper() == "HYBRID")
        )

        if is_hybrid:
            effective_mod = "SAR" if query_modality.strip().upper() in ("SAR", "S1") else "MS"
            if query_modality.strip().upper() in ("HYBRID", "DUAL", "AUTO"):
                effective_mod = "auto"
            return self.retrieve_hybrid(
                query_image_path=query_image_path,
                query_modality=effective_mod,
                subcategory=subcategory,
                k=k,
            )

        modality = query_modality.strip().upper()

        if modality in ("SAR", "S1", "SENTINEL-1"):
            sar_tensor = self.preprocess_sar(query_image_path)
            query_emb = self.encode_sar(sar_tensor)
            # SAR query searches Multispectral gallery
            results = self.search_gallery(query_emb, target_gallery="MS", k=k)
            for r in results:
                r["provenance"] = "cross-modal (SAR -> MS)"

        elif modality in ("MS", "S2", "MULTISPECTRAL", "SENTINEL-2"):
            ms_tensor = self.preprocess_ms(query_image_path)
            query_emb = self.encode_ms(ms_tensor)
            # MS query searches SAR gallery
            results = self.search_gallery(query_emb, target_gallery="SAR", k=k)
            for r in results:
                r["provenance"] = "cross-modal (MS -> SAR)"

        else:
            raise ValueError(
                f"Unsupported query modality: '{query_modality}'. Expected 'SAR', 'MS'/'Multispectral', or 'Hybrid'."
            )

        if subcategory:
            for item in results:
                item["subcategory"] = subcategory

        return results


# ============================================================
# MODULE-LEVEL CONVENIENCE
# ============================================================

_RETRIEVER_INSTANCE: Optional[CrossModalRetriever] = None


def get_retriever() -> CrossModalRetriever:
    """Singleton getter for CrossModalRetriever."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = CrossModalRetriever()
    return _RETRIEVER_INSTANCE


def retrieve(
    query_image_path: Union[str, Path, torch.Tensor, np.ndarray],
    query_modality: str,
    subcategory: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Module-level retrieval function for easy import and usage:
        from src.inference.retriever import retrieve
        results = retrieve("path/to/sar_patch", "SAR", subcategory="flood", k=10)
    """
    retriever_instance = get_retriever()
    return retriever_instance.retrieve(
        query_image_path=query_image_path,
        query_modality=query_modality,
        subcategory=subcategory,
        k=k,
    )
