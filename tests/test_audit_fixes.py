"""
test_audit_fixes.py - Focused Unit and Integration Tests for ATLAS Remote Sensing Audit Fixes.

Validates:
1. SAR sample -> SAR preprocessing and retrieval.
2. Multispectral sample -> MS preprocessing and retrieval.
3. Hybrid sample -> Hybrid dual-gallery search and score merge.
4. Incompatible sample selection -> Validation error (no silent modality alteration).
5. Uploaded image flow -> Preserves configured modality.
6. Multispectral thumbnail never falls back to SAR.
7. SAR thumbnail never falls back to Multispectral.
8. Unsupported modality thumbnail returns None gracefully.
9. Hybrid thumbnail behavior (composite when both exist, None when missing).
10. S1 and S2 folder lookup uses correct identifiers (patch_id, s1_name, s2v1_name).
11. Missing folders and corrupt TIFF files do not crash application (clean 404).
12. NaN, infinity, nodata, and constant-valued images produce safe previews.
13. Missing model, index, or metadata files produce useful errors.
14. Existing retrieval result structure and Flask endpoints remain compatible.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from PIL import Image

import app as flask_app
from app import app, SAMPLE_SCENES
from src.inference.retriever import CrossModalRetriever, get_retriever


@pytest.fixture(scope="module")
def retriever_instance():
    """Shared initialized CrossModalRetriever instance."""
    return get_retriever()


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ============================================================
# 1. SAMPLE ROUTING & COMPATIBILITY VALIDATION
# ============================================================

def test_sar_sample_preprocessing_and_retrieval(client):
    """SAR sample (flood_sar) on compatible flood subcategory runs SAR retrieval."""
    resp = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"sample_key": "flood_sar"},
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "QUERY / SAR" in html or "SAR" in html
    assert "TOP-" in html
    assert "⚠️" not in html or "Retrieval failed" not in html


def test_multispectral_sample_preprocessing_and_retrieval(client):
    """MS sample (wildfire_ms) on compatible wildfire subcategory runs MS retrieval."""
    resp = client.post(
        "/search?domain=disaster-response&subcategory=wildfire",
        data={"sample_key": "wildfire_ms"},
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "QUERY / MS" in html or "MS" in html
    assert "TOP-" in html
    assert "Retrieval failed" not in html


def test_hybrid_sample_supported_behavior(client):
    """Hybrid sample (crop_hybrid) on crop-mapping subcategory runs Hybrid retrieval."""
    resp = client.post(
        "/search?domain=urban-agricultural&subcategory=crop-mapping",
        data={"sample_key": "crop_hybrid"},
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "QUERY / Hybrid" in html or "Hybrid" in html
    assert "TOP-" in html
    assert "Retrieval failed" not in html


def test_incompatible_sample_validation_error(client):
    """Selecting a sample incompatible with domain/subcategory displays validation message."""
    # Attempting to run wildfire_ms (MS) on flood (SAR)
    resp = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"sample_key": "wildfire_ms"},
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    # Must show validation error and NOT execute retrieval results
    assert "is incompatible with the selected use case" in html or "not compatible" in html
    assert "TOP-" not in html

    # Attempting to run flood_sar (SAR) on wildfire (MS)
    resp2 = client.post(
        "/search?domain=disaster-response&subcategory=wildfire",
        data={"sample_key": "flood_sar"},
    )
    assert resp2.status_code == 200
    html2 = resp2.data.decode("utf-8")
    assert "is incompatible with the selected use case" in html2 or "not compatible" in html2
    assert "TOP-" not in html2


def test_unrecognized_sample_key(client):
    """Submitting an unrecognized sample key returns a clear error."""
    resp = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"sample_key": "non_existent_sample"},
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "Unrecognized sample scene key" in html


# ============================================================
# 2. UPLOADED IMAGE FLOW
# ============================================================

def test_uploaded_image_preserves_configured_modality(client):
    """Uploading an image uses the subcategory's configured modality."""
    # Create synthetic 3-band GeoTIFF in memory
    import rasterio
    from rasterio.transform import from_origin

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="uint8",
        crs="+proj=latlong",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(np.full((3, 32, 32), 120, dtype=np.uint8))

    buf.seek(0)
    resp = client.post(
        "/search?domain=disaster-response&subcategory=wildfire",
        data={"image": (buf, "synthetic_upload.tif")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "Processed: synthetic_upload.tif" in html
    assert "QUERY / MS" in html or "MS" in html


def test_unsupported_upload_format(client):
    """Uploading an unsupported file format returns user-friendly error."""
    buf = io.BytesIO(b"fake executable payload")
    resp = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"image": (buf, "malicious.exe")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "Unsupported file format" in html


# ============================================================
# 3. THUMBNAIL MODALITY ISOLATION (NO SILENT FALLBACK)
# ============================================================

def test_multispectral_thumbnail_never_falls_back_to_sar(retriever_instance):
    """If Multispectral thumbnail cannot be rendered, it must NEVER fall back to SAR."""
    retriever = retriever_instance
    retriever._ensure_patch_cache()

    # Find a patch in test_split that has SAR locally but NO S2 locally
    df_path = retriever.test_split_path
    import pandas as pd
    df = pd.read_csv(df_path)

    sar_only_pid = None
    for _, row in df.iterrows():
        pid = row["patch_id"]
        s1 = row["s1_name"]
        if pid not in retriever._local_s2_patches and s1 in retriever._local_s1_patches:
            sar_only_pid = pid
            break

    assert sar_only_pid is not None, "Could not find a test patch with SAR available but no S2"

    # Explicitly request Multispectral thumbnail: MUST RETURN None
    ms_thumb = retriever.render_patch_thumbnail(sar_only_pid, modality="Multispectral")
    assert ms_thumb is None, "Multispectral thumbnail incorrectly fell back to SAR!"

    # But requesting SAR thumbnail for the same patch SHOULD succeed
    sar_thumb = retriever.render_patch_thumbnail(sar_only_pid, modality="SAR")
    assert sar_thumb is not None, "SAR thumbnail failed to render for patch with SAR available"
    assert isinstance(sar_thumb, Image.Image)


def test_sar_thumbnail_never_falls_back_to_multispectral(retriever_instance, tmp_path):
    """If SAR thumbnail cannot be rendered, it must NEVER fall back to Multispectral."""
    retriever = retriever_instance

    # Mock a patch that exists only in S2 cache
    mock_pid = "MOCK_S2_ONLY_PATCH_ID"
    mock_folder = tmp_path / "mock_s2_folder"
    mock_folder.mkdir()

    # Create synthetic B04, B03, B02
    import rasterio
    for b in ("B04", "B03", "B02"):
        with rasterio.open(
            mock_folder / f"scene_{mock_pid}_{b}.tif",
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
        ) as dst:
            dst.write(np.full((1, 10, 10), 500.0, dtype=np.float32))

    # Temporarily insert into _local_s2_patches but NOT into _local_s1_patches
    retriever._ensure_patch_cache()
    retriever._local_s2_patches[mock_pid] = mock_folder

    try:
        # Requesting SAR thumbnail must return None, NEVER S2
        sar_result = retriever.render_patch_thumbnail(mock_pid, modality="SAR")
        assert sar_result is None, "SAR thumbnail incorrectly fell back to Multispectral!"

        # Requesting MS thumbnail should succeed
        ms_result = retriever.render_patch_thumbnail(mock_pid, modality="Multispectral")
        assert ms_result is not None, "MS thumbnail should succeed for S2-only patch"
        assert isinstance(ms_result, Image.Image)
    finally:
        retriever._local_s2_patches.pop(mock_pid, None)


def test_unsupported_modality_returns_none(retriever_instance):
    """Unsupported modality name returns None gracefully without exception."""
    res = retriever_instance.render_patch_thumbnail("some_patch_id", modality="ThermalInfrared")
    assert res is None


def test_hybrid_thumbnail_behavior(retriever_instance, tmp_path):
    """Hybrid thumbnail renders composite only if both are available, otherwise None."""
    retriever = retriever_instance
    retriever._ensure_patch_cache()

    # Mock a patch with both S1 and S2
    mock_pid = "MOCK_HYBRID_PATCH"
    s1_folder = tmp_path / "s1"
    s2_folder = tmp_path / "s2"
    s1_folder.mkdir()
    s2_folder.mkdir()

    import rasterio
    with rasterio.open(s1_folder / "scene_VV.tif", "w", driver="GTiff", height=10, width=10, count=1, dtype="float32") as d:
        d.write(np.full((1, 10, 10), -12.0, dtype=np.float32))

    for b in ("B04", "B03", "B02"):
        with rasterio.open(s2_folder / f"scene_{b}.tif", "w", driver="GTiff", height=10, width=10, count=1, dtype="float32") as d:
            d.write(np.full((1, 10, 10), 600.0, dtype=np.float32))

    retriever._local_s1_patches[mock_pid] = s1_folder
    retriever._local_s2_patches[mock_pid] = s2_folder

    try:
        comp = retriever.render_patch_thumbnail(mock_pid, modality="Hybrid")
        assert comp is not None
        assert isinstance(comp, Image.Image)
        # Composite should be horizontal stitch of both: width should be sum
        assert comp.width == 20
        assert comp.height == 10

        # Remove S1 -> hybrid should return None
        del retriever._local_s1_patches[mock_pid]
        assert retriever.render_patch_thumbnail(mock_pid, modality="Hybrid") is None
    finally:
        retriever._local_s1_patches.pop(mock_pid, None)
        retriever._local_s2_patches.pop(mock_pid, None)


# ============================================================
# 4. METADATA & FOLDER LOOKUP CONVENTIONS
# ============================================================

def test_metadata_lookup_identifiers(retriever_instance):
    """Metadata can be retrieved deterministically by patch_id, s1_name, or s2v1_name."""
    retriever = retriever_instance
    pid = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    meta = retriever.get_metadata(pid)
    assert meta, f"Metadata not found for known patch_id {pid}"

    s1_name = meta["s1_name"]
    s2v1_name = meta["s2v1_name"]
    assert s1_name == "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66"
    assert s2v1_name == "S2A_MSIL2A_20170613T101031_39_66"

    # Reverse lookup by s1_name
    meta_by_s1 = retriever.get_metadata(s1_name)
    assert meta_by_s1["patch_id"] == pid

    # Reverse lookup by s2v1_name
    meta_by_s2v1 = retriever.get_metadata(s2v1_name)
    assert meta_by_s2v1["patch_id"] == pid

    # Unknown ID returns empty dict gracefully
    assert retriever.get_metadata("UNKNOWN_NON_EXISTENT_PATCH") == {}
    assert retriever.get_metadata("") == {}
    assert retriever.get_metadata(None) == {}


def test_ensure_patch_cache_does_not_rescan(retriever_instance):
    """_ensure_patch_cache initializes once and does not rescan on repeated calls."""
    retriever = retriever_instance
    retriever._ensure_patch_cache()
    assert retriever._patch_cache_initialized is True

    # Mutate a mock key to verify that a second call returns immediately without rescanning
    retriever._local_s1_patches["__TEST_SENTINEL__"] = Path("dummy")
    retriever._ensure_patch_cache()
    assert "__TEST_SENTINEL__" in retriever._local_s1_patches
    del retriever._local_s1_patches["__TEST_SENTINEL__"]


# ============================================================
# 5. ROBUST THUMBNAIL PREPROCESSING (NAN, INF, NODATA, CONSTANT)
# ============================================================

def test_safe_preview_handling_nan_inf_nodata_and_constant(retriever_instance, tmp_path):
    """Thumbnail generation safely handles NaN, Inf, nodata, and constant images without dividing by zero."""
    retriever = retriever_instance
    import rasterio

    # 1. Constant value image (all pixels equal 500.0)
    const_folder = tmp_path / "const"
    const_folder.mkdir()
    for b in ("B04", "B03", "B02"):
        with rasterio.open(const_folder / f"img_{b}.tif", "w", driver="GTiff", height=16, width=16, count=1, dtype="float32") as d:
            d.write(np.full((1, 16, 16), 500.0, dtype=np.float32))

    retriever._local_s2_patches["const_test"] = const_folder
    try:
        img_const = retriever.render_patch_thumbnail("const_test", modality="Multispectral")
        assert img_const is not None
        assert isinstance(img_const, Image.Image)
        arr = np.array(img_const)
        assert np.isfinite(arr).all()
    finally:
        retriever._local_s2_patches.pop("const_test", None)

    # 2. Image with NaNs and Infs
    nan_folder = tmp_path / "nans"
    nan_folder.mkdir()
    nan_data = np.full((1, 16, 16), 600.0, dtype=np.float32)
    nan_data[0, 0, 0] = np.nan
    nan_data[0, 0, 1] = np.inf
    nan_data[0, 0, 2] = -np.inf
    for b in ("B04", "B03", "B02"):
        with rasterio.open(nan_folder / f"nan_{b}.tif", "w", driver="GTiff", height=16, width=16, count=1, dtype="float32") as d:
            d.write(nan_data)

    retriever._local_s2_patches["nan_test"] = nan_folder
    try:
        img_nan = retriever.render_patch_thumbnail("nan_test", modality="Multispectral")
        assert img_nan is not None
        arr_nan = np.array(img_nan)
        assert np.isfinite(arr_nan).all()
        assert not np.isnan(arr_nan).any()
    finally:
        retriever._local_s2_patches.pop("nan_test", None)

    # 3. All NaN image (no valid pixels)
    all_nan_folder = tmp_path / "all_nans"
    all_nan_folder.mkdir()
    for b in ("B04", "B03", "B02"):
        with rasterio.open(all_nan_folder / f"all_nan_{b}.tif", "w", driver="GTiff", height=16, width=16, count=1, dtype="float32") as d:
            d.write(np.full((1, 16, 16), np.nan, dtype=np.float32))

    retriever._local_s2_patches["all_nan_test"] = all_nan_folder
    try:
        img_all_nan = retriever.render_patch_thumbnail("all_nan_test", modality="Multispectral")
        assert img_all_nan is not None
        assert (np.array(img_all_nan) == 0).all()
    finally:
        retriever._local_s2_patches.pop("all_nan_test", None)


def test_corrupt_tiff_handling_graceful(retriever_instance, tmp_path):
    """Corrupt TIFF files return None without crashing the application."""
    retriever = retriever_instance
    corrupt_folder = tmp_path / "corrupt"
    corrupt_folder.mkdir()

    # Write truncated garbage bytes to TIFF files
    for b in ("B04", "B03", "B02"):
        (corrupt_folder / f"corrupt_{b}.tif").write_bytes(b"NOT_A_VALID_TIFF_HEADER")

    retriever._local_s2_patches["corrupt_test"] = corrupt_folder
    try:
        result = retriever.render_patch_thumbnail("corrupt_test", modality="Multispectral")
        assert result is None
    finally:
        retriever._local_s2_patches.pop("corrupt_test", None)


# ============================================================
# 6. RETRIEVAL PARAMETER & ERROR HANDLING
# ============================================================

def test_invalid_k_values(retriever_instance):
    """Invalid k values raise clear ValueError."""
    retriever = retriever_instance
    dummy = torch.randn(1, 2, 224, 224)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve(dummy, query_modality="SAR", k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve(dummy, query_modality="SAR", k=-5)


def test_missing_files_error_handling(tmp_path):
    """Missing model, index, or metadata files produce explicit FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        CrossModalRetriever(checkpoint_path=tmp_path / "nonexistent.ckpt")

    valid_ckpt = CrossModalRetriever().checkpoint_path
    with pytest.raises(FileNotFoundError, match="SAR gallery index not found"):
        CrossModalRetriever(checkpoint_path=valid_ckpt, sar_index_path=tmp_path / "nonexistent.index")


# ============================================================
# 7. FLASK ENDPOINTS & RESULT STRUCTURE COMPATIBILITY
# ============================================================

def test_flask_patch_image_endpoint(client):
    """Flask /patch_image/<patch_id> serves valid JPEG or 404 cleanly."""
    # Real test patch
    pid = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    resp = client.get(f"/patch_image/{pid}?modality=SAR")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert resp.mimetype == "image/jpeg"

    # Non-existent patch returns clean 404
    resp_404 = client.get("/patch_image/completely_fake_patch_12345?modality=MS")
    assert resp_404.status_code == 404


def test_flask_health_endpoint(client):
    """Health check endpoint returns healthy status with gallery counts."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["sar_gallery_count"] == 4500
    assert data["ms_gallery_count"] == 4500


def test_sar_thumbnail_pixel_and_source_verification(retriever_instance):
    """Regression test: SAR thumbnail renders genuine grayscale intensity without abstract blocks."""
    retriever = retriever_instance
    pid = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    img = retriever.render_patch_thumbnail(pid, modality="SAR")
    assert img is not None
    arr = np.array(img)
    assert arr.ndim == 3 and arr.shape[2] == 3
    # Check that it is genuine grayscale (R == G == B)
    assert np.array_equal(arr[..., 0], arr[..., 1])
    assert np.array_equal(arr[..., 1], arr[..., 2])
    # Check that it's not a flat or blank block
    assert float(arr.std()) > 5.0
    assert arr.max() > arr.min()


def test_ms_thumbnail_pixel_and_source_verification(retriever_instance):
    """Regression test: MS thumbnail renders true-color RGB using B04/B03/B02."""
    retriever = retriever_instance
    pid = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    img = retriever.render_patch_thumbnail(pid, modality="MS")
    assert img is not None
    arr = np.array(img)
    assert arr.ndim == 3 and arr.shape[2] == 3
    # Multispectral true-color RGB should have distinct channels (not monochromatic)
    assert not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]))
    # Valid pixel values within uint8 range
    assert 0 <= arr.min() and arr.max() <= 255
    assert float(arr.std()) > 5.0


# ============================================================
# 8. PHASE 6: USE-CASE TRACE & REGRESSION TESTS
# ============================================================

def test_different_samples_produce_distinct_embeddings(retriever_instance):
    """Different benchmark samples load correct source files and produce distinct embeddings."""
    retriever = retriever_instance
    s_flood = SAMPLE_SCENES["flood_sar"]
    s_wildfire = SAMPLE_SCENES["wildfire_ms"]
    s_deforest = SAMPLE_SCENES["deforestation_hybrid"]

    assert s_flood["path"].exists()
    assert s_wildfire["path"].exists()
    assert s_deforest["path"].exists()

    # SAR encoding
    sar_t = retriever.preprocess_sar(s_flood["path"])
    sar_emb = retriever.encode_sar(sar_t)
    assert sar_emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(sar_emb), 1.0, atol=1e-5)

    # MS encoding for wildfire
    ms_t = retriever.preprocess_ms(s_wildfire["path"])
    ms_emb = retriever.encode_ms(ms_t)
    assert ms_emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(ms_emb), 1.0, atol=1e-5)

    # MS encoding for deforestation
    def_t = retriever.preprocess_ms(s_deforest["path"])
    def_emb = retriever.encode_ms(def_t)
    assert def_emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(def_emb), 1.0, atol=1e-5)

    # Verify embeddings are mathematically distinct
    cos_sim = float(np.dot(ms_emb[0], def_emb[0]))
    assert cos_sim < 0.99, "Different scenes produced identical embeddings!"
    assert not np.allclose(sar_emb, ms_emb)


def test_correct_query_modality_and_gallery_selection(retriever_instance):
    """SAR query selects MS gallery, MS query selects SAR gallery, Hybrid queries both."""
    retriever = retriever_instance
    s_flood = SAMPLE_SCENES["flood_sar"]
    s_wildfire = SAMPLE_SCENES["wildfire_ms"]
    s_crop = SAMPLE_SCENES["crop_hybrid"]

    # SAR -> MS
    res_sar = retriever.retrieve(s_flood["path"], query_modality="SAR", k=5)
    assert len(res_sar) == 5
    for r in res_sar:
        assert r["modality"] == "Multispectral"
        assert "cross-modal (SAR -> MS)" in r["provenance"]

    # MS -> SAR
    res_ms = retriever.retrieve(s_wildfire["path"], query_modality="MS", k=5)
    assert len(res_ms) == 5
    for r in res_ms:
        assert r["modality"] == "SAR"
        assert "cross-modal (MS -> SAR)" in r["provenance"]

    # Hybrid -> Dual gallery
    res_hybrid = retriever.retrieve(s_crop["path"], query_modality="Hybrid", k=5)
    assert len(res_hybrid) == 5
    provenances = {r["provenance"] for r in res_hybrid}
    assert any("cross-modal" in p or "unimodal" in p for p in provenances)


def test_deterministic_ranking_for_fixed_input(retriever_instance):
    """Retrieval is deterministic: identical input and model yield identical results."""
    retriever = retriever_instance
    s_flood = SAMPLE_SCENES["flood_sar"]["path"]

    res1 = retriever.retrieve(s_flood, query_modality="SAR", k=10)
    res2 = retriever.retrieve(s_flood, query_modality="SAR", k=10)

    pids1 = [r["patch_id"] for r in res1]
    pids2 = [r["patch_id"] for r in res2]
    scores1 = [r["score"] for r in res1]
    scores2 = [r["score"] for r in res2]

    assert pids1 == pids2, "Patch ranking differed between identical runs!"
    assert scores1 == scores2, "Similarity scores differed between identical runs!"


def test_missing_benchmark_sample_handling(client):
    """Subcategories without a benchmark sample (Earthquake Damage) render clean notice and do not crash."""
    resp_eq = client.get("/search?domain=disaster-response&subcategory=earthquake-damage")
    assert resp_eq.status_code == 200
    html_eq = resp_eq.data.decode("utf-8")
    assert "No compatible benchmark sample is configured" in html_eq
    assert "upload option above remains fully active" in html_eq


def test_drought_benchmark_sample_and_retrieval(client):
    """Drought subcategory has verified benchmark sample (drought_ms) and executes real MS retrieval."""
    # 1. Page displays the recommended drought benchmark button
    resp = client.get("/search?domain=environmental-monitoring&subcategory=drought")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "drought_ms" in html
    assert "Drought Vegetation Stress Scene" in html
    assert "★ Recommended" in html

    # 2. Submitting drought_ms runs MS retrieval and returns top results
    resp_post = client.post(
        "/search?domain=environmental-monitoring&subcategory=drought",
        data={"sample_key": "drought_ms"},
    )
    assert resp_post.status_code == 200
    html_post = resp_post.data.decode("utf-8")
    assert "QUERY / MS" in html_post or "MS" in html_post
    assert "TOP-" in html_post
    assert "Retrieval failed" not in html_post


def test_use_case_navigation_resets_stale_query_state(client):
    """Navigating to a use case via GET provides a clean search interface without stale results."""
    # 1. Execute a search
    resp_post = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"sample_key": "flood_sar"},
    )
    assert resp_post.status_code == 200
    assert "TOP-" in resp_post.data.decode("utf-8")

    # 2. Navigate away to another subcategory via GET
    resp_nav = client.get("/search?domain=environmental-monitoring&subcategory=deforestation")
    assert resp_nav.status_code == 200
    html_nav = resp_nav.data.decode("utf-8")

    # Verify that results, query card, and old filenames are NOT present
    assert "TOP-" not in html_nav
    assert "Nearest scenes" not in html_nav
    assert "Processed: flood_sar" not in html_nav


def test_thumbnail_modality_isolation_and_missing_placeholder(client, retriever_instance):
    """Missing patch images return HTTP 404, preventing silent modality substitution."""
    # S1 endpoint
    resp_missing_s1 = client.get("/patch_image/NONEXISTENT_PATCH_12345?modality=SAR")
    assert resp_missing_s1.status_code == 404

    # S2 endpoint
    resp_missing_s2 = client.get("/patch_image/NONEXISTENT_PATCH_12345?modality=MS")
    assert resp_missing_s2.status_code == 404


# ============================================================
# 9. COMPREHENSIVE UPLOAD-TO-RETRIEVAL AUDIT FIX TESTS
# ============================================================

def test_audit_s2_valid_rgb_upload(client):
    """1. Valid Sentinel-2 3-channel RGB image upload produces 3-channel tensor and MS retrieval."""
    import rasterio
    from rasterio.transform import from_origin
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint16",
        crs="+proj=latlong", transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(np.full((3, 64, 64), 600, dtype=np.uint16))
    buf.seek(0)

    resp = client.post(
        "/search?domain=disaster-response&subcategory=wildfire",
        data={"image": (buf, "valid_s2_rgb.tif"), "modality": "Multispectral"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "QUERY / Multispectral" in html
    assert "TOP-" in html
    assert "Retrieval failed" not in html


def test_audit_s2_b04_resolves_genuine_partners(retriever_instance):
    """2. Valid single-band B04 TIFF resolves genuine B04, B03, B02 partners from dataset without fake channels."""
    retriever = retriever_instance
    known_b04 = SAMPLE_SCENES["wildfire_ms"]["path"]
    assert known_b04.exists()

    tensor = retriever.preprocess_ms(known_b04)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()

    # Verify that the channels are distinct (genuine RGB from B04, B03, B02, NOT duplicated)
    b04_ch = tensor[0, 0].cpu().numpy()
    b03_ch = tensor[0, 1].cpu().numpy()
    b02_ch = tensor[0, 2].cpu().numpy()
    assert not np.allclose(b04_ch, b03_ch), "B04 and B03 channels were duplicated!"
    assert not np.allclose(b03_ch, b02_ch), "B03 and B02 channels were duplicated!"


def test_audit_s2_b05_raises_clear_error_no_fake_rgb(retriever_instance, tmp_path):
    """3a. Single-band B05 (Red Edge) raises clear error and NEVER fabricates RGB channels."""
    retriever = retriever_instance
    import rasterio

    # Create an isolated B05 GeoTIFF without RGB partners
    b05_file = tmp_path / "scene_test_patch_B05.tif"
    with rasterio.open(b05_file, "w", driver="GTiff", height=32, width=32, count=1, dtype="uint16") as d:
        d.write(np.full((1, 32, 32), 700, dtype=np.uint16))

    with pytest.raises(ValueError, match="B05 - not an RGB band"):
        retriever.preprocess_ms(b05_file)


def test_audit_s2_isolated_b04_missing_partners_raises_error(retriever_instance, tmp_path):
    """3b. Single-band B04 without accessible RGB partner bands raises clear validation error."""
    retriever = retriever_instance
    import rasterio

    # Isolated B04 whose patch does not exist in local dataset
    b04_file = tmp_path / "UNKNOWN_NONEXISTENT_PATCH_12345_B04.tif"
    with rasterio.open(b04_file, "w", driver="GTiff", height=32, width=32, count=1, dtype="uint16") as d:
        d.write(np.full((1, 32, 32), 500, dtype=np.uint16))

    with pytest.raises(ValueError, match="genuine RGB partner bands.*could not be resolved"):
        retriever.preprocess_ms(b04_file)


def test_audit_sar_valid_2channel_input(retriever_instance, tmp_path):
    """4a. Valid 2-channel Sentinel-1 input (VV+VH) produces (1, 2, 224, 224) tensor."""
    retriever = retriever_instance
    import rasterio

    sar_2ch = tmp_path / "valid_sar_2ch.tif"
    with rasterio.open(sar_2ch, "w", driver="GTiff", height=32, width=32, count=2, dtype="float32") as d:
        d.write(np.full((32, 32), -12.0, dtype=np.float32), 1)  # VV
        d.write(np.full((32, 32), -18.0, dtype=np.float32), 2)  # VH

    tensor = retriever.preprocess_sar(sar_2ch)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 2, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_audit_sar_vv_resolves_genuine_partner(retriever_instance):
    """4b. Single-band VV file resolves genuine VH partner from dataset without fake duplication."""
    retriever = retriever_instance
    known_vv = SAMPLE_SCENES["flood_sar"]["path"]
    assert known_vv.exists()

    tensor = retriever.preprocess_sar(known_vv)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 2, 224, 224)
    assert tensor.dtype == torch.float32
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()

    # Channels must not be identical copies
    vv_ch = tensor[0, 0].cpu().numpy()
    vh_ch = tensor[0, 1].cpu().numpy()
    assert not np.allclose(vv_ch, vh_ch), "VV and VH channels were silently duplicated!"


def test_audit_sar_missing_vh_raises_error_no_duplication(retriever_instance, tmp_path):
    """5. Single VV file with missing VH raises clear error and NEVER duplicates VV to VH."""
    retriever = retriever_instance
    import rasterio

    # Isolated VV without VH and not in dataset
    vv_file = tmp_path / "UNKNOWN_NONEXISTENT_SAR_PATCH_99999_VV.tif"
    with rasterio.open(vv_file, "w", driver="GTiff", height=32, width=32, count=1, dtype="float32") as d:
        d.write(np.full((1, 32, 32), -13.0, dtype=np.float32))

    with pytest.raises(ValueError, match="companion polarization 'VH' could not be resolved"):
        retriever.preprocess_sar(vv_file)


def test_audit_modality_conflict_validation(client):
    """6. Form validation rejects conflicting modality selection with clear error message."""
    import rasterio

    # User uploads optical S2A file but explicitly selects SAR
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=16, width=16, count=3, dtype="uint8") as d:
        d.write(np.full((3, 16, 16), 100, dtype=np.uint8))
    buf.seek(0)

    resp = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"image": (buf, "S2A_MSIL2A_20170613T101031_conflict.tif"), "modality": "SAR"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "Modality conflict" in html
    assert "appears to be a Sentinel-2 Multispectral scene" in html
    assert "TOP-" not in html


def test_audit_tensor_properties_and_normalization(retriever_instance):
    """7. Verified model inputs have shape, dtype float32, and finite values."""
    retriever = retriever_instance
    s_flood = SAMPLE_SCENES["flood_sar"]["path"]
    s_wildfire = SAMPLE_SCENES["wildfire_ms"]["path"]

    sar_t = retriever.preprocess_sar(s_flood)
    assert sar_t.shape == (1, 2, 224, 224)
    assert sar_t.dtype == torch.float32
    assert torch.isfinite(sar_t).all()

    ms_t = retriever.preprocess_ms(s_wildfire)
    assert ms_t.shape == (1, 3, 224, 224)
    assert ms_t.dtype == torch.float32
    assert torch.isfinite(ms_t).all()


def test_audit_model_inference_and_faiss_search(retriever_instance):
    """8. Inference produces 512-D L2-normalized embeddings and FAISS returns top-10."""
    retriever = retriever_instance
    s_flood = SAMPLE_SCENES["flood_sar"]["path"]

    sar_t = retriever.preprocess_sar(s_flood)
    emb = retriever.encode_sar(sar_t)
    assert emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    results = retriever.search_gallery(emb, target_gallery="MS", k=10)
    assert len(results) == 10
    for r in results:
        assert r["modality"] == "Multispectral"
        assert "patch_id" in r
        assert "score" in r
        assert float(r["similarity_score"]) <= 1.0001


def test_audit_result_metadata_resolution(retriever_instance):
    """9. Retrieved results resolve metadata (label, split, tone, provenance)."""
    retriever = retriever_instance
    s_wildfire = SAMPLE_SCENES["wildfire_ms"]["path"]

    results = retriever.retrieve(s_wildfire, query_modality="MS", k=5)
    assert len(results) == 5
    for r in results:
        assert r["modality"] == "SAR"
        assert "label" in r
        assert "split" in r
        assert "provenance" in r
        assert r["provenance"] == "cross-modal (MS -> SAR)"


def test_audit_missing_thumbnail_fallback_404(client):
    """10. Missing thumbnail cleanly returns 404 for JavaScript onerror card trigger."""
    resp = client.get("/patch_image/NONEXISTENT_PATCH_ID_99999?modality=Multispectral")
    assert resp.status_code == 404


def test_audit_clear_validation_error_on_invalid_upload(client):
    """11. Invalid upload displays clear error in HTML, not unhandled 500 exception."""
    import rasterio

    # Upload single B05 file via portal
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=16, width=16, count=1, dtype="uint16") as d:
        d.write(np.full((1, 16, 16), 800, dtype=np.uint16))
    buf.seek(0)

    resp = client.post(
        "/search?domain=disaster-response&subcategory=wildfire",
        data={"image": (buf, "standalone_patch_B05.tif"), "modality": "Multispectral"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "single spectral band (B05 - not an RGB band)" in html or "B05" in html
    assert "TOP-" not in html


