"""
test_all_subcategories.py - Test suite for all 6 subcategories and Flask integration.

Performs:
1. Full test of all 6 configured subcategories with real satellite imagery:
   - Flood (SAR)
   - Wildfire (Multispectral)
   - Earthquake Damage (SAR)
   - Deforestation (Hybrid)
   - Drought (Multispectral)
   - Crop Mapping (Hybrid)
2. Records: domain, subcategory, configured modality, query modality, count, top result, score, latency.
3. Tests Flask application client:
   - GET / (domain selection)
   - GET /disaster-response (subcategory selection)
   - GET /search (search page)
   - POST /search with real SAR image (multipart upload)
   - POST /search with real MS image (multipart upload)
   - GET /patch_image/<patch_id> (thumbnail streaming)
   - GET /health (health endpoint)
4. Tests error handling (no file, bad file, missing subcategory).
5. Outputs structured JSON results to outputs/week5_subcategory_results.json.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from app import app
from src.inference.retriever import get_retriever, load_modality_config


def main() -> None:
    print("=" * 75)
    print("WEEK 5 — END-TO-END VERIFICATION: ALL 6 SUBCATEGORIES & FLASK")
    print("=" * 75)
    print()

    retriever = get_retriever()
    config = load_modality_config()

    # Patches on disk
    s1_patch = (
        PROJECT_ROOT
        / "teammate_inputs"
        / "BigEarthNet-S1-Required"
        / "S1B_IW_GRDH_1SDV_20170612T165809"
        / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66"
    )
    s2_patch = (
        PROJECT_ROOT
        / "teammate_inputs"
        / "BigEarthNet-S2-Required"
        / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
        / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    )

    assert s1_patch.exists(), f"SAR patch not found: {s1_patch}"
    assert s2_patch.exists(), f"MS patch not found: {s2_patch}"

    subcategories_to_test = [
        ("disaster-response", "flood", "SAR", s1_patch, "SAR"),
        ("disaster-response", "wildfire", "Multispectral", s2_patch, "MS"),
        ("disaster-response", "earthquake-damage", "SAR", s1_patch, "SAR"),
        ("environmental-monitoring", "deforestation", "Hybrid", s1_patch, "SAR"),
        ("environmental-monitoring", "drought", "Multispectral", s2_patch, "MS"),
        ("urban-agricultural", "crop-mapping", "Hybrid", s2_patch, "MS"),
    ]

    subcategory_results = []
    latencies = {"prep": [], "infer": [], "faiss": [], "total": []}

    print("SECTION 1: TESTING ALL 6 SUBCATEGORIES WITH REAL SATELLITE IMAGES")
    print("-" * 75)

    for domain_slug, sub_slug, expected_mod, query_path, query_mod in subcategories_to_test:
        sub_info = config["domains"][domain_slug]["subcategories"][sub_slug]
        print(f"\nTesting Subcategory: '{sub_info['name']}' ({sub_slug})")
        print(f"  Domain             : {domain_slug}")
        print(f"  Configured Modality: {expected_mod}")
        print(f"  Query Modality Used: {query_mod}")
        print(f"  Query Source Path  : {query_path.name}")

        t0 = time.perf_counter()
        results = retriever.retrieve(
            query_image_path=query_path,
            query_modality=query_mod,
            subcategory=sub_slug,
            k=10,
        )
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        latencies["total"].append(total_time_ms)

        # Basic validations
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(np.isfinite(r["similarity_score"]) for r in results), "NaN/Inf in scores!"
        scores = [r["similarity_score"] for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Scores not descending!"

        top_r = results[0]
        print(f"  Execution Status   : PASS (10 results in {total_time_ms:.2f} ms)")
        print(f"  Top-1 Patch ID     : {top_r['patch_id']}")
        print(f"  Top-1 Score        : {top_r['similarity_score']:.4f}")
        print(f"  Top-1 Modality     : {top_r['modality']}")
        print(f"  Top-1 Label        : {top_r['label']}")
        if "provenance" in top_r:
            print(f"  Provenance         : {top_r['provenance']}")

        # Semantic retrieval correctness assertions
        if sub_slug == "flood":
            gt_id = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
            retrieved_ids = [r["patch_id"] for r in results]
            assert gt_id in retrieved_ids, f"Correctness Error: Ground-truth paired scene {gt_id} missing from Top-10!"
            gt_rank = retrieved_ids.index(gt_id) + 1
            print(f"  [CORRECTNESS] Ground-truth paired MS scene '{gt_id}' verified at Rank #{gt_rank}")

        if sub_slug == "crop-mapping":
            assert results[0]["patch_id"] == "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
            assert np.isclose(results[0]["similarity_score"], 1.0, atol=1e-3)
            print(f"  [CORRECTNESS] Unimodal self-match verified at Rank #1 with score {results[0]['similarity_score']:.4f}")

        record = {
            "domain": domain_slug,
            "subcategory": sub_slug,
            "subcategory_name": sub_info["name"],
            "configured_modality": expected_mod,
            "query_modality": query_mod,
            "success": True,
            "results_count": len(results),
            "top_patch_id": top_r["patch_id"],
            "top_similarity_score": round(top_r["similarity_score"], 4),
            "top_modality": top_r["modality"],
            "top_label": top_r["label"],
            "provenance": top_r.get("provenance", "cross-modal"),
            "latency_ms": round(total_time_ms, 2),
            "retrieved_results": [
                {
                    "rank": r["rank"],
                    "patch_id": r["patch_id"],
                    "score": round(r["similarity_score"], 4),
                    "modality": r["modality"],
                    "label": r["label"],
                    "provenance": r.get("provenance", ""),
                }
                for r in results
            ],
        }
        subcategory_results.append(record)

    # Save structured results
    out_json = PROJECT_ROOT / "outputs" / "week5_subcategory_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(subcategory_results, f, indent=2)
    print(f"\n[OK] Saved structured subcategory results to: {out_json}")

    # --------------------------------------------------------
    # SECTION 2: FLASK APPLICATION CLIENT TESTS
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("SECTION 2: TESTING FLASK WEB APPLICATION (INTEGRATION TESTS)")
    print("-" * 75)

    client = app.test_client()

    # 1. Health check
    print("\n[Flask 1/7] Testing /health endpoint...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    health_data = resp.get_json()
    print(f"  Response: {health_data}")
    assert health_data["status"] == "healthy"
    assert health_data["sar_gallery_count"] == 4500
    assert health_data["ms_gallery_count"] == 4500
    print("  [PASS] /health verified.")

    # 2. Domain landing page
    print("\n[Flask 2/7] Testing GET / (Domain Landing Page)...")
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Disaster Response and Crisis Mapping" in resp.data
    assert b"Environmental Monitoring" in resp.data
    assert b"Urban and Agricultural Intelligence" in resp.data
    print("  [PASS] GET / renders 3 domains.")

    # 3. Subcategories page
    print("\n[Flask 3/7] Testing GET /<domain_slug> (Subcategory Selection)...")
    resp = client.get("/disaster-response")
    assert resp.status_code == 200
    assert b"Flood" in resp.data
    assert b"Wildfire" in resp.data
    assert b"Earthquake Damage" in resp.data
    print("  [PASS] GET /disaster-response renders subcategories.")

    # 4. Search page (GET)
    print("\n[Flask 4/7] Testing GET /search (Query & Upload Form)...")
    resp = client.get("/search?domain=disaster-response&subcategory=flood")
    assert resp.status_code == 200
    assert b"Recommended modality" in resp.data
    assert b"SAR" in resp.data
    print("  [PASS] GET /search shows recommended modality.")

    # 5. Search page (POST real image upload & sample benchmark query)
    print("\n[Flask 5/7] Testing POST /search with Real SAR Satellite Image...")
    real_sar_vv = s1_patch / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66_VV.tif"
    with open(real_sar_vv, "rb") as f:
        img_bytes = f.read()

    post_resp = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"image": (io.BytesIO(img_bytes), real_sar_vv.name)},
        content_type="multipart/form-data",
    )
    assert post_resp.status_code == 200
    assert b"TOP-10 RETRIEVED SCENES" in post_resp.data
    assert real_sar_vv.name.encode() in post_resp.data
    assert b"Nearest scenes." in post_resp.data
    # Assert expected top match for single VV input
    assert b"S2B_MSIL2A_20170817T101019_N9999_R022_T34WFS_53_29" in post_resp.data
    print("  [PASS] POST /search executes real retrieval on single-file GeoTIFF upload.")

    # Test 1-click sample scene execution with full multi-channel partner bands
    resp_sample = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"sample_key": "flood_sar"},
    )
    assert resp_sample.status_code == 200
    gt_paired_ms = b"S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    assert gt_paired_ms in resp_sample.data, f"Ground-truth paired scene {gt_paired_ms.decode()} missing from sample retrieval output!"
    assert b"S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_42_63" in resp_sample.data
    print("  [PASS] 1-Click sample benchmark retrieval verified with ground-truth paired scene.")

    # 6. Post Hybrid search (Deforestation)
    print("\n[Flask 6/7] Testing POST /search for Hybrid Subcategory (Deforestation)...")
    test_img = Image.new("RGB", (224, 224), color=(60, 80, 70))
    img_bytes2 = io.BytesIO()
    test_img.save(img_bytes2, format="PNG")
    img_bytes2.seek(0)
    post_hybrid = client.post(
        "/search?domain=environmental-monitoring&subcategory=deforestation",
        data={"image": (img_bytes2, "test_deforestation_query.png")},
        content_type="multipart/form-data",
    )
    assert post_hybrid.status_code == 200
    assert b"TOP-10 RETRIEVED SCENES" in post_hybrid.data
    print("  [PASS] POST /search Hybrid retrieval works cleanly.")

    # 7. Patch image thumbnail streaming
    print("\n[Flask 7/7] Testing GET /patch_image/<patch_id>...")
    # S2 patch 0 exists locally
    test_pid = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66"
    resp_img = client.get(f"/patch_image/{test_pid}")
    assert resp_img.status_code == 200
    assert resp_img.content_type == "image/jpeg"
    print(f"  Served JPEG thumbnail for {test_pid} ({len(resp_img.data)} bytes)")

    # Test graceful 404 on nonexistent patch
    resp_404 = client.get("/patch_image/NONEXISTENT_PATCH_ID")
    assert resp_404.status_code == 404
    print("  Graceful 404 on un-cached patch ID passed.")
    print("  [PASS] Thumbnail streaming verified.")

    # --------------------------------------------------------
    # SECTION 3: ERROR HANDLING TESTS
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("SECTION 3: ERROR HANDLING VERIFICATION")
    print("-" * 75)

    # 1. No image uploaded
    resp_no_img = client.post("/search?domain=disaster-response&subcategory=flood", data={})
    assert resp_no_img.status_code == 200
    assert b"Please choose a satellite image file" in resp_no_img.data
    print("  [PASS] Missing file handled cleanly.")

    # 2. Unsupported extension (.exe)
    bad_bytes = io.BytesIO(b"dummy")
    resp_bad_ext = client.post(
        "/search?domain=disaster-response&subcategory=flood",
        data={"image": (bad_bytes, "exploit.exe")},
        content_type="multipart/form-data",
    )
    assert resp_bad_ext.status_code == 200
    assert b"Unsupported file format" in resp_bad_ext.data
    print("  [PASS] Unsupported file extension rejected cleanly.")

    # 3. Invalid domain slug redirects to /
    resp_bad_domain = client.get("/invalid-nonexistent-domain")
    assert resp_bad_domain.status_code == 302
    print("  [PASS] Invalid domain redirects to landing page.")

    # --------------------------------------------------------
    # SECTION 4: LATENCY SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("SECTION 4: LATENCY SUMMARY ACROSS ALL 6 SUBCATEGORIES")
    print("-" * 75)
    tot_times = latencies["total"]
    print(f"Number of Real Queries: {len(tot_times)}")
    print(f"Mean Total Latency    : {np.mean(tot_times):.2f} ms")
    print(f"Median Total Latency  : {np.median(tot_times):.2f} ms")
    print(f"Min Total Latency     : {np.min(tot_times):.2f} ms")
    print(f"Max Total Latency     : {np.max(tot_times):.2f} ms")

    print("\n" + "=" * 75)
    print("ALL SUBCATEGORY & FLASK INTEGRATION TESTS PASSED 100%!")
    print("=" * 75)


if __name__ == "__main__":
    main()
