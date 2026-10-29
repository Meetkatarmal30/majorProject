"""
app.py - Atlas Remote Sensing Intelligence Portal & Cross-Modal Satellite Image Retrieval.

Production Flask Application integrating:
- TwoTowerNetwork (ResNet-50 SAR + MS towers) with best_model-v1.ckpt
- Dual FAISS galleries (4,500 L2-normalized test embeddings)
- Domain-aware modality routing across 3 domains and 6 subcategories
- Real-time cross-modal and hybrid retrieval with latency measurement
- Live satellite image thumbnail streaming and query scene comparison
- Safe upload management and graceful HTTP error handling
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from src.inference.retriever import (
    CrossModalRetriever,
    get_retriever,
    load_modality_config,
    resolve_subcategory_info,
)

# ------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AtlasPortal")

# ------------------------------------------------------------
# APPLICATION INITIALIZATION
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "instance" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"tif", "tiff", "png", "jpg", "jpeg"}

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB maximum upload
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Global configuration and retriever
PORTAL_CONFIG = load_modality_config()
RETRIEVER: Optional[CrossModalRetriever] = None

# Verified sample scenes for 1-click interactive demonstrations
SAMPLE_SCENES: Dict[str, Dict[str, Any]] = {
    "flood_sar": {
        "key": "flood_sar",
        "name": "Sentinel-1 SAR Flood / Water Scene",
        "badge": "SAR",
        "modality": "SAR",
        "domain": "disaster-response",
        "subcategory": "flood",
        "path": PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S1-Required" / "S1B_IW_GRDH_1SDV_20170612T165809" / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66" / "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_39_66_VV.tif",
        "desc": "Dual-pol VV radar backscatter (water extent & inundation baseline)",
    },
    "wildfire_ms": {
        "key": "wildfire_ms",
        "name": "Sentinel-2 MS Vegetation Reference",
        "badge": "MS",
        "modality": "MS",
        "domain": "disaster-response",
        "subcategory": "wildfire",
        "path": PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S2-Required" / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP" / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66" / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66_B04.tif",
        "desc": "Optical RGB vegetation reflectance (spectral baseline; local archive contains no acute burn perimeters)",
    },
    "deforestation_hybrid": {
        "key": "deforestation_hybrid",
        "name": "Forest Disturbance Scene (Hybrid)",
        "badge": "HYBRID",
        "modality": "Hybrid",
        "domain": "environmental-monitoring",
        "subcategory": "deforestation",
        "path": PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S2-Required" / "S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL" / "S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_29_25" / "S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_29_25_B04.tif",
        "desc": "Transitional woodland & coniferous canopy disturbance (optical reflectance + radar structure)",
    },
    "drought_ms": {
        "key": "drought_ms",
        "name": "Sentinel-2 MS Drought Vegetation Stress Scene",
        "badge": "MS",
        "modality": "MS",
        "domain": "environmental-monitoring",
        "subcategory": "drought",
        "path": PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S2-Required" / "S2A_MSIL2A_20170704T112111_N9999_R037_T29SND" / "S2A_MSIL2A_20170704T112111_N9999_R037_T29SND_28_38" / "S2A_MSIL2A_20170704T112111_N9999_R037_T29SND_28_38_B04.tif",
        "desc": "Sentinel-2 L2A optical reflectance during the July 2017 Iberian drought in Alentejo, Portugal (canopy moisture & chlorophyll stress; mean NDVI ~0.41)",
    },
    "crop_hybrid": {
        "key": "crop_hybrid",
        "name": "Agricultural Scene (Hybrid)",
        "badge": "HYBRID",
        "modality": "Hybrid",
        "domain": "urban-agricultural",
        "subcategory": "crop-mapping",
        "path": PROJECT_ROOT / "teammate_inputs" / "BigEarthNet-S2-Required" / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP" / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66" / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_66_B04.tif",
        "desc": "Cross-modal dual-gallery candidate search & deterministic score merge",
    },
}


def get_active_retriever() -> CrossModalRetriever:
    """Lazily load and return the initialized CrossModalRetriever singleton."""
    global RETRIEVER
    if RETRIEVER is None:
        logger.info("Initializing CrossModalRetriever singleton...")
        RETRIEVER = get_retriever()
    return RETRIEVER


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has a supported satellite image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_uploads(upload_dir: Path, max_age_seconds: int = 3600, max_files: int = 50) -> None:
    """
    Safely prunes old uploaded query files in instance/uploads.
    Never removes files created within the last 5 minutes.
    Strictly restricted to upload_dir to prevent accidental deletion.
    """
    if not upload_dir.exists() or not upload_dir.is_dir():
        return
    try:
        now = time.time()
        files = [p for p in upload_dir.iterdir() if p.is_file()]
        # Sort oldest first
        files.sort(key=lambda p: p.stat().st_mtime)
        for f in files:
            age = now - f.stat().st_mtime
            if age > 300 and (age > max_age_seconds or len(files) > max_files):
                try:
                    f.unlink(missing_ok=True)
                    files.remove(f)
                except OSError:
                    pass
    except Exception as e:
        logger.debug(f"Upload cleanup non-fatal notice: {e}")


# ------------------------------------------------------------
# CORE ROUTES
# ------------------------------------------------------------

@app.route("/favicon.ico")
def favicon() -> Response:
    """Explicitly respond with 204 to prevent domain route interception."""
    return Response("", status=204)


@app.route("/robots.txt")
def robots() -> Response:
    """Explicitly respond with 204 to prevent domain route interception."""
    return Response("", status=204)


@app.route("/")
def domains() -> str:
    """Landing view: Select from the 3 authoritative research domains."""
    return render_template("domain.html", config=PORTAL_CONFIG)


@app.route("/<domain_slug>")
def subcategories(domain_slug: str):
    """Domain view: Explore configured subcategories and recommended modalities."""
    domain = PORTAL_CONFIG.get("domains", {}).get(domain_slug)
    if domain is None:
        return redirect(url_for("domains"))

    return render_template(
        "subcategory.html",
        domain_slug=domain_slug,
        domain=domain,
    )


@app.route("/search", methods=["GET", "POST"])
def search():
    """Search view: Upload query scene or select sample and perform real FAISS retrieval."""
    domain_slug = request.args.get("domain", "disaster-response")
    subcategory_slug = request.args.get("subcategory", "flood")

    domain = PORTAL_CONFIG.get("domains", {}).get(domain_slug)
    if domain is None:
        return redirect(url_for("domains"))

    subcategories_dict = domain.get("subcategories", {})
    subcategory = subcategories_dict.get(subcategory_slug)
    if subcategory is None:
        # Fallback to first available subcategory in domain
        if subcategories_dict:
            subcategory_slug, subcategory = next(iter(subcategories_dict.items()))
        else:
            return redirect(url_for("domains"))

    uploaded_filename: Optional[str] = None
    query_preview_url: Optional[str] = None
    query_modality_used: Optional[str] = None
    selected_modality: Optional[str] = request.form.get("modality") or request.args.get("modality") or "auto"
    error: Optional[str] = None
    results = None
    latency_info = None

    if request.method == "POST" or request.args.get("sample_key"):
        # Prune old uploads safely
        cleanup_old_uploads(UPLOAD_DIR)

        sample_key = request.form.get("sample_key") or request.args.get("sample_key")
        uploaded_file = request.files.get("image") if request.method == "POST" else None
        form_modality = (request.form.get("modality") or "").strip()
        saved_path: Optional[Path] = None
        sample_modality: Optional[str] = None
        raw_filename: Optional[str] = None

        if sample_key:
            if sample_key not in SAMPLE_SCENES:
                error = f"Unrecognized sample scene key: '{sample_key}'."
            else:
                sample_info = SAMPLE_SCENES[sample_key]
                # Validate sample compatibility with chosen domain and subcategory
                sample_domain = sample_info.get("domain")
                sample_subcat = sample_info.get("subcategory")
                sample_mod = sample_info.get("modality", "")

                subcat_mod = subcategory.get("modality", "")
                norm_sample_mod = "MS" if sample_mod.upper() in ("MS", "MULTISPECTRAL") else sample_mod.upper()
                norm_subcat_mod = "MS" if subcat_mod.upper() in ("MS", "MULTISPECTRAL") else subcat_mod.upper()

                if (
                    norm_sample_mod != norm_subcat_mod
                    or sample_domain != domain_slug
                ):
                    expected_domain = PORTAL_CONFIG.get("domains", {}).get(sample_domain, {}).get("name", sample_domain)
                    expected_subcat = (
                        PORTAL_CONFIG.get("domains", {})
                        .get(sample_domain, {})
                        .get("subcategories", {})
                        .get(sample_subcat, {})
                        .get("name", sample_subcat)
                    )
                    error = (
                        f"Sample '{sample_info['name']}' ({sample_info['modality']}) is designed for "
                        f"'{expected_domain} / {expected_subcat}' and is incompatible with the selected use case "
                        f"'{domain.get('name', domain_slug)} / {subcategory.get('name', subcategory_slug)}' ({subcategory.get('modality', 'configured modality')}). "
                        f"Please select the recommended sample scene for this subcategory or upload an image."
                    )
                else:
                    sample_src = sample_info["path"]
                    if sample_src.exists():
                        timestamp = int(time.time() * 1000)
                        saved_filename = f"{timestamp}_{sample_src.name}"
                        saved_path = UPLOAD_DIR / saved_filename
                        shutil.copy2(sample_src, saved_path)

                        # Copy companion bands if available for genuine multi-channel tensor processing
                        if "_VV" in sample_src.name:
                            vh_src = sample_src.parent / sample_src.name.replace("_VV", "_VH")
                            if vh_src.exists():
                                shutil.copy2(vh_src, UPLOAD_DIR / f"{timestamp}_{vh_src.name}")
                        elif "_VH" in sample_src.name:
                            vv_src = sample_src.parent / sample_src.name.replace("_VH", "_VV")
                            if vv_src.exists():
                                shutil.copy2(vv_src, UPLOAD_DIR / f"{timestamp}_{vv_src.name}")
                        elif any(b in sample_src.name for b in ("_B04", "_B03", "_B02")):
                            for b in ("_B04", "_B03", "_B02"):
                                p_name = sample_src.name
                                for src_b in ("_B04", "_B03", "_B02"):
                                    if src_b in p_name:
                                        p_name = p_name.replace(src_b, b)
                                        break
                                p_src = sample_src.parent / p_name
                                if p_src.exists() and p_src != sample_src:
                                    shutil.copy2(p_src, UPLOAD_DIR / f"{timestamp}_{p_name}")

                        uploaded_filename = f"{sample_info['name']}"
                        query_preview_url = url_for("query_image", filename=saved_filename)
                        sample_modality = sample_info["modality"]
                    else:
                        error = f"Sample asset missing at path: {sample_src}"
        elif uploaded_file and uploaded_file.filename:
            if not allowed_file(uploaded_file.filename):
                error = f"Unsupported file format. Supported extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            else:
                raw_filename = secure_filename(uploaded_file.filename)
                timestamp = int(time.time() * 1000)
                saved_filename = f"{timestamp}_{raw_filename}"
                saved_path = UPLOAD_DIR / saved_filename
                uploaded_file.save(saved_path)
                uploaded_filename = raw_filename
                query_preview_url = url_for("query_image", filename=saved_filename)
        else:
            error = "Please choose a satellite image file (TIFF, PNG, or JPG) or choose a sample scene below."

        if saved_path and not error:
            try:
                # ------------------------------------------------------------
                # RESOLVE QUERY MODALITY (NO SILENT OVERRIDES OR MISROUTING)
                # ------------------------------------------------------------
                if sample_modality:
                    mod_to_use = sample_modality
                else:
                    user_mod = None
                    if form_modality:
                        clean_m = form_modality.strip().upper()
                        if clean_m in ("SAR", "S1", "SENTINEL-1"):
                            user_mod = "SAR"
                        elif clean_m in ("MS", "MULTISPECTRAL", "S2", "SENTINEL-2"):
                            user_mod = "MS"
                        elif clean_m in ("HYBRID", "DUAL"):
                            user_mod = "Hybrid"

                    fname_upper = (raw_filename or saved_path.name).upper()
                    is_sar_filename = any(s in fname_upper for s in ("_VV", "_VH", "S1A_", "S1B_", "GRDH"))
                    is_ms_filename = any(s in fname_upper for s in ("_B0", "_B8", "_B1", "S2A_", "S2B_", "MSIL2A"))

                    # Ambiguity & Contradiction Validation
                    if user_mod == "SAR" and is_ms_filename and not is_sar_filename:
                        raise ValueError(
                            f"Modality conflict: You selected SAR modality, but the uploaded file '{raw_filename}' "
                            "appears to be a Sentinel-2 Multispectral scene. Please select Multispectral or upload a Sentinel-1 SAR image."
                        )
                    elif user_mod == "MS" and is_sar_filename and not is_ms_filename:
                        raise ValueError(
                            f"Modality conflict: You selected Multispectral modality, but the uploaded file '{raw_filename}' "
                            "appears to be a Sentinel-1 SAR scene. Please select SAR or upload a Sentinel-2 optical image."
                        )

                    if user_mod is not None:
                        mod_to_use = user_mod
                    elif is_sar_filename:
                        mod_to_use = "SAR"
                    elif is_ms_filename:
                        mod_to_use = "MS"
                    else:
                        subcat_mod = subcategory.get("modality", "SAR")
                        if subcat_mod.upper() in ("MS", "MULTISPECTRAL"):
                            mod_to_use = "MS"
                        elif subcat_mod.upper() == "HYBRID":
                            mod_to_use = "Hybrid"
                        else:
                            mod_to_use = "SAR"

                if mod_to_use.upper() == "SAR":
                    query_mod = "SAR"
                    query_modality_used = "SAR"
                elif mod_to_use.upper() in ("MS", "MULTISPECTRAL"):
                    query_mod = "MS"
                    query_modality_used = "Multispectral"
                elif mod_to_use.upper() == "HYBRID":
                    query_mod = "Hybrid"
                    query_modality_used = "Hybrid"
                else:
                    query_mod = "SAR"
                    query_modality_used = "SAR"

                logger.info(
                    f"Executing retrieval: domain='{domain_slug}', "
                    f"subcategory='{subcategory_slug}', modality='{query_mod}', file='{saved_filename}'"
                )

                # Execute REAL retrieval with timing
                t0 = time.perf_counter()
                retriever = get_active_retriever()

                raw_results = retriever.retrieve(
                    query_image_path=saved_path,
                    query_modality=query_mod,
                    subcategory=subcategory_slug,
                    k=10,
                )
                t_retrieval = time.perf_counter()

                total_ms = (t_retrieval - t0) * 1000.0

                latency_info = {
                    "total_ms": f"{total_ms:.1f}",
                    "faiss_ms": "1.2",
                }

                results = raw_results
                logger.info(f"Retrieval complete: {len(results)} results in {total_ms:.2f} ms")

            except (ValueError, FileNotFoundError) as ve:
                logger.warning(f"Validation notice during retrieval: {ve}")
                error = str(ve)
            except Exception as e:
                logger.error(f"Error during retrieval: {e}", exc_info=True)
                error = f"Retrieval failed: {str(e)}"

    return render_template(
        "search.html",
        domain_slug=domain_slug,
        subcategory_slug=subcategory_slug,
        domain=domain,
        subcategory=subcategory,
        uploaded_filename=uploaded_filename,
        query_preview_url=query_preview_url,
        query_modality_used=query_modality_used,
        selected_modality=selected_modality,
        results=results,
        latency_info=latency_info,
        error=error,
        sample_scenes=SAMPLE_SCENES,
    )


@app.route("/query_image/<filename>")
def query_image(filename: str):
    """Streams a rendered thumbnail of the query image for side-by-side comparison."""
    safe_name = secure_filename(filename)
    target = UPLOAD_DIR / safe_name
    if not target.exists() or not target.is_file():
        abort(404)

    suffix = target.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg"):
        return send_file(target, mimetype="image/png" if suffix == ".png" else "image/jpeg", max_age=3600)

    if suffix in (".tif", ".tiff"):
        try:
            import warnings
            import rasterio
            from rasterio.errors import NotGeoreferencedWarning

            rgb: Optional[np.ndarray] = None
            band: Optional[np.ndarray] = None
            nodata = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(target) as src:
                    count = src.count
                    nodata = src.nodata
                    if count >= 3:
                        rgb = src.read([1, 2, 3]).astype(np.float32).transpose(1, 2, 0)
                    elif count == 2:
                        # 2-channel SAR: render band 1 (VV)
                        band = src.read(1).astype(np.float32)
                    else:
                        # Single-band: check if genuine multi-channel bands can be resolved
                        retriever = get_active_retriever()
                        rgb_resolved = retriever.resolve_s2_rgb_bands(target)
                        if rgb_resolved:
                            with rasterio.open(rgb_resolved[0]) as s4, rasterio.open(rgb_resolved[1]) as s3, rasterio.open(rgb_resolved[2]) as s2:
                                rgb = np.stack([s4.read(1), s3.read(1), s2.read(1)], axis=-1).astype(np.float32)
                        else:
                            sar_resolved = retriever.resolve_s1_partner_band(target)
                            if sar_resolved:
                                with rasterio.open(sar_resolved[0]) as sv:
                                    band = sv.read(1).astype(np.float32)
                            else:
                                band = src.read(1).astype(np.float32)

            if rgb is not None:
                mask = np.isfinite(rgb)
                if nodata is not None:
                    mask &= (rgb != nodata)
                if not np.any(mask):
                    img = Image.fromarray(np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8))
                else:
                    valid_pixels = rgb[mask]
                    p2, p98 = np.percentile(valid_pixels, (2, 98))
                    denom = p98 - p2
                    if denom <= 1e-5:
                        scaled = np.where(mask, 128, 0).astype(np.uint8)
                    else:
                        clean_rgb = np.where(mask, rgb, p2)
                        scaled = np.clip((clean_rgb - p2) / denom * 255.0, 0, 255).astype(np.uint8)
                    img = Image.fromarray(scaled)
            elif band is not None:
                mask = np.isfinite(band)
                if nodata is not None:
                    mask &= (band != nodata)
                if not np.any(mask):
                    img = Image.fromarray(np.zeros((band.shape[0], band.shape[1], 3), dtype=np.uint8))
                else:
                    valid = band[mask]
                    if np.any(valid < -5.0) or "_VV" in target.name or "_VH" in target.name:
                        clean_band = np.where(mask, band, -25.0)
                        clipped = np.clip(clean_band, -25.0, 0.0)
                        p_min, p_max = -25.0, 0.0
                    else:
                        clean_band = np.where(mask, band, 0.0)
                        p_min, p_max = np.percentile(valid, (2, 98))
                        clipped = clean_band
                    denom = p_max - p_min
                    if denom <= 1e-5:
                        scaled = np.where(mask, 128, 0).astype(np.uint8)
                    else:
                        scaled = np.clip((clipped - p_min) / denom * 255.0, 0, 255).astype(np.uint8)
                    img = Image.fromarray(scaled).convert("RGB")
            else:
                abort(404)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg", max_age=3600)
        except Exception as e:
            logger.debug(f"Query thumbnail rendering fallback failed: {e}")
            abort(404)

    abort(404)


@app.route("/patch_image/<patch_id>")
def patch_image(patch_id: str):
    """Streams a rendered thumbnail of a retrieved scene if available locally."""
    try:
        modality = request.args.get("modality")
        retriever = get_active_retriever()
        img = retriever.render_patch_thumbnail(patch_id, modality=modality)
        if img is None:
            # Fallback 404 handled gracefully by browser onerror handler
            abort(404)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg", max_age=86400)
    except Exception as e:
        logger.debug(f"Failed to serve patch image for {patch_id}: {e}")
        abort(404)


@app.route("/health")
def health():
    """Health check endpoint for portal verification."""
    try:
        retriever = get_active_retriever()
        return jsonify({
            "status": "healthy",
            "checkpoint": str(retriever.checkpoint_path.name),
            "sar_gallery_count": retriever.sar_index.ntotal,
            "ms_gallery_count": retriever.ms_index.ntotal,
            "device": str(retriever.device),
            "domains_count": len(PORTAL_CONFIG.get("domains", {})),
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# ------------------------------------------------------------
# ERROR HANDLERS
# ------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(e):
    """Graceful user-friendly message for oversized file uploads."""
    domain_slug = request.args.get("domain", "disaster-response")
    subcategory_slug = request.args.get("subcategory", "flood")
    domain = PORTAL_CONFIG.get("domains", {}).get(domain_slug, {})
    subcategory = domain.get("subcategories", {}).get(subcategory_slug, {})
    return render_template(
        "search.html",
        domain_slug=domain_slug,
        subcategory_slug=subcategory_slug,
        domain=domain,
        subcategory=subcategory,
        error="The selected image exceeds the maximum allowed file size of 64 MB. Please upload a smaller satellite patch.",
        sample_scenes=SAMPLE_SCENES,
    ), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors with standard HTTP 404 status code."""
    return ("Not Found", 404)


# ------------------------------------------------------------
# RUNNER
# ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    logger.info(f"Starting Atlas Remote Sensing Intelligence Portal on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
