"""
inspect_dataset.py - Dataset Inspection Script for BigEarthNet-S1.

This script scans the downloaded BigEarthNet-S1 dataset directory, determines
its folder hierarchy, file extensions, patch counts, band names, metadata format,
and image dimensions, and generates a structured markdown report.

Optimized to use os.scandir for high-speed listing on Windows (avoiding slow recursive rglob).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# We try to import rasterio for reading GeoTIFF properties. If not available, we fall back to PIL
HAS_RASTERIO = False
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    try:
        from PIL import Image
    except ImportError:
        pass

# Setup standardized logger
logger = logging.getLogger("DatasetInspection")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def inspect_directory(data_dir: Path) -> Dict[str, Any]:
    """
    Scans the data directory to analyze its folder hierarchy and discover files.
    Optimized for large directories using os.scandir.

    Args:
        data_dir: Path to the BigEarthNet-S1 directory.

    Returns:
        Dict[str, Any]: A dictionary containing inspection results.
    """
    logger.info(f"Starting optimized inspection of directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"The specified dataset directory does not exist: {data_dir}\n"
            "Please ensure the download is complete, the archive is extracted, "
            "and settings.py or the BIGEARTHNET_S1_DIR environment variable is set correctly."
        )

    results: Dict[str, Any] = {
        "folder_hierarchy": "Unknown",
        "file_extensions": set(),
        "total_patches": 0,
        "band_files": {"VV": [], "VH": [], "Other": []},
        "metadata_files": [],
        "sample_tree": [],
        "image_dims": "Unknown",
        "metadata_format": "None",
        "split_files": [],
    }

    # Step 1: Scan top-level elements
    top_dirs = []
    top_files = []
    try:
        with os.scandir(data_dir) as it:
            for entry in it:
                if entry.is_dir():
                    top_dirs.append(entry)
                elif entry.is_file():
                    top_files.append(entry)
    except Exception as e:
        logger.error(f"Failed to scan root directory: {e}")
        raise e

    logger.info(f"Root folder contains {len(top_dirs)} directories and {len(top_files)} files.")
    
    # Check for split files in the root
    for entry in top_files:
        suffix = os.path.splitext(entry.name)[1].lower()
        if suffix in [".txt", ".csv", ".json", ".parquet"]:
            results["split_files"].append(Path(entry.path))

    # Detect hierarchy style and traverse
    patch_folders: List[Path] = []
    total_patches = 0
    file_extensions: Set[str] = set()

    # We sample a few patch folders to check files
    sampled_patches = []

    # Let's check if the first folder contains TIFF files directly, or if it is nested
    if len(top_dirs) > 0:
        first_dir = top_dirs[0]
        # Inspect first dir
        sub_elements = []
        with os.scandir(first_dir.path) as it:
            for entry in it:
                sub_elements.append(entry)
        
        has_subdirs = any(e.is_dir() for e in sub_elements)
        
        if has_subdirs:
            results["folder_hierarchy"] = "Nested (Patch folders grouped under subdirectories)"
            logger.info("Detected nested folder hierarchy. Counting patches...")
            # We have group folders (e.g. S1A_IW_GRDH...) containing patch folders
            for group_dir in top_dirs:
                try:
                    with os.scandir(group_dir.path) as it:
                        for entry in it:
                            if entry.is_dir():
                                total_patches += 1
                                patch_path = Path(entry.path)
                                patch_folders.append(patch_path)
                                # Keep samples of first 3 patches for directory tree
                                if len(sampled_patches) < 3:
                                    sampled_patches.append(patch_path)
                except Exception as e:
                    logger.warning(f"Error scanning subdirectory {group_dir.path}: {e}")
        else:
            results["folder_hierarchy"] = "Flat (All patch folders directly under root)"
            logger.info("Detected flat folder hierarchy. Counting patches...")
            for entry in top_dirs:
                total_patches += 1
                patch_path = Path(entry.path)
                patch_folders.append(patch_path)
                if len(sampled_patches) < 3:
                    sampled_patches.append(patch_path)

    results["total_patches"] = total_patches

    # Sample leaf contents from the first few folders to get extensions, bands, metadata formats
    logger.info("Sampling leaf patch folders to inspect file structures...")
    sample_files_inspected = 0
    for patch in patch_folders[:10]:
        try:
            with os.scandir(patch) as it:
                for entry in it:
                    if entry.is_file():
                        p = Path(entry.path)
                        file_extensions.add(p.suffix.lower())
                        fname = p.name.lower()
                        
                        if p.suffix.lower() == ".json":
                            results["metadata_files"].append(p)
                        elif p.suffix.lower() in [".tif", ".tiff"]:
                            if "vv" in fname:
                                results["band_files"]["VV"].append(p)
                            elif "vh" in fname:
                                results["band_files"]["VH"].append(p)
                            else:
                                results["band_files"]["Other"].append(p)
                        sample_files_inspected += 1
        except Exception as e:
            logger.warning(f"Error sampling patch {patch}: {e}")

    results["file_extensions"] = list(file_extensions)

    # Get sample directory tree for report
    tree_lines: List[str] = []
    for patch in sampled_patches:
        tree_lines.append(f"├── {patch.name}/")
        try:
            with os.scandir(patch) as it:
                for entry in it:
                    if entry.is_file():
                        tree_lines.append(f"│   ├── {entry.name}")
        except Exception:
            pass
    results["sample_tree"] = tree_lines

    # Read image properties from a sample band file if found
    sample_img_path = None
    if results["band_files"]["VV"]:
        sample_img_path = results["band_files"]["VV"][0]
    elif results["band_files"]["VH"]:
        sample_img_path = results["band_files"]["VH"][0]
    elif results["band_files"]["Other"]:
        sample_img_path = results["band_files"]["Other"][0]

    if sample_img_path:
        results["image_dims"] = get_image_properties(sample_img_path)
        logger.info(f"Sample Image dimensions: {results['image_dims']}")
    else:
        logger.warning("No Sentinel-1 band (.tif) files found in sampled directories.")

    # Inspect metadata format if JSON files were found
    if results["metadata_files"]:
        sample_metadata_path = results["metadata_files"][0]
        try:
            with open(sample_metadata_path, "r", encoding="utf-8") as meta_f:
                meta_content = json.load(meta_f)
                results["metadata_format"] = {
                    "keys": list(meta_content.keys()),
                    "sample": meta_content
                }
        except Exception as e:
            results["metadata_format"] = f"Error reading JSON metadata: {e}"
    else:
        results["metadata_format"] = "No JSON metadata files found within sampled patch folders. (Sentinel-1 datasets typically store labels externally or share S2 metadata)."

    return results


def get_image_properties(img_path: Path) -> str:
    """
    Reads image dimensions and data type from a raster file.

    Args:
        img_path: Path to the image file.

    Returns:
        str: Description of image properties (width, height, bands, dtype).
    """
    if HAS_RASTERIO:
        try:
            with rasterio.open(img_path) as src:
                return f"Dimensions: {src.width}x{src.height}, Bands: {src.count}, Dtype: {src.dtypes[0]}"
        except Exception as e:
            return f"Error reading with rasterio: {e}"
    else:
        # Fallback to PIL
        try:
            with Image.open(img_path) as img:
                return f"Dimensions: {img.width}x{img.height}, Format: {img.format}, Mode: {img.mode}"
        except Exception as e:
            return f"Error reading with PIL: {e}"


def generate_report(results: Dict[str, Any], output_path: Path) -> None:
    """
    Generates a markdown report summarizing the dataset structure.

    Args:
        results: Inspection results.
        output_path: Target path for the report file.
    """
    os.makedirs(output_path.parent, exist_ok=True)
    
    metadata_sample_str = ""
    if isinstance(results["metadata_format"], dict):
        metadata_sample_str = json.dumps(results["metadata_format"]["sample"], indent=2)
    else:
        metadata_sample_str = str(results["metadata_format"])

    split_files_str = "\n".join([f"- {f.name}" for f in results["split_files"]]) if results["split_files"] else "None found in root directory"

    report_content = f"""# BigEarthNet-S1 Dataset Inspection Report

Generated automatically by `inspect_dataset.py` (optimized version).

## Directory Analysis

- **Folder Hierarchy**: {results["folder_hierarchy"]}
- **Total Patches Detected**: {results["total_patches"]}
- **Detected File Extensions in Patches**: {", ".join(results["file_extensions"])}

## Band File Detection (Sampled)

- **Sampled VV Band Files**: {len(results["band_files"]["VV"])}
- **Sampled VH Band Files**: {len(results["band_files"]["VH"])}
- **Sampled Other TIFF Files**: {len(results["band_files"]["Other"])}

## Image Metadata & Formats

- **Sample Image Properties**: {results["image_dims"]}
- **Split Configuration Files**: 
{split_files_str}

## Metadata Schema Detail
{metadata_sample_str}

## Sample Directory Tree
```text
{results['folder_hierarchy']}
{chr(10).join(results['sample_tree'])}
```
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Inspection report written successfully to {output_path}")


if __name__ == "__main__":
    # Import config variables
    try:
        from config.settings import DATA_DIR, BASE_DIR
        target_dir = DATA_DIR
        report_file = BASE_DIR / "outputs" / "dataset_inspection_report.md"
    except ImportError:
        target_dir = Path("./data/BigEarthNet-S1")
        report_file = Path("./outputs/dataset_inspection_report.md")

    logger.info(f"Targeting dataset directory: {target_dir}")
    
    try:
        results = inspect_directory(target_dir)
        generate_report(results, report_file)
        print("\n=== Dataset Inspection Report ===")
        print(f"Report generated at: {report_file}")
        print(f"Total Patches Found: {results['total_patches']}")
        print(f"File Extensions: {results['file_extensions']}")
        print(f"Folder Hierarchy: {results['folder_hierarchy']}")
    except FileNotFoundError as e:
        logger.error(str(e))
        print("\n[INSPECTION RUNNER NOTICE]")
        print("The dataset is currently downloading. The inspection script has been written.")
        print(f"Once the download completes, extract the files and run:")
        print(f"  python inspect_dataset.py")
        print("This will analyze the actual dataset and generate the report in outputs/dataset_inspection_report.md.")
