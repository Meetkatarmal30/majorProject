"""
create_subset.py - Script to extract a deterministic 4,500-patch validation subset.

Since val_split.csv is not present on the disk, this script extracts the first 4,500
patches from the standard deterministic validation split (15% of the dataset shuffled
with seed 42) of the BigEarthNetS1Dataset.

Copies the corresponding VV and VH GeoTIFF files into data/S1_val_subset_4500/
and packages them into S1_SAR_4500.zip.
"""

import zipfile
import shutil
import logging
from pathlib import Path

# Add parent directory to path to import dataset
import sys
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from config.settings import DATA_DIR, BASE_DIR, VV_BAND_SUFFIX, VH_BAND_SUFFIX
from dataset.dataset import BigEarthNetS1Dataset

# Set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SubsetGenerator")


def main():
    logger.info("Initializing BigEarthNetS1Dataset for validation split...")
    # Initialize validation split of the dataset
    dataset = BigEarthNetS1Dataset(split="validation")
    all_val_patches = dataset.patch_names
    
    total_val_available = len(all_val_patches)
    logger.info(f"Total validation patches available: {total_val_available}")
    
    # Select the first 4,500 patches deterministically
    target_count = 4500
    if total_val_available < target_count:
        raise ValueError(f"Not enough validation patches. Available: {total_val_available}, Required: {target_count}")
        
    selected_patches = all_val_patches[:target_count]
    logger.info(f"Selected {len(selected_patches)} patches for subsetting.")
    
    # Destination directory
    subset_dir = DATA_DIR / "S1_val_subset_4500"
    if subset_dir.exists():
        logger.info(f"Cleaning up existing subset directory: {subset_dir}")
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    copied_count = 0
    missing_count = 0
    missing_patches = []
    
    logger.info("Starting file copy operation...")
    for idx, patch_name in enumerate(selected_patches, 1):
        if patch_name not in dataset.patch_dict:
            missing_count += 1
            missing_patches.append(patch_name)
            continue
            
        patch_src_dir = Path(dataset.patch_dict[patch_name])
        
        # Verify VV and VH paths
        vv_src = next(patch_src_dir.glob(f"*{VV_BAND_SUFFIX}"), None)
        vh_src = next(patch_src_dir.glob(f"*{VH_BAND_SUFFIX}"), None)
        
        if not vv_src or not vh_src:
            logger.warning(f"Patch {patch_name} is missing VV or VH file in source directory.")
            missing_count += 1
            missing_patches.append(patch_name)
            continue
            
        # Create destination folder
        patch_dest_dir = subset_dir / patch_name
        patch_dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        shutil.copy2(vv_src, patch_dest_dir / vv_src.name)
        shutil.copy2(vh_src, patch_dest_dir / vh_src.name)
        copied_count += 1
        
        if idx % 500 == 0 or idx == target_count:
            logger.info(f"Processed {idx}/{target_count} patches...")

    # Stop and report if any are missing
    if missing_count > 0:
        logger.error(f"Failed to find or copy {missing_count} patches. Missing list: {missing_patches}")
        sys.exit(1)
        
    logger.info("Copying complete. Verifying integrity...")
    
    # Verify outputs
    subset_folders = list(subset_dir.glob("*"))
    unique_folders = set(f.name for f in subset_folders)
    
    total_vv_files = len(list(subset_dir.glob("*/*_VV.tif")))
    total_vh_files = len(list(subset_dir.glob("*/*_VH.tif")))
    
    logger.info("========================================")
    logger.info("VERIFICATION SUMMARY")
    logger.info("========================================")
    logger.info(f"Required IDs:             {target_count}")
    logger.info(f"Unique IDs Selected:      {len(unique_folders)}")
    logger.info(f"Folders Found & Copied:   {len(subset_folders)}")
    logger.info(f"Total VV Files:           {total_vv_files}")
    logger.info(f"Total VH Files:           {total_vh_files}")
    logger.info(f"Missing Patch IDs:        {missing_count}")
    logger.info(f"Original Dataset Intact:  Yes (Read-Only access)")
    logger.info("========================================")

    # Assertions to ensure exact correctness
    assert len(unique_folders) == target_count, f"Expected {target_count} unique patch folders, got {len(unique_folders)}"
    assert len(subset_folders) == target_count, f"Expected {target_count} patch folders, got {len(subset_folders)}"
    assert total_vv_files == target_count, f"Expected {target_count} VV files, got {total_vv_files}"
    assert total_vh_files == target_count, f"Expected {target_count} VH files, got {total_vh_files}"
    
    logger.info("Integrity check passed successfully. Zipping subset...")
    
    zip_path = BASE_DIR / "S1_SAR_4500.zip"
    if zip_path.exists():
        logger.info(f"Removing existing zip archive: {zip_path}")
        zip_path.unlink()
        
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for folder in subset_folders:
            for file_path in folder.glob("*"):
                # Store files matching the directory structure
                arcname = Path(folder.name) / file_path.name
                zip_file.write(file_path, arcname=arcname)
                
    logger.info(f"Successfully generated zip archive: {zip_path} (Size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")
    print("\n[SUCCESS] Subset generation and packaging completed successfully!\n")


if __name__ == "__main__":
    main()
