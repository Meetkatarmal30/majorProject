"""
build_s1_subset.py

Creates the exact 30,000-patch Sentinel-1 subset for Jainam.
- Source of truth: teammate_inputs/train_split.csv, val_split.csv, test_split.csv
- Destination: teammate_inputs/BigEarthNet-S1-Required/
- Copies only _VV.tif and _VH.tif per patch
- Preserves GROUP/S1_PATCH_NAME/ hierarchy
- Verifies post-copy completeness
- Generates manifest
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

def main():
    start_time = time.time()
    base_dir = Path.cwd()
    
    # 1. Inputs
    csv_train = base_dir / "teammate_inputs" / "train_split.csv"
    csv_val = base_dir / "teammate_inputs" / "val_split.csv"
    csv_test = base_dir / "teammate_inputs" / "test_split.csv"
    
    dest_root = base_dir / "teammate_inputs" / "BigEarthNet-S1-Required"
    s1_src_root = base_dir / "data" / "BigEarthNet-S1" / "BigEarthNet-S1"
    cache_path = base_dir / ".s1_patch_cache.json"
    
    print("Loading CSV splits...")
    df_train = pd.read_csv(csv_train)
    df_val = pd.read_csv(csv_val)
    df_test = pd.read_csv(csv_test)
    
    train_names = df_train["s1_name"].tolist()
    val_names = df_val["s1_name"].tolist()
    test_names = df_test["s1_name"].tolist()
    
    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)
    
    total_patches = len(train_names) + len(val_names) + len(test_names)
    unique_patches = len(train_set | val_set | test_set)
    
    print(f"Train: {len(train_names)} (unique: {len(train_set)})")
    print(f"Val: {len(val_names)} (unique: {len(val_set)})")
    print(f"Test: {len(test_names)} (unique: {len(test_set)})")
    print(f"Total: {total_patches}, Unique: {unique_patches}")
    
    assert len(train_names) == 21000, f"Expected 21000 train, got {len(train_names)}"
    assert len(val_names) == 4500, f"Expected 4500 val, got {len(val_names)}"
    assert len(test_names) == 4500, f"Expected 4500 test, got {len(test_names)}"
    assert unique_patches == 30000, f"Expected 30000 unique, got {unique_patches}"
    assert len(train_set & val_set) == 0, "Train and Val overlap!"
    assert len(train_set & test_set) == 0, "Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Val and Test overlap!"
    
    print("Split verification passed: exactly 30,000 unique patches, 0 overlap.")
    
    # Load cache
    print("Loading S1 directory cache...")
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache loaded with {len(cache)} entries.")
    
    # Prepare copy jobs
    # Each patch requires:
    #   GROUP = '_'.join(tokens[:5])
    #   src_dir = cache.get(s1_name) or (s1_src_root / GROUP / s1_name)
    #   dest_dir = dest_root / GROUP / s1_name
    #   vv_file: src_dir / f"{s1_name}_VV.tif" -> dest_dir / f"{s1_name}_VV.tif"
    #   vh_file: src_dir / f"{s1_name}_VH.tif" -> dest_dir / f"{s1_name}_VH.tif"
    
    all_names = train_names + val_names + test_names
    jobs = []
    
    for name in all_names:
        tokens = name.split("_")
        group = "_".join(tokens[:5])
        src_dir = Path(cache.get(name, s1_src_root / group / name))
        target_dir = dest_root / group / name
        jobs.append((name, group, src_dir, target_dir))
        
    print(f"Prepared {len(jobs)} patch copy jobs. Creating target directories...")
    dest_root.mkdir(parents=True, exist_ok=True)
    
    # Execute copy patch by patch
    # We can use ThreadPoolExecutor for high I/O throughput
    failed_copies = []
    copied_patches = 0
    copied_vv = 0
    copied_vh = 0
    total_bytes = 0

    def copy_patch(job):
        name, group, src_dir, target_dir = job
        vv_src = src_dir / f"{name}_VV.tif"
        vh_src = src_dir / f"{name}_VH.tif"
        
        target_dir.mkdir(parents=True, exist_ok=True)
        vv_dst = target_dir / f"{name}_VV.tif"
        vh_dst = target_dir / f"{name}_VH.tif"
        
        # Copy VV
        shutil.copyfile(vv_src, vv_dst)
        vv_size = vv_dst.stat().st_size
        
        # Copy VH
        shutil.copyfile(vh_src, vh_dst)
        vh_size = vh_dst.stat().st_size
        
        return (name, vv_size + vh_size)

    print("Beginning file copy across workers...")
    num_workers = 4
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(copy_patch, job): job[0] for job in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            patch_name = futures[future]
            try:
                name, b_count = future.result()
                copied_patches += 1
                copied_vv += 1
                copied_vh += 1
                total_bytes += b_count
            except Exception as e:
                failed_copies.append((patch_name, str(e)))
                
            if idx % 5000 == 0 or idx == len(jobs):
                elapsed = time.time() - start_time
                print(f"Progress: {idx}/{len(jobs)} patches processed ({copied_patches} succeeded, {len(failed_copies)} failed) [{elapsed:.1f}s]")

    copy_time = time.time() - start_time
    print(f"Copy phase completed in {copy_time:.2f}s.")
    print(f"Patches copied: {copied_patches}")
    print(f"VV files copied: {copied_vv}")
    print(f"VH files copied: {copied_vh}")
    print(f"Total size copied: {total_bytes} bytes ({total_bytes / (1024**3):.2f} GB)")
    print(f"Failed copies: {len(failed_copies)}")
    if failed_copies:
        print("First 5 failures:", failed_copies[:5])
        sys.exit(1)

    # Verification phase
    print("\n--- RUNNING POST-COPY VERIFICATION ---")
    dest_patch_count = 0
    dest_vv_count = 0
    dest_vh_count = 0
    dest_unexpected = 0
    missing_patches = []
    
    # Verify every required patch exists
    for name in all_names:
        tokens = name.split("_")
        group = "_".join(tokens[:5])
        p_dir = dest_root / group / name
        if not p_dir.is_dir():
            missing_patches.append(name)
            continue
        vv_file = p_dir / f"{name}_VV.tif"
        vh_file = p_dir / f"{name}_VH.tif"
        if not vv_file.is_file() or not vh_file.is_file():
            missing_patches.append(name)

    # Scan dest_root to ensure NO extra files or patches exist
    dest_groups = [d for d in dest_root.iterdir() if d.is_dir()]
    for g_dir in dest_groups:
        for p_dir in g_dir.iterdir():
            if p_dir.is_dir():
                dest_patch_count += 1
                for f in p_dir.iterdir():
                    if f.name.endswith("_VV.tif"):
                        dest_vv_count += 1
                    elif f.name.endswith("_VH.tif"):
                        dest_vh_count += 1
                    else:
                        dest_unexpected += 1
            else:
                dest_unexpected += 1

    print(f"Destination patch folders: {dest_patch_count} (Expected: 30000)")
    print(f"Destination VV TIFFs: {dest_vv_count} (Expected: 30000)")
    print(f"Destination VH TIFFs: {dest_vh_count} (Expected: 30000)")
    print(f"Destination unexpected files/dirs: {dest_unexpected} (Expected: 0)")
    print(f"Missing required patches: {len(missing_patches)} (Expected: 0)")

    assert dest_patch_count == 30000, f"Expected 30000 patch folders, got {dest_patch_count}"
    assert dest_vv_count == 30000, f"Expected 30000 VV TIFFs, got {dest_vv_count}"
    assert dest_vh_count == 30000, f"Expected 30000 VH TIFFs, got {dest_vh_count}"
    assert dest_unexpected == 0, f"Expected 0 unexpected files, got {dest_unexpected}"
    assert len(missing_patches) == 0, f"Expected 0 missing patches, got {len(missing_patches)}"

    # Generate Manifest / Report
    manifest_data = {
        "dataset_name": "BigEarthNet-S1-Required-30K",
        "destination_path": str(dest_root.resolve()),
        "total_patches": total_patches,
        "splits": {
            "train": len(train_names),
            "validation": len(val_names),
            "test": len(test_names)
        },
        "number_of_patch_folders_copied": copied_patches,
        "number_of_vv_files_copied": copied_vv,
        "number_of_vh_files_copied": copied_vh,
        "missing_files": len(missing_patches),
        "failed_copies": len(failed_copies),
        "total_subset_size_bytes": total_bytes,
        "total_subset_size_gb": round(total_bytes / (1024**3), 4),
        "verification_status": "PASSED"
    }

    manifest_path = base_dir / "teammate_inputs" / "s1_30k_subset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Manifest written to: {manifest_path}")

    # Also write a text report
    report_path = base_dir / "teammate_inputs" / "s1_30k_subset_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== S1 30K SUBSET GENERATION REPORT ===\n")
        f.write(f"Destination: {dest_root.resolve()}\n")
        f.write(f"Total patches: {total_patches}\n")
        f.write(f"Train: {len(train_names)}\n")
        f.write(f"Validation: {len(val_names)}\n")
        f.write(f"Test: {len(test_names)}\n")
        f.write(f"Patch folders copied: {copied_patches}\n")
        f.write(f"VV TIFF files copied: {copied_vv}\n")
        f.write(f"VH TIFF files copied: {copied_vh}\n")
        f.write(f"Missing files: {len(missing_patches)}\n")
        f.write(f"Failed copies: {len(failed_copies)}\n")
        f.write(f"Total subset size: {total_bytes} bytes ({total_bytes / (1024**3):.4f} GB)\n")
        f.write(f"Verification: PASSED\n")
        f.write(f"Completed in: {time.time() - start_time:.2f} seconds\n")
    print(f"Report written to: {report_path}")

    print("\nSUCCESS: S1 30K SUBSET CREATED AND VERIFIED.")

if __name__ == "__main__":
    main()
