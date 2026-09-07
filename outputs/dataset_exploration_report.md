# BigEarthNet-S1 Exploration & Statistics Report

This report summarizes the dataset structure and statistics discovered during the initial exploration phase.

## 1. Directory Structure & Patches
- **Dataset Path**: `C:\Users\Meet Katarmal\OneDrive\Desktop\Major-project\data\BigEarthNet-S1\BigEarthNet-S1`
- **Sub-group Directories**: `312`
- **Total Image Patches Detected**: `549,488`
- **File Extensions found inside Patch Folders**: `.tif`

## 2. Band Information
- **Sentinel-1 Polarizations**: VV and VH
- **Sample Patch Name**: `S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39`
- **VV Suffix Match**: `S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VV.tif`
- **VH Suffix Match**: `S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VH.tif`

## 3. Image Properties
- **Dimensions**: `120x120 (pixels)`
- **Data Type**: `float32`

## 4. Band Pixel Ranges & Statistics (Sampled over 50 patches)
### VV Polarization Band
- **Estimated Min (dB)**: `-38.4984`
- **Estimated Max (dB)**: `19.4053`
- **Estimated Mean (dB)**: `-11.5842`

### VH Polarization Band
- **Estimated Min (dB)**: `-45.9771`
- **Estimated Max (dB)**: `12.1303`
- **Estimated Mean (dB)**: `-17.9297`

---
*Note: S1 values are stored in Float32 format representing radar backscatter intensity in Decibel (dB) scale. Scaling constants in `config/settings.py` can be set using these ranges to clip outliers before normalizing.*
