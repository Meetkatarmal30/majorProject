"""
settings.py - Centralized configuration settings for the Sentinel-1 (SAR) data pipeline.

This module defines configuration options, paths, preprocessing parameters, and
logging configurations used across all modules. Paths are loaded dynamically
and can be overridden via environment variables to avoid hardcoding.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Base Project Directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Dataset Directory - Overridable via environment variable 'BIGEARTHNET_S1_DIR'
_RAW_DATA_DIR = Path(os.getenv("BIGEARTHNET_S1_DIR", BASE_DIR / "data" / "BigEarthNet-S1"))
if (_RAW_DATA_DIR / "BigEarthNet-S1").exists() and (_RAW_DATA_DIR / "BigEarthNet-S1").is_dir():
    DATA_DIR: Path = _RAW_DATA_DIR / "BigEarthNet-S1"
else:
    DATA_DIR: Path = _RAW_DATA_DIR

# Splits Directory (if train.txt, val.txt, test.txt are provided separately)
SPLITS_DIR: Path = Path(os.getenv("BIGEARTHNET_SPLITS_DIR", BASE_DIR / "data" / "splits"))

# Preprocessing Constants
IMAGE_SIZE: Tuple[int, int] = (120, 120)  # Default Sentinel-1 patch dimensions in pixels
PRESERVE_RESOLUTION: bool = True          # If True, keeps original image shape; otherwise, resizes to IMAGE_SIZE

# Normalization Configurations
# Supported modes: "none", "min-max", "z-score"
NORMALIZATION_MODE: str = "min-max"

# Sentinel-1 Polarization Suffixes (assumed defaults, will be confirmed in inspection)
VV_BAND_SUFFIX: str = "_VV.tif"
VH_BAND_SUFFIX: str = "_VH.tif"
METADATA_SUFFIX: str = "_labels_metadata.json"

# Normalization Clipping Boundaries (in Decibels - dB)
# These are standard clipping ranges for Sentinel-1 GRD backscatter coefficient to remove outliers
VV_CLIP_RANGE: Tuple[float, float] = (-25.0, 0.0)
VH_CLIP_RANGE: Tuple[float, float] = (-32.0, -5.0)

# Default estimated mean & standard deviation for Z-score normalization
VV_ZSCORE_STATS: Tuple[float, float] = (-12.403296, 4.895232)
VH_ZSCORE_STATS: Tuple[float, float] = (-19.085194, 5.292950)

# Training / DataLoader Defaults
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_RANDOM_STATE: int = 42

# Logging Configuration
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: int = logging.INFO

def setup_logging(name: str = "SAR_Pipeline") -> logging.Logger:
    """
    Sets up a standardized logger for team-wide integration.
    
    Args:
        name: Name of the logger, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: Standardized logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        # Create console handler and set format
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# Exported Settings Dictionary for easy reading/dumping
def get_config_summary() -> Dict[str, Any]:
    """
    Returns a dictionary summarizing the current settings.
    """
    return {
        "BASE_DIR": str(BASE_DIR),
        "DATA_DIR": str(DATA_DIR),
        "SPLITS_DIR": str(SPLITS_DIR),
        "IMAGE_SIZE": IMAGE_SIZE,
        "VV_CLIP_RANGE": VV_CLIP_RANGE,
        "VH_CLIP_RANGE": VH_CLIP_RANGE,
        "DEFAULT_BATCH_SIZE": DEFAULT_BATCH_SIZE,
        "DEFAULT_NUM_WORKERS": DEFAULT_NUM_WORKERS,
    }

if __name__ == "__main__":
    # Standard demo execution block to verify the settings import and setup
    logger = setup_logging("settings_demo")
    logger.info("Sentinel-1 SAR pipeline configuration loaded.")
    logger.info(f"Project Base Directory: {BASE_DIR}")
    logger.info(f"BigEarthNet-S1 Dataset Directory: {DATA_DIR}")
    logger.info(f"Configuration Summary: {get_config_summary()}")
