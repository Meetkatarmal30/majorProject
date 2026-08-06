"""
Configuration package initialization.
"""

from config.settings import (
    BASE_DIR,
    DATA_DIR,
    SPLITS_DIR,
    IMAGE_SIZE,
    PRESERVE_RESOLUTION,
    NORMALIZATION_MODE,
    VV_BAND_SUFFIX,
    VH_BAND_SUFFIX,
    METADATA_SUFFIX,
    VV_CLIP_RANGE,
    VH_CLIP_RANGE,
    VV_ZSCORE_STATS,
    VH_ZSCORE_STATS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RANDOM_STATE,
    setup_logging,
    get_config_summary,
)
