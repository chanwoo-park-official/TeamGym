"""Compatibility imports for the shared dataset loader in ``core.dataset``."""

from __future__ import annotations

from teamgym.core.dataset import (
    Any as Any,
    COMMON_DATAPOINT_SCHEMA_VERSION as COMMON_DATAPOINT_SCHEMA_VERSION,
    DEFAULT_LEGAL_DATASET_PATH as DEFAULT_LEGAL_DATASET_PATH,
    Iterable as Iterable,
    Iterator as Iterator,
    Path as Path,
    discover_common_datapoint_files as discover_common_datapoint_files,
    extract_common_datapoint as extract_common_datapoint,
    iter_common_datapoints as iter_common_datapoints,
    json as json,
    load_common_datapoints as load_common_datapoints,
)
