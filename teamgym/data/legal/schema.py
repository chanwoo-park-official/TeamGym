"""Compatibility imports for the shared datapoint schema in ``core.schema``."""

from __future__ import annotations

from teamgym.core.schema import (
    Any as Any,
    COMMON_DATAPOINT_SCHEMA_VERSION as COMMON_DATAPOINT_SCHEMA_VERSION,
    LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION as LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION,
    SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS as SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS,
    _check as _check,
    _participant_directory as _participant_directory,
    build_common_datapoint as build_common_datapoint,
    validate_common_datapoint as validate_common_datapoint,
)
