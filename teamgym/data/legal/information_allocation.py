"""Compatibility imports for shared allocation in ``core.information_allocation``."""

from __future__ import annotations

from teamgym.core.information_allocation import (
    ALLOCATION_MODE_ALIASES as ALLOCATION_MODE_ALIASES,
    ALLOCATION_MODES as ALLOCATION_MODES,
    DEFAULT_ALLOCATION_MODE_WEIGHTS as DEFAULT_ALLOCATION_MODE_WEIGHTS,
    Any as Any,
    Iterable as Iterable,
    ReferenceUnitAllocation as ReferenceUnitAllocation,
    _build_assignment as _build_assignment,
    _check as _check,
    _dedupe as _dedupe,
    _drop_empty as _drop_empty,
    _eligible_modes as _eligible_modes,
    _pad_empty_participants as _pad_empty_participants,
    _round_robin_assignment as _round_robin_assignment,
    _weighted_mode_order as _weighted_mode_order,
    assign_reference_units as assign_reference_units,
    dataclass as dataclass,
    normalize_allocation_mode as normalize_allocation_mode,
    owners_by_reference_unit as owners_by_reference_unit,
    random as random,
    validate_reference_allocation as validate_reference_allocation,
)
