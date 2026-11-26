"""Shared private-information allocation modes for collaborative tasks."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable

ALLOCATION_MODE_ALIASES = {
    "auto": "auto",
    "one_each": "one_each",
    "one-each": "one_each",
    "one_span_each": "one_each",
    "one-span-each": "one_each",
    "paired": "paired",
    "overlap": "overlap",
    "redundant_player": "redundant_player",
    "redundant-player": "redundant_player",
    "common_anchor": "common_anchor",
    "common-anchor": "common_anchor",
    "bottleneck": "bottleneck",
}

ALLOCATION_MODES = (
    "one_each",
    "paired",
    "overlap",
    "redundant_player",
    "common_anchor",
    "bottleneck",
)

PAPER_ALLOCATION_MODES = (
    "one_each", "overlap", "redundant_player", "common_anchor", "bottleneck",
)

DEFAULT_ALLOCATION_MODE_WEIGHTS = {
    "one_each": 0.30,
    "paired": 0.20,
    "overlap": 0.20,
    "redundant_player": 0.10,
    "common_anchor": 0.10,
    "bottleneck": 0.10,
}


@dataclass(frozen=True)
class ReferenceUnitAllocation:
    """Private/common assignment of hidden reference units."""

    mode: str
    requested_mode: str
    ordered_rui_ids: tuple[str, ...]
    private_assignment: dict[str, list[str]]
    common_rui_ids: tuple[str, ...]
    validation: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "mode": self.mode,
            "ordered_rui_ids": list(self.ordered_rui_ids),
            "private_assignment": self.private_assignment,
            "common_rui_ids": list(self.common_rui_ids),
            "validation": self.validation,
        }


def normalize_allocation_mode(mode: str) -> str:
    normalized = ALLOCATION_MODE_ALIASES.get(str(mode or "").strip().lower().replace(" ", "_"))
    if normalized is None:
        raise ValueError(f"Unknown allocation mode {mode!r}; expected one of {sorted(ALLOCATION_MODE_ALIASES)}.")
    return normalized


def assign_reference_units(
    rui_ids: Iterable[str],
    *,
    mode: str = "auto",
    rng: random.Random | None = None,
    min_participants: int = 2,
    smallest_sufficient_sets: list[list[str]] | None = None,
    mode_weights: dict[str, float] | None = None,
    require_no_single_complete: bool = True,
    allow_common_anchor: bool = True,
    allowed_modes: tuple[str, ...] = ALLOCATION_MODES,
) -> ReferenceUnitAllocation:
    """Assign reference units to participants using a benchmark-standard mode.

    The function is source-agnostic: a reference unit can be a CUAD span, a MAUD
    row excerpt, a statutory rule, or any other evaluator-side unit.
    """

    if min_participants < 2:
        raise ValueError("min_participants must be >= 2.")
    random_source = rng or random.Random()
    requested_mode = normalize_allocation_mode(mode)
    clean_ids = _dedupe(str(item) for item in rui_ids if str(item))
    if not clean_ids:
        raise ValueError("At least one reference unit is required.")
    eligible = tuple(candidate for candidate in _eligible_modes(
        len(clean_ids), allow_common_anchor=allow_common_anchor,
    ) if candidate in allowed_modes)
    if requested_mode == "auto":
        mode_order = _weighted_mode_order(eligible, random_source, mode_weights or DEFAULT_ALLOCATION_MODE_WEIGHTS)
    else:
        mode_order = [requested_mode]
    last_result: ReferenceUnitAllocation | None = None
    for candidate_mode in mode_order:
        if candidate_mode not in eligible:
            continue
        ordered = list(clean_ids)
        random_source.shuffle(ordered)
        private_assignment, common_rui_ids = _build_assignment(
            ordered,
            mode=candidate_mode,
        )
        private_assignment = _pad_empty_participants(
            private_assignment,
            min_participants=min_participants,
        )
        validation = validate_reference_allocation(
            all_rui_ids=clean_ids,
            private_assignment=private_assignment,
            common_rui_ids=common_rui_ids,
            smallest_sufficient_sets=smallest_sufficient_sets,
            require_no_single_complete=require_no_single_complete,
        )
        last_result = ReferenceUnitAllocation(
            mode=candidate_mode,
            requested_mode=requested_mode,
            ordered_rui_ids=tuple(ordered),
            private_assignment=private_assignment,
            common_rui_ids=tuple(common_rui_ids),
            validation=validation,
        )
        if requested_mode != "auto" or validation.get("valid"):
            return last_result
    if last_result is not None:
        return last_result
    raise ValueError(f"No eligible allocation mode for {len(clean_ids)} reference units.")


def validate_reference_allocation(
    *,
    all_rui_ids: Iterable[str],
    private_assignment: dict[str, list[str]],
    common_rui_ids: Iterable[str] | None = None,
    smallest_sufficient_sets: list[list[str]] | None = None,
    require_no_single_complete: bool = True,
) -> dict[str, Any]:
    all_rui = set(str(item) for item in all_rui_ids if str(item))
    common = set(str(item) for item in (common_rui_ids or []) if str(item))
    participant_sets = {
        str(participant_id): set(str(item) for item in rui_ids if str(item))
        for participant_id, rui_ids in private_assignment.items()
    }
    private_refs = set().union(*participant_sets.values()) if participant_sets else set()
    sufficient_sets = [
        set(str(item) for item in row if str(item))
        for row in (smallest_sufficient_sets or [])
        if row
    ]
    if not sufficient_sets and all_rui:
        sufficient_sets = [set(all_rui)]

    checks: list[dict[str, Any]] = []
    _check(checks, "participants_present", len(participant_sets) >= 2)
    _check(checks, "private_refs_exist", private_refs <= all_rui)
    _check(checks, "common_refs_exist", common <= all_rui)
    _check(checks, "all_non_common_refs_privately_covered", (all_rui - common) <= private_refs)
    _check(checks, "sufficient_refs_exist", all(row <= all_rui for row in sufficient_sets))
    if require_no_single_complete:
        _check(
            checks,
            "no_single_participant_has_sufficient_set",
            not any(
                sufficient and sufficient <= (participant_rui | common)
                for participant_rui in participant_sets.values()
                for sufficient in sufficient_sets
            ),
        )
    return {
        "valid": all(row["passed"] for row in checks),
        "checks": checks,
        "metrics": {
            "participant_count": len(participant_sets),
            "private_reference_unit_count": len(private_refs),
            "common_reference_unit_count": len(common),
            "participant_rui_sets": {pid: sorted(values) for pid, values in participant_sets.items()},
            "smallest_sufficient_set_count": len(sufficient_sets),
        },
    }


def owners_by_reference_unit(private_assignment: dict[str, list[str]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for participant_id, rui_ids in private_assignment.items():
        for rui_id in rui_ids:
            owners.setdefault(str(rui_id), []).append(str(participant_id))
    return {rui_id: sorted(_dedupe(participant_ids)) for rui_id, participant_ids in owners.items()}


def _eligible_modes(
    rui_count: int,
    *,
    allow_common_anchor: bool,
) -> tuple[str, ...]:
    modes: list[str] = ["one_each"]
    if rui_count >= 3:
        modes.extend(["paired", "overlap"])
    if rui_count >= 2:
        modes.append("redundant_player")
    if rui_count >= 2 and allow_common_anchor:
        modes.append("common_anchor")
    if rui_count >= 2:
        modes.append("bottleneck")
    return tuple(modes)


def _weighted_mode_order(
    eligible: tuple[str, ...],
    rng: random.Random,
    weights: dict[str, float],
) -> list[str]:
    remaining = list(eligible)
    ordered: list[str] = []
    while remaining:
        total = sum(max(0.0, float(weights.get(mode, 0.0))) for mode in remaining)
        if total <= 0:
            rng.shuffle(remaining)
            ordered.extend(remaining)
            break
        draw = rng.random() * total
        cumulative = 0.0
        chosen = remaining[-1]
        for mode in remaining:
            cumulative += max(0.0, float(weights.get(mode, 0.0)))
            if draw <= cumulative:
                chosen = mode
                break
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


def _build_assignment(
    ordered_rui_ids: list[str],
    *,
    mode: str,
) -> tuple[dict[str, list[str]], list[str]]:
    if mode == "one_each":
        return _round_robin_assignment(ordered_rui_ids, participant_count=max(1, len(ordered_rui_ids))), []
    if mode == "paired":
        participant_count = max(2, len(ordered_rui_ids) - 1)
        assignments = {f"p{idx}": [] for idx in range(1, participant_count + 1)}
        assignments["p1"].extend(ordered_rui_ids[:2])
        for offset, rui_id in enumerate(ordered_rui_ids[2:], start=2):
            participant_id = f"p{((offset - 1) % participant_count) + 1}"
            assignments[participant_id].append(rui_id)
        return _drop_empty(assignments), []
    if mode == "overlap":
        participant_count = len(ordered_rui_ids)
        assignments = {f"p{idx}": [] for idx in range(1, participant_count + 1)}
        for idx, rui_id in enumerate(ordered_rui_ids):
            participant_id = f"p{(idx % participant_count) + 1}"
            assignments[participant_id].append(rui_id)
            if idx + 1 < len(ordered_rui_ids):
                assignments[participant_id].append(ordered_rui_ids[idx + 1])
        return {pid: _dedupe(ids) for pid, ids in _drop_empty(assignments).items()}, []
    if mode == "redundant_player":
        assignments = _round_robin_assignment(ordered_rui_ids, participant_count=max(1, len(ordered_rui_ids)))
        assignments[f"p{len(assignments) + 1}"] = []
        return assignments, []
    if mode == "common_anchor":
        common = [ordered_rui_ids[0]]
        private = ordered_rui_ids[1:]
        return _round_robin_assignment(private, participant_count=max(1, len(private))), common
    if mode == "bottleneck":
        return {"p1": ordered_rui_ids[:-1], "p2": [ordered_rui_ids[-1]]}, []
    raise ValueError(f"Unhandled allocation mode: {mode}")


def _round_robin_assignment(rui_ids: list[str], *, participant_count: int) -> dict[str, list[str]]:
    participant_count = max(1, participant_count)
    assignments = {f"p{idx}": [] for idx in range(1, participant_count + 1)}
    for idx, rui_id in enumerate(rui_ids):
        assignments[f"p{(idx % participant_count) + 1}"].append(rui_id)
    return _drop_empty(assignments)


def _pad_empty_participants(
    assignments: dict[str, list[str]],
    *,
    min_participants: int,
) -> dict[str, list[str]]:
    padded = {str(pid): list(ids) for pid, ids in assignments.items()}
    target = max(min_participants, len(padded))
    next_idx = 1
    while len(padded) < target:
        participant_id = f"p{next_idx}"
        next_idx += 1
        if participant_id in padded:
            continue
        padded[participant_id] = []
    return padded


def _drop_empty(assignments: dict[str, list[str]]) -> dict[str, list[str]]:
    return {pid: ids for pid, ids in assignments.items() if ids or pid == f"p{len(assignments)}"}


def _check(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
