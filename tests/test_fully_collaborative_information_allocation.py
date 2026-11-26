from __future__ import annotations

from pathlib import Path
import random
import subprocess
import sys

import pytest

from teamgym.core import information_allocation as shared_allocation
from teamgym.core.information_allocation import (
    assign_reference_units,
    owners_by_reference_unit,
    validate_reference_allocation,
)
from teamgym.data.long_context_reasoning.seed_builder import build_one_datapoint


def test_paper_modes_exclude_paired_without_changing_legacy_allocations() -> None:
    units = ["a", "b", "c"]
    paper_modes = shared_allocation.PAPER_ALLOCATION_MODES
    assert paper_modes == ("one_each", "overlap", "redundant_player", "common_anchor", "bottleneck")
    assert assign_reference_units(units, mode="paired").mode == "paired"
    with pytest.raises(ValueError, match="No eligible allocation mode"):
        assign_reference_units(units, mode="paired", allowed_modes=paper_modes)
    restricted = assign_reference_units(
        units, allowed_modes=paper_modes, rng=random.Random(1), mode_weights={"paired": 1.0},
    )
    assert restricted.mode in paper_modes


def test_shared_allocation_imports_without_domain_or_training_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import teamgym.core.information_allocation; "
            "blocked = ('teamgym.data', "
            "'teamgym.interaction', 'rllm', 'verl', "
            "'torch', 'pandas', 'openai', 'tiktoken'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_one_each_allocates_all_reference_units_without_shortcut() -> None:
    allocation = assign_reference_units(
        ["s1", "s2", "s3", "s4"],
        mode="one_each",
        rng=random.Random(1),
        smallest_sufficient_sets=[["s1", "s2", "s3", "s4"]],
    )

    assert allocation.mode == "one_each"
    assert allocation.validation["valid"] is True
    assert sorted(owners_by_reference_unit(allocation.private_assignment)) == ["s1", "s2", "s3", "s4"]
    assert all(len(values) == 1 for values in allocation.private_assignment.values())


def test_common_anchor_records_common_reference_unit_and_validates_effective_information() -> None:
    allocation = assign_reference_units(
        ["s1", "s2", "s3"],
        mode="common_anchor",
        rng=random.Random(2),
        smallest_sufficient_sets=[["s1", "s2", "s3"]],
    )

    assert allocation.mode == "common_anchor"
    assert len(allocation.common_rui_ids) == 1
    assert allocation.validation["valid"] is True
    common = set(allocation.common_rui_ids)
    for participant_rui_ids in allocation.private_assignment.values():
        assert not {"s1", "s2", "s3"} <= (set(participant_rui_ids) | common)


def test_auto_skips_invalid_bottleneck_when_small_subset_is_sufficient() -> None:
    allocation = assign_reference_units(
        ["s1", "s2", "s3"],
        mode="auto",
        rng=random.Random(8),
        smallest_sufficient_sets=[["s1", "s2"]],
        mode_weights={"bottleneck": 1.0, "one_each": 0.1},
    )

    assert allocation.validation["valid"] is True


def test_min_participants_pads_with_empty_context_reviewers() -> None:
    allocation = assign_reference_units(
        ["s1", "s2"],
        mode="one_each",
        rng=random.Random(1),
        min_participants=4,
        smallest_sufficient_sets=[["s1", "s2"]],
    )

    assert allocation.validation["valid"] is True
    assert len(allocation.private_assignment) == 4
    assert sorted(allocation.private_assignment) == ["p1", "p2", "p3", "p4"]
    assert sum(1 for values in allocation.private_assignment.values() if not values) == 2


def test_validation_detects_single_participant_shortcut_with_common_information() -> None:
    validation = validate_reference_allocation(
        all_rui_ids=["s1", "s2", "s3"],
        private_assignment={"p1": ["s2", "s3"], "p2": []},
        common_rui_ids=["s1"],
        smallest_sufficient_sets=[["s1", "s2", "s3"]],
    )

    assert validation["valid"] is False
    assert any(
        row["name"] == "no_single_participant_has_sufficient_set" and not row["passed"]
        for row in validation["checks"]
    )


def test_long_context_common_anchor_is_visible_to_every_participant() -> None:
    datapoint = build_one_datapoint(
        spec={
            "instance_id": "common_anchor_test",
            "source_id": "source",
            "source_text": "FIRST ANCHOR. SECOND PRIVATE. THIRD PRIVATE.",
            "question": "What follows from all three passages?",
            "answer": "yes",
            "required_rui_ids": ["s1", "s2", "s3"],
            "rius": [
                {"rui_id": "s1", "source_span": "FIRST ANCHOR."},
                {"rui_id": "s2", "source_span": "SECOND PRIVATE."},
                {"rui_id": "s3", "source_span": "THIRD PRIVATE."},
            ],
        },
        index=0,
        allocation_mode="common_anchor",
        target_context_tokens=500,
        min_participants=3,
        rng=random.Random(3),
    )

    common_id = datapoint["generation_metadata"]["common_rui_ids"][0]
    assert all(common_id in participant["owned_rui_ids"] for participant in datapoint["participants"])
    common_rui = next(row for row in datapoint["reference_information_units"] if row["rui_id"] == common_id)
    assert sorted(common_rui["owned_by"]) == sorted(row["participant_id"] for row in datapoint["participants"])


def test_long_context_distributes_masked_units_without_requiring_them() -> None:
    datapoint = build_one_datapoint(
        spec={
            "instance_id": "masked_distribution_test",
            "source_id": "source",
            "source_text": "FIRST. SECOND. MASKED.",
            "question": "What follows?",
            "answer": "yes",
            "required_rui_ids": ["s1", "s2"],
            "distribution_rui_ids": ["s1", "s2", "s3"],
            "rius": [
                {"rui_id": "s1", "source_span": "FIRST."},
                {"rui_id": "s2", "source_span": "SECOND."},
                {
                    "rui_id": "s3",
                    "source_span": "MASKED.",
                    "criticality_class": "derivation_active_masked",
                },
            ],
        },
        index=0,
        allocation_mode="one_each",
        target_context_tokens=500,
        min_participants=3,
        rng=random.Random(3),
    )

    assert datapoint["oracle"]["required_rui_ids"] == ["s1", "s2"]
    assert {row["rui_id"] for row in datapoint["reference_information_units"]} == {
        "s1",
        "s2",
        "s3",
    }
    masked = next(
        row for row in datapoint["reference_information_units"] if row["rui_id"] == "s3"
    )
    assert masked["owned_by"]
    assert masked["metadata"]["criticality_class"] == "derivation_active_masked"
