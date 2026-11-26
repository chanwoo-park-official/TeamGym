from __future__ import annotations

import json
from pathlib import Path

from teamgym.data.medical.evidencebench.evaluation import (
    build_evaluation_datapoint,
)
from teamgym.data.medical.export_schema import validate_common_datapoint
from teamgym.core.schema import LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION
from teamgym.scoring.written_answers import is_written_answer_task


def test_build_evaluation_datapoint_preserves_disjoint_packets(tmp_path: Path) -> None:
    root = tmp_path / "evidencebench_test_id_1"
    root.mkdir()
    _write(root / "summary.json", {"status": "accepted", "instance_id": "evidencebench_test_id_1"})
    _write(
        root / "02_task_rubric.json",
        {
            "public_task": {"title": "T", "question": "Q", "response_format_note": "A"},
            "canonical_answer": "Both.",
            "rubric": {
                "required_claims": [
                    {"claim_id": "c1", "criterion": "One", "supporting_candidate_ids": ["cand_1"]},
                    {"claim_id": "c2", "criterion": "Two", "supporting_candidate_ids": ["cand_2"]},
                ],
                "disqualifying_errors": [],
            },
        },
    )
    _write(
        root / "05_final_rius.hidden.json",
        [
            {"riu_id": "r1", "candidate_id": "cand_1", "hidden_claim": "One", "required_claim_ids": ["c1"], "evidence_atoms": [{"text": "one", "sentence_indices": [1]}]},
            {"riu_id": "r2", "candidate_id": "cand_2", "hidden_claim": "Two", "required_claim_ids": ["c2"], "evidence_atoms": [{"text": "two-a", "sentence_indices": [2]}, {"text": "two-b", "sentence_indices": [3]}]},
        ],
    )
    _write(
        root / "06_participant_packets.hidden.json",
        {
            "allocation_policy": "one_each_disjoint_full_paper",
            "participants": [
                {"participant_id": "p1", "hidden_owned_riu_ids": ["r1"], "context_ids": ["x1", "pad"]},
                {"participant_id": "p2", "hidden_owned_riu_ids": ["r2"], "context_ids": ["x2", "x3"]},
            ],
            "contexts": [
                {"context_id": "x1", "text": "one", "sentence_index": 1, "hidden_contains_riu_ids": ["r1"]},
                {"context_id": "x2", "text": "two-a", "sentence_index": 2, "hidden_contains_riu_ids": ["r2"]},
                {"context_id": "pad", "text": "padding", "sentence_index": 4, "hidden_contains_riu_ids": []},
                {"context_id": "x3", "text": "two-b", "sentence_index": 3, "hidden_contains_riu_ids": ["r2"]},
            ],
        },
    )

    from tests.medical.certified_fixtures import complete_fixture
    complete_fixture(root, "evidencebench")
    row = build_evaluation_datapoint(root)
    assert row["format_validation"]["valid"]
    assert is_written_answer_task(row)
    row["schema_version"] = LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION
    assert validate_common_datapoint(row)["valid"]
    assert [p["private_context_unit_ids"] for p in row["participants"]] == [
        ["x1", "pad"],
        ["x2", "x3"],
    ]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
