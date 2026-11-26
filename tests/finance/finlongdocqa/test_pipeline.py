from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from teamgym.data.finance.finlongdocqa.partition import (
    build_partitioned_datapoint,
    build_one_each_datapoint,
    repartition_common_datapoint,
)
from teamgym.data.finance.finlongdocqa.pipeline import (
    minimize_redundant_riu,
    run_blind_solver_audit,
    run_packet_leakage_repair,
)
from teamgym.data.finance.finlongdocqa.source import (
    index_report,
    validate_report_index,
)
from teamgym.data.finance.finlongdocqa.validation import (
    hydrate_task,
    summarize_free_response_audit,
    validate_evidence_consolidation,
    validate_generated_task,
)
from teamgym.core.information_allocation import (
    PAPER_ALLOCATION_MODES,
)
from teamgym.core.schema import LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION
from teamgym.data.finance.finlongdocqa.schema import validate_common_datapoint
from teamgym.scoring.written_answers import is_written_answer_task


def _report_index() -> dict:
    report = """# Page 1

## Revenue

Revenue was $120 million in 2024.

# Page 2

## Margin

Operating income was $30 million in 2024.

# Page 3

## Background

The company serves enterprise customers and repeated revenue of $120 million.
"""
    indexed = index_report("TEST_2024", report)
    assert validate_report_index(indexed, report) == []
    return indexed


def _mappings() -> list[dict]:
    blocks = _report_index()["blocks"]
    return [
        {
            "qa_id": "qa_1",
            "required_block_ids": [
                blocks[0]["block_id"],
                blocks[1]["block_id"],
            ],
        },
        {
            "qa_id": "qa_2",
            "required_block_ids": [blocks[2]["block_id"]],
        },
    ]


def _task() -> dict:
    blocks = _report_index()["blocks"]
    proposed = {
        "schema": "finlongdocqa_free_response_task_v1",
        "source_qa_ids_used": ["qa_1"],
        "public_task": {
            "title": "2024 operating review",
            "report_context": "Prepare a concise analyst note.",
            "question": (
                "State 2024 revenue and calculate operating margin as operating "
                "income divided by revenue."
            ),
            "response_requirements": ["Show values, formula, unit, and result."],
        },
        "evidence_units": [
            {
                "source_block_id": blocks[0]["block_id"],
                "information_feature": "2024 revenue",
                "hidden_summary": "$120 million",
                "supports_claim_ids": ["claim_01", "claim_02"],
            },
            {
                "source_block_id": blocks[1]["block_id"],
                "information_feature": "2024 operating income",
                "hidden_summary": "$30 million",
                "supports_claim_ids": ["claim_02"],
            },
        ],
        "rubric": {
            "rubric_version": "finlongdocqa_claim_rubric_v1",
            "required_claims": [
                {
                    "claim_id": "claim_01",
                    "description": "States 2024 revenue.",
                    "expected_answer_element": "$120 million",
                    "supporting_unit_ids": ["riu_01"],
                    "grading_rule": "Require value, unit, and year.",
                },
                {
                    "claim_id": "claim_02",
                    "description": "Calculates operating margin.",
                    "expected_answer_element": "30 / 120 = 25%",
                    "supporting_unit_ids": ["riu_01", "riu_02"],
                    "grading_rule": "Accept 25% with the correct operands.",
                },
            ],
            "prohibited_errors": ["Do not reverse the ratio."],
        },
        "oracle": {
            "reference_answer": (
                "Revenue was $120 million. Operating margin was "
                "$30 million / $120 million = 25%."
            )
        },
    }
    return hydrate_task(proposed, blocks)


def _judge(
    *,
    claim_01: bool,
    claim_02: bool,
    complete: bool,
    evidence_count: int,
) -> dict:
    return {
        "claim_results": [
            {
                "claim_id": "claim_01",
                "status": "satisfied" if claim_01 else "missing",
                "content_correct": claim_01,
                "grounded_in_available_material": claim_01,
                "supporting_evidence_numbers": [1] if claim_01 else [],
                "reason": "synthetic",
            },
            {
                "claim_id": "claim_02",
                "status": "satisfied" if claim_02 else "missing",
                "content_correct": claim_02,
                "grounded_in_available_material": claim_02,
                "supporting_evidence_numbers": (
                    list(range(1, evidence_count + 1)) if claim_02 else []
                ),
                "reason": "synthetic",
            },
        ],
        "has_material_error": False,
        "overall_complete_and_correct": complete,
        "overall_reason": "synthetic",
    }


def _audit_inputs() -> tuple[dict, dict, list[dict]]:
    full = {
        "solver_response": {"answer": "complete"},
        "judge_result": _judge(
            claim_01=True,
            claim_02=True,
            complete=True,
            evidence_count=2,
        ),
    }
    none = {
        "solver_response": {"answer": "underdetermined"},
        "judge_result": _judge(
            claim_01=False,
            claim_02=False,
            complete=False,
            evidence_count=0,
        ),
    }
    leave_one_out = [
        {
            "omitted_unit_id": "riu_01",
            "solver_response": {"answer": "underdetermined"},
            "judge_result": _judge(
                claim_01=False,
                claim_02=False,
                complete=False,
                evidence_count=1,
            ),
        },
        {
            "omitted_unit_id": "riu_02",
            "solver_response": {"answer": "revenue only"},
            "judge_result": _judge(
                claim_01=True,
                claim_02=False,
                complete=False,
                evidence_count=1,
            ),
        },
    ]
    return full, none, leave_one_out


def _common() -> dict:
    full, none, leave_one_out = _audit_inputs()
    audit, errors = summarize_free_response_audit(
        task=_task(),
        full_result=full,
        no_evidence_result=none,
        leave_one_out_results=leave_one_out,
    )
    assert errors == []
    return build_one_each_datapoint(
        report_id="TEST_2024",
        company="TEST",
        year="2024",
        report_source={"kind": "test"},
        report_index=_report_index(),
        task=_task(),
        qa_mappings=_mappings(),
        solver_audit=audit,
    )


def _three_riu_task() -> dict:
    task = deepcopy(_task())
    block = _report_index()["blocks"][2]
    task["evidence_units"].append(
        {
            "unit_id": "riu_03",
            "source_block_id": block["block_id"],
            "information_feature": "enterprise customer background",
            "hidden_summary": "Serves enterprise customers.",
            "supports_claim_ids": ["claim_03"],
            "text": block["text"],
            "source_ref": {
                "source_block_id": block["block_id"],
                "page_number": block["page_number"],
            },
        }
    )
    return task


def test_task_allows_one_qa_two_rius_and_many_to_many_claims() -> None:
    task = _task()
    assert (
        validate_generated_task(
            task,
            candidate_blocks=_report_index()["blocks"],
            qa_mappings=_mappings(),
        )
        == []
    )
    assert task["source_qa_ids_used"] == ["qa_1"]
    assert task["evidence_units"][0]["supports_claim_ids"] == [
        "claim_01",
        "claim_02",
    ]

    invalid = deepcopy(task)
    invalid["source_qa_ids_used"] = ["qa_2"]
    errors = validate_generated_task(
        invalid,
        candidate_blocks=_report_index()["blocks"],
        qa_mappings=_mappings(),
    )
    assert any("used source QA" in error for error in errors)


def test_free_response_full_none_and_leave_one_out_audit() -> None:
    full, none, leave_one_out = _audit_inputs()
    audit, errors = summarize_free_response_audit(
        task=_task(),
        full_result=full,
        no_evidence_result=none,
        leave_one_out_results=leave_one_out,
    )
    assert errors == []
    assert audit["accept"] is True
    assert audit["leave_one_out"][0]["broken_claim_ids"] == [
        "claim_01",
        "claim_02",
    ]


def test_blind_solver_and_oracle_aware_judge_are_separate(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_json(self, **kwargs: object) -> dict:
            self.calls.append(kwargs)
            prompt_id = str(kwargs["prompt_id"])
            prompt = str(kwargs["user_prompt"])
            trace = str(kwargs["trace_dir"])
            if prompt_id.endswith("03_blind_solver_user"):
                assert "HIDDEN RUBRIC" not in prompt
                assert "REFERENCE ANSWER" not in prompt
                return {
                    "answer": "synthetic",
                    "evidence_used": [],
                    "unresolved_requirements": [],
                }
            assert prompt_id.endswith("04_free_response_judge_user")
            if "/full/" in trace:
                return _judge(
                    claim_01=True,
                    claim_02=True,
                    complete=True,
                    evidence_count=2,
                )
            if "/without_riu_02/" in trace:
                return _judge(
                    claim_01=True,
                    claim_02=False,
                    complete=False,
                    evidence_count=1,
                )
            return _judge(
                claim_01=False,
                claim_02=False,
                complete=False,
                evidence_count=0 if "/none/" in trace else 1,
            )

    client = FakeClient()
    audit, errors = run_blind_solver_audit(
        client=client,
        task=_task(),
        trace_root=tmp_path,
    )
    assert errors == []
    assert audit["accept"] is True
    assert len(client.calls) == 8


def test_partition_is_free_response_one_each() -> None:
    common = _common()
    assert common["format_validation"]["valid"] is True
    assert is_written_answer_task(common)
    common["schema_version"] = LEGACY_COMMON_DATAPOINT_SCHEMA_VERSION
    assert validate_common_datapoint(common)["valid"]
    assert common["decision_set"] == ["free_response"]
    assert common["task_description"]["answer_format"]["type"] == "free_response"
    assert len(common["participants"]) == 2
    assert all(
        len(participant["owned_rui_ids"]) == 1
        for participant in common["participants"]
    )


def test_all_five_allocation_modes_build_and_packet_audit_smoke(
    tmp_path: Path,
) -> None:
    all_rui_ids = {"riu_01", "riu_02", "riu_03"}
    for mode in PAPER_ALLOCATION_MODES:
        common = build_partitioned_datapoint(
            report_id="TEST_2024",
            company="TEST",
            year="2024",
            report_source={"kind": "test"},
            report_index=_report_index(),
            task=_three_riu_task(),
            qa_mappings=_mappings(),
            solver_audit={},
            allocation_mode=mode,
            allocation_seed=7,
        )
        assert common["format_validation"]["valid"] is True
        assert common["generation_metadata"]["partition_strategy"] == mode
        assert common["generation_metadata"][
            "private_information_allocation"
        ]["validation"]["valid"] is True

        participants = {
            row["participant_id"]: row for row in common["participants"]
        }
        shared = {
            row["rui_id"]
            for row in common["initial_common_information"].get(
                "common_reference_units", []
            )
        }
        for context in common["private_context_units"]:
            hidden = set(context.get("hidden_contains_rui_ids") or [])
            for participant_id in context["assigned_to"]:
                authorized = (
                    set(participants[participant_id]["owned_rui_ids"])
                    | shared
                )
                assert hidden <= authorized

        class NoLeakClient:
            def generate_json(self, **kwargs: object) -> dict:
                participant_id = str(kwargs["trace_dir"]).rsplit("/", 1)[-1]
                authorized = (
                    set(participants[participant_id]["owned_rui_ids"])
                    | shared
                )
                return {
                    "participant_id": participant_id,
                    "foreign_rui_results": [
                        {
                            "rui_id": rui_id,
                            "reconstructable_from_packet": False,
                            "minimal_support_context_ids": [],
                            "removable_noncanonical_context_ids": [],
                            "canonical_overlap": False,
                            "reason": "synthetic smoke test",
                        }
                        for rui_id in sorted(all_rui_ids - authorized)
                    ],
                }

        repaired, audits, removed, errors = run_packet_leakage_repair(
            client=NoLeakClient(),
            datapoint=common,
            public_task=_three_riu_task()["public_task"],
            trace_root=tmp_path / mode,
            max_workers=4,
            max_rounds=1,
        )
        assert errors == []
        assert removed == set()
        assert len(audits) == len(common["participants"])
        assert repaired["format_validation"]["valid"] is True


def test_frozen_task_repartition_preserves_question_rius_and_oracle() -> None:
    base = build_partitioned_datapoint(
        report_id="TEST_2024",
        company="TEST",
        year="2024",
        report_source={"kind": "test"},
        report_index=_report_index(),
        task=_three_riu_task(),
        qa_mappings=_mappings(),
        solver_audit={},
        allocation_mode="one_each",
        preserve_rui_order=True,
    )
    frozen = {
        "task_description": deepcopy(base["task_description"]),
        "reference_information_units": deepcopy(
            base["reference_information_units"]
        ),
        "oracle": deepcopy(base["oracle"]),
        "dataset_specific": deepcopy(base["dataset_specific"]),
    }
    for mode in PAPER_ALLOCATION_MODES:
        repartitioned = repartition_common_datapoint(
            base,
            allocation_mode=mode,
            allocation_seed=11,
        )
        assert repartitioned["format_validation"]["valid"] is True
        assert repartitioned["generation_metadata"][
            "partition_strategy"
        ] == mode
        for key, value in frozen.items():
            assert repartitioned[key] == value


def test_noncanonical_blocks_are_balanced_before_page_locality() -> None:
    report_index = deepcopy(_report_index())
    for index in range(4, 104):
        report_index["blocks"].append(
            {
                "block_id": f"p{index:03d}_paragraph_01",
                "report_id": "TEST_2024",
                "page_number": index,
                "block_type": "paragraph",
                "heading": "Later disclosure",
                "text": " ".join(["balanced"] * 20),
                "raw_start": 0,
                "raw_end": 1,
            }
        )
    full, none, leave_one_out = _audit_inputs()
    audit, errors = summarize_free_response_audit(
        task=_task(),
        full_result=full,
        no_evidence_result=none,
        leave_one_out_results=leave_one_out,
    )
    assert errors == []
    common = build_one_each_datapoint(
        report_id="TEST_2024",
        company="TEST",
        year="2024",
        report_source={"kind": "test"},
        report_index=report_index,
        task=_task(),
        qa_mappings=_mappings(),
        solver_audit=audit,
    )
    loads = [
        participant["private_word_count"]
        for participant in common["participants"]
    ]
    assert max(loads) - min(loads) <= 20


def test_packet_audit_removes_noncanonical_combination_leak(
    tmp_path: Path,
) -> None:
    common = _common()
    leaking_context = next(
        row
        for row in common["private_context_units"]
        if not row["hidden_contains_rui_ids"]
    )
    leaking_context_id = leaking_context["context_unit_id"]
    leaking_participant_id = leaking_context["assigned_to"][0]

    class FakeClient:
        def generate_json(self, **kwargs: object) -> dict:
            trace = str(kwargs["trace_dir"])
            participant_id = trace.rsplit("/", 1)[-1]
            foreign_rui = (
                "riu_02" if participant_id == "participant_a" else "riu_01"
            )
            leak = (
                "round_01" in trace
                and participant_id == leaking_participant_id
            )
            return {
                "participant_id": participant_id,
                "foreign_rui_results": [
                    {
                        "rui_id": foreign_rui,
                        "reconstructable_from_packet": leak,
                        "minimal_support_context_ids": (
                            [leaking_context_id] if leak else []
                        ),
                        "removable_noncanonical_context_ids": (
                            [leaking_context_id] if leak else []
                        ),
                        "canonical_overlap": False,
                        "reason": "synthetic",
                    }
                ],
            }

    repaired, audits, removed, errors = run_packet_leakage_repair(
        client=FakeClient(),
        datapoint=common,
        public_task=_task()["public_task"],
        trace_root=tmp_path,
        max_workers=2,
        max_rounds=2,
    )
    assert errors == []
    assert removed == {leaking_context_id}
    assert len(audits) == 4
    remaining = {
        row["context_unit_id"] for row in repaired["private_context_units"]
    }
    assert leaking_context_id not in remaining
    assert repaired["format_validation"]["valid"] is True


def test_large_packet_rejects_incomplete_shortlist_certification(
    tmp_path: Path,
) -> None:
    common = _common()
    noncanonical = next(
        row
        for row in common["private_context_units"]
        if not row["hidden_contains_rui_ids"]
    )
    noncanonical["text"] += " " + ("large-packet-context " * 1000)

    class FakeClient:
        def generate_json(self, **kwargs: object) -> dict:
            prompt_id = str(kwargs["prompt_id"])
            prompt = str(kwargs["user_prompt"])
            participant_id = next(
                value
                for value in str(kwargs["trace_dir"]).split("/")
                if value.startswith("participant_")
            )
            foreign_rui = (
                "riu_02" if participant_id == "participant_a" else "riu_01"
            )
            if prompt_id.endswith("05_packet_leakage_candidate_user"):
                return {
                    "participant_id": participant_id,
                    "foreign_rui_candidates": [
                        {
                            "rui_id": foreign_rui,
                            "candidate_context_unit_ids": [],
                            "reason": "none",
                        }
                    ],
                }
            assert prompt_id.endswith("05_packet_leakage_audit_user")
            assert "COMPLETE PRIVATE PACKET" in prompt
            return {
                "participant_id": participant_id,
                "foreign_rui_results": [
                    {
                        "rui_id": foreign_rui,
                        "reconstructable_from_packet": False,
                        "minimal_support_context_ids": [],
                        "removable_noncanonical_context_ids": [],
                        "canonical_overlap": False,
                        "reason": "none",
                    }
                ],
            }

    _, audits, _, errors = run_packet_leakage_repair(
        client=FakeClient(),
        datapoint=common,
        public_task=_task()["public_task"],
        trace_root=tmp_path,
        max_workers=2,
        max_rounds=1,
        max_prompt_chars=1000,
    )
    assert any("complete packet exceeds whole-packet audit budget" in error for error in errors)
    assert "complete_view_over_budget_rejected" in {row["audit_mode"] for row in audits}


def test_adjacent_label_and_value_are_normalized_as_exact_raw_span() -> None:
    report = """# Page 1

## Valuation

Enterprise Value

$40B
"""
    indexed = index_report("TEST_2024", report)
    blocks = indexed["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["block_type"] == "composite"
    assert blocks[0]["text"] == "Enterprise Value\n\n$40B"
    assert blocks[0]["normalization"]["kind"] == "adjacent_label_value_merge"
    assert validate_report_index(indexed, report) == []


def test_evidence_consolidation_excludes_only_fully_subsumed_blocks() -> None:
    blocks = _report_index()["blocks"]
    result = {
        "report_id": "TEST_2024",
        "overlap_groups": [
            {
                "information": "Revenue is repeated.",
                "relation": "information_contained",
                "provider_block_ids": [
                    blocks[0]["block_id"],
                    blocks[2]["block_id"],
                ],
                "preferred_block_id": blocks[2]["block_id"],
                "fully_subsumed_block_ids": [blocks[0]["block_id"]],
                "reason": "The background block repeats revenue.",
            }
        ],
        "retained_block_ids": [
            blocks[1]["block_id"],
            blocks[2]["block_id"],
        ],
        "excluded_block_ids": [blocks[0]["block_id"]],
    }
    assert (
        validate_evidence_consolidation(
            result,
            report_id="TEST_2024",
            candidate_blocks=blocks,
        )
        == []
    )


def test_loo_redundancy_can_remove_one_riu_and_relink_claims() -> None:
    task = _task()
    task["rubric"]["required_claims"][0]["supporting_unit_ids"] = [
        "riu_01",
        "riu_02",
    ]
    task["rubric"]["required_claims"][1]["supporting_unit_ids"] = [
        "riu_01",
        "riu_02",
    ]
    for unit in task["evidence_units"]:
        unit["supports_claim_ids"] = ["claim_01", "claim_02"]
    full, none, _ = _audit_inputs()
    redundant_judge = _judge(
        claim_01=True,
        claim_02=True,
        complete=True,
        evidence_count=1,
    )
    audit = {
        "full_evidence": full,
        "no_evidence": none,
        "leave_one_out": [
            {
                "omitted_unit_id": "riu_01",
                "linked_claim_ids": ["claim_01", "claim_02"],
                "broken_claim_ids": [],
                "judge_result": redundant_judge,
            },
            {
                "omitted_unit_id": "riu_02",
                "linked_claim_ids": ["claim_01", "claim_02"],
                "broken_claim_ids": ["claim_02"],
                "judge_result": _judge(
                    claim_01=True,
                    claim_02=False,
                    complete=False,
                    evidence_count=1,
                ),
            },
        ],
    }
    repaired, detail = minimize_redundant_riu(
        task=task,
        solver_audit=audit,
        solver_errors=[
            "riu_01: leave-one-out response remained complete",
            "riu_01: all linked claims remained correct and grounded",
        ],
        candidate_blocks=_report_index()["blocks"],
        qa_mappings=_mappings(),
        consolidation={"overlap_groups": []},
    )
    # A valid task must still have at least two RIUs, so this two-RIU task is not
    # reducible even though one unit is empirically redundant.
    assert repaired is None
    assert detail == {}


def test_loo_redundancy_repair_keeps_two_rius_and_relinks_citations() -> None:
    blocks = _report_index()["blocks"]
    proposed = _task()
    proposed["source_qa_ids_used"] = ["qa_1", "qa_2"]
    proposed["evidence_units"].append(
        {
            "unit_id": "riu_03",
            "source_block_id": blocks[2]["block_id"],
            "information_feature": "repeated revenue",
            "hidden_summary": "$120 million",
            "supports_claim_ids": ["claim_01"],
            "text": blocks[2]["text"],
            "source_ref": {
                "source_block_id": blocks[2]["block_id"],
                "page_number": blocks[2]["page_number"],
            },
        }
    )
    proposed["rubric"]["required_claims"][0]["supporting_unit_ids"] = [
        "riu_01",
        "riu_03",
    ]
    proposed["rubric"]["required_claims"][1]["supporting_unit_ids"] = [
        "riu_01",
        "riu_02",
    ]
    proposed["evidence_units"][0]["supports_claim_ids"] = [
        "claim_01",
        "claim_02",
    ]
    proposed["evidence_units"][1]["supports_claim_ids"] = ["claim_02"]
    loo_judge = _judge(
        claim_01=True,
        claim_02=True,
        complete=True,
        evidence_count=2,
    )
    loo_judge["claim_results"][0]["supporting_evidence_numbers"] = [2]
    loo_judge["claim_results"][1]["supporting_evidence_numbers"] = [1]
    full, none, _ = _audit_inputs()
    repaired, detail = minimize_redundant_riu(
        task=proposed,
        solver_audit={
            "full_evidence": full,
            "no_evidence": none,
            "leave_one_out": [
                {
                    "omitted_unit_id": "riu_01",
                    "linked_claim_ids": ["claim_01", "claim_02"],
                    "broken_claim_ids": [],
                    "judge_result": loo_judge,
                }
            ],
        },
        solver_errors=[
            "riu_01: leave-one-out response remained complete",
            "riu_01: all linked claims remained correct and grounded",
        ],
        candidate_blocks=blocks,
        qa_mappings=_mappings(),
        consolidation={"overlap_groups": []},
    )
    assert detail["removed_unit_id"] == "riu_01"
    assert [unit["unit_id"] for unit in repaired["evidence_units"]] == [
        "riu_02",
        "riu_03",
    ]
    claims = repaired["rubric"]["required_claims"]
    assert claims[0]["supporting_unit_ids"] == ["riu_03"]
    assert claims[1]["supporting_unit_ids"] == ["riu_02"]
    assert (
        validate_generated_task(
            repaired,
            candidate_blocks=blocks,
            qa_mappings=_mappings(),
        )
        == []
    )


def _source_args(tmp_path: Path):
    import argparse
    import json
    import zipfile
    from teamgym.data.finance.finlongdocqa.runtime import write_jsonl
    from teamgym.data.finance.finlongdocqa.source import ReportStore
    report = "# Page 1\n\n## Revenue\n\nRevenue was $120 million in 2024.\n\n# Page 2\n\n## Margin\n\nOperating income was $30 million in 2024.\n"
    archive = tmp_path / "reports.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("reports/TEST/2024.md", report)
    qa_path = tmp_path / "qa.jsonl"
    write_jsonl(qa_path, [{"id": "qa_1", "company": "TEST", "year": "2024", "question": "Review revenue and margin.", "answer": 25, "page_numbers": [1, 2]}])
    store = ReportStore(archive)
    _, source = store.load("TEST", "2024")
    store.close()
    indexed = index_report("TEST_2024", report)
    indexed["report_source"] = source
    index_path = tmp_path / "index.jsonl"
    write_jsonl(index_path, [indexed])
    return argparse.Namespace(output_root=str(tmp_path / "run"), qa_path=str(qa_path), reports_zip=str(archive),
        block_index_jsonl=str(index_path), report_ids="TEST_2024", sample_reports=0, max_reports=0, sample_seed=0,
        generation_attempts=1, audit_workers=1, packet_repair_rounds=1, packet_audit_max_chars=500000,
        allocation_mode="one_each", model="offline", provider="offline")


def test_generation_retries_failed_commits_and_retains_history(tmp_path, monkeypatch):
    from teamgym.data.finance.finlongdocqa import pipeline
    args = _source_args(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "build_client", lambda *a: object())
    def process(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"report_id": "TEST_2024", "status": "failed", "errors": ["temporary provider error"]}, None
        return {"report_id": "TEST_2024", "status": "accepted"}, _common()
    monkeypatch.setattr(pipeline, "process_report", process)
    assert pipeline.generate_dataset(args)["accepted_count"] == 0
    result = pipeline.generate_dataset(args)
    assert result["accepted_count"] == 1 and result["attempt_count"] == 2
    assert len(calls) == 2
    assert calls[0]["trace_root"] != calls[1]["trace_root"]


def test_export_rebuild_recovers_interruption_after_atomic_commit(tmp_path, monkeypatch):
    import pytest
    from teamgym.data.finance.finlongdocqa import pipeline
    args = _source_args(tmp_path)
    monkeypatch.setattr(pipeline, "build_client", lambda *a: object())
    monkeypatch.setattr(pipeline, "process_report", lambda **kw: ({"report_id": "TEST_2024", "status": "accepted"}, _common()))
    original = pipeline.rebuild_exports
    calls = []
    def interrupted(*args):
        calls.append(True)
        if len(calls) == 2:
            raise KeyboardInterrupt("after committed accepted result, before derived exports")
        return original(*args)
    monkeypatch.setattr(pipeline, "rebuild_exports", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pipeline.generate_dataset(args)
    monkeypatch.setattr(pipeline, "rebuild_exports", original)
    monkeypatch.setattr(pipeline, "build_client", lambda *a: (_ for _ in ()).throw(AssertionError("accepted commit must resume without provider")))
    # Malformed projections are derived data and are repaired from the commit.
    (Path(args.output_root) / "common_datapoints.jsonl").write_text('{"interrupted":')
    result = pipeline.generate_dataset(args)
    assert result["accepted_count"] == 1
    from teamgym.data.finance.finlongdocqa.runtime import read_jsonl
    assert len(read_jsonl(result["common_path"])) == 1


def test_changed_source_or_index_cannot_reuse_accepted_commit(tmp_path, monkeypatch):
    import json
    import pytest
    from teamgym.data.finance.finlongdocqa import pipeline
    args = _source_args(tmp_path)
    monkeypatch.setattr(pipeline, "build_client", lambda *a: object())
    monkeypatch.setattr(pipeline, "process_report", lambda **kw: ({"report_id": "TEST_2024", "status": "accepted"}, _common()))
    pipeline.generate_dataset(args)
    qa = json.loads(Path(args.qa_path).read_text())
    qa["question"] = "Changed question with same report ID"
    Path(args.qa_path).write_text(json.dumps(qa) + "\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        pipeline.generate_dataset(args)
    qa["page_numbers"] = [99]
    Path(args.qa_path).write_text(json.dumps(qa) + "\n")
    with pytest.raises(ValueError, match="annotated pages absent"):
        pipeline.generate_dataset(args)
    qa["page_numbers"] = [1, 2]
    Path(args.qa_path).write_text(json.dumps(qa) + "\n")
    index = json.loads(Path(args.block_index_jsonl).read_text())
    index["blocks"][0]["page_number"] = 88
    Path(args.block_index_jsonl).write_text(json.dumps(index) + "\n")
    with pytest.raises(ValueError, match="deterministic reparse"):
        pipeline.generate_dataset(args)


def test_empty_unrecognized_report_format_is_invalid():
    report = "## Page 1\n\nThis marker is not supported."
    assert any("recognized pages" in error for error in validate_report_index(index_report("TEST", report), report))


def test_exact_table_row_restriction_preserves_provenance_and_one_piu_per_block():
    import pytest
    from teamgym.data.finance.finlongdocqa.source import restrict_table_rows
    report = "# Page 1\n\n| Item | 2024 million |\n|---|---:|\n| Revenue | 120 |\n| Unrelated income | 30 |"
    block = index_report("TEST", report)["blocks"][0]
    restriction = restrict_table_rows(block, [1, 2, 3])
    assert "Unrelated income" not in restriction["text"]
    for span in restriction["source_row_spans"]:
        assert report[span["raw_start"]:span["raw_end"]] == span["text"]
    with pytest.raises(ValueError, match="header"):
        restrict_table_rows(block, [3])
    with pytest.raises(ValueError, match="original rows"):
        restrict_table_rows(block, [1, 2, 300])
    proposed = {"evidence_units": [{"unit_id": "original", "source_block_id": block["block_id"], "source_row_numbers": [1, 2, 3]}]}
    hydrated = hydrate_task(proposed, [block])
    assert len(hydrated["evidence_units"]) == 1
    assert hydrated["evidence_units"][0]["text"] == restriction["text"]
    assert hydrated["evidence_units"][0]["source_ref"]["source_row_spans"] == restriction["source_row_spans"]


def test_canonical_table_overlap_restricts_exact_rows_then_reaudits_whole_packet(tmp_path):
    report = "# Page 1\n\n| Item | 2024 million |\n|---|---:|\n| Revenue | 120 |\n| Income | 30 |\n\n# Page 2\n\nOperating income was $30 million in 2024.\n"
    indexed = index_report("TEST_2024", report)
    blocks = indexed["blocks"]
    proposed = _task()
    for unit, block in zip(proposed["evidence_units"], blocks):
        unit["source_block_id"] = block["block_id"]
    task = hydrate_task(proposed, blocks)
    common = build_one_each_datapoint(report_id="TEST_2024", company="TEST", year="2024", report_source={"kind": "test"},
        report_index=indexed, task=task, qa_mappings=[], solver_audit={})
    calls = []
    class Client:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            if str(kwargs["prompt_id"]).endswith("06_table_row_restriction_user"):
                return {"restrictions": [{"source_block_id": blocks[0]["block_id"], "source_row_numbers": [1, 2, 3], "reason": "Keep revenue; omit unrelated income row."}]}
            pid = str(kwargs["trace_dir"]).rsplit("/", 1)[-1]
            leaks = pid == "participant_a" and "round_01" in str(kwargs["trace_dir"])
            context = next(row["context_unit_id"] for row in common["private_context_units"] if pid in row["assigned_to"])
            return {"participant_id": pid, "foreign_rui_results": [{"rui_id": "riu_02" if pid == "participant_a" else "riu_01",
                "reconstructable_from_packet": leaks, "minimal_support_context_ids": [context] if leaks else [],
                "removable_noncanonical_context_ids": [], "canonical_overlap": leaks, "reason": "Table repeats foreign data." if leaks else "No foreign value."}]}
    repaired, audits, _, errors = run_packet_leakage_repair(client=Client(), datapoint=common, public_task=task["public_task"],
        trace_root=tmp_path, max_rounds=2, task=task, report_index=indexed)
    assert not errors
    assert len(audits) == 4  # Whole packet repeated after row change.
    assert "| Income |" not in task["evidence_units"][0]["text"]
    assert repaired["reference_information_units"][0]["text"] == task["evidence_units"][0]["text"]
    assert repaired["generation_metadata"]["canonical_table_restrictions"]


def test_initial_public_plus_local_check_detects_complete_answer_path(tmp_path):
    from teamgym.data.finance.finlongdocqa.pipeline import run_initial_view_audit
    class Client:
        def generate_json(self, **kwargs):
            if str(kwargs["prompt_id"]).endswith("03_blind_solver_user"):
                return {"answer": "An alternative complete answer", "evidence_used": [{"evidence_number": 1, "use": "complete packet"}], "unresolved_requirements": []}
            return _judge(claim_01=True, claim_02=True, complete=True, evidence_count=1)
    rows, errors = run_initial_view_audit(client=Client(), datapoint=_common(), task=_task(), trace_root=tmp_path)
    assert len(rows) == 4
    assert sum("complete task" in error for error in errors) == 4
    assert all("hidden_summary" not in row["consumer_private_input"] and "riu_" not in row["consumer_private_input"] for row in rows)


def test_consolidation_substitution_preserves_qa_links_and_original_provenance():
    from teamgym.data.finance.finlongdocqa.validation import consolidated_qa_mappings
    mappings = [{"qa_id": "qa-a", "evidence_block_ids": ["revenue", "cash"],
                 "block_contributions": [{"block_id": "revenue", "role": "calculation_input", "contribution": "revenue denominator"}]}]
    result = consolidated_qa_mappings(mappings, {"excluded_block_ids": ["revenue"],
        "retained_block_ids": ["income-table", "cash"], "overlap_groups": [{"preferred_block_id": "income-table",
            "relation": "information_contained", "fully_subsumed_block_ids": ["revenue"]}]})
    assert result[0]["evidence_block_ids"] == ["income-table", "cash"]
    assert result[0]["original_evidence_block_ids"] == ["revenue", "cash"]
    assert result[0]["block_contributions"][0]["block_id"] == "income-table"
    assert mappings[0]["evidence_block_ids"] == ["revenue", "cash"]


def test_judge_cannot_certify_negative_local_result_with_all_claims_satisfied():
    from teamgym.data.finance.finlongdocqa.validation import validate_judge_result
    judge = _judge(claim_01=True, claim_02=True, complete=False, evidence_count=1)
    assert any("disagrees" in error for error in validate_judge_result(judge,
        claim_ids={"claim_01", "claim_02"}, available_evidence_count=1))
    judge["overall_complete_and_correct"] = 0
    assert any("must be boolean" in error for error in validate_judge_result(judge,
        claim_ids={"claim_01", "claim_02"}, available_evidence_count=1))


def test_packet_repair_removes_multiple_independent_routes_and_reuses_only_identical_input(tmp_path):
    common = _common()
    first = next(row for row in common["private_context_units"] if not row["hidden_contains_rui_ids"])
    second = deepcopy(first)
    second["context_unit_id"] += "_independent"
    second["text"] = "An independent report restatement of the same revenue."
    common["private_context_units"].append(second)
    pid = first["assigned_to"][0]
    participant = next(row for row in common["participants"] if row["participant_id"] == pid)
    participant["private_context_unit_ids"].append(second["context_unit_id"])
    calls = []
    class Client:
        def generate_json(self, **kwargs):
            calls.append(kwargs)
            current_pid = str(kwargs["trace_dir"]).rsplit("/", 1)[-1]
            leak = current_pid == pid and "round_01" in str(kwargs["trace_dir"])
            route = {"minimal_support_context_ids": [second["context_unit_id"]],
                     "removable_noncanonical_context_ids": [second["context_unit_id"]],
                     "canonical_overlap": False, "reason": "Independent numerical restatement."}
            return {"participant_id": current_pid, "foreign_rui_results": [{
                "rui_id": "riu_02" if current_pid == "participant_a" else "riu_01",
                "reconstructable_from_packet": leak,
                "minimal_support_context_ids": [first["context_unit_id"]] if leak else [],
                "removable_noncanonical_context_ids": [first["context_unit_id"]] if leak else [],
                "canonical_overlap": False, "reason": "synthetic",
                "additional_reconstruction_routes": [route] if leak else []}]}
    _, audits, removed, errors = run_packet_leakage_repair(client=Client(), datapoint=common,
        public_task=_task()["public_task"], trace_root=tmp_path, max_workers=2, max_rounds=1)
    assert not errors
    assert removed == {first["context_unit_id"], second["context_unit_id"]}
    assert len(audits) == 4
    assert len(calls) == 3  # Unchanged other participant's exact full prompt reused.
    assert sum(row["audit_mode"] == "direct_identical_input_reuse" for row in audits) == 1
    by_pid = {}
    for audit in audits:
        by_pid.setdefault(audit["participant_id"], []).append(audit["audit_input_sha256"])
    assert len(set(by_pid[pid])) == 2  # Changed packet was actually re-audited.


def test_additional_reconstruction_routes_cannot_remove_canonical_or_unknown_contexts():
    from teamgym.data.finance.finlongdocqa.validation import validate_packet_leakage_audit
    audit = {"participant_id": "p", "foreign_rui_results": [{"rui_id": "foreign",
        "reconstructable_from_packet": True, "minimal_support_context_ids": ["background"],
        "removable_noncanonical_context_ids": ["background"], "canonical_overlap": False, "reason": "Primary route",
        "additional_reconstruction_routes": [{"minimal_support_context_ids": ["canonical", "unknown"],
            "removable_noncanonical_context_ids": ["canonical", "unknown"], "canonical_overlap": False,
            "reason": "Invalid extra route"}]}]}
    errors = validate_packet_leakage_audit(audit, participant_id="p", packet_context_unit_ids={"background", "canonical"},
        canonical_context_unit_ids={"canonical"}, foreign_rui_ids={"foreign"})
    assert any("outside packet" in error for error in errors)
    assert any("must be non-canonical" in error for error in errors)


def test_context_materialization_applies_paper_cap_without_cutting_canonical_source():
    from teamgym.data.finance.finlongdocqa.packet_budget import materialize_context_cap, complete_private_text
    from teamgym.interaction.actor_utils import count_tokens
    common = _common()
    for index in range(4):
        owner = common["participants"][index % 2]
        context = {"context_unit_id": f"large-background-{index}", "text": "annual background detail\n" * 12000,
                   "assigned_to": [owner["participant_id"]], "hidden_contains_rui_ids": [], "word_count": 36000,
                   "source_ref": {"block_id": f"large-{index}"}}
        common["private_context_units"].append(context)
        owner["private_context_unit_ids"].append(context["context_unit_id"])
    import pytest
    from teamgym.data.finance.finlongdocqa.packet_budget import certify_materialized_views, PacketBudgetError
    with pytest.raises(PacketBudgetError, match="exact complete source differs"):
        certify_materialized_views(common)
    materialize_context_cap(common)
    receipt = common["generation_metadata"]["source_packet_materialization"]
    assert receipt["context_token_budget"] == 40000
    assert len(receipt["excluded_contexts"]) == 4
    assert all(row["reason"] == "complete_noncanonical_context_exceeds_fixed_initial_local_cap" for row in receipt["excluded_contexts"])
    for participant in common["participants"]:
        assert count_tokens(complete_private_text(common, participant)) <= 40000
        owned = participant["owned_rui_ids"]
        for unit in common["reference_information_units"]:
            if unit["rui_id"] in owned:
                assert unit["text"] in complete_private_text(common, participant)
    assert all("large-background" not in context["context_unit_id"] for context in common["private_context_units"])


def test_overbudget_canonical_piu_is_rejected_instead_of_silently_truncated():
    import pytest
    from teamgym.data.finance.finlongdocqa.packet_budget import materialize_context_cap, PacketBudgetError
    common = _common()
    canonical = next(context for context in common["private_context_units"] if context["hidden_contains_rui_ids"])
    original = "pivotal exact source row\n" * 12000
    canonical["text"] = original
    unit = next(unit for unit in common["reference_information_units"] if unit["rui_id"] in canonical["hidden_contains_rui_ids"])
    unit["text"] = original
    with pytest.raises(PacketBudgetError, match="complete canonical evidence exceeds"):
        materialize_context_cap(common)
    assert canonical["text"] == original and unit["text"] == original


def test_local_audit_rejects_an_actor_budget_different_from_the_materialized_budget(tmp_path):
    from teamgym.data.finance.finlongdocqa.packet_budget import materialize_context_cap
    from teamgym.data.finance.finlongdocqa.pipeline import run_initial_view_audit
    common = materialize_context_cap(_common())
    class Client:
        def generate_json(self, **kwargs):
            raise AssertionError("budget mismatch must fail before any model call")
    rows, errors = run_initial_view_audit(client=Client(), datapoint=common, task=_task(), trace_root=tmp_path,
                                         context_token_budget=20000)
    assert len(rows) == 4
    assert len(errors) == 4 and all("budget differs" in error for error in errors)


def test_actual_input_salience_schema_does_not_accept_missing_or_positive_flags(tmp_path):
    from teamgym.data.finance.finlongdocqa.packet_budget import materialize_context_cap
    from teamgym.data.finance.finlongdocqa.pipeline import run_packet_salience_audit
    common = materialize_context_cap(_common())
    class Client:
        def generate_json(self, **kwargs):
            assert "ACTUAL MODEL INPUT" in kwargs["user_prompt"]
            assert "private_source_material" in kwargs["user_prompt"]
            return {"leaks_constructor_metadata": True, "reveals_answer_in_identity": False,
                    "marks_private_piu_as_important": False, "rationale": "Injected positive control."}
    rows, errors = run_packet_salience_audit(client=Client(), datapoint=common, trace_root=tmp_path)
    assert len(rows) == 2
    assert len(errors) == 2 and all("artificial construction" in error for error in errors)
