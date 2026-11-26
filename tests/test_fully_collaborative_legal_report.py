from __future__ import annotations

import json
from pathlib import Path

from teamgym.visualization.legal_dataset_report import build_report


def test_legal_dataset_report_renders_roles_and_prompts(tmp_path: Path) -> None:
    input_root = tmp_path / "generated"
    dataset_dir = input_root / "mini"
    dataset_dir.mkdir(parents=True)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "mini.prompt.system.txt").write_text("System prompt text.", encoding="utf-8")
    (trace_dir / "mini.prompt.user.txt").write_text("User prompt text.", encoding="utf-8")
    prompt_root = tmp_path / "prompts"
    (prompt_root / "mini").mkdir(parents=True)
    (prompt_root / "mini" / "system.txt").write_text("Editable system template.", encoding="utf-8")

    row = {
        "instance_id": "mini_001",
        "dataset": "mini",
        "scenario_family": "unit_test",
        "public_task": {
            "title": "Mini task",
            "objective": "Decide whether the answer is yes.",
            "decision_set": ["yes", "no"],
        },
        "participants": [
            {
                "participant_id": "p1",
                "public_role": "first_reviewer",
                "private_packet": {"title": "First packet", "evidence_unit_ids": ["u1"]},
            },
            {
                "participant_id": "p2",
                "public_role": "second_reviewer",
                "private_packet": {"title": "Second packet", "evidence_unit_ids": ["u2"]},
            },
        ],
        "evidence_units": [
            {"unit_id": "u1", "criticality": "critical", "token_count": 5, "text": "A"},
            {"unit_id": "u2", "criticality": "critical", "token_count": 7, "text": "B"},
        ],
        "hidden_world": {"answer": "yes", "required_evidence_unit_ids": ["u1", "u2"]},
        "validation": {"valid": True},
        "token_accounting": {"participant_private_tokens": {"p1": 5, "p2": 7}},
        "llm_trace": {
            "model": "test-model",
            "trace_dir": str(trace_dir),
            "request_meta": {"usage": {"total_tokens": 42}, "response_id": "resp_test"},
        },
    }
    (dataset_dir / "examples_1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = build_report(input_root=input_root, output_path=tmp_path / "report.html", prompt_root=prompt_root)
    html = output.read_text(encoding="utf-8")

    assert "Role And Private Information Amount" in html
    assert "first_reviewer" in html
    assert "Editable system template." in html
    assert "Rendered Prompt Trace" in html
    assert "User prompt text." in html


def test_legal_dataset_report_renders_span_necessity(tmp_path: Path) -> None:
    input_root = tmp_path / "generated"
    dataset_dir = input_root / "cuad"
    dataset_dir.mkdir(parents=True)
    row = {
        "instance_id": "cuad_001",
        "dataset": "cuad",
        "scenario_family": "span_required",
        "public_task": {
            "title": "Span task",
            "objective": "Classify the package.",
            "decision_set": ["a", "b"],
        },
        "participants": [
            {"participant_id": "p1", "public_role": "r1", "private_packet": {"title": "P1", "evidence_unit_ids": ["u1"]}},
            {"participant_id": "p2", "public_role": "r2", "private_packet": {"title": "P2", "evidence_unit_ids": ["u2"]}},
        ],
        "evidence_units": [
            {
                "unit_id": "u1",
                "criticality": "critical",
                "token_count": 5,
                "text": "A",
                "decision_relevance": [{"decision": "b", "relation": "supports", "reason": "A supports b."}],
            },
            {
                "unit_id": "u2",
                "criticality": "critical",
                "token_count": 7,
                "text": "B",
                "decision_relevance": [{"decision": "b", "relation": "required_for", "reason": "B is required."}],
            },
        ],
        "hidden_world": {
            "answer": "b",
            "required_evidence_unit_ids": ["u1", "u2"],
            "rui_necessity_mode": "strict_all_rui_leave_one_out",
            "minimum_sufficient_rui_set_size": 2,
            "smallest_deciding_rui_sets": [
                {
                    "unit_ids": ["u1", "u2"],
                    "answer": "b",
                    "why_sufficient": "Together they decide b.",
                    "why_minimal": "Removing either unit leaves ambiguity.",
                }
            ],
            "oracle_necessity_tests": [
                {
                    "unit_id": "u1",
                    "omitted_unit_id": "u1",
                    "remaining_unit_ids": ["u2"],
                    "can_determine_oracle_without_omitted": False,
                    "effect": "changes_answer",
                    "without_this_span": "A is missing.",
                    "ambiguous_or_plausible_decisions_without_it": ["a", "b"],
                    "why_required": "A is needed.",
                }
            ],
            "necessity_validation": {
                "valid": True,
                "leave_one_out_results": [
                    {
                        "omitted_unit_id": "u1",
                        "remaining_unit_ids": ["u2"],
                        "can_determine_oracle_without_omitted": False,
                        "plausible_decisions_without_omitted": ["a", "b"],
                        "why_required": "A is needed.",
                    }
                ],
            },
            "native_metadata_hidden_from_participants": {
                "candidate_assessment": {
                    "selected_span_indices": [1, 2],
                    "question_design_rationale": "Needs both spans.",
                    "coverage_rationale": "No exclusions.",
                }
            },
        },
        "validation": {"valid": True},
        "token_accounting": {"participant_private_tokens": {"p1": 5, "p2": 7}},
        "llm_trace": {"model": "test-model", "request_meta": {"usage": {"total_tokens": 42}}},
    }
    (dataset_dir / "examples_1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = build_report(input_root=input_root, output_path=tmp_path / "report.html", prompt_root=tmp_path / "missing")
    html = output.read_text(encoding="utf-8")

    assert "Span Necessity" in html
    assert "Smallest Deciding RUI Set" in html
    assert "Decision Relevance" in html
    assert "Needs both spans." in html
