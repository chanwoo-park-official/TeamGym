from copy import deepcopy

import pytest

from teamgym.data.medical.mlcr import cli
from teamgym.data.medical.mlcr import pipeline
from teamgym.data.medical.mlcr.construction import validate_task


@pytest.mark.parametrize("minimum", [1, 2])
def test_ownership_repairs_preserve_earlier_leakage_merges(tmp_path, monkeypatch, minimum):
    unit_ids = ["a", "b", "c", "d"]
    question_ids = ["q01"] if minimum == 1 else ["q01", "q02"]
    task = {
        "source_question_ids_used": question_ids,
        "public_task": {"question": "Summarize the recorded facts."},
        "evidence_units": [
            {"unit_id": unit_id, "candidate_id": unit_id, "text": f"Fact {unit_id}."}
            for unit_id in unit_ids
        ],
        "rubric": {
            "required_claims": [
                {"supporting_candidate_ids": [unit_id]} for unit_id in unit_ids
            ]
        },
    }
    case = {
        "case_uuid": "case",
        "source_questions": [{"prompt_id": prompt_id} for prompt_id in question_ids],
    }

    def structured_call(**kwargs):
        if kwargs["stage"].startswith("map_qa."):
            return {
                "answer_reconstructable": True,
                "candidate_spans": [{"candidate_id": "first"}, {"candidate_id": "second"}],
            }, []
        if kwargs["stage"].startswith("generate_composite_task."):
            assert kwargs["variables"]["MIN_SOURCE_QUESTIONS"] == minimum
            assert kwargs["validator"](task) == []
            return deepcopy(task), []
        return {"passed": True}, []

    requested_groups = []

    def partition(**kwargs):
        groups = deepcopy(kwargs["ownership_groups"])
        requested_groups.append(groups)
        return {
            "participants": [
                {
                    "participant_id": f"participant_{index}",
                    "hidden_owned_unit_ids": group,
                    "private_context": "Source material.",
                }
                for index, group in enumerate(groups)
            ]
        }

    repairs = iter(([["a", "b"]], [["c", "d"]], []))

    def audit(**kwargs):
        pairs = next(repairs)
        return kwargs["partition"], {
            "passed": not pairs, "ownership_merge_pairs": pairs, "errors": [],
        }

    monkeypatch.setattr(pipeline, "public_source_atoms", lambda _: [])
    monkeypatch.setattr(pipeline, "structured_call", structured_call)
    monkeypatch.setattr(pipeline, "hydrate_candidates", lambda row, **_: row["candidate_spans"])
    monkeypatch.setattr(pipeline, "hydrate_task_evidence", lambda row, **_: row)
    validated_minimums = []

    def validate(*args, min_source_questions, **kwargs):
        validated_minimums.append(min_source_questions)
        return []

    monkeypatch.setattr(pipeline, "validate_task", validate)
    monkeypatch.setattr(pipeline, "run_boundaries", lambda **_: {"passed": True})
    monkeypatch.setattr(pipeline, "build_token_budgeted_partition", partition)
    monkeypatch.setattr(pipeline, "audit_and_sanitize_partition", audit)
    monkeypatch.setattr(pipeline, "audit_participant_answers", lambda **_: {"passed": True})

    result = pipeline.run_case(
        case_index=case, output_root=tmp_path,
        generation_client=None, verification_client=None,
        ownership_repair_rounds=3,
        min_source_questions=minimum,
    )

    assert requested_groups == [
        [["a"], ["b"], ["c"], ["d"]],
        [["a", "b"], ["c"], ["d"]],
        [["a", "b"], ["c", "d"]],
    ]
    assert result["status"] == "accepted"
    assert result["min_source_questions"] == minimum
    assert validated_minimums == [minimum, minimum]


@pytest.mark.parametrize(("question_ids", "minimum", "expected_errors"), [
    (["q01"], 1, []),
    (["q01"], 2, ["at least 2 unique source questions must be used"]),
    (["q01", "q02"], 3, ["at least 3 unique source questions must be used"]),
])
def test_task_validation_uses_the_configured_source_question_minimum(
    question_ids, minimum, expected_errors,
):
    candidates = [
        {"candidate_id": f"c{index}", "source_prompt_id": question_ids[index % len(question_ids)]}
        for index in range(2)
    ]
    task = {
        "source_question_ids_used": question_ids,
        "selected_candidate_ids": ["c0", "c1"],
        "public_task": {"question": "Summarize the recorded findings."},
        "rubric": {"required_claims": [{
            "claim_id": "claim", "supporting_candidate_ids": ["c0", "c1"],
            "description": "Combine both findings.",
            "expected_answer_element": "Both findings.", "grading_rule": "Require both.",
        }]},
        "evidence_units": [
            {"unit_id": f"u{index}", "candidate_id": f"c{index}", "supports_claim_ids": ["claim"]}
            for index in range(2)
        ],
        "oracle": {"reference_answer": "Both findings."},
    }

    assert validate_task(
        task, candidates=candidates, valid_prompt_ids=set(question_ids),
        min_source_questions=minimum,
    ) == expected_errors


@pytest.mark.parametrize("minimum", [1, 3])
def test_cli_forwards_the_source_question_minimum(tmp_path, monkeypatch, minimum):
    monkeypatch.setattr(cli, "load_inspectable", lambda _: ([], []))
    monkeypatch.setattr(cli, "build_case_indexes", lambda *_: [{"case_uuid": "case"}])
    monkeypatch.setattr(cli, "OpenAIJsonClient", lambda _: None)
    received = {}

    def run(**kwargs):
        received.update(kwargs)
        return {"status": "task_generation_failed"}

    monkeypatch.setattr(cli, "run_case", run)
    monkeypatch.setattr("sys.argv", [
        "mlcr", "--source-dir", str(tmp_path), "--case-id", "case",
        "--output-root", str(tmp_path / "output"), "--min-source-questions", str(minimum),
    ])
    cli.main()

    assert received["min_source_questions"] == minimum
    assert received["strict_one_each"] is True


@pytest.mark.parametrize("minimum", [0, -1])
def test_cli_rejects_invalid_minimum_before_reading_sources(tmp_path, monkeypatch, minimum):
    monkeypatch.setattr(cli, "load_inspectable", lambda _: pytest.fail("unexpected source read"))
    monkeypatch.setattr("sys.argv", [
        "mlcr", "--source-dir", str(tmp_path), "--case-id", "case",
        "--output-root", str(tmp_path / "output"), "--min-source-questions", str(minimum),
    ])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
