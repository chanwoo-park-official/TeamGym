from __future__ import annotations

import json
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from teamgym.data.software.swe_atlas.pipeline import (
    CanonicalArtifactOverlapError,
    PipelineConfig,
    PipelineError,
    _is_distributable_path,
    _build_reselection_feedback,
    _display_context_view,
    _parse_args,
    allocate_rius,
    build_experiment_logs,
    merge_overlapping_rius,
    validate_consolidation,
    validate_packet_audit,
)
from teamgym.data.software.swe_atlas.export_common import export_build
from teamgym.data.software.swe_atlas.schema import (
    EvaluationSpec,
    PublicTask,
    RepositorySpec,
    Rubric,
    RuntimeSpec,
    SWEAtlasQATask,
)


@pytest.mark.parametrize("output_flag", ["--output-root", "--output-dir"])
def test_pipeline_output_root_names_the_task_run_directory(monkeypatch, output_flag):
    monkeypatch.setattr(sys, "argv", ["swe-atlas", "--task-dir", "qa-task", output_flag, "run"])
    assert _parse_args().output_dir == "run"


def _task() -> SWEAtlasQATask:
    return SWEAtlasQATask(
        task_id="task-x",
        source_dir=Path("/tmp/task-x"),
        repository=RepositorySpec("owner/repo", "abc", "go", "QA"),
        runtime=RuntimeSpec("image", False, 1, 100, 100, 1, 1, 1),
        public=PublicTask("question", "instruction"),
        evaluation=EvaluationSpec(
            rubrics=(
                Rubric("r1", "claim one", "positive", "must have", {}),
                Rubric("r2", "claim two", "positive", "must have", {}),
            ),
            reference_answer="answer",
            judge_system_prompt="system",
            judge_user_prompt_template="user",
        ),
    )


def _bundle():
    candidates = [
        {
            "candidate_id": "c1",
            "claim": "one",
            "evidence_mode": "source",
            "supports_rubric_ids": ["r1"],
            "refutes_rubric_ids": [],
            "evidence": [{"artifact_id": "s1", "kind": "source", "quote": "code one"}],
        },
        {
            "candidate_id": "c2",
            "claim": "two",
            "evidence_mode": "hybrid",
            "supports_rubric_ids": ["r2"],
            "refutes_rubric_ids": [],
            "evidence": [
                {"artifact_id": "s2", "kind": "source", "quote": "code two"},
                {"artifact_id": "run", "kind": "runtime", "quote": "PASS"},
            ],
        },
    ]
    artifacts = [
        {"artifact_id": "s1", "kind": "source", "path": "a.go", "content": "code one", "verified": True},
        {"artifact_id": "s2", "kind": "source", "path": "b.go", "content": "code two", "verified": True},
        {"artifact_id": "run", "kind": "runtime", "content": "PASS", "command": "go test", "verified": True},
    ]
    return candidates, artifacts


def test_consolidation_preserves_exact_evidence_and_rubric_coverage() -> None:
    candidates, artifacts = _bundle()
    response = {
        "groups": [
            {
                "group_id": "g1",
                "hidden_claim": "one",
                "candidate_ids": ["c1"],
                "supports_rubric_ids": ["r1"],
                "refutes_rubric_ids": [],
                "selected_evidence": [{"artifact_id": "s1", "kind": "source", "quote": "code one"}],
            },
            {
                "group_id": "g2",
                "hidden_claim": "two",
                "candidate_ids": ["c2"],
                "supports_rubric_ids": ["r2"],
                "refutes_rubric_ids": [],
                "selected_evidence": [
                    {"artifact_id": "s2", "kind": "source", "quote": "code two"},
                    {"artifact_id": "run", "kind": "runtime", "quote": "PASS"},
                ],
            },
        ]
    }
    rius = validate_consolidation(response, task=_task(), candidates=candidates, artifacts=artifacts)
    assert [row["riu_id"] for row in rius] == ["riu_01", "riu_02"]
    assert {atom["kind"] for atom in rius[1]["evidence_atoms"]} == {"source", "runtime"}


def test_consolidation_rejects_canonical_path_shared_across_groups_before_fallback() -> None:
    candidates, artifacts = _bundle()
    artifacts[1]["path"] = "a.go"
    response = {
        "groups": [
            {
                "group_id": "g1", "hidden_claim": "one", "candidate_ids": ["c1"],
                "supports_rubric_ids": ["r1"], "refutes_rubric_ids": [],
                "selected_evidence": [{"artifact_id": "s1", "kind": "source", "quote": "code one"}],
            },
            {
                "group_id": "g2", "hidden_claim": "two", "candidate_ids": ["c2"],
                "supports_rubric_ids": ["r2"], "refutes_rubric_ids": [],
                "selected_evidence": [
                    {"artifact_id": "s2", "kind": "source", "quote": "code two"},
                    {"artifact_id": "run", "kind": "runtime", "quote": "PASS"},
                ],
            },
        ]
    }
    with pytest.raises(CanonicalArtifactOverlapError):
        validate_consolidation(response, task=_task(), candidates=candidates, artifacts=artifacts)
    excerpt_rius = validate_consolidation(
        response,
        task=_task(),
        candidates=candidates,
        artifacts=artifacts,
        enforce_artifact_exclusivity=False,
    )
    assert len(excerpt_rius) == 2
    rius = validate_consolidation(
        response,
        task=_task(),
        candidates=candidates,
        artifacts=artifacts,
        allow_artifact_merge=True,
    )
    assert len(rius) == 1
    assert rius[0]["supports_rubric_ids"] == ["r1", "r2"]


def test_consolidation_rejects_runtime_overlap_then_allows_final_fallback() -> None:
    candidates, artifacts = _bundle()
    candidates[0]["evidence_mode"] = "hybrid"
    candidates[0]["evidence"].append({"artifact_id": "run", "kind": "runtime", "quote": "PASS"})
    response = {
        "groups": [
            {
                "group_id": "g1", "hidden_claim": "one", "candidate_ids": ["c1"],
                "supports_rubric_ids": ["r1"], "refutes_rubric_ids": [],
                "selected_evidence": [
                    {"artifact_id": "s1", "kind": "source", "quote": "code one"},
                    {"artifact_id": "run", "kind": "runtime", "quote": "PASS"},
                ],
            },
            {
                "group_id": "g2", "hidden_claim": "two", "candidate_ids": ["c2"],
                "supports_rubric_ids": ["r2"], "refutes_rubric_ids": [],
                "selected_evidence": [
                    {"artifact_id": "s2", "kind": "source", "quote": "code two"},
                    {"artifact_id": "run", "kind": "runtime", "quote": "PASS"},
                ],
            },
        ]
    }
    with pytest.raises(CanonicalArtifactOverlapError):
        validate_consolidation(response, task=_task(), candidates=candidates, artifacts=artifacts)
    rius = validate_consolidation(
        response,
        task=_task(),
        candidates=candidates,
        artifacts=artifacts,
        allow_artifact_merge=True,
    )
    assert len(rius) == 1
    assert {atom["artifact_id"] for atom in rius[0]["evidence_atoms"]} == {"s1", "s2", "run"}


def test_allocation_policy_and_context_cap_are_configurable() -> None:
    rius = [
        {"riu_id": f"riu_{index:02d}", "evidence_atoms": [{"quote": "x" * index}]}
        for index in range(1, 5)
    ]
    one_each = allocate_rius(rius, PipelineConfig(allocation_policy="one_each").checked())
    balanced = allocate_rius(
        rius,
        PipelineConfig(
            allocation_policy="fixed_team_balanced",
            participant_count=2,
            participant_context_token_cap=1234,
        ).checked(),
    )
    assert len(set(one_each.values())) == 4
    assert len(set(balanced.values())) == 2
    assert PipelineConfig(participant_context_token_cap=1234).checked().participant_context_token_cap == 1234
    assert PipelineConfig(stop_after_riu_audit=True).checked().stop_after_riu_audit is True
    assert PipelineConfig().checked().riu_reselection_attempts == 3


def test_reselection_feedback_accumulates_failure_reason_for_three_retries() -> None:
    audit = {
        "errors": ["ALL did not satisfy every rubric"],
        "all": {
            "judge_result": {
                "overall_complete_and_correct": False,
                "rubric_results": [
                    {
                        "rubric_id": "r1",
                        "status": "missing",
                        "reason": "missing source proof",
                        "content_correct": False,
                        "grounded_in_available_material": False,
                    }
                ],
            }
        },
        "none": {"judge_result": {"overall_complete_and_correct": False}},
        "leave_one_out": [],
    }
    first = _build_reselection_feedback(
        task=_task(),
        audit=audit,
        final_rius=[{"riu_id": "riu_01"}],
        fallback_error="ALL failed",
        failed_selection_round=0,
        max_retries=3,
    )
    second = _build_reselection_feedback(
        task=_task(),
        audit=audit,
        final_rius=[{"riu_id": "riu_01"}, {"riu_id": "riu_02"}],
        fallback_error="ALL still failed",
        failed_selection_round=1,
        max_retries=3,
        prior_history=first["failure_history"],
    )
    assert first["retries_remaining"] == 3
    assert first["next_retry_number"] == 1
    assert second["retries_remaining"] == 2
    assert len(second["failure_history"]) == 2
    assert second["failure_history"][0]["reason"] == "ALL failed"
    assert second["failed_positive_rubrics"][0]["rubric_id"] == "r1"


def test_nested_target_source_is_not_confused_with_root_build_output() -> None:
    assert _is_distributable_path("internal/target/queue/queue.go")
    assert not _is_distributable_path("target/generated.rs")


def test_canonical_overlap_merge_preserves_atoms_and_rubrics() -> None:
    rius = [
        {
            "riu_id": "riu_01", "source_group_id": "g1", "hidden_claim": "one",
            "candidate_ids": ["c1"], "supports_rubric_ids": ["r1"],
            "refutes_rubric_ids": [],
            "evidence_atoms": [{"atom_id": "a1", "artifact_id": "s1", "kind": "source", "quote": "one"}],
        },
        {
            "riu_id": "riu_02", "source_group_id": "g2", "hidden_claim": "two",
            "candidate_ids": ["c2"], "supports_rubric_ids": ["r2"],
            "refutes_rubric_ids": [],
            "evidence_atoms": [{"atom_id": "a2", "artifact_id": "s2", "kind": "source", "quote": "two"}],
        },
        {
            "riu_id": "riu_03", "source_group_id": "g3", "hidden_claim": "three",
            "candidate_ids": ["c3"], "supports_rubric_ids": ["r3"],
            "refutes_rubric_ids": [],
            "evidence_atoms": [{"atom_id": "a3", "artifact_id": "s3", "kind": "source", "quote": "three"}],
        },
    ]
    merged = merge_overlapping_rius(rius, [["riu_01", "riu_03"]])
    assert len(merged) == 2
    assert merged[0]["supports_rubric_ids"] == ["r1", "r3"]
    assert {atom["artifact_id"] for atom in merged[0]["evidence_atoms"]} == {"s1", "s3"}


def test_public_runtime_context_hides_construction_labels() -> None:
    row = {
        "context_unit_id": "log_x", "context_type": "runtime_log", "text": "[run]",
        "source_ref": {"artifact_id": "R1", "family": "canonical"},
        "assigned_to": ["participant_A"],
    }
    public = _display_context_view(row, include_assignee=True)
    assert "source_ref" not in public
    assert public["assigned_to"] == ["participant_A"]


def test_common_export_keeps_hidden_claims_out_of_solver_material(tmp_path: Path, monkeypatch) -> None:
    build = tmp_path / "build"
    (build / "05_final_rius").mkdir(parents=True)
    (build / "07_partition").mkdir(parents=True)
    (build / "pipeline_report.json").write_text(
        json.dumps(
            {
                "task_id": "task-x",
                "status": "benchmark_ready",
                "all_none_loo_accept": True,
                "semantic_leakage_accept": True,
                "salience_accept": True,
            }
        ),
        encoding="utf-8",
    )
    rius = [
        {
            "riu_id": "riu_01",
            "hidden_claim": "generated answer sentence one",
            "supports_rubric_ids": ["r1"],
            "refutes_rubric_ids": [],
            "evidence_atoms": [
                {
                    "atom_id": "atom_01",
                    "artifact_id": "source_a",
                    "kind": "source",
                    "quote": "func One() bool { return true }",
                    "source_path": "a.go",
                    "source_line_range": [1, 1],
                    "command": None,
                }
            ],
        },
        {
            "riu_id": "riu_02",
            "hidden_claim": "generated answer sentence two",
            "supports_rubric_ids": ["r2"],
            "refutes_rubric_ids": [],
            "evidence_atoms": [
                {
                    "atom_id": "atom_02",
                    "artifact_id": "source_b",
                    "kind": "source",
                    "quote": "func Two() bool { return false }",
                    "source_path": "b.go",
                    "source_line_range": [1, 1],
                    "command": None,
                }
            ],
        },
    ]
    (build / "05_final_rius" / "final_rius.hidden.json").write_text(
        json.dumps({"rius": rius}), encoding="utf-8"
    )
    from teamgym.data.software.swe_atlas.integrity import atomic_json, task_identity
    atomic_json(build / 'input_identity.json', {'source': task_identity(_task())})
    atomic_json(build / '05_final_rius/audit_summary.json', {'accept': True})
    atomic_json(build / '05_final_rius/selected_artifacts.hidden.json', {'artifacts': []})
    partition = {
        "allocation_policy": "one_each",
        "participant_context_token_cap": 40000,
        "riu_ownership": {"riu_01": "participant_A", "riu_02": "participant_B"},
        "participants": [
            {
                "participant_id": "participant_A",
                "role_name": "repository reviewer 1",
                "private_context_unit_ids": ["code_a"],
            },
            {
                "participant_id": "participant_B",
                "role_name": "repository reviewer 2",
                "private_context_unit_ids": ["code_b"],
            },
        ],
        "contexts": [
            {
                "context_unit_id": "code_a",
                "context_type": "code",
                "text": "[file]\npath: a.go\ncontent:\nfunc One() bool { return true }",
                "assigned_to": ["participant_A"],
                "source_ref": {"path": "a.go"},
                "hidden_contains_riu_ids": ["riu_01"],
            },
            {
                "context_unit_id": "code_b",
                "context_type": "code",
                "text": "[file]\npath: b.go\ncontent:\nfunc Two() bool { return false }",
                "assigned_to": ["participant_B"],
                "source_ref": {"path": "b.go"},
                "hidden_contains_riu_ids": ["riu_02"],
            },
        ],
    }
    (build / "07_partition" / "final_partition.hidden.json").write_text(
        json.dumps(partition), encoding="utf-8"
    )
    monkeypatch.setattr(
        "teamgym.data.software.swe_atlas.export_common.load_qa_task",
        lambda _: _task(),
    )

    from teamgym.data.software.swe_atlas.integrity import write_ready_receipt
    write_ready_receipt(build, _task())

    full, riu_only = export_build(build_dir=build, task_dir=tmp_path / "task-x")

    visible = json.dumps(
        {
            "full_contexts": full["private_context_units"],
            "riu_only_contexts": riu_only["private_context_units"],
            "reference_information_units": full["reference_information_units"],
        },
        ensure_ascii=False,
    )
    assert "generated answer sentence" not in visible
    assert "func One() bool" in visible
    assert all(not row["text"].startswith("RIU ") for row in full["private_context_units"])
    assert full["format_validation"]["valid"] is True
    assert riu_only["format_validation"]["valid"] is True



def test_invalid_distractor_plan_is_retried_before_execution(tmp_path, monkeypatch):
    invalid = {
        "experiment_id": "bad-path",
        "family": "environment_probe",
        "command": "printf '%s' ready > /app/output",
    }
    corrected = {
        "experiment_id": "readiness",
        "family": "environment_probe",
        "command": "printf '%s' ready",
    }
    responses = iter(({"experiments": [invalid]}, {"experiments": [corrected]}))
    feedback = []

    def call_prompt(client, system, user, variables, **kwargs):
        feedback.append(variables["FEEDBACK_JSON"])
        return next(responses)

    def collect(requests):
        assert [request["command"] for request in requests] == [corrected["command"]]
        return [{"request_id": "readiness", "artifact_id": "log-1", "verified": True,
                 "command": corrected["command"], "content": "ready"}]

    monkeypatch.setattr(
        "teamgym.data.software.swe_atlas.pipeline._call_prompt",
        call_prompt,
    )
    result = build_experiment_logs(
        client=None, task=_task(), rius=[], artifacts=[],
        collector=SimpleNamespace(collect=collect), tool_paths={},
        output_dir=tmp_path,
        config=PipelineConfig(min_distractor_logs=1, experiment_repair_attempts=1),
    )
    assert len(result["distractor_logs"]) == 1
    assert "may only write under /tmp" in feedback[1]["validation_error"]


@pytest.mark.parametrize("results_key", ["foreign_riu_results", "foreign_rui_results"])
def test_packet_audit_accepts_prompt_key_and_preserves_report_format(results_key):
    participant = {"participant_id": "participant_A"}
    foreign_rius = [{"riu_id": "riu_02"}]
    result = {
        "riu_id": "riu_02",
        "reconstructable_from_packet": False,
        "minimal_support_context_ids": [],
        "removable_noncanonical_context_ids": [],
        "canonical_overlap": False,
        "reason": "The packet does not contain the required evidence.",
    }
    audit = {**participant, results_key: [result]}

    assert validate_packet_audit(audit, participant, [], foreign_rius) == {
        **participant, "foreign_rui_results": [result],
    }


@pytest.mark.parametrize("overlap_kind", ["source_file", "runtime_log", "semantic"])
@pytest.mark.parametrize("reselection_succeeds", [True, False])
def test_build_repairs_overlap_and_exports_the_final_evidence(
    tmp_path, monkeypatch, overlap_kind, reselection_succeeds,
):
    from teamgym.data.software.swe_atlas import pipeline
    from teamgym.data.software.swe_atlas import integrity
    monkeypatch.setattr(integrity, 'image_identity', lambda _: {'id': 'sha256:fixture'})

    base_task = _task()
    task = replace(
        base_task,
        evaluation=replace(
            base_task.evaluation,
            rubrics=base_task.evaluation.rubrics + (
                Rubric("r3", "claim three", "positive", "must have", {}),
            ),
        ),
    )
    artifacts = [
        {"artifact_id": "s1", "kind": "source", "path": "a.go",
         "content": "code one", "verified": True},
        {"artifact_id": "s2", "kind": "source",
         "path": "a.go" if overlap_kind == "source_file" else "b.go",
         "content": "code two", "verified": True},
        {"artifact_id": "s3", "kind": "source", "path": "c.go",
         "content": "code three", "verified": True},
    ]
    candidates = [
        {"candidate_id": f"c{i}", "claim": artifact["content"],
         "evidence_mode": "source", "supports_rubric_ids": [f"r{i}"],
         "refutes_rubric_ids": [],
         "evidence": [{"artifact_id": artifact["artifact_id"],
                       "kind": "source", "quote": artifact["content"]}]}
        for i, artifact in enumerate(artifacts, start=1)
    ]
    if overlap_kind == "runtime_log":
        artifacts.append(
            {"artifact_id": "run", "kind": "runtime", "command": "go test",
             "content": "PASS", "verified": True},
        )
        for candidate in candidates[:2]:
            candidate["evidence_mode"] = "hybrid"
            candidate["evidence"].append(
                {"artifact_id": "run", "kind": "runtime", "quote": "PASS"},
            )
    groups = [
        {"group_id": f"g{i}", "hidden_claim": candidate["claim"],
         "candidate_ids": [candidate["candidate_id"]],
         "supports_rubric_ids": candidate["supports_rubric_ids"],
         "refutes_rubric_ids": [], "selected_evidence": candidate["evidence"]}
        for i, candidate in enumerate(candidates, start=1)
    ]
    merged_groups = [
        {"group_id": "merged", "hidden_claim": "claims one and two",
         "candidate_ids": ["c1", "c2"], "supports_rubric_ids": ["r1", "r2"],
         "refutes_rubric_ids": [],
         "selected_evidence": candidates[0]["evidence"] + candidates[1]["evidence"]},
        groups[2],
    ]
    build = tmp_path / "build"
    candidate_dir = build / "03_candidates"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "riu_candidates.json").write_text(
        json.dumps({"task_id": task.task_id, "candidates": candidates}),
    )
    (candidate_dir / "evidence_artifacts.json").write_text(
        json.dumps({"artifacts": artifacts}),
    )
    consolidation_calls = []

    def consolidate(**kwargs):
        consolidation_calls.append(kwargs)
        if len(consolidation_calls) == 1:
            selected_groups = groups
        elif not reselection_succeeds:
            raise pipeline.PipelineError("no independent alternative excerpts")
        else:
            selected_groups = merged_groups
        selected_groups = deepcopy(selected_groups)
        mapping = {a.get('local_artifact_id', a['artifact_id']): a['artifact_id'] for a in kwargs['artifacts']}
        for group in selected_groups:
            for atom in group['selected_evidence']:
                atom['artifact_id'] = mapping[atom['artifact_id']]
        return {'groups': selected_groups}

    corpus_text = {}
    for artifact in artifacts:
        if artifact["kind"] == "source":
            path = artifact["path"]
            corpus_text[path] = corpus_text.get(path, "") + artifact["content"] + "\n"
    corpus = [
        {"path": path, "text": text, "module_id": "root", "estimated_tokens": 10}
        for path, text in corpus_text.items()
    ]
    leakage_calls = []

    def leakage(**kwargs):
        leakage_calls.append(kwargs)
        if overlap_kind == "semantic" and len(leakage_calls) == 1:
            raise pipeline.CanonicalOverlapError([["riu_01", "riu_02"]], [])
        return kwargs["partition"], {
            "accept": True, "rounds_run": 1, "removed_context_ids": [],
        }

    monkeypatch.setattr(pipeline, "_consolidate_candidates", consolidate)
    monkeypatch.setattr(
        pipeline, "run_riu_audit",
        lambda **_: {"accept": True, "errors": []},
    )
    monkeypatch.setattr(
        pipeline, "DockerEvidenceCollector",
        lambda *a, **kw: SimpleNamespace(inventory=lambda: {}, tool_paths=lambda: {}),
    )
    monkeypatch.setattr(
        pipeline, "build_experiment_logs",
        lambda **_: {"canonical_logs": [], "distractor_logs": []},
    )
    monkeypatch.setattr(pipeline, "collect_code_corpus", lambda **_: corpus)
    monkeypatch.setattr(pipeline, "run_leakage_repair", leakage)
    monkeypatch.setattr(
        pipeline, "run_salience_repair",
        lambda **kw: (kw["partition"], {"accept": True}),
    )
    client = SimpleNamespace(cfg=SimpleNamespace(model='fake'))
    config = PipelineConfig(riu_reselection_attempts=0, audit_repair_attempts=0)
    identity = integrity.run_identity(task, client, config.checked(), docker_timeout_sec=900)
    integrity.atomic_json(build / 'input_identity.json', identity)
    integrity.write_checkpoint(build, identity)
    report = pipeline.build_task(task=task, output_dir=build, client=client, config=config)
    assert report["status"] == "benchmark_ready"
    assert report["final_riu_count"] == 2
    assert len(consolidation_calls) == 2

    evaluation = json.loads((build / "09_benchmark" / "evaluation.hidden.json").read_text())
    checkpoint = json.loads((build / "05_final_rius" / "final_rius.hidden.json").read_text())
    assert checkpoint["rius"] == evaluation["final_rius"]
    monkeypatch.setattr(
        "teamgym.data.software.swe_atlas.export_common.load_qa_task", lambda _: task,
    )
    full, riu_only = export_build(build_dir=build, task_dir=tmp_path / task.task_id)
    for row in (full, riu_only):
        assert row["format_validation"]["valid"] is True
        assert len(row["reference_information_units"]) == 2
        assert row["reference_information_units"][0]["supports_rubric_ids"] == ["r1", "r2"]
        if overlap_kind == "runtime_log":
            assert len(row['private_experiment_registry']) == 1
            registry = next(iter(row["private_experiment_registry"].values()))
            assert registry["rui_ids"] == ["riu_01"]
            assert registry["owned_by"] == ["participant_A"]
