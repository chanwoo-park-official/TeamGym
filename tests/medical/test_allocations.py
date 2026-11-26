"""Exercise all Medical distributions through packet audit and export gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
from teamgym.data.medical import allocations
from teamgym.data.medical.allocation_source import load_allocation_source, file_hash
from teamgym.data.medical.allocation_packets import (build_partition, build_datapoint,
    material_snapshot, render_contexts, AllocationNotApplicable, validate_partition)
from tests.medical.certified_fixtures import complete_fixture, seal_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def qualified_run(tmp_path, dataset, *, count=4):
    """Real source parsers plus explicit synthetic successful construction audits."""
    root = tmp_path / dataset; root.mkdir()
    raw = tmp_path / f"{dataset}_raw"; raw.mkdir()
    texts = [f"Recorded fact {i}." for i in range(1, count + 1)]
    if dataset == "evidencebench":
        record = {"instance_id": "fixture", "paper_id": "paper", "hypothesis": "Recorded findings.",
            "paper_as_candidate_pool": [*texts, "Unrelated background."],
            "sentence_types_in_candidate_pool": ["results"] * count + ["discussion"]}
        path = raw / "evidencebench_test_set.json"; write(path, {"fixture": record})
        lineage = {"source_file": str(path), "source_file_sha256": file_hash(path), "instance_id": "fixture",
            "record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        units = [{"riu_id": f"u{i}", "candidate_id": f"cand{i}", "hidden_claim": text,
            "required_claim_ids": [f"claim{i}"], "evidence_atoms": [{"text": text, "sentence_indices": [i-1]}]}
            for i, text in enumerate(texts, 1)]
        contexts = [{"context_id": f"x{i}", "text": text, "sentence_index": i-1}
                    for i, text in enumerate(texts, 1)]
    else:
        from teamgym.data.medical.mlcr.source import build_case_indexes, load_inspectable
        summaries = [{"case_uuid": "case", "summary_file": f"{i:03d}.txt", "summary_text": text}
                     for i, text in enumerate([*texts, "Unrelated background."], 1)]
        questions = [{"case_uuid": "case", "prompt_id": f"q{i}", "question": f"What is fact {i}?", "answer": text}
                     for i, text in enumerate(texts, 1)]
        for name, rows in [("cases_summaries.jsonl", summaries), ("questions.jsonl", questions)]:
            (raw / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        lineage = {"source_files": {name: {"path": str(raw / name), "sha256": file_hash(raw / name)}
                                  for name in ["cases_summaries.jsonl", "questions.jsonl"]}}
        case = build_case_indexes(*load_inspectable(raw))[0]
        units, contexts = [], []
        for i, doc in enumerate(case["documents"][:count], 1):
            atom = doc["atoms"][0]
            units.append({"unit_id": f"u{i}", "candidate_id": f"cand{i}", "text": doc["text"],
                "atom_ids": [atom["atom_id"]], "supports_claim_ids": [f"claim{i}"],
                "source_document_id": doc["document_id"], "raw_start": 0, "raw_end": len(doc["text"]),
                "sha256": hashlib.sha256(doc["text"].encode()).hexdigest()})
            contexts.append({"context_id": f"x{i}", "text": doc["text"], "source_document_id": doc["document_id"],
                "source_file": doc["summary_file"], "document_order": i, "source_order": 1})
    claims = [{"claim_id": f"claim{i}", "criterion": text, "description": text,
        "expected_answer_element": text, "grading_rule": "Preserve the recorded fact.",
        "support_logic": "ALL_REQUIRED", "supporting_candidate_ids": [f"cand{i}"],
        "candidate_roles": {f"cand{i}": "recorded finding"}}
        for i, text in enumerate(texts, 1)]
    task = {"public_task": {"question": "State every recorded finding."},
        "selected_candidate_ids": [f"cand{i}" for i in range(1, count+1)],
        "rubric": {"required_claims": claims, "scope_constraints": ["Preserve clinical scope."]},
        "canonical_answer": " ".join(texts), "oracle": {"reference_answer": " ".join(texts)}}
    participants = [{"participant_id": f"p{i}", "hidden_owned_unit_ids": [f"u{i}"],
        "hidden_owned_riu_ids": [f"u{i}"], "context_ids": [f"x{i}"], "context_tokens": 10}
        for i in range(1, count+1)]
    if dataset == "evidencebench":
        write(root / "summary.json", {"status": "partition_leakage_failed", "instance_id": "fixture"})
        write(root / "02_task_rubric.json", task)
        write(root / "05_final_rius.hidden.json", units)
        write(root / "06_participant_packets.hidden.json", {"allocation_policy": "one_each_disjoint_full_paper",
              "participants": participants, "contexts": contexts})
    else:
        task["evidence_units"] = units
        write(root / "run_report.json", {"status": "packet_leakage_failed", "case_id": "case", "instance_id": "fixture",
            "source_lineage": lineage, "task": task,
            "partition": {"participants": participants, "contexts": contexts, "excluded_context_ids": []}})
    complete_fixture(root, dataset)
    write(root / "run_manifest.json", {"source_lineage": lineage, "execution_identity": {"fixture": True}})
    if dataset == "mlcr":
        report = json.loads((root / "run_report.json").read_text())
        write(root / "02_task.json", {"task": report["task"]})
        write(root / "03_grounding_audit.json", {"response": report["grounding_audit"]})
        write(root / "04_boundary_audits.json", report["boundary_audits"])
    from teamgym.data.medical.certification import construction_artifacts, certify_export
    for path in construction_artifacts(root, dataset).values():
        if not path.exists(): write(path, {"fixture": True})
    qualified_task = (json.loads((root / "02_task_rubric.json").read_text()) if dataset == "evidencebench"
                      else json.loads((root / "run_report.json").read_text())["task"])
    qualified = {"status": "construction_qualified", "task": qualified_task, "units": units,
                 "source_lineage": lineage, "execution_identity": {"fixture": True}}
    qualified["export_certification"] = certify_export(qualified, construction_artifacts(root, dataset))
    write(root / "qualified_construction.json", qualified)
    seal_fixture(root, dataset)
    return root


class SemanticFixtureClient:
    cfg = SimpleNamespace(model="fixture", provider="fixture")
    last_request_meta = {}
    def __init__(self): self.requests = []
    def generate_json(self, **kw):
        payload = json.loads(kw["user_prompt"]); self.requests.append(payload)
        if "foreign_units" in payload:
            results = []
            for unit in payload["foreign_units"]:
                support = [c["context_id"] for c in payload["source_contexts"] if unit["exact_evidence"] in c["text"]]
                removable = sorted(set(support) - set(payload["protected_context_ids"]))
                results.append({"unit_id": unit["unit_id"], "reconstructable": bool(support),
                    "support_context_ids": support, "removable_context_ids": removable,
                    "unremovable": bool(support) and not removable, "reason": "Synthetic semantic match"})
            return {"foreign_unit_results": results}
        if "solver_response" not in payload: return {"answer": "Recorded findings available in this packet."}
        snapshot = payload["actual_consumer_material"]
        private = snapshot["private_source_material"]
        common = "\n".join(r["text"] for r in snapshot["initial_common_information"].get("common_reference_units", []))
        results = []
        for claim in payload["rubric"]["required_claims"]:
            fact = claim["criterion"]
            cites = (["private"] if fact in private else []) + (["common"] if fact in common else [])
            results.append({"claim_id": claim["claim_id"], "content_correct": bool(cites),
                            "grounded_in_available_material": bool(cites), "material_numbers": cites})
        complete = all(r["content_correct"] for r in results)
        return {"content_complete": complete, "grounded_complete": complete, "claim_results": results, "critical_errors": []}


@pytest.mark.parametrize("dataset", ["evidencebench", "mlcr"])
def test_all_five_modes_run_distribution_specific_audits_and_export(tmp_path, dataset):
    original = qualified_run(tmp_path, dataset)
    client = SemanticFixtureClient(); out = tmp_path / "out"
    result = allocations.run_allocations(dataset=dataset, source_run=original,
        output_root=out, client=client, modes=PAPER_ALLOCATION_MODES, token_budget=1000)
    assert [r["status"] for r in result["results"]] == ["accepted"] * len(PAPER_ALLOCATION_MODES)
    rows = [json.loads(s) for s in (out / "common_datapoints.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["instance_id"] for r in rows}) == len(PAPER_ALLOCATION_MODES)
    assert all(r["format_validation"]["valid"] for r in rows)
    assert len({json.dumps(r["task_description"], sort_keys=True) for r in rows}) == 1
    assert all(r["schema_version"] == "teamgym_release_datapoint_v1" for r in rows)
    assert len({json.dumps(r["pivotal_information_units"], sort_keys=True) for r in rows}) == 1
    for mode in PAPER_ALLOCATION_MODES:
        loaded = allocations.build_evaluation_datapoint(out / mode / "report.json")
        assert loaded["dataset_specific"]["allocation_mode"] == mode
    redundant = rows[PAPER_ALLOCATION_MODES.index("redundant_player")]
    assert any(not p["owned_piu_ids"] for p in redundant["participants"])
    overlap = rows[PAPER_ALLOCATION_MODES.index("overlap")]
    assert sum(len(p["owned_piu_ids"]) for p in overlap["participants"]) > 4
    common = rows[PAPER_ALLOCATION_MODES.index("common_anchor")]
    assert len(common["initial_common_information"]["common_pius"]) == 1
    common_id = common["initial_common_information"]["common_pius"][0]["piu_id"]
    assert all(common_id not in p["owned_piu_ids"] for p in common["participants"])
    assert all(common_id not in [u["unit_id"] for u in req["foreign_units"]]
               for req in client.requests if "foreign_units" in req and common_id in req["allowed_unit_ids"])


@pytest.mark.parametrize("dataset", ["evidencebench", "mlcr"])
@pytest.mark.parametrize("mode", PAPER_ALLOCATION_MODES)
def test_real_actor_observes_exactly_the_audited_private_and_common_material(tmp_path, dataset, mode):
    from teamgym.interaction.llm_actors import LLMParticipantActor, PromptTraceRecorder
    from teamgym.interaction.types import BudgetSpec
    source = load_allocation_source(dataset, qualified_run(tmp_path, dataset))
    partition = build_partition(source, mode=mode, seed=1, token_budget=1000)
    row = build_datapoint(source, partition)
    snapshots = material_snapshot(row, token_budget=1000)
    calls = []
    class Capture:
        last_request_meta = {}
        def generate_json(self, **kw): calls.append(json.loads(kw["user_prompt"])); return {"utterance": "ok"}
    actor = LLMParticipantActor(Capture(), PromptTraceRecorder())
    for p in row["participants"]:
        actor.respond(datapoint=row, participant_id=p["participant_id"], state={"turn": 0, "messages": []},
                      request=None, budgets=BudgetSpec(participant_context_token_budget=1000))
        actual = calls[-1]
        assert actual["private_source_material"] == snapshots[p["participant_id"]]["private_source_material"]
        assert actual["task"]["initial_common_information"] == snapshots[p["participant_id"]]["initial_common_information"]
        assert "unit_id" not in json.dumps(actual["task"]["initial_common_information"])


@pytest.mark.parametrize("dataset", ["evidencebench", "mlcr"])
def test_small_piu_set_is_skipped_without_falling_back_or_model_calls(tmp_path, dataset):
    original = qualified_run(tmp_path, dataset, count=2)
    client = SemanticFixtureClient()
    result = allocations.run_allocations(dataset=dataset, source_run=original, output_root=tmp_path / "out",
        client=client, modes=["overlap", "common_anchor"])
    assert [r["status"] for r in result["results"]] == ["skipped"] * 2
    assert not client.requests
    assert not (tmp_path / "out/common_datapoints.jsonl").read_text()


@pytest.mark.parametrize("dataset", ["evidencebench", "mlcr"])
@pytest.mark.parametrize("change", ["task", "source", "audit"])
def test_redistribution_rejects_changed_or_unqualified_source(tmp_path, dataset, change):
    root = qualified_run(tmp_path, dataset)
    if change == "source":
        path = tmp_path / f"{dataset}_raw" / ("questions.jsonl" if dataset == "mlcr" else "evidencebench_test_set.json")
        path.write_text(path.read_text() + " ")
    elif dataset == "mlcr":
        path = root / "run_report.json"; value = json.loads(path.read_text())
        if change == "task": value["task"]["public_task"]["question"] = "Changed"
        else: value["grounding_audit"]["passed"] = False
        write(path, value)
    else:
        path = root / ("04_boundaries.json" if change == "audit" else "02_task_rubric.json")
        value = json.loads(path.read_text())
        if change == "audit": value["passed"] = False
        else: value["selected_candidate_ids"].pop()
        write(path, value)
    with pytest.raises(ValueError): load_allocation_source(dataset, root)


@pytest.mark.parametrize("change", ["partition", "missing_participant", "common", "string_passed", "claim_flags"])
def test_export_rejects_modified_or_incomplete_distribution_audits(tmp_path, change):
    source_run = qualified_run(tmp_path, "mlcr"); out = tmp_path / "out"
    allocations.run_allocations(dataset="mlcr", source_run=source_run, output_root=out,
        client=SemanticFixtureClient(), modes=["common_anchor"])
    path = out / "common_anchor/report.json"; report = json.loads(path.read_text())
    if change == "partition": report["partition"]["participants"][0]["context_ids"] = []
    elif change == "missing_participant": report["participant_audit"]["participants"].pop()
    elif change == "common": report["participant_audit"]["participants"][0]["actual_consumer_material"]["initial_common_information"] = {}
    elif change == "string_passed": report["participant_audit"]["passed"] = "false"
    else:
        report["participant_audit"]["participants"][0]["judgment"]["content_complete"] = True
    # A fresh digest cannot replace semantic/coverage validation.
    report["export_certification"] = allocations.certify_export(report, allocations._proof_files(path.parent))
    write(path, report)
    with pytest.raises(ValueError): allocations.build_evaluation_datapoint(path)


def test_removable_foreign_information_is_removed_and_rechecked(tmp_path, monkeypatch):
    original = qualified_run(tmp_path, "evidencebench")
    source = load_allocation_source("evidencebench", original)
    # Put a repeated foreign fact in the noncanonical background. Its owner is
    # chosen after seeing the deterministic packet assignment.
    base = build_partition(source, mode="one_each", seed=1, token_budget=40000)
    filler = next(r for r in source["contexts"] if "Unrelated" in r["text"])
    owner = next(p for p in base["participants"] if filler["context_id"] in p["context_ids"])
    foreign = next(u for u in source["units"] if u["unit_id"] not in owner["owned_unit_ids"])
    filler["text"] = foreign["text"]
    monkeypatch.setattr(allocations, "load_allocation_source", lambda *_a, **_kw: deepcopy(source))
    out = tmp_path / "out"
    result = allocations.run_allocations(dataset="evidencebench", source_run=original, output_root=out,
        client=SemanticFixtureClient(), modes=["one_each"], seed=1, repair_rounds=1)
    assert result["results"][0]["status"] == "accepted"
    report = json.loads((out / "one_each/report.json").read_text())
    assert len(report["leakage_audit"]["rounds"]) == 2
    assert filler["context_id"] in report["partition"]["excluded_context_ids"]


def test_unallocated_protected_reconstruction_is_rejected_without_merging_dist(tmp_path):
    class Unremovable(SemanticFixtureClient):
        def generate_json(self, **kw):
            value = super().generate_json(**kw); p = json.loads(kw["user_prompt"])
            if "foreign_units" in p and p["protected_context_ids"]:
                r = value["foreign_unit_results"][0]
                r.update(reconstructable=True, support_context_ids=[p["protected_context_ids"][0]],
                         removable_context_ids=[], unremovable=True)
            return value
    result = allocations.run_allocations(dataset="mlcr", source_run=qualified_run(tmp_path, "mlcr"),
        output_root=tmp_path / "out", client=Unremovable(), modes=["bottleneck"])
    assert result["results"][0]["status"] == "packet_leakage_failed"
    assert not (tmp_path / "out/bottleneck/common_datapoints.jsonl").exists()


@pytest.mark.parametrize("mode", PAPER_ALLOCATION_MODES)
def test_private_and_common_evidence_share_the_configured_budget(tmp_path, mode):
    source = load_allocation_source("mlcr", qualified_run(tmp_path, "mlcr"))
    with pytest.raises(ValueError, match="budget"):
        build_partition(source, mode=mode, seed=1, token_budget=1)


def test_correct_ungrounded_guess_is_rejected_even_without_grounding(tmp_path):
    class Guesser(SemanticFixtureClient):
        def generate_json(self, **kw):
            value = super().generate_json(**kw)
            if "claim_results" in value:
                for r in value["claim_results"]:
                    r.update(content_correct=True, grounded_in_available_material=False, material_numbers=[])
                value.update(content_complete=True, grounded_complete=False)
            return value
    result = allocations.run_allocations(dataset="mlcr", source_run=qualified_run(tmp_path, "mlcr"),
        output_root=tmp_path / "out", client=Guesser(), modes=["common_anchor"])
    assert result["results"][0]["status"] == "participant_answer_failed"
    assert not (tmp_path / "out/common_anchor/common_datapoints.jsonl").exists()


@pytest.mark.parametrize("dataset", ["evidencebench", "mlcr"])
def test_failed_base_distribution_requires_certified_construction_to_reuse(tmp_path, dataset):
    original = qualified_run(tmp_path, dataset)
    (original / "qualified_construction.json").unlink()
    with pytest.raises(ValueError, match="Uncertified construction"):
        load_allocation_source(dataset, original)


def test_common_and_owned_pius_survive_the_real_piu_only_transform(tmp_path):
    from teamgym.interaction.rui_context import rui_only_datapoint, all_rui_context_text
    source = load_allocation_source("mlcr", qualified_run(tmp_path, "mlcr"))
    for mode in PAPER_ALLOCATION_MODES:
        partition = build_partition(source, mode=mode, seed=1, token_budget=1000)
        full = build_datapoint(source, partition)
        piu_only = rui_only_datapoint(full)
        assert piu_only["initial_common_information"] == full["initial_common_information"]
        assert [p["owned_rui_ids"] for p in piu_only["participants"]] == [p["owned_rui_ids"] for p in full["participants"]]
        assert len(piu_only["participants"]) == len(full["participants"])
        assert all(u["text"] in all_rui_context_text(full) for u in source["units"])


@pytest.mark.parametrize("dataset", ["evidencebench", "mlcr"])
def test_native_cli_routes_fixed_construction_to_all_distributions(monkeypatch, tmp_path, dataset):
    import sys
    if dataset == "evidencebench":
        from teamgym.data.medical.evidencebench import pipeline as cli
        argv = ["run", "--source-path", "raw", "--output-root", str(tmp_path),
                "--audit-model", "auditor", "--allocation-modes", "all"]
        name = "run_pipeline"
    else:
        from teamgym.data.medical.mlcr import cli
        argv = ["run", "--source-dir", "raw", "--case-id", "case", "--output-root", str(tmp_path),
                "--audit-model", "auditor", "--allocation-modes", "all"]
        name = "run_case"
        monkeypatch.setattr(cli, "load_inspectable", lambda _: ([], []))
        monkeypatch.setattr(cli, "build_case_indexes", lambda *_: [{"case_uuid": "case"}])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cli, "OpenAIJsonClient", lambda cfg: SimpleNamespace(cfg=cfg))
    seen = {}
    def construct(**kw): seen["construction"] = kw; return {"status": "construction_qualified"}
    def sweep(**kw): seen["sweep"] = kw; return {"results": []}
    monkeypatch.setattr(cli, name, construct)
    monkeypatch.setattr(allocations, "run_allocations", sweep)
    cli.main()
    assert seen["construction"]["construction_only"] is True
    assert seen["sweep"]["modes"] == PAPER_ALLOCATION_MODES
    assert seen["sweep"]["client"].cfg.model == "auditor"
    assert seen["sweep"]["output_root"] == tmp_path / "allocations"


def test_source_whitespace_uses_the_real_consumers_packet_boundary_normalization(tmp_path):
    source = load_allocation_source("evidencebench", qualified_run(tmp_path, "evidencebench"))
    for context in source["contexts"]:
        context["text"] = "\n  " + context["text"] + "  \n"
    for mode in ("one_each", "common_anchor"):
        partition = build_partition(source, mode=mode, seed=1, token_budget=1000)
        row = build_datapoint(source, partition)
        snapshot = material_snapshot(row, token_budget=1000)
        for p in partition["participants"]:
            assert snapshot[p["participant_id"]]["private_source_material"] == render_contexts(source, p["context_ids"])
