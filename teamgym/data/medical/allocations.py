"""Audit and export each allocation of a fixed, qualified Medical task."""
from __future__ import annotations

import argparse
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path

from teamgym.core.release import write_release_datapoints
from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
from teamgym.tokenization import count_tokens
from .allocation_source import load_allocation_source, content_hash, read_json, file_hash
from .allocation_packets import (AllocationNotApplicable, build_partition, build_datapoint,
    material_snapshot, visible_context_ids, validate_partition)
from .certification import certify_export, validate_export_certification, require_passed_audit, positive_token_budget

PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "medical" / "allocations"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _proof_files(root):
    return {"source.json": root.parent / "source.json", "run_manifest.json": root / "run_manifest.json"}


def _call(*, source, client, cache, root, stage, prompt, payload, validator, attempts):
    feedback, response, errors = None, {}, ["audit did not run"]
    system = (PROMPTS / prompt).read_text()
    for attempt in range(1, attempts + 1):
        user = json.dumps({**payload, "format_feedback": feedback}, ensure_ascii=False)
        tokens = count_tokens(system) + count_tokens(user)
        if tokens > cache.max_input_tokens:
            return {}, [f"input budget exceeded: {tokens} > {cache.max_input_tokens}; no model call made"]
        try:
            response = cache.call(client=client, stage=f"{stage}.attempt_{attempt}",
                **{("instance_id" if source["dataset"] == "evidencebench" else "case_id"): source["instance_id"]},
                system_prompt=system, user_prompt=user, trace_dir=root / "_llm_traces")
            errors = validator(response)
        except (ValueError, TypeError) as exc:
            response, errors = {}, [f"invalid audit response: {exc}"]
        if not errors:
            return response, []
        feedback = {"errors": errors, "instruction": "Correct the JSON for the unchanged task and material."}
    return response, errors


def validate_leakage(value, *, foreign, available, protected):
    if not isinstance(value, dict) or not isinstance(value.get("foreign_unit_results"), list):
        return ["foreign_unit_results must be an array"]
    rows = value["foreign_unit_results"]
    if (len(rows) != len(foreign) or any(not isinstance(r, dict) for r in rows)
            or {r.get("unit_id") for r in rows} != set(foreign)):
        return ["foreign PIU coverage mismatch"]
    errors = []
    for r in rows:
        for key in ("reconstructable", "unremovable"):
            if type(r.get(key)) is not bool: errors.append(f"{key} must be boolean")
        support, removable = r.get("support_context_ids"), r.get("removable_context_ids")
        if any(not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
               for ids in (support, removable)):
            errors.append("support and removal IDs must be arrays of strings"); continue
        support, removable = set(support), set(removable)
        if not support <= available or not removable <= support - protected:
            errors.append("unavailable support or removal of protected/non-supporting context")
        if r.get("reconstructable") is True:
            if not support: errors.append("reconstruction requires actual supporting context citations")
            if support and support <= protected:
                if r.get("unremovable") is not True or removable:
                    errors.append("protected-only reconstruction must be unremovable")
            elif r.get("unremovable") is not False or not removable:
                errors.append("noncanonical reconstruction requires removable support")
        elif removable or r.get("unremovable") is not False:
            errors.append("non-reconstructable PIU cannot request removals or be unremovable")
    return errors


def _leakage_inputs(source, partition, participant, snapshot):
    units = {u["unit_id"]: u for u in source["units"]}
    allowed = set(participant["owned_unit_ids"]) | set(partition["allocation"]["common_rui_ids"])
    foreign = set(units) - allowed
    visible = set(visible_context_ids(partition, participant))
    protected = {c for uid in allowed for c in units[uid]["context_ids"]}
    payload = {"public_task": source["task"]["public_task"], "actual_consumer_material": snapshot,
        "allowed_unit_ids": sorted(allowed), "protected_context_ids": sorted(protected),
        "foreign_units": [{"unit_id": uid, "exact_evidence": units[uid]["text"],
            "affected_claims": [{k: v for k, v in c.items() if k not in
                                 {"supporting_candidate_ids", "candidate_roles", "support_logic"}}
                for c in source["task"]["rubric"]["required_claims"]
                if c["claim_id"] in units[uid]["supports_claim_ids"]]} for uid in sorted(foreign)],
        "source_contexts": [{"context_id": r["context_id"], "source_file": r["source_file"], "text": r["text"]}
                            for r in source["contexts"] if r["context_id"] in visible]}
    return payload, {"foreign": foreign, "available": visible, "protected": protected}


def audit_packets(source, partition, *, client, cache, root, attempts, repair_rounds):
    partition = deepcopy(partition)
    rounds = []
    for number in range(1, repair_rounds + 2):
        row = build_datapoint(source, partition)
        snapshots = material_snapshot(row, token_budget=partition["participant_token_budget"])
        records, removals, errors, dirty = [], {}, [], False
        for p in partition["participants"]:
            pid = p["participant_id"]
            payload, validation = _leakage_inputs(source, partition, p, snapshots[pid])
            response, problems = _call(source=source, client=client, cache=cache, root=root,
                stage=f"leakage.round_{number}.{pid}", prompt="leakage_system.txt", payload=payload,
                validator=lambda v, kw=validation: validate_leakage(v, **kw), attempts=attempts)
            remove = set()
            if not problems:
                for r in response["foreign_unit_results"]:
                    if r["reconstructable"]:
                        dirty = True
                        if r["unremovable"]:
                            problems.append(f"{pid}: unallocated {r['unit_id']} reconstructable from protected evidence")
                        remove.update(r["removable_context_ids"])
            removals[pid] = sorted(remove)
            errors.extend(problems)
            records.append({"participant_id": pid, "actual_consumer_material": snapshots[pid],
                            "response": response, "errors": problems})
        rounds.append({"round": number, "participants": records, "removals": removals, "errors": errors})
        if errors or not dirty:
            return partition, {"passed": not errors and not dirty, "rounds": rounds, "errors": errors}
        if number > repair_rounds:
            break
        for p in partition["participants"]:
            p["context_ids"] = [c for c in p["context_ids"] if c not in removals[p["participant_id"]]]
        included = set(partition["common_context_ids"]) | {c for p in partition["participants"] for c in p["context_ids"]}
        partition["excluded_context_ids"] = sorted({r["context_id"] for r in source["contexts"]} - included)
        validate_partition(source, partition)
    return partition, {"passed": False, "rounds": rounds, "errors": ["leakage repair budget exhausted after final recheck"]}


def validate_judgment(value, *, claims, material_numbers):
    if not isinstance(value, dict): return ["judgment must be an object"]
    rows = value.get("claim_results")
    if (not isinstance(rows, list) or len(rows) != len(claims) or any(not isinstance(r, dict) for r in rows)
            or {r.get("claim_id") for r in rows} != set(claims)):
        return ["judgment must cover every claim exactly once"]
    errors = []
    for r in rows:
        if any(type(r.get(k)) is not bool for k in ("content_correct", "grounded_in_available_material")):
            errors.append("claim correctness and grounding must be boolean")
        cited = r.get("material_numbers")
        if not isinstance(cited, list) or any(not isinstance(c, str) for c in cited) or not set(cited) <= material_numbers:
            errors.append("judge cited unavailable material")
        if r.get("grounded_in_available_material") is True and not cited:
            errors.append("grounded claims require actual material citations")
    complete = all(r.get("content_correct") is True for r in rows)
    grounded = all(r.get("content_correct") is True and r.get("grounded_in_available_material") is True for r in rows)
    if type(value.get("content_complete")) is not bool or value["content_complete"] is not complete:
        errors.append("content_complete contradicts claim results")
    if type(value.get("grounded_complete")) is not bool or value["grounded_complete"] is not grounded:
        errors.append("grounded_complete contradicts claim results")
    if not isinstance(value.get("critical_errors"), list): errors.append("critical_errors must be an array")
    return errors


def _judge_inputs(source, snapshot, solver):
    rubric = deepcopy(source["task"]["rubric"])
    for claim in rubric["required_claims"]:
        for key in ("supporting_candidate_ids", "candidate_roles", "support_logic"):
            claim.pop(key, None)
    numbers = set()
    if snapshot["private_source_material"]: numbers.add("private")
    if snapshot["initial_common_information"].get("common_reference_units"): numbers.add("common")
    task = source["task"]
    return {"public_task": task["public_task"], "rubric": rubric,
        "reference_answer": task.get("canonical_answer") if source["dataset"] == "evidencebench" else task["oracle"]["reference_answer"],
        "actual_consumer_material": snapshot, "solver_response": solver}, {
            "claims": {c["claim_id"] for c in rubric["required_claims"]}, "material_numbers": numbers}


def audit_participants(source, partition, *, client, cache, root, attempts):
    snapshots = material_snapshot(build_datapoint(source, partition), token_budget=partition["participant_token_budget"])
    results = []
    for p in partition["participants"]:
        pid = p["participant_id"]; snapshot = snapshots[pid]
        solver, errors = _call(source=source, client=client, cache=cache, root=root,
            stage=f"participant_solver.{pid}", prompt="solver_system.txt",
            payload={"public_task": source["task"]["public_task"], **snapshot},
            validator=lambda r: [] if isinstance(r, dict) and isinstance(r.get("answer"), str) and r["answer"].strip()
                                  else ["solver answer must be nonempty text"], attempts=attempts)
        judgment = {}
        if not errors:
            payload, validation = _judge_inputs(source, snapshot, solver)
            judgment, errors = _call(source=source, client=client, cache=cache, root=root,
                stage=f"participant_judge.{pid}", prompt="judge_system.txt", payload=payload,
                validator=lambda v, kw=validation: validate_judgment(v, **kw), attempts=attempts)
        results.append({"participant_id": pid, "actual_consumer_material": snapshot,
            "participant_context_token_budget": partition["participant_token_budget"],
            "solver": solver, "judgment": judgment, "errors": errors,
            "passed": not errors and judgment.get("content_complete") is False})
    return {"passed": bool(results) and all(r["passed"] for r in results), "participants": results,
            "limitation": "finite model evidence, not proof over every possible answer"}


def _complete_records(rows, partition):
    expected = {p["participant_id"] for p in partition["participants"]}
    if (not isinstance(rows, list) or len(rows) != len(expected) or any(not isinstance(r, dict) for r in rows)
            or {r.get("participant_id") for r in rows} != expected):
        raise ValueError("Missing or duplicate participant audit records")
    return {r["participant_id"]: r for r in rows}


def build_evaluation_datapoint(report_path):
    path = Path(report_path).resolve(); report = read_json(path)
    if report.get("status") != "accepted":
        raise ValueError("Medical allocation has not passed its distribution-specific audits")
    validate_export_certification(report, _proof_files(path.parent))
    source = read_json(path.parent.parent / "source.json")
    if report["source_sha256"] != content_hash(source):
        raise ValueError("Frozen allocation source changed")
    partition = report["partition"]
    manifest = read_json(path.parent / "run_manifest.json")
    if (report["mode"] != partition["allocation"]["mode"] or manifest["mode"] != report["mode"]
            or manifest["source_sha256"] != report["source_sha256"]
            or report["execution_identity"]["participant_context_token_budget"] != partition["participant_token_budget"]):
        raise ValueError("Distribution or budget differs from its execution manifest")
    datapoint = build_datapoint(source, partition, execution_identity=report["execution_identity"])
    snapshots = material_snapshot(datapoint, token_budget=partition["participant_token_budget"])
    require_passed_audit(report["leakage_audit"], "distribution leakage audit")
    rounds = report["leakage_audit"].get("rounds")
    if not isinstance(rounds, list) or not rounds: raise ValueError("Missing final leakage recheck")
    if rounds[-1].get("errors"): raise ValueError("Final leakage recheck reported errors")
    leak = _complete_records(rounds[-1]["participants"], partition)
    require_passed_audit(report["participant_audit"], "distribution participant-answer audit")
    answers = _complete_records(report["participant_audit"]["participants"], partition)
    for p in partition["participants"]:
        pid = p["participant_id"]; a, l = answers[pid], leak[pid]
        if a.get("actual_consumer_material") != snapshots[pid] or l.get("actual_consumer_material") != snapshots[pid]:
            raise ValueError("Export differs from the actually audited common/private material")
        _, validation = _leakage_inputs(source, partition, p, snapshots[pid])
        if (l.get("errors") or validate_leakage(l["response"], **validation)
                or any(r["reconstructable"] for r in l["response"]["foreign_unit_results"])):
            raise ValueError("Final leakage audit is incomplete or failed")
        require_passed_audit(a, f"participant {pid}")
        _, validation = _judge_inputs(source, snapshots[pid], a["solver"])
        if (a.get("participant_context_token_budget") != partition["participant_token_budget"]
                or validate_judgment(a["judgment"], **validation) or a["judgment"]["content_complete"] is not False):
            raise ValueError("Participant final-answer audit failed or used a different budget")
    return datapoint


def run_allocations(*, dataset, source_run, output_root, client, modes=PAPER_ALLOCATION_MODES,
                    source_dir=None, seed=20260913, token_budget=40000,
                    max_input_tokens=180000, attempts=2, repair_rounds=3):
    modes = list(dict.fromkeys(modes))
    if not modes or any(m not in PAPER_ALLOCATION_MODES for m in modes):
        raise ValueError("Select explicit supported allocation modes")
    positive_token_budget(token_budget)
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    if type(attempts) is not int or attempts < 1 or type(repair_rounds) is not int or repair_rounds < 0:
        raise ValueError("Invalid audit or repair budget")
    source = load_allocation_source(dataset, source_run, source_dir=source_dir)
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty output directory for an allocation audit run")
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "source.json", source)
    llm = import_module(f"teamgym.data.medical.{dataset}.llm")
    execution = import_module(f"teamgym.data.medical.{dataset}.execution")
    results, exported = [], []
    for mode in modes:
        directory = root / mode; directory.mkdir()
        cache = llm.ResponseCache(directory / "llm_cache.jsonl", max_input_tokens=max_input_tokens)
        identity = {"dependencies": cache.dependencies, "audit": llm.inference_identity(client),
                    "participant_context_token_budget": token_budget}
        _write(directory / "run_manifest.json", {"source_sha256": content_hash(source), "mode": mode,
            "seed": seed, "token_budget": token_budget, "max_input_tokens": max_input_tokens,
            "attempts": attempts, "repair_rounds": repair_rounds, "execution_identity": identity})
        report = {"schema": "medical_allocation_report_v1", "dataset": dataset, "mode": mode,
            "source_sha256": content_hash(source), "execution_identity": identity}
        try:
            partition = build_partition(source, mode=mode, seed=seed, token_budget=token_budget)
            partition, leakage = audit_packets(source, partition, client=client, cache=cache,
                root=directory, attempts=attempts, repair_rounds=repair_rounds)
            report.update(partition=partition, leakage_audit=leakage, status="packet_leakage_failed")
            if leakage["passed"]:
                answer_audit = audit_participants(source, partition, client=client, cache=cache,
                    root=directory, attempts=attempts)
                report.update(participant_audit=answer_audit,
                              status="accepted" if answer_audit["passed"] else "participant_answer_failed")
            execution.assert_frozen_dependencies(cache.dependencies)
            if any(file_hash(r["path"]) != r["sha256"] for r in source["source_artifacts"].values()):
                raise RuntimeError("Construction source artifacts changed during allocation audits")
            if report["status"] == "accepted":
                report["export_certification"] = certify_export(report, _proof_files(directory))
        except AllocationNotApplicable as exc:
            report.update(status="skipped", reason=str(exc))
        except Exception as exc:
            report.pop("export_certification", None)
            report.update(status="execution_error", error=f"{type(exc).__name__}: {exc}")
        _write(directory / "report.json", report)
        if report["status"] == "accepted":
            try:
                row = build_evaluation_datapoint(directory / "report.json")
                write_release_datapoints(directory / "common_datapoints.jsonl", [row])
                exported.append(row)
            except Exception as exc:
                report.pop("export_certification", None)
                report.update(status="export_failed", error=f"{type(exc).__name__}: {exc}")
                _write(directory / "report.json", report)
        results.append({"mode": mode, "status": report["status"], "report": str(directory / "report.json"),
                        **{k: report[k] for k in ("reason", "error") if k in report}})
        _write(root / "allocation_results.json", {"dataset": dataset, "source_sha256": content_hash(source), "results": results})
    write_release_datapoints(root / "common_datapoints.jsonl", exported)
    return {"dataset": dataset, "source_sha256": content_hash(source), "results": results}


def main():
    from teamgym.llm.construction_client import OpenAIJsonClient, OpenAIJsonClientConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("evidencebench", "mlcr"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", nargs="+", choices=(*PAPER_ALLOCATION_MODES, "all"), default=["all"])
    parser.add_argument("--audit-model", "--verification-model", dest="audit_model", default="gpt-5.5")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--provider"); parser.add_argument("--base-url"); parser.add_argument("--api-key-env")
    parser.add_argument("--participant-token-budget", type=int, default=40000)
    parser.add_argument("--max-input-tokens", type=int, default=180000)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--leakage-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if min(args.participant_token_budget, args.max_input_tokens, args.max_output_tokens, args.validation_attempts) < 1 or args.leakage_rounds < 0:
        parser.error("Token and validation budgets must be positive; repair rounds nonnegative")
    options = {k: getattr(args, k) for k in ("provider", "base_url", "api_key_env") if getattr(args, k)}
    client = OpenAIJsonClient(OpenAIJsonClientConfig(model=args.audit_model, env_file=args.env_file,
        temperature=None, reasoning_effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens,
        request_log_path=str(args.output_root / "llm_requests.jsonl"), **options))
    result = run_allocations(dataset=args.dataset, source_run=args.source_run, source_dir=args.source_dir,
        output_root=args.output_root, client=client, modes=PAPER_ALLOCATION_MODES if "all" in args.mode else args.mode,
        seed=args.seed, token_budget=args.participant_token_budget, max_input_tokens=args.max_input_tokens,
        attempts=args.validation_attempts, repair_rounds=args.leakage_rounds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(r["status"] in {"execution_error", "export_failed"} for r in result["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
