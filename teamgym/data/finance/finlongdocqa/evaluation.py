"""Versioned evaluator-metadata export from a frozen fresh FinLongDocQA producer.

The producer's native claim rubric is already used by its live construction
checks. This deterministic stage maps that same rubric to the generic written
answer grader, while preserving every source/public/model observation byte.
It does not edit the producer manifest or reuse an old task as new generation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from teamgym.core.schema import validate_common_datapoint
from teamgym.interaction.llm_actors import LLMParticipantActor, PromptTraceRecorder
from teamgym.interaction.rui_context import rui_only_datapoint
from teamgym.interaction.run_single_agent_rui_evaluation import run_single_agent_rui_episode
from teamgym.interaction.types import BudgetSpec
from teamgym.scoring.short_answers import judge_one_answer

from .provenance import REPO_ROOT, digest, file_digest, read_commits, source_identity
from .runtime import write_json, write_jsonl
from .source import load_qa_rows, group_qas_by_report

SCHEMA = "finlongdocqa_generic_grader_export_v1"


def evaluator_datapoint(common: dict) -> dict:
    """Add only evaluator metadata; keep all native criteria and source links."""
    out = deepcopy(common)
    rubric = deepcopy(out["dataset_specific"]["free_response_rubric"])
    claims = rubric["required_claims"]
    ids = [row["claim_id"] for row in claims]
    if not claims or len(set(ids)) != len(ids):
        raise ValueError("native claim rubric must be nonempty and unique")
    known_units = {row["rui_id"] for row in out["reference_information_units"]}
    for row in claims:
        if not row.get("description") or not row.get("expected_answer_element") or not row.get("grading_rule"):
            raise ValueError("native claim criterion is incomplete")
        if not row.get("supporting_unit_ids") or not set(row["supporting_unit_ids"]) <= known_units:
            raise ValueError("native claim source links are invalid")
    oracle = out["oracle"]
    canonical = oracle.get("answer_text") or oracle.get("canonical_response")
    if not canonical:
        raise ValueError("frozen native reference answer missing")
    steps = [{"step": index, "step_id": row["claim_id"], "claim_id": row["claim_id"],
              "claim": row["description"], "expected_answer_element": row["expected_answer_element"],
              "grading_rule": row["grading_rule"], "uses": list(row["supporting_unit_ids"])}
             for index, row in enumerate(claims, 1)]
    oracle.update(canonical_response=canonical, required_decision_steps=steps,
                  judge_rubric={**rubric, "require_all_steps_for_pass": True, "passing_score": 1.0},
                  disqualifying_errors=deepcopy(rubric.get("prohibited_errors") or []),
                  pass_policy="all_required_decision_steps",
                  semantic_judge={"enabled": True, "primary_answer_field": "written_answer",
                                  "fallback_answer_fields": ["rationale"]})
    out.setdefault("generation_metadata", {})["evaluator_export"] = {
        "schema": SCHEMA, "native_rubric_sha256": digest(rubric),
        "model_observations_unchanged": True, "producer_manifest_rewritten": False}
    if not validate_common_datapoint(out)["valid"]:
        raise ValueError("invalid evaluator datapoint")
    # Whitelist the entire transform, not just selected prompt strings.
    before = deepcopy(common)
    stripped = deepcopy(out)
    for key in ("canonical_response", "required_decision_steps", "judge_rubric", "disqualifying_errors", "pass_policy", "semantic_judge"):
        if key in before["oracle"]:
            stripped["oracle"][key] = deepcopy(before["oracle"][key])
        else:
            stripped["oracle"].pop(key, None)
    stripped["generation_metadata"].pop("evaluator_export")
    if stripped != before:
        raise ValueError("grader export modified non-evaluator data")
    return out


class _CaptureClient:
    last_request_meta = {"offline_capture": True}
    def generate_json(self, **kwargs):
        return {"utterance": "Capture only.", "answer": "free_response",
                "written_answer": "Capture only.", "confidence": 0}


def capture_model_observations(row: dict, *, budget: int = 40000) -> list[dict]:
    captures = []
    for mode, view in (("multi_agent_full_context", row), ("multi_agent_piu_only", rui_only_datapoint(row))):
        for participant in view["participants"]:
            recorder = PromptTraceRecorder()
            LLMParticipantActor(_CaptureClient(), recorder).respond(datapoint=view,
                participant_id=participant["participant_id"], state={"turn": 0, "messages": []}, request=None,
                budgets=BudgetSpec(participant_context_token_budget=budget))
            request = recorder.prompt_records[0]
            captures.append({"mode": mode, "participant_id": participant["participant_id"],
                             **{key: request[key] for key in ("system_prompt", "user_prompt")}})
    recorder = PromptTraceRecorder()
    run_single_agent_rui_episode(row, client=_CaptureClient(), recorder=recorder,
        args=argparse.Namespace(context_token_budget=budget, final_reasoning_token_budget=256,
                                final_rationale_token_budget=120))
    captures.append({"mode": "single_agent_all_piu", "participant_id": "single_agent",
                     **{key: recorder.prompt_records[0][key] for key in ("system_prompt", "user_prompt")}})
    return captures


def capture_grader_payload(row: dict, answer: str) -> dict:
    class Captured(Exception):
        pass
    class Client:
        def generate_json(self, **kwargs):
            self.request = kwargs
            raise Captured()
    client = Client()
    try:
        judge_one_answer(client=client, datapoint=row, answer_row={"answer_text": answer}, trace_dir=None, index=1)
    except Captured:
        return {"system_prompt": client.request["system_prompt"],
                "payload": json.loads(client.request["user_prompt"])}
    raise AssertionError("actual grader did not invoke its structured client")


def _execution_identity() -> dict:
    package = Path(__file__).parent
    paths = [*package.glob("*.py"), *(REPO_ROOT / "teamgym/llm").glob("*.py"),
             *(REPO_ROOT / "teamgym/core").glob("*.py"), *(REPO_ROOT / "teamgym/interaction").glob("*.py"),
             *(REPO_ROOT / "teamgym/scoring").glob("*.py"), REPO_ROOT / "teamgym/tokenization.py",
             REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"]
    packages = {}
    for name in ("openai", "tiktoken", "httpx", "pydantic", "pyarrow", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    from teamgym.tokenization import TOKENIZER, encoding
    vocab = hashlib.sha256()
    for token, rank in sorted(encoding()._mergeable_ranks.items(), key=lambda item: item[1]):
        vocab.update(len(token).to_bytes(4, "big") + token + rank.to_bytes(8, "big"))
    identity = {"files": {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in sorted(set(paths))},
                "packages": packages, "python": sys.version,
                "tokenizer": {"name": TOKENIZER, "vocabulary_sha256": vocab.hexdigest()}}
    return {**identity, "sha256": digest(identity)}


def _producer_evidence(run: Path) -> tuple[dict, dict, dict]:
    generated = run / "generated"
    manifest = json.loads((generated / "run_manifest.json").read_text())
    expected = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if digest(expected) != manifest["fingerprint"]:
        raise ValueError("producer manifest fingerprint mismatch")
    snapshot = {}
    for relative, sha in manifest["code_and_prompts"].items():
        stored = run / "code_snapshot" / relative
        if file_digest(stored) != sha or file_digest(REPO_ROOT / relative) != sha:
            raise ValueError(f"frozen producer dependency changed: {relative}")
        snapshot[relative] = sha
    raw_rows = [json.loads(line) for line in (generated / "common_datapoints.jsonl").read_text().splitlines() if line.strip()]
    if len(raw_rows) != 1:
        raise ValueError("release stage expects exactly one frozen accepted producer")
    raw = raw_rows[0]
    rid = raw["source"]["report_id"]
    commits, invalid = read_commits(generated, manifest["fingerprint"], [rid])
    accepted = [r for r in commits if r["result"].get("status") == "accepted"]
    if invalid or len(accepted) != 1 or accepted[0]["common"] != raw:
        raise ValueError("raw producer export disagrees with accepted atomic commit")
    result = accepted[0]["result"]
    if not result["solver_audit"].get("accept") or result.get("errors"):
        raise ValueError("producer lacks passed qualification")
    units = {row["unit_id"] for row in result["task"]["evidence_units"]}
    solver = result["solver_audit"]
    if solver["full_evidence"]["judge_result"].get("overall_complete_and_correct") is not True or solver["no_evidence"]["judge_result"].get("overall_complete_and_correct") is not False:
        raise ValueError("producer ALL/NONE gate is incomplete")
    loo = solver["leave_one_out"]
    if len(loo) != len(units) or {row["omitted_unit_id"] for row in loo} != units or any(not row["broken_claim_ids"] for row in loo):
        raise ValueError("producer leave-one-out coverage is incomplete")
    pids = {row["participant_id"] for row in raw["participants"]}
    local = result["initial_view_audits"]
    expected_local = {(mode, pid) for mode in ("multi_agent_full_context", "multi_agent_piu_only") for pid in pids}
    if len(local) != len(expected_local) or {(r["mode"], r["participant_id"]) for r in local} != expected_local or any(
            r["context_token_budget"] != 40000 or r["judge_result"].get("overall_complete_and_correct") is not False for r in local):
        raise ValueError("producer participant-only validation is incomplete")
    salience = result["packet_salience_audits"]
    if len(salience) != len(pids) or {r["participant_id"] for r in salience} != pids or any(r["errors"] or any(
            r["audit"][key] is not False for key in ("leaks_constructor_metadata", "reveals_answer_in_identity", "marks_private_piu_as_important")) for r in salience):
        raise ValueError("producer salience gate is incomplete")
    audits = result["packet_leakage_audits"]
    last_round = max(r["round"] for r in audits)
    final = [r for r in audits if r["round"] == last_round]
    if len(final) != len(pids) or {r["participant_id"] for r in final} != pids or any(
            q["reconstructable_from_packet"] is not False for r in final for q in r["audit"]["foreign_rui_results"]):
        raise ValueError("producer final whole-packet gate is incomplete")
    from .packet_budget import certify_materialized_views
    checked = deepcopy(raw)
    certify_materialized_views(checked, token_budget=40000)
    if checked != raw:
        raise ValueError("producer source budget receipt is stale")
    index = json.loads((generated / "report_block_indexes.jsonl").read_text())
    sources = source_identity(argparse.Namespace(qa_path=run / "source/dataset_qa.jsonl", reports_zip=run / "source/reports.zip"),
        dict(group_qas_by_report(load_qa_rows(run / "source/dataset_qa.jsonl"))), {rid: index}, [rid])
    if sources != manifest["sources"]:
        raise ValueError("producer source changed")
    inputs = {str(p.relative_to(run)): file_digest(p) for p in sorted((generated / "_llm_traces").rglob("*"))
              if p.is_file() and (p.name.endswith(".user.txt") or p.name.endswith(".system.txt"))}
    return raw, manifest, {"producer_manifest_sha256": file_digest(generated / "run_manifest.json"),
        "producer_fingerprint": manifest["fingerprint"], "producer_snapshot_sha256": digest(snapshot),
        "producer_snapshot_files": snapshot, "raw_common_sha256": file_digest(generated / "common_datapoints.jsonl"),
        "accepted_commit_file_sha256": file_digest(generated / "_commits" / rid / f"{accepted[0]['attempt_number']:05d}.json"),
        "accepted_commit_payload_sha256": digest(accepted[0]), "live_model_input_files": inputs,
        "live_model_inputs_sha256": digest(inputs)}


def write_release(run: str | Path, output: str | Path) -> dict:
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("release output must be a new empty directory")
    raw, manifest, producer = _producer_evidence(run)
    before_identity = _execution_identity()
    final = evaluator_datapoint(raw)
    before, after = capture_model_observations(raw), capture_model_observations(final)
    saved = json.loads((run / "consumer_inputs.json").read_text())
    frozen_requests = {(r["mode"], r["participant_id"]): {key: r[key] for key in ("system_prompt", "user_prompt")}
                       for r in saved["inputs"]}
    if saved.get("materialized_context_token_budget") != 40000 or len(frozen_requests) != len(before) or any(
            frozen_requests.get((r["mode"], r["participant_id"])) != {key: r[key] for key in ("system_prompt", "user_prompt")} for r in before):
        raise ValueError("current producer observations differ from frozen verified live-equivalent observations")
    if before != after:
        raise ValueError("evaluator metadata changed actual model observations")
    canonical = raw["oracle"]["answer_text"]
    grader = capture_grader_payload(final, canonical)
    p = grader["payload"]
    claims = raw["dataset_specific"]["free_response_rubric"]["required_claims"]
    if p["canonical_response"] != canonical or p["pass_policy"]["require_all_steps_for_pass"] is not True:
        raise ValueError("actual generic grader has wrong canonical answer or policy")
    if p["judge_rubric"]["required_claims"] != claims or len(p["required_decision_steps"]) != len(claims):
        raise ValueError("actual generic grader lost native criteria")
    if _execution_identity() != before_identity or _producer_evidence(run)[2] != producer:
        raise ValueError("producer or export dependencies changed during release")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "common_datapoints.jsonl", [final])
    write_json(output / "consumer_model_inputs.json", after)
    write_json(output / "generic_grader_payload.json", grader)
    receipt = {"schema": SCHEMA, "status": "verified_deterministic_evaluator_export",
        "producer": producer, "export_execution_identity": before_identity,
        "transform_scope": ["oracle evaluator metadata", "generation_metadata.evaluator_export"],
        "raw_producer_identity_rewritten": False, "new_generation_claimed": False,
        "native_rubric_sha256": digest(raw["dataset_specific"]["free_response_rubric"]),
        "frozen_producer_consumer_inputs_sha256": file_digest(run / "consumer_inputs.json"),
        "frozen_live_equivalent_observations_preserved": True,
        "model_observations_byte_equal": True, "all_live_producer_model_input_files_unchanged": True,
        "captured_observation_count": len(after), "context_token_budget": 40000,
        "outputs": {name: file_digest(output / name) for name in
            ("common_datapoints.jsonl", "consumer_model_inputs.json", "generic_grader_payload.json")}}
    write_json(output / "release_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_release(args.producer_run, args.output_dir)
    print(json.dumps({"status": result["status"], "producer_fingerprint": result["producer"]["producer_fingerprint"],
                      "same_model_observations": result["model_observations_byte_equal"], "export_sha256": result["outputs"]["common_datapoints.jsonl"]}))


if __name__ == "__main__":
    main()
