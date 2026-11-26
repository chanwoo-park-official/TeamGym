"""Independent EvidenceBench-derived PIU construction and audit pipeline."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any
from .evaluation import write_evaluation_jsonl
from .execution import assert_frozen_dependencies, export_artifacts
from teamgym.data.medical.certification import certify_export, positive_token_budget, construction_artifacts

from teamgym.data.medical.evidencebench.boundary import (
    boundary_case_outcome,
    display_materials,
    final_rius,
    run_boundaries,
    audit_participant_answers,
)
from teamgym.data.medical.evidencebench.llm import (
    ResponseCache,
    structured_call,
    write_json,
    inference_identity,
)
from teamgym.data.medical.evidencebench.partition import (
    audit_and_sanitize_packets,
    build_disjoint_packets,
    materialize_modes,
)
from teamgym.data.medical.evidencebench.source import (
    load_records,
    optimal_evidence_indices,
    sentence_blocks,
)
from teamgym.data.medical.evidencebench.validation import (
    string_list,
    validate_boundary_judgment,
    validate_candidates,
    validate_task_audit,
    validate_task_rubric,
)
from teamgym.llm.construction_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


SCHEMA = "evidencebench_riu_pipeline_v1"
DEFAULT_MODEL = "openai.gpt-5.5"
DEFAULT_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
DEFAULT_ENV_FILE = ".env"


def _source_snapshot(
    record: dict[str, Any],
    *,
    neighbor_radius: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indices = optimal_evidence_indices(record)
    blocks = sentence_blocks(record)
    by_index = {int(row["sentence_index"]): row for row in blocks}
    anchors: list[dict[str, Any]] = []
    for position, index in enumerate(indices, start=1):
        if index not in by_index:
            raise ValueError(f"optimal evidence sentence {index} is absent")
        neighbors = [
            by_index[value]
            for value in range(index - neighbor_radius, index + neighbor_radius + 1)
            if value in by_index
        ]
        anchors.append(
            {
                "anchor_id": f"anchor_{position:02d}",
                "sentence_index": index,
                "text": by_index[index]["text"],
                "sentence_type": by_index[index]["block_type"],
                "adjacent_context": [
                    {
                        "sentence_index": row["sentence_index"],
                        "text": row["text"],
                        "sentence_type": row["block_type"],
                    }
                    for row in neighbors
                ],
            }
        )
    snapshot = {
        "instance_id": record["instance_id"],
        "paper_id": record.get("paper_id"),
        "split": "test",
        "original_hypothesis": record["hypothesis"],
        "optimal_sentence_indices": indices,
        "paper_sentence_count": len(blocks),
    }
    return snapshot, anchors


def run_pipeline(
    *,
    source_path: str | Path,
    output_root: str | Path,
    generation_client: Any,
    verification_client: Any,
    instance_id: str | None = None,
    record_index: int = 0,
    neighbor_radius: int = 2,
    generation_attempts: int = 2,
    validation_attempts: int = 2,
    leakage_rounds: int = 3,
    seed: int = 20260830,
    source_revision: str | None = None,
    participant_token_budget: int = 40_000,
    max_input_tokens: int = 180_000,
    construction_only: bool = False,
) -> dict[str, Any]:
    positive_token_budget(participant_token_budget)
    if type(neighbor_radius) is not int or not 0 <= neighbor_radius <= 2:
        raise ValueError("neighbor_radius must be between 0 and 2 under the paper protocol")
    if min(generation_attempts, validation_attempts) < 1 or leakage_rounds < 0:
        raise ValueError("generation/validation attempts must be positive and repair budget nonnegative")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(output / "llm_cache.jsonl", max_input_tokens=max_input_tokens)
    trace_root = output / "_llm_traces"
    records = load_records(source_path, split="test")
    if instance_id:
        matches = [row for row in records if str(row["instance_id"]) == instance_id]
        if not matches:
            raise ValueError(f"unknown EvidenceBench instance_id: {instance_id}")
        record = matches[0]
    else:
        if record_index < 0 or record_index >= len(records):
            raise ValueError(f"record_index out of range: {record_index}")
        record = records[record_index]
    instance_id = str(record["instance_id"])
    snapshot, anchors = _source_snapshot(record, neighbor_radius=neighbor_radius)
    from .source import SPLIT_FILES, _sha256_file
    source_file = Path(source_path).expanduser().resolve() / SPLIT_FILES["test"]
    lineage = {"source_file": str(source_file), "source_file_sha256": _sha256_file(source_file),
               "source_revision": source_revision, "split": "test", "instance_id": instance_id,
               "paper_id": record.get("paper_id"), "record_sha256": hashlib.sha256(
                   json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    snapshot["source_lineage"] = lineage
    execution_identity = {"dependencies": cache.dependencies, "generation": inference_identity(generation_client),
                          "verification": inference_identity(verification_client), "participant_context_token_budget": participant_token_budget}
    package_root = Path(__file__).parent
    prompt_root = package_root.parents[2] / "prompts" / "medical" / "evidencebench"
    code_prompt_files = [*package_root.glob("*.py"), *prompt_root.glob("*.txt")]
    write_json(output / "run_manifest.json", {"schema": "evidencebench_run_manifest_v2", "execution_identity": execution_identity,
        "source_lineage": lineage, "generation": inference_identity(generation_client),
        "verification": inference_identity(verification_client), "seed": seed,
        "limits": {"generation_attempts": generation_attempts, "validation_attempts": validation_attempts,
                   "leakage_repairs": leakage_rounds, "neighbor_radius": neighbor_radius,
                   "participant_token_budget": participant_token_budget, "max_input_tokens": max_input_tokens},
        "code_prompt_sha256": {str(p): _sha256_file(p) for p in code_prompt_files}})
    write_json(output / "00_source.json", {"snapshot": snapshot, "anchors": anchors})

    candidates_response, errors = structured_call(
        client=generation_client,
        cache=cache,
        trace_root=trace_root,
        instance_id=instance_id,
        stage="normalize_candidates",
        user_prompt_name="01_normalize_candidates_user.txt",
        variables={"SOURCE_RECORD_JSON": snapshot, "ANCHORS_JSON": anchors},
        validator=lambda row: validate_candidates(row, record=record, anchors=anchors),
        attempts=validation_attempts,
    )
    write_json(output / "01_candidates.json", {"response": candidates_response, "errors": errors})
    if errors:
        summary = {
            "schema": SCHEMA,
            "instance_id": instance_id,
            "status": "candidate_normalization_failed",
            "errors": errors,
        }
        write_json(output / "summary.json", summary)
        return summary
    candidates = candidates_response["candidate_rius"]

    feedback: Any = "None"
    task_rubric: dict[str, Any] = {}
    task_audit: dict[str, Any] = {}
    boundaries: dict[str, Any] = {}
    selected_candidates: list[dict[str, Any]] = []
    accepted_attempt: int | None = None
    generation_history: list[dict[str, Any]] = []
    for attempt in range(1, generation_attempts + 1):
        task_rubric, task_errors = structured_call(
            client=generation_client,
            cache=cache,
            trace_root=trace_root,
            instance_id=instance_id,
            stage=f"generate_task_rubric.generation_{attempt}",
            user_prompt_name="02_generate_task_rubric_user.txt",
            variables={
                "ORIGINAL_HYPOTHESIS_JSON": {"hypothesis": record["hypothesis"]},
                "CANDIDATES_JSON": candidates,
            },
            validator=lambda row: validate_task_rubric(row, candidates=candidates),
            attempts=validation_attempts,
            initial_feedback=feedback,
        )
        if task_errors:
            generation_history.append({"attempt": attempt, "stage": "task_generation", "errors": task_errors})
            feedback = task_errors
            continue
        selected_ids = set(string_list(task_rubric["selected_candidate_ids"]))
        selected_candidates = [row for row in candidates if str(row["candidate_id"]) in selected_ids]
        task_audit, audit_errors = structured_call(
            client=verification_client,
            cache=cache,
            trace_root=trace_root,
            instance_id=instance_id,
            stage=f"audit_task_rubric.generation_{attempt}",
            user_prompt_name="03_audit_task_rubric_user.txt",
            variables={
                "TASK_RUBRIC_JSON": task_rubric,
                "NORMALIZED_CANDIDATES_JSON": candidates,
            },
            validator=lambda row: validate_task_audit(row, task_rubric=task_rubric),
            attempts=validation_attempts,
        )
        if audit_errors:
            generation_history.append({"attempt": attempt, "stage": "task_audit", "errors": audit_errors})
            feedback = {"task_audit": task_audit, "errors": audit_errors}
            continue
        boundaries = run_boundaries(
            instance_id=instance_id,
            task_rubric=task_rubric,
            selected_candidates=selected_candidates,
            client=verification_client,
            cache=cache,
            trace_root=trace_root,
            attempts=validation_attempts,
            seed=seed + attempt * 1000,
        )
        if not boundaries["passed"]:
            generation_history.append({"attempt": attempt, "stage": "boundaries", "boundaries": boundaries})
            feedback = {
                "boundary_failures": [
                    {"case_id": row["case_id"], "outcome": row["outcome"], "errors": row["errors"]}
                    for row in boundaries["cases"]
                    if row["outcome"].get("condition_ok") is not True
                ]
            }
            continue
        accepted_attempt = attempt
        break

    write_json(output / "02_task_rubric.json", task_rubric)
    write_json(output / "03_task_audit.json", task_audit)
    write_json(output / "04_boundaries.json", boundaries)
    write_json(output / "generation_history.json", generation_history)
    if accepted_attempt is None:
        summary = {
            "schema": SCHEMA,
            "instance_id": instance_id,
            "status": "riu_boundary_audit_failed",
            "generation_attempts": generation_history,
        }
        write_json(output / "summary.json", summary)
        return summary

    accepted_rius = final_rius(
        task_rubric=task_rubric,
        selected_candidates=selected_candidates,
        boundaries=boundaries,
    )
    write_json(output / "05_final_rius.hidden.json", accepted_rius)
    assert_frozen_dependencies(cache.dependencies)
    qualified = {"status": "construction_qualified", "task": task_rubric, "units": accepted_rius,
                 "source_lineage": lineage, "execution_identity": execution_identity}
    qualified["export_certification"] = certify_export(qualified, construction_artifacts(output, "evidencebench"))
    write_json(output / "qualified_construction.json", qualified)
    if construction_only:
        summary = {"schema": SCHEMA, "instance_id": instance_id, "status": "construction_qualified",
                   "source_lineage": lineage, "execution_identity": execution_identity,
                   "final_riu_count": len(accepted_rius)}
        write_json(output / "summary.json", summary)
        return summary
    partition = build_disjoint_packets(record=record, final_rius=accepted_rius, seed=seed)
    partition["participant_token_budget"] = participant_token_budget
    sanitized_partition, leakage = audit_and_sanitize_packets(
        instance_id=instance_id,
        public_task=task_rubric["public_task"],
        final_rius=accepted_rius,
        partition=partition,
        client=verification_client,
        cache=cache,
        trace_root=trace_root,
        attempts=validation_attempts,
        max_rounds=leakage_rounds,
    )
    write_json(output / "06_participant_packets.hidden.json", sanitized_partition)
    write_json(output / "07_leakage_audit.json", leakage)
    if not leakage["passed"]:
        summary = {
            "schema": SCHEMA,
            "instance_id": instance_id,
            "status": "partition_leakage_failed",
            "final_riu_count": len(accepted_rius),
            "errors": leakage["errors"],
        }
        write_json(output / "summary.json", summary)
        return summary
    participant_answers = audit_participant_answers(
        instance_id=instance_id, task_rubric=task_rubric, partition=sanitized_partition,
        client=verification_client, cache=cache, trace_root=trace_root,
        attempts=validation_attempts)
    write_json(output / "09_participant_final_answer_audit.json", participant_answers)
    if not participant_answers["passed"]:
        summary = {"schema": SCHEMA, "instance_id": instance_id,
                   "status": "participant_final_answer_audit_failed", "source_lineage": lineage}
        write_json(output / "summary.json", summary)
        return summary
    modes = materialize_modes(
        final_rius=accepted_rius,
        partition=sanitized_partition,
        public_task=task_rubric["public_task"],
        seed=seed,
    )
    write_json(output / "08_benchmark_modes.json", modes)
    assert_frozen_dependencies(cache.dependencies)
    summary = {
        "schema": SCHEMA,
        "execution_identity": execution_identity,
        "source_lineage": lineage,
        "instance_id": instance_id,
        "status": "accepted",
        "accepted_generation_attempt": accepted_attempt,
        "generation_model": generation_client.cfg.model,
        "verification_model": verification_client.cfg.model,
        "original_optimal_anchor_count": len(anchors),
        "normalized_candidate_count": len(candidates),
        "final_riu_count": len(accepted_rius),
        "boundary_case_count": len(boundaries["cases"]),
        "participant_count": len(sanitized_partition["participants"]),
        "paper_context_count": len(sentence_blocks(record)),
        "retained_context_count": len(sanitized_partition["contexts"]),
        "removed_leakage_context_count": len(leakage["removed_context_ids"]),
        "leakage_round_count": len(leakage["rounds"]),
        "participant_final_answer_audit_passed": participant_answers["passed"],
    }
    summary["export_certification"] = certify_export(summary, export_artifacts(output))
    write_json(output / "summary.json", summary)
    write_evaluation_jsonl([output], output / "common_datapoints.jsonl")
    return summary


def _parse_args() -> argparse.Namespace:
    from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--instance-id")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--source-revision")
    parser.add_argument("--neighbor-radius", type=int, default=2)
    parser.add_argument("--generation-attempts", type=int, default=2)
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--leakage-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audit-model", "--verification-model", dest="verification_model",
                        help="Model for solver/judge, grounding and leakage checks; defaults to --model.")
    parser.add_argument("--participant-token-budget", type=int, default=40_000)
    parser.add_argument("--allocation-modes", nargs="+", choices=(*PAPER_ALLOCATION_MODES, "all"),
                        help="Audit additional distributions of the fixed task; 'all' requests all five modes.")
    parser.add_argument("--construction-only", action="store_true",
                        help="Freeze task/PIUs after grounding and boundary checks, before distribution.")
    parser.add_argument("--max-input-tokens", type=int, default=180_000)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    args = parser.parse_args()
    if min(args.participant_token_budget, args.max_input_tokens) < 1:
        parser.error("--participant-token-budget and --max-input-tokens must be positive")
    return args


def main() -> None:
    args = _parse_args()
    output = Path(args.output_root).expanduser().resolve()
    generation_client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            model=args.model,
            base_url=args.base_url,
            endpoint="/responses",
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            temperature=None,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(output / "llm_requests.jsonl"),
        )
    )
    verification_client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider="bedrock_openai",
            model=args.verification_model or args.model,
            base_url=args.base_url,
            endpoint="/responses",
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            temperature=None,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(output / "llm_requests.jsonl"),
        )
    )
    summary = run_pipeline(
        source_path=args.source_path,
        output_root=output,
        generation_client=generation_client,
        verification_client=verification_client,
        instance_id=args.instance_id,
        record_index=args.record_index,
        neighbor_radius=args.neighbor_radius,
        generation_attempts=args.generation_attempts,
        validation_attempts=args.validation_attempts,
        leakage_rounds=args.leakage_rounds,
        seed=args.seed,
        source_revision=args.source_revision,
        participant_token_budget=args.participant_token_budget,
        max_input_tokens=args.max_input_tokens,
        construction_only=args.construction_only or bool(args.allocation_modes),
    )
    result = summary
    if args.allocation_modes:
        from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
        from teamgym.data.medical.allocations import run_allocations
        allocations = run_allocations(dataset="evidencebench", source_run=output,
            output_root=output / "allocations", client=verification_client,
            modes=PAPER_ALLOCATION_MODES if "all" in args.allocation_modes else args.allocation_modes,
            seed=args.seed, token_budget=args.participant_token_budget, max_input_tokens=args.max_input_tokens,
            attempts=args.validation_attempts, repair_rounds=args.leakage_rounds)
        result = {"construction": summary, "allocations": allocations}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
