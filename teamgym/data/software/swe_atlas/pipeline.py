"""Build audited collaborative SWE-Atlas QA benchmark material from PIU candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import subprocess
import tarfile
from typing import Any, Iterable

from teamgym.tokenization import count_tokens as estimate_tokens
from .export_common import build_evaluation_rows, write_evaluation_rows
from teamgym.llm.construction_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)

from .candidate import (
    CandidateCollectionExhausted,
    CandidateValidationError,
    DockerEvidenceCollector,
    _artifact_evidence_content,
    _artifact_for_prompt,
    _validate_runtime_command,
    generate_riu_candidates,
)
from .schema import Rubric, SWEAtlasQATask
from .source import load_qa_task
from .integrity import (atomic_json, canonical_hash, prepare_run, run_identity,
                        validate_checkpoint, write_checkpoint, write_ready_receipt)


SCHEMA = "swe_atlas_collaborative_pipeline_v1"
PROMPT_DIR = (
    Path(__file__).resolve().parents[3] / "prompts" / "software" / "swe_atlas"
)
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html", ".ini",
    ".java", ".js", ".jsx", ".json", ".kt", ".md", ".mod", ".proto",
    ".py", ".repo", ".rs", ".scss", ".sh", ".sql", ".sum", ".toml",
    ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
TEXT_FILENAMES = {
    ".dockerignore", ".editorconfig", ".gitignore", "CODEOWNERS", "Dockerfile",
    "GNUmakefile", "Makefile",
}
EXCLUDED_PREFIXES = (
    ".git/", "node_modules/", "vendor/", "dist/", "build/", "target/",
    ".cache/", ".venv/", "venv/",
)


class PipelineError(ValueError):
    """Raised when a construction stage cannot satisfy its hard contract."""


class CanonicalOverlapError(PipelineError):
    """Raised when owned canonical evidence reconstructs a foreign RIU."""

    def __init__(self, merge_groups: list[list[str]], audits: list[dict[str, Any]]):
        super().__init__(f"Canonical RIU overlap requires upstream regrouping: {merge_groups}")
        self.merge_groups = merge_groups
        self.audits = audits


class CanonicalArtifactOverlapError(PipelineError):
    """Raised when proposed RIUs still share one full-context artifact."""

    def __init__(self, merge_groups: list[list[str]]):
        super().__init__(
            "Canonical artifacts span multiple RIUs and require evidence reselection: "
            f"{merge_groups}"
        )
        self.merge_groups = merge_groups


@dataclass(frozen=True)
class PipelineConfig:
    allocation_policy: str = "one_each"
    participant_count: int | None = None
    participant_context_token_cap: int = 40_000
    audit_repair_attempts: int = 2
    # Number of retries after the initial RIU selection.
    riu_reselection_attempts: int = 3
    candidate_runtime_repair_attempts: int = 1
    experiment_repair_attempts: int = 2
    leakage_repair_rounds: int = 3
    max_workers: int = 8
    min_distractor_logs: int = 2
    leakage_scan_threshold_chars: int = 300_000
    leakage_chunk_chars: int = 80_000
    seed: int = 0
    stop_after_riu_audit: bool = False

    def checked(self) -> "PipelineConfig":
        policy = str(self.allocation_policy)
        if policy not in {"one_each", "fixed_team_balanced"}:
            raise PipelineError(f"Unsupported allocation policy: {policy}")
        count = self.participant_count
        if policy == "fixed_team_balanced" and (count is None or int(count) < 2):
            raise PipelineError("fixed_team_balanced requires participant_count >= 2")
        return PipelineConfig(
            allocation_policy=policy,
            participant_count=None if count is None else int(count),
            participant_context_token_cap=max(1, int(self.participant_context_token_cap)),
            audit_repair_attempts=max(0, int(self.audit_repair_attempts)),
            riu_reselection_attempts=max(0, int(self.riu_reselection_attempts)),
            candidate_runtime_repair_attempts=max(
                0, int(self.candidate_runtime_repair_attempts)
            ),
            experiment_repair_attempts=max(0, int(self.experiment_repair_attempts)),
            leakage_repair_rounds=max(1, int(self.leakage_repair_rounds)),
            max_workers=max(1, int(self.max_workers)),
            min_distractor_logs=max(0, int(self.min_distractor_logs)),
            leakage_scan_threshold_chars=max(1, int(self.leakage_scan_threshold_chars)),
            leakage_chunk_chars=max(1, int(self.leakage_chunk_chars)),
            seed=int(self.seed),
            stop_after_riu_audit=bool(self.stop_after_riu_audit),
        )


def build_task(
    *,
    task: SWEAtlasQATask,
    output_dir: str | Path,
    client: OpenAIJsonClient,
    config: PipelineConfig,
    candidate_dir: str | Path | None = None,
    docker_timeout_sec: int = 900,
) -> dict[str, Any]:
    """Run construction through three-mode materialization for one QA task."""

    cfg = config.checked()
    output = Path(output_dir).expanduser().resolve()
    identity = run_identity(task, client, cfg, docker_timeout_sec=docker_timeout_sec)
    prepare_run(output, identity)
    execution_task = replace(task, runtime=replace(task.runtime, docker_image=identity['image']['id']))
    _write_json(
        output / "manifest.started.json",
        {
            "schema": SCHEMA,
            "task_id": task.task_id,
            "started_at_utc": _now(),
            "model": client.cfg.model,
            "config": cfg.__dict__,
            "stages": [],
        },
    )

    initial_candidate_root = (
        Path(candidate_dir).expanduser().resolve()
        if candidate_dir
        else output / "03_candidates"
    )
    if (initial_candidate_root / "riu_candidates.json").is_file():
        if initial_candidate_root != output / '03_candidates':
            bound = _read_json(initial_candidate_root / 'input_identity.json')
            if bound != identity:
                raise PipelineError('External candidate bundle belongs to different construction inputs')
            validate_checkpoint(initial_candidate_root, identity)
        candidate_result = _read_json(initial_candidate_root / "riu_candidates.json")
        artifacts = _read_json(initial_candidate_root / "evidence_artifacts.json")["artifacts"]
        candidate_source = "reused_verified_candidate_bundle"
    else:
        candidate_result = generate_riu_candidates(
            task=execution_task,
            output_dir=initial_candidate_root,
            client=client,
            docker_timeout_sec=docker_timeout_sec,
            runtime_repair_attempts=cfg.candidate_runtime_repair_attempts,
        )
        artifacts = _read_json(initial_candidate_root / "evidence_artifacts.json")["artifacts"]
        candidate_source = "generated_in_pipeline"
    candidate_result, artifacts = _scope_candidate_bundle(candidate_result, artifacts)
    _write_json(initial_candidate_root / 'riu_candidates.json', candidate_result)
    _write_json(initial_candidate_root / 'evidence_artifacts.json', {'artifacts': artifacts})
    _validate_candidate_bundle(task, candidate_result, artifacts)

    final_rius: list[dict[str, Any]] = []
    audit_summary: dict[str, Any] = {}
    final_riu_checkpoint = output / "05_final_rius" / "final_rius.hidden.json"
    audit_checkpoint = output / "05_final_rius" / "audit_summary.json"
    if final_riu_checkpoint.is_file() and audit_checkpoint.is_file():
        checkpoint_audit = _read_json(audit_checkpoint)
        if checkpoint_audit.get("accept") is True:
            final_rius = _read_json(final_riu_checkpoint)["rius"]
            audit_summary = checkpoint_audit
            artifacts = _read_json(output / '05_final_rius/selected_artifacts.hidden.json')['artifacts']

    reselection_feedback: dict[str, Any] = {}
    reselection_history: list[dict[str, Any]] = []
    if not final_rius:
        selection_succeeded = False
        for selection_round in range(cfg.riu_reselection_attempts + 1):
            selection_dir = output / "03_candidates" / f"selection_{selection_round:02d}"
            selection_error = ""
            audit_summary = {}
            final_rius = []
            if selection_round:
                try:
                    candidate_result = generate_riu_candidates(
                        task=execution_task,
                        output_dir=selection_dir,
                        client=client,
                        docker_timeout_sec=docker_timeout_sec,
                        runtime_repair_attempts=cfg.candidate_runtime_repair_attempts,
                        construction_feedback=reselection_feedback,
                    )
                except CandidateCollectionExhausted as exc:
                    selection_error = f"candidate collection failed: {exc}"
                else:
                    artifacts = _read_json(selection_dir / "evidence_artifacts.json")["artifacts"]
                    candidate_result, artifacts = _scope_candidate_bundle(candidate_result, artifacts)
                    _write_json(selection_dir / 'riu_candidates.json', candidate_result)
                    _write_json(selection_dir / 'evidence_artifacts.json', {'artifacts': artifacts})
                    candidate_source = f"{candidate_source}+rubric_guided_reselection_{selection_round}"
                    _validate_candidate_bundle(task, candidate_result, artifacts)
            if not selection_error:
                _write_json(selection_dir / "candidate_snapshot.json", candidate_result)
                _write_json(selection_dir / "artifact_snapshot.json", {"artifacts": artifacts})

            audit_feedback: dict[str, Any] = {
                "candidate_reselection_feedback": reselection_feedback
            } if reselection_feedback else {}
            audit_attempts = 0 if selection_error else cfg.audit_repair_attempts + 1
            for attempt in range(audit_attempts):
                attempt_dir = (
                    output / "04_riu_audit"
                    / f"selection_{selection_round:02d}"
                    / f"attempt_{attempt:02d}"
                )
                consolidation = _consolidate_candidates(
                    client=client,
                    task=task,
                    candidate_result=candidate_result,
                    artifacts=artifacts,
                    audit_feedback=audit_feedback,
                    trace_dir=attempt_dir / "consolidation",
                )
                try:
                    final_rius = validate_consolidation(
                        consolidation,
                        task=task,
                        candidates=candidate_result["candidates"],
                        artifacts=artifacts,
                        enforce_artifact_exclusivity=False,
                    )
                except CanonicalArtifactOverlapError as exc:
                    audit_feedback = {
                        "errors": [str(exc)],
                        "overlap_groups": exc.merge_groups,
                        "instruction": (
                            "Do not merge yet. Reselect member evidence or grouping so each "
                            "full source file/runtime artifact belongs to only one RIU."
                        ),
                        "previous_raw_consolidation": consolidation,
                    }
                    _write_json(attempt_dir / "validation_error.json", audit_feedback)
                    if attempt < cfg.audit_repair_attempts:
                        continue
                    # A physical merge is permitted only after all regrouping attempts.
                    final_rius = validate_consolidation(
                        consolidation,
                        task=task,
                        candidates=candidate_result["candidates"],
                        artifacts=artifacts,
                        allow_artifact_merge=True,
                    )
                    _write_json(
                        attempt_dir / "conservative_merge_fallback.json",
                        {"merge_groups": exc.merge_groups, "reason": "regrouping attempts exhausted"},
                    )
                except PipelineError as exc:
                    selection_error = f"consolidation validation failed: {exc}"
                    audit_feedback = {
                        "errors": [selection_error],
                        "previous_raw_consolidation": consolidation,
                    }
                    _write_json(attempt_dir / "validation_error.json", audit_feedback)
                    if attempt >= cfg.audit_repair_attempts:
                        break
                    continue

                _write_json(attempt_dir / "consolidation.json", {"groups": final_rius})
                if len(final_rius) < 2:
                    selection_error = "Consolidation produced fewer than two collaborative RIUs"
                    audit_feedback = {
                        "errors": [selection_error],
                        "previous_groups": final_rius,
                    }
                    if attempt >= cfg.audit_repair_attempts:
                        break
                    continue
                audit_summary = run_riu_audit(
                    client=client,
                    task=task,
                    rius=final_rius,
                    trace_dir=attempt_dir / "conditions",
                    max_workers=cfg.max_workers,
                )
                _write_json(attempt_dir / "summary.json", audit_summary)
                if audit_summary["accept"]:
                    if not cfg.stop_after_riu_audit:
                        paths, logs = _canonical_artifact_membership(final_rius)
                        ownership = allocate_rius(final_rius, cfg)
                        overlaps = _cross_owner_artifact_groups(paths, logs, ownership)
                        if overlaps and len(merge_overlapping_rius(final_rius, overlaps)) < 2:
                            selection_error = (
                                "Canonical artifacts would collapse all collaborative RIUs "
                                "under the requested allocation. Select independent full files or logs."
                            )
                            audit_feedback = {
                                "errors": [selection_error],
                                "overlap_groups": overlaps,
                                "previous_groups": final_rius,
                            }
                            _write_json(attempt_dir / "partitionability_failure.json", audit_feedback)
                            continue
                    selection_succeeded = True
                    break
                selection_error = "; ".join(audit_summary["errors"][:8])
                audit_feedback = {
                    "errors": audit_summary["errors"],
                    "previous_groups": final_rius,
                    "condition_summary": _compact_audit_feedback(audit_summary),
                }
            if selection_succeeded:
                break
            reselection_feedback = _build_reselection_feedback(
                task=task,
                audit=audit_summary,
                final_rius=final_rius,
                fallback_error=selection_error,
                failed_selection_round=selection_round,
                max_retries=cfg.riu_reselection_attempts,
                prior_history=reselection_history,
            )
            reselection_history = list(reselection_feedback["failure_history"])
            _write_json(
                output / "04_riu_audit" / f"selection_{selection_round:02d}" / "reselection_feedback.json",
                reselection_feedback,
            )
        if not selection_succeeded:
            _write_json(
                output / "04_riu_audit" / "reselection_exhausted.json",
                {
                    "status": "reselection_exhausted",
                    "initial_attempts": 1,
                    "retry_attempts": cfg.riu_reselection_attempts,
                    "total_selection_attempts": cfg.riu_reselection_attempts + 1,
                    "failure_history": reselection_history,
                    "final_feedback": reselection_feedback,
                },
            )
            raise PipelineError(
                "RIU construction failed after candidate reselection: "
                + str(reselection_feedback.get("reason") or "unknown failure")
            )

    # The selected artifact bundle is authoritative. Failed selections remain
    # in their history directories and cannot shadow the selected references.
    _write_json(output / '05_final_rius/selected_artifacts.hidden.json', {'artifacts': artifacts})
    artifacts = _load_selection_artifact_universe(
        initial_candidate_root=initial_candidate_root,
        output_candidate_root=output / "03_candidates",
        initial_artifacts=artifacts,
    )
    _validate_final_riu_artifacts(final_rius, artifacts)

    _write_json(output / "05_final_rius" / "final_rius.hidden.json", {"rius": final_rius})
    _write_json(output / "05_final_rius" / "audit_summary.json", audit_summary)

    if cfg.stop_after_riu_audit:
        if run_identity(task, client, cfg, docker_timeout_sec=docker_timeout_sec) != identity:
            raise PipelineError('Construction inputs changed during execution')
        report = {
            "schema": SCHEMA,
            "task_id": task.task_id,
            "status": "riu_audit_complete",
            "completed_at_utc": _now(),
            "model": client.cfg.model,
            "candidate_source": candidate_source,
            "candidate_count": len(candidate_result["candidates"]),
            "final_riu_count": len(final_rius),
            "all_none_loo_accept": audit_summary["accept"],
            "audit_errors": audit_summary["errors"],
        }
        _write_json(output / "pipeline_report.json", report)
        write_checkpoint(output, identity)
        return report

    collector = DockerEvidenceCollector(
        execution_task.runtime.docker_image,
        task.repository.base_commit,
        timeout_sec=docker_timeout_sec,
        runtime_workers=min(4, cfg.max_workers),
    )
    inventory = collector.inventory()
    tool_paths = collector.tool_paths()
    experiment_checkpoint = output / "06_logs" / "display_logs.hidden.json"
    if experiment_checkpoint.is_file():
        experiment_bundle = _read_json(experiment_checkpoint)
    else:
        experiment_bundle = build_experiment_logs(
            client=client,
            task=task,
            rius=final_rius,
            artifacts=artifacts,
            collector=collector,
            tool_paths=tool_paths,
            output_dir=output / "06_logs",
            config=cfg,
        )

    code_corpus = collect_code_corpus(
        image=execution_task.runtime.docker_image,
        expected_commit=task.repository.base_commit,
        timeout_sec=docker_timeout_sec,
    )
    _write_json(output / "07_partition" / "code_corpus.json", _corpus_manifest(code_corpus))
    regroup_history: list[dict[str, Any]] = []
    for regroup_round in range(1, max(2, len(final_rius)) + 1):
        try:
            partition = build_partition(
                task=task,
                rius=final_rius,
                artifacts=artifacts,
                code_corpus=code_corpus,
                experiment_bundle=experiment_bundle,
                config=cfg,
            )
            _write_json(output / "07_partition" / "initial_partition.hidden.json", partition)
            mechanical = validate_partition_mechanically(partition, cfg)
            _write_json(output / "08_leakage" / "mechanical.json", mechanical)
            if not mechanical["valid"]:
                raise PipelineError("Mechanical partition audit failed: " + "; ".join(mechanical["errors"]))
            partition, leakage = run_leakage_repair(
                client=client,
                task=task,
                rius=final_rius,
                partition=partition,
                output_dir=output / "08_leakage",
                config=cfg,
            )
            break
        except CanonicalOverlapError as exc:
            if regroup_round == 1:
                regroup_feedback = {
                    "errors": [str(exc)],
                    "canonical_overlap_groups": exc.merge_groups,
                    "trigger_audits": exc.audits,
                    "instruction": (
                        "Do not merge RIUs. Conservatively reselect exact excerpts and regroup "
                        "the existing candidates so a participant's full canonical source/log "
                        "cannot reconstruct a foreign RIU. Keep at least two necessary RIUs."
                    ),
                }
                regroup_record = {
                    "round": regroup_round,
                    "action": "reselect_before_merge",
                    "merge_groups": exc.merge_groups,
                    "before_riu_count": len(final_rius),
                    "trigger_audits": exc.audits,
                }
                try:
                    reconsolidation = _consolidate_candidates(
                        client=client,
                        task=task,
                        candidate_result=candidate_result,
                        artifacts=artifacts,
                        audit_feedback=regroup_feedback,
                        trace_dir=output / "04_riu_audit" / "post_partition_reselection_01" / "consolidation",
                    )
                    reselected = validate_consolidation(
                        reconsolidation,
                        task=task,
                        candidates=candidate_result["candidates"],
                        artifacts=artifacts,
                    )
                    if len(reselected) < 2:
                        raise PipelineError("Conservative reselection left fewer than two RIUs")
                    reselected_audit = run_riu_audit(
                        client=client,
                        task=task,
                        rius=reselected,
                        trace_dir=output / "04_riu_audit" / "post_partition_reselection_01" / "conditions",
                        max_workers=cfg.max_workers,
                    )
                    regroup_record["strict_riu_audit"] = reselected_audit
                    if not reselected_audit["accept"]:
                        raise PipelineError(
                            "Conservative reselection failed strict ALL/NONE/LOO: "
                            + "; ".join(reselected_audit["errors"])
                        )
                    final_rius = reselected
                    audit_summary = reselected_audit
                    _write_json(final_riu_checkpoint, {"rius": final_rius})
                    _write_json(audit_checkpoint, audit_summary)
                    regroup_record["after_riu_count"] = len(reselected)
                    regroup_record["status"] = "accepted_for_repartition"
                    regroup_history.append(regroup_record)
                    _write_json(
                        output / "08_leakage" / "regroup_01.hidden.json",
                        regroup_record,
                    )
                    continue
                except PipelineError as reselection_exc:
                    regroup_record["status"] = "failed_then_merge_fallback"
                    regroup_record["reselection_error"] = str(reselection_exc)
                    regroup_history.append(regroup_record)
                    _write_json(
                        output / "08_leakage" / "regroup_01.hidden.json",
                        regroup_record,
                    )
            previous_count = len(final_rius)
            merged = merge_overlapping_rius(final_rius, exc.merge_groups)
            regroup_record = {
                "round": regroup_round,
                "action": "merge_after_reselection_failure",
                "merge_groups": exc.merge_groups,
                "before_riu_count": previous_count,
                "after_riu_count": len(merged),
                "trigger_audits": exc.audits,
            }
            if len(merged) >= previous_count or len(merged) < 2:
                _write_json(output / "08_leakage" / "regroup_failure.json", regroup_record)
                raise PipelineError("Canonical overlap could not be repaired while preserving collaboration") from exc
            merged_audit = run_riu_audit(
                client=client,
                task=task,
                rius=merged,
                trace_dir=output / "04_riu_audit" / f"post_partition_merge_{regroup_round:02d}",
                max_workers=cfg.max_workers,
            )
            regroup_record["strict_riu_audit"] = merged_audit
            regroup_history.append(regroup_record)
            _write_json(
                output / "08_leakage" / f"regroup_{regroup_round:02d}.hidden.json",
                regroup_record,
            )
            if not merged_audit["accept"]:
                raise PipelineError(
                    "Canonical-overlap merge failed strict ALL/NONE/LOO: "
                    + "; ".join(merged_audit["errors"])
                ) from exc
            final_rius = merged
            audit_summary = merged_audit
            _write_json(final_riu_checkpoint, {"rius": final_rius})
            _write_json(audit_checkpoint, audit_summary)
    else:
        raise PipelineError("Canonical overlap remained after all possible regrouping rounds")

    _write_json(output / "08_leakage" / "regroup_history.hidden.json", {"regroups": regroup_history})
    partition, salience = run_salience_repair(
        client=client,
        task=task,
        partition=partition,
        output_dir=output / "08_leakage" / "salience",
        config=cfg,
    )
    _write_json(output / "08_leakage" / "salience.json", salience)

    _write_json(output / "07_partition" / "final_partition.hidden.json", partition)
    benchmark_paths = materialize_modes(
        task=task,
        rius=final_rius,
        partition=partition,
        output_dir=output / "09_benchmark",
        config=cfg,
    )
    report = {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "status": "benchmark_ready",
        "completed_at_utc": _now(),
        "model": client.cfg.model,
        "candidate_source": candidate_source,
        "candidate_count": len(candidate_result["candidates"]),
        "final_riu_count": len(final_rius),
        "all_none_loo_accept": audit_summary["accept"],
        "canonical_runtime_log_count": len(experiment_bundle["canonical_logs"]),
        "distractor_log_count": len(experiment_bundle["distractor_logs"]),
        "allocation_policy": cfg.allocation_policy,
        "participant_count": len(partition["participants"]),
        "participant_context_token_cap": cfg.participant_context_token_cap,
        "assigned_context_count": len(partition["contexts"]),
        "assigned_code_tokens": sum(
            row["estimated_tokens"] for row in partition["contexts"] if row["context_type"] == "code"
        ),
        "excluded_code_count": len(partition["excluded_code"]),
        "excluded_code_tokens": sum(row["estimated_tokens"] for row in partition["excluded_code"]),
        "leakage_repair_rounds": leakage["rounds_run"],
        "leakage_removed_context_count": len(leakage["removed_context_ids"]),
        "mechanical_partition_valid": mechanical["valid"],
        "semantic_leakage_accept": leakage["accept"],
        "salience_accept": salience["accept"],
        "benchmark_paths": benchmark_paths,
    }
    if run_identity(task, client, cfg, docker_timeout_sec=docker_timeout_sec) != identity:
        raise PipelineError('Construction inputs changed during execution')
    _write_json(output / "pipeline_report.json", report)
    write_ready_receipt(output, task)
    full, piu_only = build_evaluation_rows(output, task)
    write_evaluation_rows(output / "evaluation", [full], [piu_only])
    write_checkpoint(output, identity)
    return report


def _validate_candidate_bundle(
    task: SWEAtlasQATask,
    result: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> None:
    if result.get("task_id") != task.task_id:
        raise PipelineError("Candidate bundle task_id mismatch")
    if not result.get("candidates"):
        raise PipelineError("Candidate bundle contains no candidates")
    artifact_by_id = {str(row.get("artifact_id")): row for row in artifacts}
    if not artifact_by_id or not all(row.get("verified") for row in artifacts):
        raise PipelineError("Candidate bundle has missing or unverified artifacts")
    for candidate in result["candidates"]:
        for excerpt in candidate.get("evidence") or []:
            artifact = artifact_by_id.get(str(excerpt.get("artifact_id")))
            quote = str(excerpt.get("quote") or "")
            if artifact is None or not quote.strip() or quote not in _artifact_evidence_content(artifact):
                raise PipelineError("Candidate excerpt is not exact in verified artifact")


def _scope_candidate_bundle(result, artifacts):
    """Content and provenance determine identity; planner IDs remain local."""
    result, artifacts = deepcopy(result), deepcopy(artifacts)
    mapping = {}
    for artifact in artifacts:
        old = str(artifact.get('artifact_id') or '')
        if not old or old in mapping:
            raise PipelineError('Missing or duplicated local artifact identity')
        payload = {k: v for k, v in artifact.items()
                   if k not in {'artifact_id', 'local_artifact_id', 'artifact_identity_sha256'}}
        identity = canonical_hash(payload)
        new = 'artifact_' + identity
        mapping[old] = new
        artifact['local_artifact_id'] = artifact.get('local_artifact_id', old)
        artifact['artifact_identity_sha256'] = identity
        artifact['artifact_id'] = new
    for candidate in result['candidates']:
        for excerpt in candidate.get('evidence', []):
            if excerpt['artifact_id'] not in mapping:
                raise PipelineError('Candidate references unknown local artifact')
            excerpt['artifact_id'] = mapping[excerpt['artifact_id']]
    return result, artifacts


def _load_selection_artifact_universe(
    *,
    initial_candidate_root: Path,
    output_candidate_root: Path,
    initial_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve only the authoritative selected bundle; preserve failed history separately."""

    by_id: dict[str, dict[str, Any]] = {}

    def add(rows: list[dict[str, Any]], source: Path | None) -> None:
        for row in rows:
            artifact_id = str(row.get("artifact_id") or "")
            if not artifact_id or row.get("verified") is not True:
                continue
            prior = by_id.get(artifact_id)
            if prior is not None and prior != row:
                raise PipelineError(
                    f"Verified artifact id {artifact_id!r} has conflicting selection contents"
                )
            by_id[artifact_id] = row

    add(initial_artifacts, None)
    return list(by_id.values())


def _validate_final_riu_artifacts(
    rius: list[dict[str, Any]], artifacts: list[dict[str, Any]]
) -> None:
    artifact_by_id = {
        str(row.get("artifact_id") or ""): row
        for row in artifacts
        if row.get("verified") is True
    }
    for riu in rius:
        for atom in riu.get("evidence_atoms") or []:
            artifact_id = str(atom.get("artifact_id") or "")
            artifact = artifact_by_id.get(artifact_id)
            quote = str(atom.get("quote") or "")
            if artifact is None:
                raise PipelineError(
                    f"Final RIU {riu.get('riu_id')} cites missing artifact {artifact_id!r}"
                )
            if not quote.strip() or quote not in _artifact_evidence_content(artifact):
                raise PipelineError(
                    f"Final RIU {riu.get('riu_id')} contains an inexact artifact excerpt"
                )


def _consolidate_candidates(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    candidate_result: dict[str, Any],
    artifacts: list[dict[str, Any]],
    audit_feedback: dict[str, Any],
    trace_dir: Path,
) -> dict[str, Any]:
    artifact_by_id = {row["artifact_id"]: row for row in artifacts}
    candidates = []
    for row in candidate_result["candidates"]:
        candidates.append(
            {
                **row,
                "evidence": [
                    {
                        **excerpt,
                        "source_path": artifact_by_id[excerpt["artifact_id"]].get("path"),
                    }
                    for excerpt in row["evidence"]
                ],
            }
        )
    return _call_prompt(
        client,
        "candidate_consolidation_system.txt",
        "candidate_consolidation_user.txt",
        {
            "PUBLIC_TASK_JSON": {"question": task.public.question},
            "RUBRICS_JSON": [_rubric_json(row) for row in task.evaluation.rubrics],
            "CANDIDATES_JSON": candidates,
            "VERIFIED_ARTIFACTS_JSON": [_artifact_for_prompt(row, include_runtime_command=False) for row in artifacts],
            "AUDIT_FEEDBACK_JSON": audit_feedback,
        },
        prompt_id=f"{task.task_id}.candidate_consolidation",
        trace_dir=trace_dir,
    )


def validate_consolidation(
    response: dict[str, Any],
    *,
    task: SWEAtlasQATask,
    candidates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    allow_artifact_merge: bool = False,
    enforce_artifact_exclusivity: bool = True,
) -> list[dict[str, Any]]:
    groups = response.get("groups")
    if not isinstance(groups, list) or not groups:
        raise PipelineError("Consolidation returned no groups")
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    valid_candidate_ids = set(candidate_by_id)
    valid_positive = {row.rubric_id for row in task.evaluation.rubrics if row.polarity == "positive"}
    valid_negative = {row.rubric_id for row in task.evaluation.rubrics if row.polarity == "negative"}
    artifact_by_id = {row["artifact_id"]: row for row in artifacts if row.get("verified")}
    seen_candidates: list[str] = []
    seen_groups: set[str] = set()
    positive_coverage: set[str] = set()
    negative_coverage: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(groups, start=1):
        group_id = str(raw.get("group_id") or "")
        member_ids = _string_list(raw.get("candidate_ids"))
        selected = raw.get("selected_evidence")
        if not group_id or group_id in seen_groups:
            raise PipelineError(f"Missing or duplicate consolidation group: {group_id!r}")
        if not member_ids or not set(member_ids) <= valid_candidate_ids:
            raise PipelineError(f"Invalid candidate membership for {group_id}")
        supports = {
            rubric_id
            for candidate_id in member_ids
            for rubric_id in candidate_by_id[candidate_id].get("supports_rubric_ids") or []
        }
        refutes = {
            rubric_id
            for candidate_id in member_ids
            for rubric_id in candidate_by_id[candidate_id].get("refutes_rubric_ids") or []
        }
        if not supports <= valid_positive or not refutes <= valid_negative:
            raise PipelineError(f"Invalid candidate rubric mapping for {group_id}")
        for field, rubric_ids in (("supports_rubric_ids", supports), ("refutes_rubric_ids", refutes)):
            if field in raw and set(_string_list(raw[field])) != rubric_ids:
                raise PipelineError(f"Group {group_id} rubric mapping must equal member-candidate union")
        if not isinstance(selected, list) or not selected:
            raise PipelineError(f"Group {group_id} has no selected evidence")
        evidence: list[dict[str, Any]] = []
        kinds: set[str] = set()
        for excerpt in selected:
            key = (
                str(excerpt.get("artifact_id") or ""),
                str(excerpt.get("kind") or ""),
                str(excerpt.get("quote") or ""),
            )
            artifact = artifact_by_id.get(key[0])
            if (
                artifact is None
                or not key[2].strip()
                or key[1] != artifact.get("kind")
                or key[2] not in _artifact_evidence_content(artifact)
            ):
                raise PipelineError(f"Group {group_id} selected unverified or inexact evidence")
            kinds.add(key[1])
            source_path = str(artifact.get("path") or "") if key[1] == "source" else ""
            evidence.append(
                {
                    "atom_id": f"atom_{index:02d}_{len(evidence) + 1:02d}",
                    "artifact_id": key[0],
                    "kind": key[1],
                    "quote": key[2],
                    "source_path": source_path or None,
                    "source_line_range": artifact.get("line_range") if source_path else None,
                    "command": artifact.get("command") if key[1] == "runtime" else None,
                }
            )
        if "source" not in kinds:
            raise PipelineError(f"Group {group_id} lacks a source/test anchor")
        if any(candidate_by_id[cid].get("evidence_mode") == "hybrid" for cid in member_ids) and "runtime" not in kinds:
            raise PipelineError(f"Hybrid group {group_id} lacks runtime evidence")
        hidden_claim = str(raw.get("hidden_claim") or "").strip()
        if not hidden_claim:
            raise PipelineError(f"Group {group_id} has no hidden claim")
        seen_groups.add(group_id)
        seen_candidates.extend(member_ids)
        positive_coverage.update(supports)
        negative_coverage.update(refutes)
        normalized.append(
            {
                "riu_id": f"riu_{index:02d}",
                "source_group_id": group_id,
                "hidden_claim": hidden_claim,
                "candidate_ids": member_ids,
                "supports_rubric_ids": sorted(supports),
                "refutes_rubric_ids": sorted(refutes),
                "evidence_atoms": evidence,
                "reason": str(raw.get("reason") or ""),
            }
        )
    if len(seen_candidates) != len(set(seen_candidates)) or set(seen_candidates) != valid_candidate_ids:
        raise PipelineError("Consolidation must assign every candidate exactly once")
    if positive_coverage != valid_positive:
        raise PipelineError(f"Consolidation misses positive rubrics: {sorted(valid_positive - positive_coverage)}")
    if negative_coverage != valid_negative:
        raise PipelineError(f"Consolidation misses negative rubrics: {sorted(valid_negative - negative_coverage)}")
    if enforce_artifact_exclusivity:
        # A later full-context packet contains a whole source file or runtime
        # log. Initial RIU sufficiency auditing, however, operates on exact
        # excerpts and must not pre-merge RIUs before partition construction.
        artifact_to_rius: dict[tuple[str, str], set[str]] = {}
        for riu in normalized:
            for atom in riu["evidence_atoms"]:
                identity = (
                    "source_path",
                    str(atom["source_path"]),
                ) if atom.get("source_path") else (
                    "runtime_artifact",
                    str(atom["artifact_id"]),
                )
                artifact_to_rius.setdefault(identity, set()).add(riu["riu_id"])
        overlap_groups = [
            sorted(owners) for owners in artifact_to_rius.values() if len(owners) > 1
        ]
        if overlap_groups:
            if not allow_artifact_merge:
                raise CanonicalArtifactOverlapError(overlap_groups)
            normalized = merge_overlapping_rius(normalized, overlap_groups)
    return normalized


def run_riu_audit(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    trace_dir: Path,
    max_workers: int,
) -> dict[str, Any]:
    conditions = [
        {"kind": "all", "omitted_riu_id": "", "rius": rius},
        {"kind": "none", "omitted_riu_id": "", "rius": []},
        *[
            {
                "kind": "leave_one_out",
                "omitted_riu_id": row["riu_id"],
                "rius": [other for other in rius if other["riu_id"] != row["riu_id"]],
            }
            for row in rius
        ],
    ]

    def run(condition: dict[str, Any]) -> dict[str, Any]:
        label = condition["kind"] if not condition["omitted_riu_id"] else f"without_{condition['omitted_riu_id']}"
        material = _audit_material(condition["rius"])
        solver = _call_prompt(
            client,
            "riu_audit_solver_system.txt",
            "riu_audit_solver_user.txt",
            {
                "PUBLIC_TASK_JSON": {"question": task.public.question},
                "AVAILABLE_EVIDENCE_JSON": material,
            },
            prompt_id=f"{task.task_id}.riu_audit.{label}.solver",
            trace_dir=trace_dir / label / "solver",
        )
        judge = _call_prompt(
            client,
            "riu_audit_judge_system.txt",
            "riu_audit_judge_user.txt",
            {
                "PUBLIC_TASK_JSON": {"question": task.public.question},
                "RUBRICS_JSON": [_rubric_json(row) for row in task.evaluation.rubrics],
                "REFERENCE_ANSWER": task.evaluation.reference_answer,
                "AVAILABLE_EVIDENCE_JSON": material,
                "SOLVER_RESPONSE_JSON": solver,
            },
            prompt_id=f"{task.task_id}.riu_audit.{label}.judge",
            trace_dir=trace_dir / label / "judge",
        )
        judge_errors = validate_audit_judge(judge, task.evaluation.rubrics, len(material))
        if judge_errors:
            judge = _call_prompt(
                client,
                "riu_audit_judge_repair_system.txt",
                "riu_audit_judge_repair_user.txt",
                {
                    "RUBRICS_JSON": [
                        _rubric_json(row) for row in task.evaluation.rubrics
                    ],
                    "EVIDENCE_COUNT": len(material),
                    "PREVIOUS_JUDGE_JSON": judge,
                    "VALIDATION_ERRORS_JSON": judge_errors,
                    "PUBLIC_TASK_JSON": {"question": task.public.question},
                    "AVAILABLE_EVIDENCE_JSON": material,
                    "SOLVER_RESPONSE_JSON": solver,
                    "REFERENCE_ANSWER": task.evaluation.reference_answer,
                },
                prompt_id=f"{task.task_id}.riu_audit.{label}.judge_repair",
                trace_dir=trace_dir / label / "judge_repair",
            )
            judge_errors = validate_audit_judge(
                judge, task.evaluation.rubrics, len(material)
            )
        return {
            "kind": condition["kind"],
            "omitted_riu_id": condition["omitted_riu_id"],
            "available_riu_ids": [row["riu_id"] for row in condition["rius"]],
            "material": material,
            "solver_response": solver,
            "judge_result": judge,
            "judge_validation_errors": judge_errors,
        }

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(conditions)))) as executor:
        results = list(executor.map(run, conditions))
    return summarize_riu_audit(task, rius, results)


def validate_audit_judge(
    judge: dict[str, Any], rubrics: tuple[Rubric, ...], evidence_count: int
) -> list[str]:
    errors: list[str] = []
    expected = {row.rubric_id for row in rubrics}
    positive = {row.rubric_id for row in rubrics if row.polarity == "positive"}
    rows = judge.get("rubric_results")
    if not isinstance(rows, list):
        return ["judge.rubric_results must be a list"]
    seen: list[str] = []
    for row in rows:
        rubric_id = str(row.get("rubric_id") or "")
        seen.append(rubric_id)
        if rubric_id not in expected:
            errors.append(f"judge returned unknown rubric {rubric_id!r}")
        if row.get("status") not in {"satisfied", "partial", "missing", "incorrect"}:
            errors.append(f"{rubric_id}: invalid status")
        if not isinstance(row.get("content_correct"), bool):
            errors.append(f"{rubric_id}: content_correct must be boolean")
        if not isinstance(row.get("grounded_in_available_material"), bool):
            errors.append(f"{rubric_id}: grounded flag must be boolean")
        satisfied = row.get("content_correct") is True and row.get("grounded_in_available_material") is True
        if (row.get("status") == "satisfied") != satisfied:
            errors.append(f"{rubric_id}: status must match correctness and grounding")
        numbers = row.get("supporting_evidence_numbers", [])
        if not isinstance(numbers, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > evidence_count for value in numbers
        ):
            errors.append(f"{rubric_id}: invalid evidence number")
        if rubric_id in positive and satisfied and isinstance(numbers, list) and not numbers:
            errors.append(f"{rubric_id}: satisfied positive rubric requires supporting evidence")
    if len(seen) != len(set(seen)) or set(seen) != expected:
        errors.append("judge must grade every rubric exactly once")
    if not isinstance(judge.get("has_material_error"), bool):
        errors.append("has_material_error must be boolean")
    if not isinstance(judge.get("overall_complete_and_correct"), bool):
        errors.append("overall_complete_and_correct must be boolean")
    return _dedupe(errors)


def summarize_riu_audit(
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    positive_rubric_ids = {
        rubric.rubric_id for rubric in task.evaluation.rubrics if rubric.polarity == "positive"
    }
    by_kind = {(row["kind"], row["omitted_riu_id"]): row for row in results}
    full = by_kind[("all", "")]
    none = by_kind[("none", "")]
    errors.extend(f"all: {value}" for value in full["judge_validation_errors"])
    if not full["judge_validation_errors"]:
        if full["judge_result"].get("overall_complete_and_correct") is not True:
            errors.append("ALL did not produce a complete and correct grounded answer")
        if full["judge_result"].get("has_material_error") is True:
            errors.append("ALL answer contains a material error")
        if not all(_audit_row_satisfied(row) for row in full["judge_result"].get("rubric_results") or []):
            errors.append("ALL did not satisfy every rubric")
    errors.extend(f"none: {value}" for value in none["judge_validation_errors"])
    if not none["judge_validation_errors"]:
        if none["judge_result"].get("overall_complete_and_correct") is True:
            errors.append("NONE remained complete and correct")
        if any(
            str(row.get("rubric_id") or "") in positive_rubric_ids
            and _audit_row_satisfied(row)
            for row in none["judge_result"].get("rubric_results") or []
        ):
            errors.append("NONE grounded at least one answer-specific rubric")

    loo_rows: list[dict[str, Any]] = []
    for riu in rius:
        omitted = riu["riu_id"]
        row = by_kind[("leave_one_out", omitted)]
        errors.extend(f"{omitted}: {value}" for value in row["judge_validation_errors"])
        # Negative rubrics constrain hallucinations in final evaluation; mere
        # omission satisfies them and therefore cannot establish RIU necessity.
        linked = set(riu["supports_rubric_ids"])
        judged = {
            str(result.get("rubric_id") or ""): result
            for result in row["judge_result"].get("rubric_results") or []
        }
        broken = sorted(rubric_id for rubric_id in linked if not _audit_row_satisfied(judged.get(rubric_id) or {}))
        if not row["judge_validation_errors"]:
            if row["judge_result"].get("overall_complete_and_correct") is True:
                errors.append(f"{omitted}: leave-one-out response remained complete")
            if not broken:
                errors.append(f"{omitted}: all linked rubrics remained correct and grounded")
        loo_rows.append({**row, "linked_rubric_ids": sorted(linked), "broken_rubric_ids": broken})
    return {
        "schema": "swe_atlas_strict_riu_audit_v1",
        "accept": not errors,
        "errors": _dedupe(errors),
        "all": full,
        "none": none,
        "leave_one_out": loo_rows,
    }


def build_experiment_logs(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    collector: DockerEvidenceCollector,
    tool_paths: dict[str, str],
    output_dir: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    artifact_by_id = {row["artifact_id"]: row for row in artifacts}
    runtime_ids = {
        atom["artifact_id"]
        for riu in rius
        for atom in riu["evidence_atoms"]
        if atom["kind"] == "runtime"
    }
    canonical = [artifact_by_id[artifact_id] for artifact_id in sorted(runtime_ids)]
    canonical_commands = {str(row.get("command") or "") for row in canonical}
    feedback: dict[str, Any] = {}
    verified: list[dict[str, Any]] = []
    plan: dict[str, Any] = {"experiments": []}
    for attempt in range(config.experiment_repair_attempts + 1):
        plan = _call_prompt(
            client,
            "experiment_family_plan_system.txt",
            "experiment_family_plan_user.txt",
            {
                "PUBLIC_TASK_JSON": {"question": task.public.question},
                "REPOSITORY_JSON": {
                    "name": task.repository.repository,
                    "base_commit": task.repository.base_commit,
                    "language": task.repository.language,
                },
                "TOOL_PATHS_JSON": tool_paths,
                "FINAL_RIUS_JSON": [
                    {
                        "riu_id": row["riu_id"],
                        "hidden_claim": row["hidden_claim"],
                        "source_paths": sorted(
                            {atom["source_path"] for atom in row["evidence_atoms"] if atom.get("source_path")}
                        ),
                    }
                    for row in rius
                ],
                "CANONICAL_RUNTIME_JSON": [
                    {"artifact_id": row["artifact_id"], "command": row.get("command"), "content": row.get("content")}
                    for row in canonical
                ],
                "FEEDBACK_JSON": feedback,
            },
            prompt_id=f"{task.task_id}.experiment_family_plan.attempt_{attempt}",
            trace_dir=output_dir / "llm_traces" / f"attempt_{attempt:02d}",
        )
        try:
            requests = validate_experiment_plan(plan, canonical_commands)
        except (PipelineError, CandidateValidationError) as exc:
            feedback = {"validation_error": str(exc), "previous_plan": plan}
            if attempt >= config.experiment_repair_attempts:
                raise PipelineError(f"Unable to create a valid distractor plan: {exc}") from exc
            continue
        collected = collector.collect(requests)
        request_metadata = {row["request_id"]: row for row in requests}
        for artifact in collected:
            metadata = request_metadata.get(str(artifact.get("artifact_id") or ""), {})
            artifact["family"] = metadata.get("family")
            artifact["why_not_canonical"] = metadata.get("why_not_canonical")
        _write_json(output_dir / f"all_collected_artifacts.attempt_{attempt:02d}.hidden.json",
                    {"artifacts": collected, "planned_count": len(requests)})
        verified = [row for row in collected if row.get("verified")]
        if len(verified) >= config.min_distractor_logs:
            break
        feedback = {
            "validation_error": f"only {len(verified)} verified distractor logs; need {config.min_distractor_logs}",
            "failed_artifacts": [row for row in collected if not row.get("verified")],
            "previous_plan": plan,
        }
        if attempt >= config.experiment_repair_attempts:
            raise PipelineError("Unable to create enough verified distractor logs")
    canonical_logs = [_display_log(row, canonical=True) for row in canonical]
    distractor_logs = [_display_log(row, canonical=False) for row in verified]
    bundle = {
        "experiment_plan": plan,
        "canonical_logs": canonical_logs,
        "distractor_logs": distractor_logs,
        "raw_distractor_artifacts": verified,
    }
    _write_json(output_dir / "experiment_plan.json", plan)
    _write_json(output_dir / "raw_logs.json", {"canonical": canonical, "distractors": verified})
    _write_json(output_dir / "display_logs.hidden.json", bundle)
    return bundle


def validate_experiment_plan(
    plan: dict[str, Any], canonical_commands: set[str]
) -> list[dict[str, Any]]:
    rows = plan.get("experiments")
    if not isinstance(rows, list):
        raise PipelineError("Experiment plan must contain experiments")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    valid_families = {
        "parameter_neighbor", "negative_control", "boundary_case", "same_module_test",
        "adjacent_module_test", "environment_probe", "expected_failure",
    }
    for raw in rows:
        experiment_id = str(raw.get("experiment_id") or "")
        command = str(raw.get("command") or "").strip()
        family = str(raw.get("family") or "")
        if not experiment_id or experiment_id in seen:
            raise PipelineError(f"Missing or duplicate experiment id {experiment_id!r}")
        if family not in valid_families:
            raise PipelineError(f"Invalid experiment family {family!r}")
        _validate_runtime_command(command)
        if command in canonical_commands:
            raise PipelineError("Distractor experiment duplicates canonical command")
        seen.add(experiment_id)
        normalized.append(
            {
                "request_id": experiment_id,
                "kind": "runtime",
                "command": command,
                "expected_exit_code": int(raw.get("expected_exit_code", 0)),
                "purpose": str(raw.get("purpose") or ""),
                "supports_rubric_ids": [],
                "family": family,
                "why_not_canonical": str(raw.get("why_not_canonical") or ""),
            }
        )
    return normalized


def collect_code_corpus(
    *, image: str, expected_commit: str, timeout_sec: int
) -> list[dict[str, Any]]:
    script = (
        "git config --global --add safe.directory /app >/dev/null 2>&1 || true; "
        "cd /app && test \"$(git rev-parse HEAD)\" = "
        + _shell_quote(expected_commit)
        + " && git archive --format=tar HEAD"
    )
    try:
        completed = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "/bin/bash", image, "-lc", script],
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("Docker git archive timed out") from exc
    if completed.returncode != 0:
        raise PipelineError("Docker git archive failed: " + completed.stderr.decode("utf-8", errors="replace")[:1000])
    rows: list[dict[str, Any]] = []
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _is_distributable_path(member.name):
                continue
            handle = archive.extractfile(member)
            raw = handle.read() if handle else b""
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            rows.append(
                {
                    "path": member.name,
                    "text": text,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "estimated_tokens": estimate_tokens(text),
                    "module_id": _module_id(member.name),
                }
            )
    if not rows:
        raise PipelineError("Distributable code corpus is empty")
    return rows


def _canonical_artifact_membership(rius: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    paths: dict[str, set[str]] = {}
    logs: dict[str, set[str]] = {}
    for riu in rius:
        for atom in riu["evidence_atoms"]:
            if atom["kind"] == "source" and atom.get("source_path"):
                paths.setdefault(atom["source_path"], set()).add(riu["riu_id"])
            if atom["kind"] == "runtime":
                logs.setdefault(atom["artifact_id"], set()).add(riu["riu_id"])
    return paths, logs


def _cross_owner_artifact_groups(
    paths: dict[str, set[str]], logs: dict[str, set[str]], ownership: dict[str, str],
) -> list[list[str]]:
    groups = {tuple(sorted(rius)) for rius in [*paths.values(), *logs.values()]
              if len({ownership[riu_id] for riu_id in rius}) > 1}
    return [list(group) for group in sorted(groups)]


def build_partition(
    *,
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    code_corpus: list[dict[str, Any]],
    experiment_bundle: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    ownership = allocate_rius(rius, config)
    participant_ids = sorted(set(ownership.values()))
    participants = [
        {
            "participant_id": participant_id,
            "role_name": f"repository reviewer {index}",
            "owned_riu_ids": sorted(riu_id for riu_id, owner in ownership.items() if owner == participant_id),
            "private_context_unit_ids": [],
            "private_context_tokens": 0,
        }
        for index, participant_id in enumerate(participant_ids, start=1)
    ]
    participant_by_id = {row["participant_id"]: row for row in participants}
    artifact_by_id = {row["artifact_id"]: row for row in artifacts}
    path_to_rius, runtime_to_rius = _canonical_artifact_membership(rius)
    overlap_groups = _cross_owner_artifact_groups(path_to_rius, runtime_to_rius, ownership)
    if overlap_groups:
        raise CanonicalOverlapError(overlap_groups, [])

    corpus_by_path = {row["path"]: row for row in code_corpus}
    missing_paths = sorted(set(path_to_rius) - set(corpus_by_path))
    if missing_paths:
        raise PipelineError(f"Canonical source paths missing from corpus: {missing_paths}")
    contexts: list[dict[str, Any]] = []
    assigned_paths: set[str] = set()

    def assign_context(context: dict[str, Any], participant_id: str) -> None:
        tokens = int(context["estimated_tokens"])
        participant = participant_by_id[participant_id]
        if participant["private_context_tokens"] + tokens > config.participant_context_token_cap:
            raise PipelineError(
                f"Canonical/reserved context exceeds cap for {participant_id}: {context['context_unit_id']}"
            )
        context["assigned_to"] = [participant_id]
        contexts.append(context)
        participant["private_context_unit_ids"].append(context["context_unit_id"])
        participant["private_context_tokens"] += tokens

    for path in sorted(path_to_rius):
        linked = sorted(path_to_rius[path])
        owner = ownership[linked[0]]
        row = corpus_by_path[path]
        assign_context(_code_context(row, linked, canonical=True), owner)
        assigned_paths.add(path)

    canonical_log_by_artifact = {
        row["artifact_id"]: row for row in experiment_bundle["canonical_logs"]
    }
    for artifact_id in sorted(runtime_to_rius):
        linked = sorted(runtime_to_rius[artifact_id])
        owner = ownership[linked[0]]
        display = canonical_log_by_artifact.get(artifact_id)
        if display is None:
            display = _display_log(artifact_by_id[artifact_id], canonical=True)
        assign_context(_log_context(display, linked, canonical=True), owner)

    for display in experiment_bundle["distractor_logs"]:
        owner = min(
            participant_ids,
            key=lambda value: (participant_by_id[value]["private_context_tokens"], value),
        )
        context = _log_context(display, [], canonical=False)
        if participant_by_id[owner]["private_context_tokens"] + context["estimated_tokens"] <= config.participant_context_token_cap:
            assign_context(context, owner)

    canonical_modules = {corpus_by_path[path]["module_id"] for path in path_to_rius}
    remaining = [row for row in code_corpus if row["path"] not in assigned_paths]
    remaining.sort(
        key=lambda row: (
            0 if _near_canonical_module(row["module_id"], canonical_modules) else 1,
            row["path"],
        )
    )
    excluded: list[dict[str, Any]] = []
    seen_text_hashes = {hashlib.sha256(row["text"].encode("utf-8")).hexdigest() for row in contexts}
    for row in remaining:
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            excluded.append({**_corpus_row_without_text(row), "reason": "duplicate_text"})
            continue
        context = _code_context(row, [], canonical=False)
        candidates = sorted(
            participant_ids,
            key=lambda value: (participant_by_id[value]["private_context_tokens"], value),
        )
        owner = next(
            (
                value
                for value in candidates
                if participant_by_id[value]["private_context_tokens"] + context["estimated_tokens"]
                <= config.participant_context_token_cap
            ),
            None,
        )
        if owner is None:
            excluded.append({**_corpus_row_without_text(row), "reason": "capacity"})
            continue
        assign_context(context, owner)
        seen_text_hashes.add(text_hash)

    rng = random.Random(f"{task.task_id}:{config.seed}:packet_order")
    for participant in participants:
        ids = list(participant["private_context_unit_ids"])
        rng.shuffle(ids)
        participant["private_context_unit_ids"] = ids
    partition = {
        "schema": "swe_atlas_partition_v1",
        "allocation_policy": config.allocation_policy,
        "participant_context_token_cap": config.participant_context_token_cap,
        "riu_ownership": ownership,
        "participants": participants,
        "contexts": contexts,
        "excluded_code": excluded,
    }
    partition, reader_fit = fit_actual_reader_budget(partition, config)
    partition["reader_budget_fit"] = reader_fit
    return partition



def fit_actual_reader_budget(partition: dict[str, Any], config: PipelineConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep whole canonical evidence and fit the actual segmented source reader.

    Separate tokenization and join separators can exceed the sum of stored unit
    estimates. Removing a minimal neutral whole unit is preferable to silently
    window-truncating many contexts at the actual model boundary.
    """
    from teamgym.interaction.actor_utils import count_tokens
    from teamgym.interaction.llm_actors import _private_context_payload
    from teamgym.interaction.types import BudgetSpec
    current = deepcopy(partition)
    packet_ids = [context_id for participant in current["participants"]
                  for context_id in participant["private_context_unit_ids"]]
    packet_order = {context_id: index for index, context_id in enumerate(packet_ids)}
    # The evaluation reader filters the exported context list in list order.
    # Match the participant-local order used in leakage and salience audits.
    current["contexts"].sort(key=lambda row: packet_order.get(row["context_unit_id"], len(packet_order)))
    cap = config.participant_context_token_cap
    removed = []
    reports = []
    for participant in current['participants']:
        def units():
            allowed = set(participant['private_context_unit_ids'])
            return [row for row in current['contexts'] if row['context_unit_id'] in allowed and str(row.get('text') or '').strip()]
        def segmented(rows):
            return sum(count_tokens(str(row['text']).strip()) for row in rows) + max(0, len(rows) - 1) * count_tokens('\n\n')
        before = segmented(units())
        while segmented(units()) > cap:
            rows = units()
            surplus = segmented(rows) - cap
            neutral = [row for row in rows if not row.get('hidden_contains_riu_ids')]
            if not neutral:
                raise PipelineError(f"Canonical material exceeds actual reader cap for {participant['participant_id']}")
            enough = [row for row in neutral if count_tokens(str(row['text']).strip()) + count_tokens('\n\n') >= surplus]
            chosen = min(enough, key=lambda row: (count_tokens(str(row['text']).strip()), row['context_unit_id'])) if enough else max(
                neutral, key=lambda row: (count_tokens(str(row['text']).strip()), row['context_unit_id']))
            removed.append({'participant_id': participant['participant_id'],
                'context_unit_id': chosen['context_unit_id'], 'source_ref': chosen.get('source_ref'),
                'text_sha256': hashlib.sha256(chosen['text'].encode()).hexdigest(),
                'reason': 'actual_reader_capacity_including_segment_boundaries'})
            _remove_contexts(current, {chosen['context_unit_id']})
        rows = units()
        datapoint = {'private_context_units': [{**row,
            'hidden_contains_rui_ids': list(row.get('hidden_contains_riu_ids') or [])} for row in current['contexts']]}
        actual = _private_context_payload(datapoint, participant, BudgetSpec(participant_context_token_budget=cap))
        expected = '\n\n'.join(str(row['text']).strip() for row in rows)
        if actual != expected or count_tokens(actual) > cap:
            raise PipelineError('Actual model reader still truncates or changes the certified source packet')
        reports.append({'participant_id': participant['participant_id'], 'segmented_tokens_before': before,
                        'segmented_tokens_after': segmented(rows), 'actual_source_tokens': count_tokens(actual),
                        'byte_exact_untruncated': True})
    return current, {'schema': 'swe_atlas_actual_reader_fit_v1', 'cap': cap,
                     'removed_neutral_contexts': removed, 'canonical_preserved': True,
                     'participant_results': reports}


def allocate_rius(rius: list[dict[str, Any]], config: PipelineConfig) -> dict[str, str]:
    if len(rius) < 2:
        raise PipelineError("Collaborative allocation requires at least two RIUs")
    if config.allocation_policy == "one_each":
        return {
            row["riu_id"]: _participant_id(index)
            for index, row in enumerate(rius, start=1)
        }
    assert config.participant_count is not None
    count = min(config.participant_count, len(rius))
    participant_ids = [_participant_id(index) for index in range(1, count + 1)]
    loads = {participant_id: 0 for participant_id in participant_ids}
    ownership: dict[str, str] = {}
    for riu in sorted(
        rius,
        key=lambda row: -sum(estimate_tokens(atom["quote"]) for atom in row["evidence_atoms"]),
    ):
        owner = min(participant_ids, key=lambda value: (loads[value], value))
        ownership[riu["riu_id"]] = owner
        loads[owner] += sum(estimate_tokens(atom["quote"]) for atom in riu["evidence_atoms"])
    return ownership


def merge_overlapping_rius(
    rius: list[dict[str, Any]], merge_groups: list[list[str]]
) -> list[dict[str, Any]]:
    """Merge RIUs connected by canonical cross-packet reconstruction."""

    by_id = {row["riu_id"]: row for row in rius}
    parent = {riu_id: riu_id for riu_id in by_id}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for group in merge_groups:
        unknown = set(group) - set(by_id)
        if unknown:
            raise PipelineError(f"Canonical-overlap merge cites unknown RIUs: {sorted(unknown)}")
        for riu_id in group[1:]:
            union(group[0], riu_id)

    components: dict[str, list[dict[str, Any]]] = {}
    for row in rius:
        components.setdefault(find(row["riu_id"]), []).append(row)

    merged_rows: list[dict[str, Any]] = []
    for index, component in enumerate(components.values(), start=1):
        atoms: list[dict[str, Any]] = []
        seen_atoms: set[tuple[str, str, str]] = set()
        for row in component:
            for atom in row["evidence_atoms"]:
                key = (str(atom["artifact_id"]), str(atom["kind"]), str(atom["quote"]))
                if key in seen_atoms:
                    continue
                seen_atoms.add(key)
                atoms.append(dict(atom))
        for atom_index, atom in enumerate(atoms, start=1):
            atom["atom_id"] = f"atom_{index:02d}_{atom_index:02d}"
        source_ids = [row["riu_id"] for row in component]
        merged_rows.append(
            {
                "riu_id": f"riu_{index:02d}",
                "source_group_id": "+".join(
                    str(row.get("source_group_id") or row["riu_id"]) for row in component
                ),
                "hidden_claim": " ".join(str(row["hidden_claim"]) for row in component),
                "candidate_ids": sorted(
                    {candidate_id for row in component for candidate_id in row["candidate_ids"]}
                ),
                "supports_rubric_ids": sorted(
                    {rubric_id for row in component for rubric_id in row["supports_rubric_ids"]}
                ),
                "refutes_rubric_ids": sorted(
                    {rubric_id for row in component for rubric_id in row["refutes_rubric_ids"]}
                ),
                "evidence_atoms": atoms,
                "reason": f"Merged after canonical packet overlap connected {source_ids}.",
            }
        )
    return merged_rows


def validate_partition_mechanically(
    partition: dict[str, Any], config: PipelineConfig
) -> dict[str, Any]:
    errors: list[str] = []
    participants = partition["participants"]
    contexts = partition["contexts"]
    participant_ids = {row["participant_id"] for row in participants}
    context_ids = [row["context_unit_id"] for row in contexts]
    if len(context_ids) != len(set(context_ids)):
        errors.append("duplicate context_unit_id")
    for context in contexts:
        assigned = context.get("assigned_to") or []
        if len(assigned) != 1 or assigned[0] not in participant_ids:
            errors.append(f"{context['context_unit_id']}: context must have exactly one valid owner")
    referenced = [
        context_id
        for participant in participants
        for context_id in participant["private_context_unit_ids"]
    ]
    if len(referenced) != len(set(referenced)) or set(referenced) != set(context_ids):
        errors.append("participant context references are not disjoint and exhaustive")
    hashes: dict[str, str] = {}
    for context in contexts:
        digest = hashlib.sha256(_normalize_context_text(context["text"]).encode("utf-8")).hexdigest()
        if digest in hashes:
            errors.append(f"normalized duplicate contexts: {hashes[digest]} and {context['context_unit_id']}")
        hashes[digest] = context["context_unit_id"]
    for participant in participants:
        if participant["private_context_tokens"] > config.participant_context_token_cap:
            errors.append(f"{participant['participant_id']}: context token cap exceeded")
        owned = set(participant["owned_riu_ids"])
        for context_id in participant["private_context_unit_ids"]:
            context = next(row for row in contexts if row["context_unit_id"] == context_id)
            if not set(context.get("hidden_contains_riu_ids") or []) <= owned:
                errors.append(f"{context_id}: canonical evidence assigned to a non-owner")
    return {"valid": not errors, "errors": _dedupe(errors)}


def run_leakage_repair(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    partition: dict[str, Any],
    output_dir: Path,
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = deepcopy(partition)
    removed: set[str] = set()
    all_rounds: list[dict[str, Any]] = []
    for round_index in range(1, config.leakage_repair_rounds + 1):
        audits = _run_packet_audits(
            client=client,
            task=task,
            rius=rius,
            partition=current,
            trace_dir=output_dir / f"round_{round_index:02d}" / "traces",
            config=config,
        )
        _write_json(output_dir / f"round_{round_index:02d}" / "audits.json", {"audits": audits})
        all_rounds.append({"round": round_index, "audits": audits})
        leaking = [
            (audit, result)
            for audit in audits
            for result in audit["foreign_rui_results"]
            if result["reconstructable_from_packet"]
        ]
        if not leaking:
            summary = {
                "accept": True,
                "rounds_run": round_index,
                "removed_context_ids": sorted(removed),
                "rounds": all_rounds,
            }
            _write_json(output_dir / "summary.json", summary)
            return current, summary
        canonical = [
            (audit["participant_id"], result["riu_id"])
            for audit, result in leaking
            if result["canonical_overlap"]
        ]
        if canonical:
            contexts_by_id = {row["context_unit_id"]: row for row in current["contexts"]}
            participants_by_id = {
                row["participant_id"]: row for row in current["participants"]
            }
            merge_groups: list[list[str]] = []
            for audit, result in leaking:
                if not result["canonical_overlap"]:
                    continue
                support_owned = {
                    riu_id
                    for context_id in result["minimal_support_context_ids"]
                    for riu_id in contexts_by_id[context_id].get("hidden_contains_riu_ids", [])
                }
                if not support_owned:
                    support_owned.update(
                        participants_by_id[audit["participant_id"]]["owned_riu_ids"]
                    )
                merge_groups.append(sorted(support_owned | {result["riu_id"]}))
            raise CanonicalOverlapError(merge_groups, audits)
        removable = {
            context_id
            for _, result in leaking
            for context_id in result["removable_noncanonical_context_ids"]
        }
        if not removable:
            raise PipelineError("Semantic leakage found no removable noncanonical context")
        removed.update(removable)
        _remove_contexts(current, removable)
        mechanical = validate_partition_mechanically(current, config)
        if not mechanical["valid"]:
            raise PipelineError("Leakage repair broke partition: " + "; ".join(mechanical["errors"]))
    raise PipelineError("Semantic packet leakage remained after maximum repair rounds")


def _run_packet_audits(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    partition: dict[str, Any],
    trace_dir: Path,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    contexts_by_id = {row["context_unit_id"]: row for row in partition["contexts"]}
    ownership = partition["riu_ownership"]

    def audit_participant(participant: dict[str, Any]) -> dict[str, Any]:
        participant_id = participant["participant_id"]
        packet = [contexts_by_id[value] for value in participant["private_context_unit_ids"]]
        foreign = [row for row in rius if ownership[row["riu_id"]] != participant_id]
        foreign_view = [
            {
                "riu_id": row["riu_id"],
                "hidden_claim": row["hidden_claim"],
                "answer_contribution_atoms": [atom["quote"] for atom in row["evidence_atoms"]],
            }
            for row in foreign
        ]
        candidate_hints: list[dict[str, Any]] = []
        if sum(len(row["text"]) for row in packet) > config.leakage_scan_threshold_chars:
            chunks = _context_chunks(packet, config.leakage_chunk_chars)
            for index, chunk in enumerate(chunks, start=1):
                scan = _call_prompt(
                    client,
                    "packet_leakage_candidate_system.txt",
                    "packet_leakage_candidate_user.txt",
                    {
                        "PUBLIC_TASK_JSON": {"question": task.public.question},
                        "PARTICIPANT_JSON": {"participant_id": participant_id},
                        "FOREIGN_RIUS_JSON": foreign_view,
                        "PACKET_CONTEXT_CHUNK_JSON": [_audit_context_view(row) for row in chunk],
                    },
                    prompt_id=f"{task.task_id}.{participant_id}.leakage_scan.{index}",
                    trace_dir=trace_dir / participant_id / f"scan_{index:02d}",
                )
                candidate_hints.extend(scan.get("foreign_rui_candidates") or [])
        audit = _call_prompt(
            client,
            "packet_leakage_audit_system.txt",
            "packet_leakage_audit_user.txt",
            {
                "PUBLIC_TASK_JSON": {"question": task.public.question},
                "PARTICIPANT_JSON": {
                    "participant_id": participant_id,
                    "owned_riu_ids": participant["owned_riu_ids"],
                },
                "FOREIGN_RIUS_JSON": foreign_view,
                "PACKET_CONTEXTS_JSON": [_audit_context_view(row) for row in packet],
                "CANDIDATE_HINTS_JSON": candidate_hints,
            },
            prompt_id=f"{task.task_id}.{participant_id}.packet_leakage_audit",
            trace_dir=trace_dir / participant_id / "final",
        )
        return validate_packet_audit(audit, participant, packet, foreign)

    with ThreadPoolExecutor(max_workers=max(1, min(config.max_workers, len(partition["participants"])))) as executor:
        return list(executor.map(audit_participant, partition["participants"]))


def validate_packet_audit(
    audit: dict[str, Any],
    participant: dict[str, Any],
    packet: list[dict[str, Any]],
    foreign_rius: list[dict[str, Any]],
) -> dict[str, Any]:
    participant_id = participant["participant_id"]
    if str(audit.get("participant_id") or "") != participant_id:
        raise PipelineError("Packet audit participant_id mismatch")
    packet_by_id = {row["context_unit_id"]: row for row in packet}
    foreign_ids = {row["riu_id"] for row in foreign_rius}
    rows = audit.get("foreign_riu_results", audit.get("foreign_rui_results"))
    if not isinstance(rows, list):
        raise PipelineError("Packet audit has no foreign_riu_results")
    seen: list[str] = []
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        riu_id = str(raw.get("riu_id") or "")
        seen.append(riu_id)
        reconstructable = raw.get("reconstructable_from_packet")
        canonical_overlap = raw.get("canonical_overlap")
        support = _string_list(raw.get("minimal_support_context_ids"))
        removable = _string_list(raw.get("removable_noncanonical_context_ids"))
        if riu_id not in foreign_ids:
            raise PipelineError(f"Packet audit returned unknown foreign RIU {riu_id}")
        if reconstructable not in {True, False} or canonical_overlap not in {True, False}:
            raise PipelineError(f"Packet audit booleans invalid for {riu_id}")
        if not set(support) <= set(packet_by_id) or not set(removable) <= set(packet_by_id):
            raise PipelineError(f"Packet audit cites context outside packet for {riu_id}")
        if any(packet_by_id[value].get("hidden_contains_riu_ids") for value in removable):
            raise PipelineError(f"Packet audit tried to remove canonical context for {riu_id}")
        if reconstructable is False and (support or removable or canonical_overlap):
            raise PipelineError(f"Non-reconstructable result has support/removal for {riu_id}")
        if canonical_overlap and not reconstructable:
            raise PipelineError(f"canonical_overlap requires reconstructable=true for {riu_id}")
        normalized.append(
            {
                "riu_id": riu_id,
                "reconstructable_from_packet": reconstructable,
                "minimal_support_context_ids": support,
                "removable_noncanonical_context_ids": removable,
                "canonical_overlap": canonical_overlap,
                "reason": str(raw.get("reason") or ""),
            }
        )
    if len(seen) != len(set(seen)) or set(seen) != foreign_ids:
        raise PipelineError("Packet audit must grade every foreign RIU exactly once")
    return {"participant_id": participant_id, "foreign_rui_results": normalized}


def run_salience_audit(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    partition: dict[str, Any],
    trace_dir: Path,
    max_workers: int,
) -> dict[str, Any]:
    contexts_by_id = {row["context_unit_id"]: row for row in partition["contexts"]}

    def run(participant: dict[str, Any]) -> dict[str, Any]:
        packet = [
            _display_context_view(contexts_by_id[value])
            for value in participant["private_context_unit_ids"]
        ]
        result = _call_prompt(
            client,
            "context_salience_audit_system.txt",
            "context_salience_audit_user.txt",
            {
                "PUBLIC_TASK_JSON": {"question": task.public.question},
                "PACKET_CONTEXTS_JSON": packet,
            },
            prompt_id=f"{task.task_id}.{participant['participant_id']}.salience_audit",
            trace_dir=trace_dir / participant["participant_id"],
        )
        if result.get("format_leakage") not in {True, False}:
            raise PipelineError("Salience audit format_leakage must be boolean")
        valid_ids = {row["context_unit_id"] for row in packet}
        suspicious = _string_list(result.get("suspicious_context_ids"))
        if not set(suspicious) <= valid_ids:
            raise PipelineError("Salience audit cites context outside packet")
        return {"participant_id": participant["participant_id"], **result}

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(partition["participants"])))) as executor:
        results = list(executor.map(run, partition["participants"]))
    return {
        "accept": all(row.get("format_leakage") is False for row in results),
        "participant_results": results,
    }


def run_salience_repair(
    *,
    client: OpenAIJsonClient,
    task: SWEAtlasQATask,
    partition: dict[str, Any],
    output_dir: Path,
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove only noncanonical format outliers and re-audit packet salience."""

    current = deepcopy(partition)
    removed: set[str] = set()
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, config.leakage_repair_rounds + 1):
        result = run_salience_audit(
            client=client,
            task=task,
            partition=current,
            trace_dir=output_dir / f"round_{round_index:02d}" / "traces",
            max_workers=config.max_workers,
        )
        rounds.append({"round": round_index, **result})
        _write_json(output_dir / f"round_{round_index:02d}" / "audit.json", result)
        if result["accept"]:
            summary = {
                "accept": True,
                "rounds_run": round_index,
                "removed_context_ids": sorted(removed),
                "rounds": rounds,
            }
            _write_json(output_dir / "summary.json", summary)
            return current, summary
        suspicious = {
            context_id
            for participant_result in result["participant_results"]
            if participant_result.get("format_leakage") is True
            for context_id in participant_result.get("suspicious_context_ids", [])
        }
        by_id = {row["context_unit_id"]: row for row in current["contexts"]}
        canonical = sorted(
            context_id
            for context_id in suspicious
            if by_id.get(context_id, {}).get("hidden_contains_riu_ids")
        )
        if canonical:
            raise PipelineError(f"Canonical contexts remain format-salient: {canonical}")
        removable = {context_id for context_id in suspicious if context_id in by_id}
        if not removable:
            raise PipelineError("Salience audit found no removable noncanonical context")
        removed.update(removable)
        _remove_contexts(current, removable)
        mechanical = validate_partition_mechanically(current, config)
        if not mechanical["valid"]:
            raise PipelineError("Salience repair broke partition: " + "; ".join(mechanical["errors"]))
    raise PipelineError("Context salience leakage remained after maximum repair rounds")


def materialize_modes(
    *,
    task: SWEAtlasQATask,
    rius: list[dict[str, Any]],
    partition: dict[str, Any],
    output_dir: Path,
    config: PipelineConfig,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ownership = partition["riu_ownership"]
    public_task = {
        "task_id": task.task_id,
        "question": task.public.question,
        "repository": task.repository.repository,
        "base_commit": task.repository.base_commit,
    }
    single_materials = []
    riu_only_materials = []
    for riu in rius:
        owner = ownership[riu["riu_id"]]
        for atom in riu["evidence_atoms"]:
            material = {
                "material_id": f"single_{atom['atom_id']}",
                "material_type": atom["kind"],
                "text": atom["quote"],
                "source_ref": {
                    "path": atom.get("source_path"),
                    "line_range": atom.get("source_line_range"),
                    "command": atom.get("command"),
                },
            }
            single_materials.append(material)
            riu_only_materials.append(
                {
                    **material,
                    "material_id": f"riu_only_{owner}_{atom['atom_id']}",
                    "assigned_to": [owner],
                }
            )
    base = {
        "schema": "swe_atlas_collaborative_benchmark_public_v1",
        "public_task": public_task,
        "construction_profile": {
            "allocation_policy": config.allocation_policy,
            "participant_context_token_cap": config.participant_context_token_cap,
        },
    }
    single = {**base, "mode": "single_all_rius", "materials": single_materials}
    riu_only = {
        **base,
        "mode": "multi_riu_only",
        "participants": _public_participants(partition["participants"]),
        "materials": riu_only_materials,
    }
    full = {
        **base,
        "mode": "multi_full_context",
        "participants": _public_participants(partition["participants"]),
        "materials": [_display_context_view(row, include_assignee=True) for row in partition["contexts"]],
    }
    paths = {
        "single_all_rius": str(output_dir / "single_all_rius.jsonl"),
        "multi_riu_only": str(output_dir / "multi_riu_only.jsonl"),
        "multi_full_context": str(output_dir / "multi_full_context.jsonl"),
        "hidden_evaluation": str(output_dir / "evaluation.hidden.json"),
    }
    _write_jsonl(Path(paths["single_all_rius"]), [single])
    _write_jsonl(Path(paths["multi_riu_only"]), [riu_only])
    _write_jsonl(Path(paths["multi_full_context"]), [full])
    _write_json(
        Path(paths["hidden_evaluation"]),
        {
            "task_id": task.task_id,
            "rubrics": [_rubric_json(row) for row in task.evaluation.rubrics],
            "reference_answer": task.evaluation.reference_answer,
            "final_rius": rius,
            "riu_ownership": ownership,
        },
    )
    return paths


def _audit_material(rius: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for riu in rius:
        for atom in riu["evidence_atoms"]:
            result.append(
                {
                    "evidence_number": len(result) + 1,
                    "material_type": atom["kind"],
                    "source_ref": {
                        "path": atom.get("source_path"),
                        "line_range": atom.get("source_line_range"),
                        "command": atom.get("command"),
                    },
                    "text": atom["quote"],
                }
            )
    return result


def _compact_audit_feedback(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "all_complete": audit["all"]["judge_result"].get("overall_complete_and_correct"),
        "all_overall_reason": audit["all"]["judge_result"].get("overall_reason"),
        "all_failed_rubrics": [
            {key: row.get(key) for key in (
                "rubric_id", "status", "content_correct", "grounded_in_available_material", "reason")}
            for row in audit["all"]["judge_result"].get("rubric_results", [])
            if not _audit_row_satisfied(row)
        ],
        "none_complete": audit["none"]["judge_result"].get("overall_complete_and_correct"),
        "leave_one_out": [
            {
                "omitted_riu_id": row["omitted_riu_id"],
                "overall_complete": row["judge_result"].get("overall_complete_and_correct"),
                "linked_rubric_ids": row["linked_rubric_ids"],
                "broken_rubric_ids": row["broken_rubric_ids"],
                "answer": row["solver_response"].get("answer"),
            }
            for row in audit["leave_one_out"]
        ],
    }


def _build_reselection_feedback(
    *,
    task: SWEAtlasQATask,
    audit: dict[str, Any],
    final_rius: list[dict[str, Any]],
    fallback_error: str,
    failed_selection_round: int = 0,
    max_retries: int = 3,
    prior_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Turn failed strict-audit findings into binding candidate-search guidance."""

    rubric_by_id = {row.rubric_id: row for row in task.evaluation.rubrics}
    failed_positive: list[dict[str, Any]] = []
    if audit.get("all"):
        for result in audit["all"].get("judge_result", {}).get("rubric_results") or []:
            rubric_id = str(result.get("rubric_id") or "")
            rubric = rubric_by_id.get(rubric_id)
            if rubric is None or rubric.polarity != "positive" or _audit_row_satisfied(result):
                continue
            failed_positive.append(
                {
                    "rubric_id": rubric_id,
                    "rubric": _rubric_json(rubric),
                    "audit_status": result.get("status"),
                    "audit_reason": result.get("reason"),
                    "content_correct": result.get("content_correct"),
                    "grounded_in_available_material": result.get("grounded_in_available_material"),
                }
            )
    reason = fallback_error or "; ".join(audit.get("errors") or [])
    all_result = audit.get("all", {}).get("judge_result", {})
    none_result = audit.get("none", {}).get("judge_result", {})
    failed_loo = []
    for row in audit.get("leave_one_out") or []:
        if row.get("judge_result", {}).get("overall_complete_and_correct") is not True:
            continue
        failed_loo.append(
            {
                "omitted_riu_id": row.get("omitted_riu_id"),
                "linked_rubric_ids": row.get("linked_rubric_ids") or [],
                "reason": "Answer remained complete after this RIU was omitted",
            }
        )
    attempt_record = {
        "selection_round": failed_selection_round,
        "attempt_kind": "initial" if failed_selection_round == 0 else "retry",
        "reason": reason,
        "riu_count": len(final_rius),
        "all_complete": all_result.get("overall_complete_and_correct"),
        "none_complete": none_result.get("overall_complete_and_correct"),
        "failed_positive_rubrics": failed_positive,
        "unnecessary_riu_findings": failed_loo,
        "audit_errors": audit.get("errors") or [],
    }
    failure_history = [*(prior_history or []), attempt_record]
    retries_remaining = max(0, int(max_retries) - int(failed_selection_round))
    return {
        "reason": reason,
        "failed_selection_round": failed_selection_round,
        "next_retry_number": (
            failed_selection_round + 1 if retries_remaining else None
        ),
        "max_retries": max_retries,
        "retries_remaining": retries_remaining,
        "failed_positive_rubrics": failed_positive,
        "unnecessary_riu_findings": failed_loo,
        "previous_riu_count": len(final_rius),
        "previous_audit_errors": audit.get("errors") or [],
        "failure_history": failure_history,
        "instruction": (
            "Inspect the repository again and collect new exact source/test excerpts and, "
            "where required, short targeted runtime observations for every failed positive "
            "rubric. Do not merely regroup or paraphrase the previous evidence. Prefer "
            "independent physical source/runtime artifacts so RIUs remain partitionable. "
            "Use failure_history to avoid repeating approaches that already failed. If the "
            "previous result had fewer than two RIUs, search for a genuinely independent "
            "answer contribution rather than splitting one artifact artificially."
        ),
    }


def _display_log(artifact: dict[str, Any], *, canonical: bool) -> dict[str, Any]:
    command = str(artifact.get("command") or "")
    display = "\n".join(
        [
            "[run]",
            f"command: {command}",
            f"commit: {artifact.get('observed_commit') or ''}",
            f"exit_code: {artifact.get('observed_exit_code')}",
            "output:",
            _artifact_evidence_content(artifact),
        ]
    ).strip()
    return {
        "artifact_id": artifact["artifact_id"],
        "text": display,
        "estimated_tokens": estimate_tokens(display),
        "hidden_canonical": canonical,
        "family": artifact.get("family") or ("canonical" if canonical else "distractor"),
    }


def _code_context(row: dict[str, Any], riu_ids: list[str], *, canonical: bool) -> dict[str, Any]:
    text = f"[file]\npath: {row['path']}\ncontent:\n{row['text']}"
    return {
        "context_unit_id": "code_" + hashlib.sha256(row["path"].encode()).hexdigest()[:16],
        "context_type": "code",
        "text": text,
        "estimated_tokens": estimate_tokens(text),
        "source_ref": {"path": row["path"], "module_id": row["module_id"]},
        "hidden_contains_riu_ids": list(riu_ids),
        "hidden_canonical": canonical,
    }


def _log_context(row: dict[str, Any], riu_ids: list[str], *, canonical: bool) -> dict[str, Any]:
    return {
        "context_unit_id": "log_" + hashlib.sha256(row["artifact_id"].encode()).hexdigest()[:16],
        "context_type": "runtime_log",
        "text": row["text"],
        "estimated_tokens": int(row["estimated_tokens"]),
        "source_ref": {"artifact_id": row["artifact_id"], "family": row.get("family")},
        "hidden_contains_riu_ids": list(riu_ids),
        "hidden_canonical": canonical,
    }


def _remove_contexts(partition: dict[str, Any], context_ids: set[str]) -> None:
    by_id = {row["context_unit_id"]: row for row in partition["contexts"]}
    if any(by_id[value].get("hidden_contains_riu_ids") for value in context_ids if value in by_id):
        raise PipelineError("Cannot remove canonical context")
    partition["contexts"] = [row for row in partition["contexts"] if row["context_unit_id"] not in context_ids]
    for participant in partition["participants"]:
        participant["private_context_unit_ids"] = [
            value for value in participant["private_context_unit_ids"] if value not in context_ids
        ]
        participant["private_context_tokens"] = sum(
            row["estimated_tokens"]
            for row in partition["contexts"]
            if participant["participant_id"] in row.get("assigned_to", [])
        )


def _context_chunks(rows: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    used = 0
    for row in rows:
        length = len(row["text"])
        if current and used + length > max_chars:
            chunks.append(current)
            current = []
            used = 0
        current.append(row)
        used += length
    if current:
        chunks.append(current)
    return chunks


def _audit_context_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_unit_id": row["context_unit_id"],
        "context_type": row["context_type"],
        "text": row["text"],
        "canonical_for_owned_riu": bool(row.get("hidden_contains_riu_ids")),
    }


def _display_context_view(row: dict[str, Any], *, include_assignee: bool = False) -> dict[str, Any]:
    result = {
        "context_unit_id": row["context_unit_id"],
        "context_type": row["context_type"],
        "text": row["text"],
    }
    # Only expose an ordinary file path. Runtime artifact IDs, construction
    # families, and canonical/distractor labels are construction metadata and
    # would make the answer-bearing log visually identifiable.
    if row["context_type"] == "code" and (row.get("source_ref") or {}).get("path"):
        result["source_ref"] = {"path": row["source_ref"]["path"]}
    if include_assignee:
        result["assigned_to"] = list(row.get("assigned_to") or [])
    return result


def _public_participants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "participant_id": row["participant_id"],
            "role_name": row["role_name"],
            "private_context_unit_ids": row["private_context_unit_ids"],
            "private_context_tokens": row["private_context_tokens"],
        }
        for row in rows
    ]


def _corpus_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path_count": len(rows),
        "estimated_tokens": sum(row["estimated_tokens"] for row in rows),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "files": [_corpus_row_without_text(row) for row in rows],
    }


def _corpus_row_without_text(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "text"}


def _is_distributable_path(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered = path.casefold()
    if any(lowered.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    # build/dist/target are excluded only as repository-root output trees via
    # EXCLUDED_PREFIXES. They are also legitimate nested source module names
    # (for example internal/target/queue).
    if any(part in {"node_modules", "vendor", "__pycache__"} for part in pure.parts):
        return False
    return pure.name in TEXT_FILENAMES or pure.suffix.casefold() in TEXT_SUFFIXES


def _module_id(path: str) -> str:
    pure = PurePosixPath(path)
    parent = str(pure.parent)
    return "root" if parent == "." else parent


def _near_canonical_module(module_id: str, canonical_modules: set[str]) -> bool:
    return any(
        module_id == canonical
        or module_id.startswith(canonical + "/")
        or canonical.startswith(module_id + "/")
        for canonical in canonical_modules
    )


def _normalize_context_text(text: str) -> str:
    return " ".join(str(text).split())


def _audit_row_satisfied(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "satisfied"
        and row.get("content_correct") is True
        and row.get("grounded_in_available_material") is True
    )


def _participant_id(index: int) -> str:
    value = index
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"participant_{letters}"


def _rubric_json(row: Rubric) -> dict[str, str]:
    return {
        "id": row.rubric_id,
        "title": row.title,
        "polarity": row.polarity,
        "importance": row.importance,
    }


def _call_prompt(
    client: OpenAIJsonClient,
    system_name: str,
    user_name: str,
    variables: dict[str, Any],
    *,
    prompt_id: str,
    trace_dir: Path,
) -> dict[str, Any]:
    system = (PROMPT_DIR / system_name).read_text(encoding="utf-8")
    user = (PROMPT_DIR / user_name).read_text(encoding="utf-8")
    for key, value in variables.items():
        replacement = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        user = user.replace("{{" + key + "}}", replacement)
    return client.generate_json(
        prompt_id=prompt_id,
        system_prompt=system,
        user_prompt=user,
        trace_dir=trace_dir,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    atomic_json(path, payload)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument(
        "--output-root", "--output-dir", dest="output_dir", required=True,
        help="Output directory for this task's construction run.",
    )
    parser.add_argument("--candidate-dir", default=None)
    parser.add_argument("--allocation-policy", choices=("one_each", "fixed_team_balanced"), default="one_each")
    parser.add_argument("--participant-count", type=int, default=None)
    parser.add_argument("--participant-context-token-cap", type=int, default=40_000)
    parser.add_argument("--audit-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--riu-reselection-attempts",
        type=int,
        default=3,
        help="PIU reselection retries after the initial attempt (default: 3).",
    )
    parser.add_argument("--candidate-runtime-repair-attempts", type=int, default=1)
    parser.add_argument("--experiment-repair-attempts", type=int, default=2)
    parser.add_argument("--leakage-repair-rounds", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-distractor-logs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-after-riu-audit", action="store_true")
    parser.add_argument("--docker-timeout-sec", type=int, default=900)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    parser.add_argument("--reasoning-effort", default="medium")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            credentials_file=args.credentials_file,
            max_output_tokens=12_000,
            reasoning_effort=args.reasoning_effort,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(Path(args.output_dir) / "llm_requests.jsonl"),
        )
    )
    report = build_task(
        task=load_qa_task(args.task_dir),
        output_dir=args.output_dir,
        client=client,
        config=PipelineConfig(
            allocation_policy=args.allocation_policy,
            participant_count=args.participant_count,
            participant_context_token_cap=args.participant_context_token_cap,
            audit_repair_attempts=args.audit_repair_attempts,
            riu_reselection_attempts=args.riu_reselection_attempts,
            candidate_runtime_repair_attempts=args.candidate_runtime_repair_attempts,
            experiment_repair_attempts=args.experiment_repair_attempts,
            leakage_repair_rounds=args.leakage_repair_rounds,
            max_workers=args.max_workers,
            min_distractor_logs=args.min_distractor_logs,
            seed=args.seed,
            stop_after_riu_audit=args.stop_after_riu_audit,
        ),
        candidate_dir=args.candidate_dir,
        docker_timeout_sec=args.docker_timeout_sec,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
