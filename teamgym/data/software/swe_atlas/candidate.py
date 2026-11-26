"""Construct mechanically verified PIU candidates for SWE-Atlas QA tasks.

This module intentionally stops before All/None/LOO sufficiency auditing.  It
uses only the public question to plan repository inspection and runtime
observations. Hidden criteria are available later to a constructor that groups
already collected evidence, without repository search access.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
from typing import Any
from types import SimpleNamespace
from uuid import uuid4

from teamgym.llm.construction_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)

from .schema import Rubric, SWEAtlasQATask
from .source import load_qa_task
from .integrity import canonical_hash, run_identity


SCHEMA = "swe_atlas_riu_candidates_v1"
PROMPT_DIR = (
    Path(__file__).resolve().parents[3] / "prompts" / "software" / "swe_atlas"
)


class CandidateValidationError(ValueError):
    """Raised when planned or synthesized evidence is not mechanically valid."""


class CandidateCollectionExhausted(CandidateValidationError):
    """The current plan ran out of repairs before collecting adequate evidence."""


class DockerEvidenceCollector:
    """Collect immutable source and clean runtime observations from one image."""

    def __init__(
        self,
        image: str,
        expected_commit: str,
        *,
        timeout_sec: int = 900,
        runtime_workers: int = 4,
    ):
        self.image = image
        self.expected_commit = expected_commit
        self.timeout_sec = timeout_sec
        self.runtime_workers = max(1, int(runtime_workers))
        self._preflight_cache: dict[str, Any] | None = None

    def preflight(self) -> dict[str, Any]:
        """Inspect commit, paths, tools, and config shape in one cheap container run."""

        if self._preflight_cache is not None:
            return self._preflight_cache
        result = self._run(
            r'''
cd /app
printf '__COMMIT__=%s\n' "$(git rev-parse HEAD)"
printf '__RUNTIME_UID__=%s\n' "$(id -u)"
printf '__RUNTIME_USER__=%s\n' "$(id -un)"
printf '__PATHS_BEGIN__\n'
git ls-tree -r --name-only HEAD
printf '__PATHS_END__\n'
printf '__TOOLS_BEGIN__\n'
for item in \
  /app/venv/bin/python /app/venv/bin/python3 \
  /app/venv/bin/alembic /app/venv/bin/pytest /app/venv/bin/flask \
  /app/.venv/bin/python /app/.venv/bin/python3 \
  /app/.venv/bin/alembic /app/.venv/bin/pytest /app/.venv/bin/flask \
  /usr/lib/postgresql/*/bin/postgres /usr/lib/postgresql/*/bin/initdb \
  /usr/lib/postgresql/*/bin/pg_ctl /usr/lib/postgresql/*/bin/psql \
  /usr/lib/postgresql/*/bin/createdb /usr/bin/redis-server \
  /usr/local/go/bin/go /usr/local/bin/python3 /usr/bin/python3 \
  /usr/local/bin/python /usr/bin/python /usr/bin/node /usr/bin/npm; do
  if [ -x "$item" ]; then printf '%s\n' "$item"; fi
done
printf '__TOOLS_END__\n'
printf '__CONFIG_BEGIN__\n'
git ls-tree -r --name-only HEAD \
  | grep -E '(^|/)([^/]*\.env([^/]*)?|\.env([^/]*)?|Dockerfile|docker-compose[^/]*|compose\.(ya?ml|json))$' \
  | head -40 \
  | while IFS= read -r cfg; do
      printf '__CONFIG_FILE__=%s\n' "$cfg"
      git show "HEAD:$cfg" 2>/dev/null \
        | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' \
        | sort -u \
        | while IFS= read -r key; do printf '__CONFIG_KEY__=%s:%s\n' "$cfg" "$key"; done
    done
printf '__CONFIG_END__\n'
'''
        )
        if result["runner_exit_code"] != 0:
            raise CandidateValidationError("Unable to preflight /app in task image")
        content = result["content"]
        commit = _marker_text(content, "__COMMIT__=")
        if commit != self.expected_commit:
            raise CandidateValidationError(
                f"Container commit {commit!r} != expected {self.expected_commit!r}"
            )
        paths = _between_markers(content, "__PATHS_BEGIN__", "__PATHS_END__")
        tool_rows = _between_markers(content, "__TOOLS_BEGIN__", "__TOOLS_END__")
        tools: dict[str, str] = {}
        for path in tool_rows:
            if not path.startswith("/"):
                continue
            key = (
                f"repository_{PurePosixPath(path).name}"
                if path.startswith(("/app/venv/", "/app/.venv/"))
                else PurePosixPath(path).name
            )
            tools.setdefault(key, path)
        config_files: dict[str, list[str]] = {}
        for line in _between_markers(content, "__CONFIG_BEGIN__", "__CONFIG_END__"):
            if line.startswith("__CONFIG_FILE__="):
                config_files.setdefault(line.split("=", 1)[1], [])
            elif line.startswith("__CONFIG_KEY__="):
                payload = line.split("=", 1)[1]
                path, _, key = payload.rpartition(":")
                if path and key:
                    config_files.setdefault(path, []).append(key)
        self._preflight_cache = {
            "commit": commit,
            "runtime_uid": _marker_int(content, "__RUNTIME_UID__="),
            "runtime_user": _marker_text(content, "__RUNTIME_USER__="),
            "paths": paths,
            "tool_paths": tools,
            "config_files": [
                {"path": path, "keys": sorted(set(keys))}
                for path, keys in config_files.items()
            ],
        }
        return self._preflight_cache

    def inventory(self) -> dict[str, Any]:
        preflight = self.preflight()
        return {"commit": preflight["commit"], "paths": preflight["paths"]}

    def tool_paths(self) -> dict[str, str]:
        """Report common tools, including image-installed tools omitted from PATH."""

        return dict(self.preflight()["tool_paths"])

    def collect(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any] | None] = [None] * len(requests)
        runtime_rows: list[tuple[int, dict[str, Any]]] = []
        for index, request in enumerate(requests):
            if request["kind"] == "source":
                artifacts[index] = self._source(request)
            else:
                runtime_rows.append((index, request))
        if runtime_rows:
            workers = min(self.runtime_workers, len(runtime_rows))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                collected = executor.map(self._runtime, [row[1] for row in runtime_rows])
                for (index, _), artifact in zip(runtime_rows, collected, strict=True):
                    artifacts[index] = artifact
        return [row for row in artifacts if row is not None]

    def _source(self, request: dict[str, Any]) -> dict[str, Any]:
        path = _safe_repo_path(request.get("path"))
        start = max(1, int(request.get("line_start") or 1))
        end_raw = request.get("line_end")
        if end_raw is None:
            command = f"cd /app && git show HEAD:{shlex.quote(path)}"
            line_range = "full_file"
        else:
            end = max(start, int(end_raw))
            command = (
                f"cd /app && git show HEAD:{shlex.quote(path)} "
                f"| sed -n {shlex.quote(f'{start},{end}p')}"
            )
            line_range = f"{start}-{end}"
        result = self._run(command)
        verified = result["runner_exit_code"] == 0
        return {
            "artifact_id": request["request_id"],
            "kind": "source",
            "path": path,
            "line_range": line_range,
            "purpose": request.get("purpose", ""),
            "supports_rubric_ids": request["supports_rubric_ids"],
            "content": result["content"],
            "sha256": _sha256(result["content"]),
            "verified": verified,
        }

    def _runtime(self, request: dict[str, Any]) -> dict[str, Any]:
        command = str(request.get("command") or "").strip()
        _validate_runtime_command(command)
        expected_exit = int(request.get("expected_exit_code", 0))
        expected_output = _string_list(request.get("expected_output_substrings"))
        # Encode command output before placing it beside collector metadata.
        # Printed marker-like lines remain ordinary observations, never controls.
        wrapped = f"""
cd /app || exit 125
_swe_atlas_capture_dir=$(mktemp -d /tmp/swe-atlas-capture.XXXXXX) || exit 125
trap 'rm -rf "$_swe_atlas_capture_dir"' EXIT
_swe_atlas_commit=$(git rev-parse HEAD) || exit 125
_swe_atlas_dirty_before=$(git status --porcelain | wc -l | tr -d ' ')
set +e
(
{command}
) >"$_swe_atlas_capture_dir/output" 2>&1
_swe_atlas_command_exit=$?
set -e
printf '__CAPTURE_SCHEMA__=swe_atlas_runtime_capture_v1\\n'
printf '__COMMIT__=%s\\n' "$_swe_atlas_commit"
printf '__DIRTY_BEFORE__=%s\\n' "$_swe_atlas_dirty_before"
printf '__DIRTY_AFTER__=%s\\n' "$(git status --porcelain | wc -l | tr -d ' ')"
printf '__OUTPUT_BASE64__='
base64 <"$_swe_atlas_capture_dir/output" | tr -d '\\n'
printf '\\n'
exit "$_swe_atlas_command_exit"
"""
        try:
            result = self._run(wrapped)
        except CandidateValidationError as exc:
            content = f"__COLLECTION_ERROR__={exc}"
            return {
                "artifact_id": request["request_id"],
                "kind": "runtime",
                "command": command,
                "expected_exit_code": expected_exit,
                "expected_output_substrings": expected_output,
                "observed_exit_code": None,
                "repository_dirty_before": None,
                "repository_dirty_after": None,
                "observed_commit": None,
                "purpose": request.get("purpose", ""),
                "supports_rubric_ids": request["supports_rubric_ids"],
                "content": content,
                "sha256": _sha256(content),
                "verified": False,
                "collection_error": str(exc),
            }
        content = result["content"]
        observed_exit = result["runner_exit_code"]
        capture_error = None
        try:
            capture = _decode_runtime_capture(content)
        except CandidateValidationError as exc:
            capture = {}
            capture_error = str(exc)
        dirty_before = capture.get("dirty_before")
        dirty_after = capture.get("dirty_after")
        commit = capture.get("commit")
        observed_content = capture.get("output", "")
        missing_expected_output = [
            value for value in expected_output if value not in observed_content
        ]
        verified = (
            capture_error is None
            and type(observed_exit) is int
            and observed_exit == expected_exit
            and dirty_before is not None
            and dirty_before == dirty_after
            and commit == self.expected_commit
            and bool(observed_content.strip())
            and not missing_expected_output
        )
        return {
            "artifact_id": request["request_id"],
            "kind": "runtime",
            "command": command,
            "expected_exit_code": expected_exit,
            "expected_output_substrings": expected_output,
            "missing_expected_output_substrings": missing_expected_output,
            "observed_exit_code": observed_exit,
            "collection_error": capture_error,
            "repository_dirty_before": dirty_before,
            "repository_dirty_after": dirty_after,
            "observed_commit": commit,
            "purpose": request.get("purpose", ""),
            "supports_rubric_ids": request["supports_rubric_ids"],
            "content": content,
            "observed_content": observed_content,
            "sha256": _sha256(content),
            "verified": verified,
        }

    def _run(self, script: str) -> dict[str, Any]:
        container = f"teamgym-atlas-evidence-{uuid4().hex}"
        script = (
            "export PATH=/usr/local/go/bin:/usr/local/bin:$PATH\n"
            "git config --global --add safe.directory /app >/dev/null 2>&1 || true\n"
            + script
        )
        try:
            completed = subprocess.run(
                [
                    "docker", "run", "--rm", "--name", container,
                    "--platform", "linux/amd64", "--network", "none",
                    "--entrypoint", "/bin/bash", self.image, "-lc", script,
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            message = f"Docker evidence command timed out after {self.timeout_sec}s"
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "--force", container],
                    text=True, capture_output=True, check=False, timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                raise CandidateValidationError(
                    f"{message}; could not remove {container}: {type(cleanup_error).__name__}"
                ) from exc
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                raise CandidateValidationError(
                    f"{message}; could not remove {container}: {cleanup.stderr.strip()}"
                ) from exc
            raise CandidateValidationError(message) from exc
        return {
            "runner_exit_code": completed.returncode,
            "content": completed.stdout + completed.stderr,
        }


def generate_riu_candidates(
    *,
    task: SWEAtlasQATask,
    output_dir: str | Path,
    client: OpenAIJsonClient,
    docker_timeout_sec: int = 900,
    max_repository_paths: int = 4000,
    repair_attempts: int = 2,
    runtime_repair_attempts: int = 1,
    reuse_plan: bool = False,
    runtime_workers: int = 4,
    construction_feedback: dict[str, Any] | None = None,
    synthesis_client: OpenAIJsonClient | None = None,
) -> dict[str, Any]:
    """Plan, collect, synthesize, and validate candidates for one QA task."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_dir = output / "llm_traces"
    collector = DockerEvidenceCollector(
        task.runtime.docker_image,
        task.repository.base_commit,
        timeout_sec=docker_timeout_sec,
        runtime_workers=runtime_workers,
    )
    preflight = collector.preflight()
    inventory = {"commit": preflight["commit"], "paths": preflight["paths"]}
    tool_paths = preflight["tool_paths"]
    _write_json(output / "repository_inventory.json", inventory)
    _write_json(output / "runtime_preflight.json", preflight)
    paths = _bounded_paths(
        inventory["paths"],
        _construction_text(task),
        max_repository_paths,
    )
    planning_rubrics = (Rubric('public_question', task.public.question, 'positive', 'public task', {}),)
    # The blind searcher judges whether the public question requires execution.
    # Private rubric wording must not determine its collection requests.
    evidence_mode_constraints = {}
    planner_input = {
        "task_id": task.task_id,
        "question": task.public.question,
        "rubrics": [_rubric_json(row) for row in planning_rubrics],
        "repository": {
            "name": task.repository.repository,
            "base_commit": task.repository.base_commit,
            "language": task.repository.language,
        },
        "repository_paths": paths,
        "container_tool_paths": tool_paths,
        "runtime_preflight": {
            "config_files": preflight["config_files"],
            "runtime_uid": preflight["runtime_uid"],
            "runtime_user": preflight["runtime_user"],
            "runtime_request_limit": 1,
            "runtime_repair_attempts": max(0, int(runtime_repair_attempts)),
        },
        "evidence_mode_constraints": evidence_mode_constraints,
        "construction_feedback": {'previous_attempt_needs_broader_public_evidence': bool(construction_feedback)},
        "response_schema": _plan_schema(),
    }
    plan_path = output / "evidence_plan.json"
    plan_identity = run_identity(task, client, SimpleNamespace(
        max_repository_paths=max_repository_paths,
        repair_attempts=repair_attempts, runtime_repair_attempts=runtime_repair_attempts,
        runtime_workers=runtime_workers,
        synthesis_model=(synthesis_client or client).cfg.model), docker_timeout_sec=docker_timeout_sec)
    plan_receipt_path = output / 'plan_cache_receipt.json'
    candidate_client = synthesis_client or client
    if reuse_plan and plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        try:
            receipt = json.loads(plan_receipt_path.read_text())
            if receipt != {'identity_sha256': canonical_hash(plan_identity), 'plan_sha256': canonical_hash(plan)}:
                raise ValueError('plan inputs changed')
        except (OSError, ValueError) as exc:
            raise CandidateValidationError('Missing/stale plan receipt; generate a fresh evidence plan') from exc
    else:
        plan = client.generate_json(
            prompt_id=f"{task.task_id}.candidate_plan",
            system_prompt=_read_prompt("candidate_plan_system.txt"),
            user_prompt=_read_prompt("candidate_plan_user.txt").replace(
                "{{INPUT_JSON}}", json.dumps(planner_input, ensure_ascii=False)
            ),
            trace_dir=trace_dir,
        )
    for attempt in range(repair_attempts + 1):
        try:
            requests = validate_plan(
                plan,
                planning_rubrics,
                set(inventory["paths"]),
                evidence_mode_constraints=evidence_mode_constraints,
            )
            break
        except CandidateValidationError as exc:
            if attempt >= repair_attempts:
                raise
            plan_repair_client = (
                candidate_client
                if synthesis_client is not None and attempt + 1 >= repair_attempts
                else client
            )
            plan = plan_repair_client.generate_json(
                prompt_id=f"{task.task_id}.candidate_plan_repair_{attempt + 1}",
                system_prompt=_read_prompt("candidate_plan_system.txt"),
                user_prompt=json.dumps(
                    {
                        **planner_input,
                        "previous_response": plan,
                        "validation_error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                trace_dir=trace_dir,
            )
    _write_json(output / "evidence_plan.json", plan)
    _write_json(plan_receipt_path, {'identity_sha256': canonical_hash(plan_identity),
                                  'plan_sha256': canonical_hash(plan)})
    runtime_repair_attempts = max(0, int(runtime_repair_attempts))
    coverage_history = []
    for collection_attempt in range(runtime_repair_attempts + 1):
        artifacts = collector.collect(requests)
        _write_json(output / "evidence_artifacts.json", {"artifacts": artifacts})
        _write_json(output / f"collection_{collection_attempt:02d}.artifacts.json", {"artifacts": artifacts})
        failed = [row for row in artifacts if row.get("verified") is not True]
        remaining = runtime_repair_attempts - collection_attempt
        if not failed:
            coverage = _review_public_coverage(
                client=client, task_id=task.task_id, planner_input=planner_input,
                plan=plan, artifacts=artifacts, remaining_revisions=remaining,
                trace_dir=trace_dir, round_number=collection_attempt,
                validation_retries=repair_attempts)
            coverage_history.append(coverage)
            _write_json(output / f"public_coverage_{collection_attempt:02d}.json", coverage)
            if coverage["adequate_for_public_question"]:
                (output / "runtime_failure.json").unlink(missing_ok=True)
                break
            if not remaining:
                raise CandidateCollectionExhausted(
                    "Public-question evidence coverage incomplete after bounded collection revisions"
                )
            plan = coverage['replacement_plan']
        else:
            if not remaining:
                _write_json(output / "runtime_failure.json", {
                    "status": "runtime_evidence_failed", "attempts_run": collection_attempt + 1,
                    "failed_artifacts": [_artifact_for_prompt(row) for row in failed]})
                raise CandidateCollectionExhausted(
                    f"Unverified evidence artifacts: {[row['artifact_id'] for row in failed]}"
                )
            plan = client.generate_json(
                prompt_id=f"{task.task_id}.candidate_collection_repair_{collection_attempt + 1}",
                system_prompt=_read_prompt("candidate_plan_system.txt"),
                user_prompt=json.dumps({**planner_input, "previous_response": plan,
                    "failed_artifacts": [_public_collection_artifact(row) for row in failed],
                    "verified_source_artifacts": [_public_collection_artifact(row) for row in artifacts
                                                  if row.get('verified') is True and row['kind'] == 'source'],
                    "repair_instruction": "Return one complete replacement plan grounded in the supplied public source. Repair the failed probe without dropping required setup/cleanup; use installed tools, no network/install, and at most one integrated runtime request."}, ensure_ascii=False),
                trace_dir=trace_dir)
        # The same public-only validator is used for coverage and runtime repair.
        # Neither a failed hidden criterion nor a hidden judge rationale is sent.
        for validation_attempt in range(repair_attempts + 1):
            try:
                requests = validate_plan(plan, planning_rubrics, set(inventory["paths"]),
                                         evidence_mode_constraints=evidence_mode_constraints)
                break
            except CandidateValidationError as exc:
                if validation_attempt >= repair_attempts:
                    raise
                plan = client.generate_json(
                    prompt_id=f"{task.task_id}.candidate_collection_plan_validation_repair_{collection_attempt + 1}_{validation_attempt + 1}",
                    system_prompt=_read_prompt("candidate_plan_system.txt"),
                    user_prompt=json.dumps({**planner_input, "previous_response": plan,
                        "validation_error": str(exc), "verified_source_artifacts": [
                            _public_collection_artifact(row) for row in artifacts
                            if row.get('verified') is True and row['kind'] == 'source']}, ensure_ascii=False),
                    trace_dir=trace_dir)
        _write_json(output / f"evidence_plan.collection_revision_{collection_attempt + 1}.json", plan)
        _write_json(plan_path, plan)
        _write_json(plan_receipt_path, {'identity_sha256': canonical_hash(plan_identity),
                                      'plan_sha256': canonical_hash(plan)})

    # Criteria-aware grouping occurs only after blind collection has finished.
    # This constructor sees frozen artifacts and has no search/execute tools.
    requirement_modes = _source_first_evidence_mode_constraints(task.evaluation.rubrics)
    hidden_requirements = [{'rubric_id': r.rubric_id,
        'evidence_mode': requirement_modes[r.rubric_id]} for r in task.evaluation.rubrics]
    synthesis_input = {
        "task_id": task.task_id,
        "question": task.public.question,
        "rubrics": [_rubric_json(row) for row in task.evaluation.rubrics],
        "evidence_requirements": hidden_requirements,
        "private_construction_feedback": construction_feedback or {},
        "verified_artifacts": [
            _artifact_for_prompt(row, include_runtime_command=False)
            for row in artifacts
        ],
        "response_schema": _candidate_schema(),
    }
    candidate_response = candidate_client.generate_json(
        prompt_id=f"{task.task_id}.candidate_synthesis",
        system_prompt=_read_prompt("candidate_synthesis_system.txt"),
        user_prompt=_read_prompt("candidate_synthesis_user.txt").replace(
            "{{INPUT_JSON}}", json.dumps(synthesis_input, ensure_ascii=False)
        ),
        trace_dir=trace_dir,
    )
    for attempt in range(repair_attempts + 1):
        try:
            candidates = validate_candidates(
                candidate_response,
                task.evaluation.rubrics,
                hidden_requirements,
                artifacts,
            )
            break
        except CandidateValidationError as exc:
            if attempt >= repair_attempts:
                raise
            candidate_response = candidate_client.generate_json(
                prompt_id=f"{task.task_id}.candidate_synthesis_repair_{attempt + 1}",
                system_prompt=_read_prompt("candidate_synthesis_system.txt"),
                user_prompt=json.dumps(
                    {
                        **synthesis_input,
                        "previous_response": candidate_response,
                        "validation_error": str(exc),
                    },
                    ensure_ascii=False,
                ),
                trace_dir=trace_dir,
            )
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task.task_id,
        "question": task.public.question,
        "rubrics": [_rubric_json(row) for row in task.evaluation.rubrics],
        "reference_answer_used_for_construction": False,
        "repository_search_policy": "public_question_only; criteria_private_to_frozen_evidence_constructor",
        "planner_model": client.cfg.model,
        "synthesis_model": candidate_client.cfg.model,
        "model": candidate_client.cfg.model,
        "evidence_plan": plan,
        "public_coverage_history": coverage_history,
        "collection_attempt_limit": runtime_repair_attempts + 1,
        "candidates": candidates,
        "audit_status": "not_run_candidate_stage_only",
    }
    _write_json(output / "riu_candidates.json", result)
    return result



def _public_collection_artifact(row: dict[str, Any]) -> dict[str, Any]:
    # Raw observations only; exclude rubric mappings and any constructor fields.
    keys = ('artifact_id', 'kind', 'path', 'line_range', 'command', 'content',
            'observed_content', 'observed_exit_code', 'verified')
    return {key: row[key] for key in keys if key in row}


def _review_public_coverage(*, client, task_id, planner_input, plan, artifacts,
                            remaining_revisions, trace_dir, round_number,
                            validation_retries=2) -> dict[str, Any]:
    system = _read_prompt('candidate_coverage_system.txt')
    public_input = _read_prompt('candidate_coverage_user.txt').replace(
            '{{PUBLIC_PLAN_INPUT_JSON}}', json.dumps(planner_input, ensure_ascii=False)).replace(
            '{{PLAN_JSON}}', json.dumps(plan, ensure_ascii=False)).replace(
            '{{ARTIFACTS_JSON}}', json.dumps([_public_collection_artifact(row) for row in artifacts], ensure_ascii=False)).replace(
            '{{REMAINING_REVISIONS}}', str(remaining_revisions))
    user = public_input
    for attempt in range(max(0, int(validation_retries)) + 1):
        response = None
        suffix = '' if attempt == 0 else f'.format_repair_{attempt}'
        try:
            response = client.generate_json(
                prompt_id=f"{task_id}.candidate_public_coverage_{round_number}{suffix}",
                system_prompt=system, user_prompt=user, trace_dir=trace_dir)
            return _validate_public_coverage_response(response)
        except ValueError as exc:
            if attempt >= max(0, int(validation_retries)):
                raise CandidateValidationError(
                    f'Public coverage format retries exhausted: {exc}') from exc
            user = json.dumps({
                'public_coverage_input': public_input,
                'previous_response': response,
                'validation_error': str(exc),
                'instruction': 'Return the required coverage-review JSON only. Commands and tool requests '
                               'are not executed. If more evidence is needed, use missing_public_aspects '
                               'and replacement_plan; do not request a shell tool.',
            }, ensure_ascii=False)
    raise AssertionError('Unreachable coverage-review loop')


def _validate_public_coverage_response(response) -> dict[str, Any]:
    if not isinstance(response, dict) or not isinstance(response.get('adequate_for_public_question'), bool):
        raise CandidateValidationError('Public coverage decision must be a strict boolean')
    missing = response.get('missing_public_aspects')
    if not isinstance(missing, list) or not all(isinstance(value, str) for value in missing):
        raise CandidateValidationError('Public coverage missing aspects must be a string list')
    if response['adequate_for_public_question']:
        if missing or response.get('replacement_plan') is not None:
            raise CandidateValidationError('Adequate public coverage contradicts requested replacement/missing aspects')
    elif not missing or not isinstance(response.get('replacement_plan'), dict):
        raise CandidateValidationError('Incomplete public coverage requires missing aspects and a replacement plan')
    return response


def validate_plan(
    plan: dict[str, Any],
    rubrics: tuple[Rubric, ...],
    repository_paths: set[str],
    *,
    evidence_mode_constraints: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    valid_ids = {row.rubric_id for row in rubrics}
    positive_ids = {row.rubric_id for row in rubrics if row.polarity == "positive"}
    requirements = plan.get("rubric_requirements")
    requests = plan.get("evidence_requests")
    if not isinstance(requirements, list) or not isinstance(requests, list) or not requests:
        raise CandidateValidationError("Plan needs rubric_requirements and evidence_requests")
    requirement_by_id: dict[str, dict[str, Any]] = {}
    for row in requirements:
        rubric_id = str(row.get("rubric_id") or "")
        mode = str(row.get("evidence_mode") or "")
        if rubric_id not in valid_ids or rubric_id in requirement_by_id:
            raise CandidateValidationError(f"Invalid requirement rubric id: {rubric_id!r}")
        if mode not in {"source", "hybrid"}:
            raise CandidateValidationError(f"Rubric {rubric_id} has invalid evidence_mode {mode!r}")
        required_mode = (evidence_mode_constraints or {}).get(rubric_id)
        if required_mode and mode != required_mode:
            raise CandidateValidationError(
                f"Rubric {rubric_id} must use source-first mode {required_mode!r}, got {mode!r}"
            )
        requirement_by_id[rubric_id] = row
    if set(requirement_by_id) != valid_ids:
        raise CandidateValidationError("Every rubric needs exactly one evidence requirement")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    coverage: dict[str, set[str]] = {rubric_id: set() for rubric_id in valid_ids}
    for raw in requests:
        if not isinstance(raw, dict):
            raise CandidateValidationError("Evidence request must be an object")
        request_id = str(raw.get("request_id") or "")
        kind = str(raw.get("kind") or "")
        support = _string_list(raw.get("supports_rubric_ids"))
        if not request_id or request_id in seen:
            raise CandidateValidationError(f"Missing or duplicate request_id {request_id!r}")
        if kind not in {"source", "runtime"}:
            raise CandidateValidationError(f"Invalid request kind {kind!r}")
        if not support or not set(support) <= valid_ids:
            raise CandidateValidationError(f"Invalid rubric mapping for {request_id}")
        if kind == "source":
            path = _safe_repo_path(raw.get("path"))
            if path not in repository_paths:
                raise CandidateValidationError(f"Unknown source path: {path}")
        else:
            _validate_runtime_command(str(raw.get("command") or ""))
            expected_output = _string_list(raw.get("expected_output_substrings"))
            if not expected_output:
                raise CandidateValidationError(
                    f"Runtime request {request_id} needs expected_output_substrings"
                )
        seen.add(request_id)
        for rubric_id in support:
            coverage[rubric_id].add(kind)
        normalized.append(raw)
    runtime_requests = [row for row in normalized if row.get("kind") == "runtime"]
    if len(runtime_requests) > 1:
        raise CandidateValidationError(
            "Evidence plan may contain at most one integrated runtime request"
        )
    constrained_source_ids = {
        rubric_id
        for rubric_id, mode in (evidence_mode_constraints or {}).items()
        if mode == "source"
    }
    runtime_source_overlap = sorted(
        rubric_id
        for row in runtime_requests
        for rubric_id in _string_list(row.get("supports_rubric_ids"))
        if rubric_id in constrained_source_ids
    )
    if runtime_source_overlap:
        raise CandidateValidationError(
            "Integrated runtime request includes source-only rubrics: "
            f"{runtime_source_overlap}"
        )
    for rubric_id in positive_ids:
        kinds = coverage[rubric_id]
        mode = requirement_by_id[rubric_id]["evidence_mode"]
        if "source" not in kinds:
            raise CandidateValidationError(f"Rubric {rubric_id} has no source anchor")
        if mode == "hybrid" and "runtime" not in kinds:
            raise CandidateValidationError(f"Hybrid rubric {rubric_id} has no runtime observation")
    return normalized


def validate_candidates(
    response: dict[str, Any],
    rubrics: tuple[Rubric, ...],
    requirements: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = response.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise CandidateValidationError("Synthesizer returned no candidates")
    artifact_by_id = {row["artifact_id"]: row for row in artifacts if row["verified"]}
    positive_ids = {row.rubric_id for row in rubrics if row.polarity == "positive"}
    negative_ids = {row.rubric_id for row in rubrics if row.polarity == "negative"}
    requirement_mode = {row["rubric_id"]: row["evidence_mode"] for row in requirements}
    seen: set[str] = set()
    covered: set[str] = set()
    refuted: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        candidate_id = str(raw.get("candidate_id") or "")
        supports = set(_string_list(raw.get("supports_rubric_ids")))
        refutes = set(_string_list(raw.get("refutes_rubric_ids")))
        evidence = raw.get("evidence")
        if not candidate_id or candidate_id in seen:
            raise CandidateValidationError(f"Missing or duplicate candidate id {candidate_id!r}")
        if not supports <= positive_ids or not refutes <= negative_ids or not (supports or refutes):
            raise CandidateValidationError(f"Invalid rubric mapping for {candidate_id}")
        if not isinstance(evidence, list) or not evidence:
            raise CandidateValidationError(f"Candidate {candidate_id} has no evidence")
        kinds: set[str] = set()
        normalized_evidence: list[dict[str, str]] = []
        for excerpt in evidence:
            artifact_id = str(excerpt.get("artifact_id") or "")
            quote = str(excerpt.get("quote") or "")
            artifact = artifact_by_id.get(artifact_id)
            if artifact is None or not quote.strip():
                raise CandidateValidationError(f"Candidate {candidate_id} references invalid evidence")
            if quote not in _artifact_evidence_content(artifact):
                raise CandidateValidationError(
                    f"Candidate {candidate_id} quote is not exact in {artifact_id}"
                )
            kinds.add(artifact["kind"])
            normalized_evidence.append(
                {"artifact_id": artifact_id, "kind": artifact["kind"], "quote": quote}
            )
        if "source" not in kinds:
            raise CandidateValidationError(f"Candidate {candidate_id} is bare runtime output")
        hybrid_ids = {rid for rid in supports | refutes if requirement_mode[rid] == "hybrid"}
        if hybrid_ids and "runtime" not in kinds:
            raise CandidateValidationError(
                f"Candidate {candidate_id} lacks runtime evidence for {sorted(hybrid_ids)}"
            )
        claim = str(raw.get("claim") or "").strip()
        if not claim:
            raise CandidateValidationError(f"Candidate {candidate_id} has no claim")
        seen.add(candidate_id)
        covered.update(supports)
        refuted.update(refutes)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "claim": claim,
                "evidence_mode": "hybrid" if "runtime" in kinds else "source",
                "supports_rubric_ids": sorted(supports),
                "refutes_rubric_ids": sorted(refutes),
                "evidence": normalized_evidence,
            }
        )
    if covered != positive_ids:
        raise CandidateValidationError(
            f"Candidates miss positive rubrics: {sorted(positive_ids - covered)}"
        )
    if refuted != negative_ids:
        raise CandidateValidationError(
            f"Candidates miss negative rubrics: {sorted(negative_ids - refuted)}"
        )
    return normalized


def _validate_runtime_command(command: str) -> None:
    if not command.strip():
        raise CandidateValidationError("Runtime request has no command")
    if re.match(r"\s*(?:/bin/)?bash\s+-lc\b", command):
        raise CandidateValidationError(
            "Runtime commands already run under bash; do not wrap them in nested bash -lc"
        )
    lowered = command.casefold()
    forbidden = (
        "solution/answer.txt", "/logs/agent/answer", "git reset", "git checkout",
        "git clean", "git commit", "git apply", "sed -i", "curl ", "wget ",
        "pip install", "apt-get", "dnf ", "yum ", "apk add",
    )
    hit = next((token for token in forbidden if token in lowered), None)
    if hit:
        raise CandidateValidationError(f"Unsafe or answer-leaking runtime command contains {hit!r}")
    shell_surface = _without_heredoc_bodies(command)
    for redirect in re.findall(r"(?:>>|>)\s*([^\s;&|]+)", shell_surface):
        target = redirect.strip("'\"")
        # A redirect inside command substitution commonly ends as `$OUT)`;
        # the closing parenthesis is shell syntax, not part of the path.
        target = target.rstrip(")")
        variable = re.fullmatch(
            r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?(?:/[^\s]*)?", target
        )
        variable_is_tmp = bool(
            variable and _shell_variable_resolves_under_tmp(variable.group(1), shell_surface)
        )
        if (
            not target.startswith("/tmp/")
            and target != "/dev/null"
            and not variable_is_tmp
        ):
            raise CandidateValidationError(f"Runtime command may only write under /tmp: {target}")


def _shell_variable_resolves_under_tmp(
    name: str, shell_surface: str, *, seen: set[str] | None = None
) -> bool:
    """Recognize direct and one-or-more-level /tmp-derived shell variables."""

    visited = set(seen or ())
    if name in visited:
        return False
    visited.add(name)
    direct = re.search(
        rf"(?:^|[;\s]){re.escape(name)}=(?:['\"])?/tmp/",
        shell_surface,
    ) or re.search(
        rf"(?:^|[;\s]){re.escape(name)}=\$\([^)]*\bmktemp\b[^)]*/tmp/",
        shell_surface,
    )
    if direct:
        return True
    derived = re.search(
        rf"(?:^|[;\s]){re.escape(name)}=(?:['\"])?\$\{{?([A-Za-z_][A-Za-z0-9_]*)\}}?/",
        shell_surface,
    )
    return bool(
        derived
        and _shell_variable_resolves_under_tmp(
            derived.group(1), shell_surface, seen=visited
        )
    )


def _without_heredoc_bodies(command: str) -> str:
    """Keep shell command lines while excluding embedded heredoc language bodies."""

    kept: list[str] = []
    delimiter: str | None = None
    strip_tabs = False
    for line in command.splitlines():
        if delimiter is not None:
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate.strip() == delimiter:
                delimiter = None
                strip_tabs = False
            continue
        kept.append(line)
        match = re.search(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2", line)
        if match:
            strip_tabs = match.group(1) == "-"
            delimiter = match.group(3)
    return "\n".join(kept)


def _safe_repo_path(value: Any) -> str:
    path = str(value or "").strip()
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts:
        raise CandidateValidationError(f"Unsafe repository path: {path!r}")
    return path


def _bounded_paths(paths: list[str], hint: str, limit: int) -> list[str]:
    if len(paths) <= limit:
        return paths
    terms = {token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", hint.casefold())}
    scored = []
    for index, path in enumerate(paths):
        lowered = path.casefold()
        score = sum(1 for term in terms if term in lowered)
        if lowered.endswith(("test.py", "_test.go", ".spec.ts", ".test.ts")):
            score += 1
        scored.append((-score, index, path))
    return [row[2] for row in sorted(scored)[:limit]]


def _construction_text(task: SWEAtlasQATask) -> str:
    return task.public.question


def _source_first_evidence_mode_constraints(
    rubrics: tuple[Rubric, ...],
) -> dict[str, str]:
    """Reserve hybrid mode for irreducibly observed outputs, not runtime wording."""

    irreducible_output_patterns = (
        r"\blog messages? indicating\b",
        r"\breports? (?:the )?(?:sequence of )?log (?:line|lines|message|messages|entry|entries)\b",
        r"\bexact (?:runtime|command|test) output\b",
        r"\bobserved (?:stdout|stderr|output|test result)\b",
        r"\b(?:actual )?terminal output\b",
        r"\btest execution output\b",
        r"\btest output markers?\b",
        r"\bverbose .{0,40}\btest output\b",
        r"\bruntime (?:dissection|evidence|output)\b",
        r"\bserialized .{0,50}\b(?:bytes?|hex(?:adecimal)?|length)\b",
        r"\b(?:actual )?on-the-wire bytes?\b",
        r"\b(?:complete )?packet (?:bytes?|hex dump)\b",
        r"\b(?:checksum value|byte positions?|actual byte count|raw byte length measurements?|packet structure|layer traversal output)\b",
        r"\bpayload size threshold\b",
        r"\bat runtime returns?\b",
        r"\braises? .{0,100}\bduring .{0,30}\b(?:construction|execution|serialization)\b",
        r"\bafter (?:the )?(?:first )?(?:build|serialization|construction)\b",
        r"\b(?:remains unchanged|changes?|produces identical behavior) when\b",
        r"\b(?:passes|fails|passed|failed) when (?:run|executed)\b",
    )
    constraints: dict[str, str] = {}
    for rubric in rubrics:
        title = rubric.title.casefold()
        requires_observed_output = any(
            re.search(pattern, title) for pattern in irreducible_output_patterns
        )
        constraints[rubric.rubric_id] = (
            "hybrid" if requires_observed_output else "source"
        )
    return constraints


def _rubric_json(row: Rubric) -> dict[str, str]:
    return {
        "id": row.rubric_id,
        "title": row.title,
        "polarity": row.polarity,
        "importance": row.importance,
    }


def _artifact_for_prompt(
    row: dict[str, Any],
    *,
    include_runtime_command: bool = True,
) -> dict[str, Any]:
    keep = {
        "artifact_id", "kind", "path", "line_range", "command", "purpose",
        "supports_rubric_ids", "content", "verified",
    }
    result = {key: value for key, value in row.items() if key in keep}
    if row.get("kind") == "runtime":
        if not include_runtime_command:
            result.pop("command", None)
        result["content"] = _artifact_evidence_content(row)
    return result


def _artifact_evidence_content(artifact: dict[str, Any]) -> str:
    content = str(artifact.get("content") or "")
    if artifact.get("kind") == "runtime":
        return _runtime_observation_content(content)
    return content


def _decode_runtime_capture(content: str) -> dict[str, Any]:
    """Parse only the collector envelope; the command body is encoded data."""
    prefixes = ("__CAPTURE_SCHEMA__=", "__COMMIT__=", "__DIRTY_BEFORE__=",
                "__DIRTY_AFTER__=", "__OUTPUT_BASE64__=")
    lines = content.splitlines()
    if len(lines) != len(prefixes):
        raise CandidateValidationError("Invalid runtime capture envelope")
    values = {}
    for prefix in prefixes:
        matches = [line[len(prefix):] for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise CandidateValidationError("Missing or duplicate runtime capture metadata")
        values[prefix] = matches[0]
    if values[prefixes[0]] != "swe_atlas_runtime_capture_v1":
        raise CandidateValidationError("Unsupported runtime capture schema")
    if not all(re.fullmatch(r"[0-9]+", values[key]) for key in prefixes[2:4]):
        raise CandidateValidationError("Invalid runtime checkout metadata")
    try:
        output = base64.b64decode(values[prefixes[4]], validate=True).decode("utf-8", errors="replace").strip()
    except ValueError as exc:
        raise CandidateValidationError("Invalid encoded runtime output") from exc
    return {"commit": values[prefixes[1]], "dirty_before": int(values[prefixes[2]]),
            "dirty_after": int(values[prefixes[3]]), "output": output}


def _runtime_observation_content(content: str) -> str:
    """Hide collector bookkeeping so only command output can become RIU evidence."""

    if content.startswith("__CAPTURE_SCHEMA__="):
        try:
            return _decode_runtime_capture(content)["output"]
        except CandidateValidationError:
            return ""
    prefixes = (
        "__COMMIT__=", "__DIRTY_BEFORE__=", "__COMMAND__=",
        "__COMMAND_EXIT__=", "__DIRTY_AFTER__=",
    )
    return "\n".join(
        line for line in content.splitlines() if not line.startswith(prefixes)
    ).strip()


def _plan_schema() -> dict[str, Any]:
    return {
        "rubric_requirements": [
            {
                "rubric_id": "exact rubric id",
                "evidence_mode": "source or hybrid",
                "required_observation": "what must be established",
            }
        ],
        "evidence_requests": [
            {
                "request_id": "unique id",
                "kind": "source or runtime",
                "path": "source-only repository path",
                "line_start": "optional integer",
                "line_end": "optional integer",
                "command": "runtime-only read-only shell command",
                "expected_exit_code": 0,
                "expected_output_substrings": [
                    "literal stable output proving the required observation"
                ],
                "purpose": "why this is needed",
                "supports_rubric_ids": ["rubric ids"],
            }
        ],
    }


def _candidate_schema() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "candidate_id": "candidate_01",
                "claim": "precise answer-bearing proposition",
                "supports_rubric_ids": ["positive rubric ids"],
                "refutes_rubric_ids": ["negative rubric ids"],
                "evidence": [
                    {"artifact_id": "verified artifact id", "quote": "exact contiguous quote"}
                ],
            }
        ]
    }


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def _marker_text(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _between_markers(text: str, start: str, end: str) -> list[str]:
    collecting = False
    rows: list[str] = []
    for line in text.splitlines():
        if line == start:
            collecting = True
            continue
        if line == end:
            break
        if collecting and line:
            rows.append(line)
    return rows


def _marker_int(text: str, prefix: str) -> int | None:
    value = _marker_text(text, prefix)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--synthesis-model",
        default="",
        help="Optional stronger model for PIU synthesis; defaults to --model.",
    )
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    parser.add_argument("--docker-timeout-sec", type=int, default=900)
    parser.add_argument("--max-repository-paths", type=int, default=4000)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--runtime-repair-attempts", type=int, default=1)
    parser.add_argument("--runtime-workers", type=int, default=4)
    parser.add_argument("--reuse-plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    task = load_qa_task(args.task_dir)
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            env_file=args.env_file,
            credentials_file=args.credentials_file,
            max_output_tokens=10000,
            timeout_sec=600,
            request_log_path=str(Path(args.output_dir) / "llm_requests.jsonl"),
        )
    )
    synthesis_client = None
    if args.synthesis_model and args.synthesis_model != args.model:
        synthesis_client = OpenAIJsonClient(
            OpenAIJsonClientConfig(
                provider=args.provider,
                model=args.synthesis_model,
                base_url=args.base_url,
                endpoint=args.endpoint,
                api_key_env=args.api_key_env,
                env_file=args.env_file,
                credentials_file=args.credentials_file,
                max_output_tokens=10000,
                timeout_sec=600,
                request_log_path=str(Path(args.output_dir) / "llm_requests.jsonl"),
            )
        )
    result = generate_riu_candidates(
        task=task,
        output_dir=args.output_dir,
        client=client,
        docker_timeout_sec=args.docker_timeout_sec,
        max_repository_paths=args.max_repository_paths,
        repair_attempts=args.repair_attempts,
        runtime_repair_attempts=args.runtime_repair_attempts,
        reuse_plan=args.reuse_plan,
        runtime_workers=args.runtime_workers,
        synthesis_client=synthesis_client,
    )
    print(json.dumps({
        "task_id": result["task_id"],
        "candidate_count": len(result["candidates"]),
        "output": str(Path(args.output_dir).resolve() / "riu_candidates.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
