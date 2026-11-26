#!/usr/bin/env python3
"""Collect runtime observations from the pinned base image.

Each observation command runs in its own isolated container. The collector
compares the tracked checkout before and after execution and rejects changes
made by the command. Results are candidates for evidence construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from teamgym.data.software.deep_swe import load_task
from .evidence import usable_observations
from .checkout_integrity import CHECKOUT_HELPERS, checkout_unchanged
from .provenance import content_hash, visible_observation, audit_solver_visible
from .model_config import add_model_arguments, model_config, model_options
from teamgym.llm.construction_client import OpenAIJsonClient
from .integrity import atomic_write_json

RUNNER = CHECKOUT_HELPERS + r'''
import hashlib, json, pathlib, shutil, subprocess, sys
request = json.load(sys.stdin)
before = checkout_snapshot(request["expected_commit"])
assert before["clean"], "Base checkout is not clean"
head = before["head"]
if request["kind"] == "source":
    files = {}
    for path in request["paths"]:
        raw = subprocess.check_output(["git", "show", head + ":" + path])
        files[path] = {"text": raw.decode("utf-8"), "sha256": hashlib.sha256(raw).hexdigest()}
    result = {"files": files}
else:
    argv = list(request["argv"])
    # Execute exactly the audited, solver-visible argv. Shell failure handling
    # belongs in that argv, never in a hidden collector rewrite.
    try:
        run = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                             timeout=request["timeout_sec"])
        result = {"exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                  "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        result = {"exit_code": None, "stdout": text(exc.stdout), "stderr": text(exc.stderr),
                  "timed_out": True}
    except OSError as exc:
        result = {"exit_code": 127, "stdout": "", "stderr": str(exc),
                  "timed_out": False, "failure_kind": "setup_error"}
after = checkout_snapshot()
result.update(base_commit=request["expected_commit"], resolved_base_commit=head, git_status_before=before["status"], git_status_after=after["status"],
              checkout_before=before, checkout_after=after,
              tracked_checkout_unchanged=(before == after),
              tooling_inventory={
                  "programs": {name: shutil.which(name) for name in
                               ("python3", "node", "pnpm", "npm", "go", "cargo", "deno", "tsx")},
                  "local_bin": sorted(str(p) for p in pathlib.Path("node_modules/.bin").glob("*"))})
print(json.dumps(result))
'''


def run(task, request: dict, image_id: str) -> dict:
    request = {**request, "expected_commit": task.repository.base_commit_hash}
    argv = ["docker", "run", "--rm", "-i", "--platform", "linux/amd64",
            "--network", "none", "--cpus", str(task.runtime.cpus),
            "--memory", f"{task.runtime.memory_mb}m", "--workdir", "/app",
            "--entrypoint", "python3", image_id, "-c", RUNNER]
    started = time.monotonic()
    completed = subprocess.run(argv, input=json.dumps(request), capture_output=True, text=True,
                               timeout=int(request.get("timeout_sec", 60)) + 60)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    return {**json.loads(completed.stdout), "elapsed_seconds": round(time.monotonic() - started, 3)}


def validate_collection_plan(plan, task) -> None:
    if not isinstance(plan, dict) or plan.get("task_id") != task.task_id or plan.get("base_commit") != task.repository.base_commit_hash:
        raise ValueError("observation plan task/base identity mismatch")
    if not isinstance(plan.get("observations"), list):
        raise ValueError("observation plan requires an observations array")
    seen = set()
    for row in plan["observations"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("observation plan has invalid or duplicate IDs")
        seen.add(row["id"])
        argv = row.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a and "\x00" not in a for a in argv):
            raise ValueError("invalid observation argv")
        if type(row.get("timeout_sec")) is not int or not 1 <= row["timeout_sec"] <= 1800:
            raise ValueError("invalid observation timeout")
        artifact = {"task_id": task.task_id, "base_commit": plan["base_commit"], "argv": argv}
        audit = row.get("proposal_audit") or {}
        if (row.get("task_id") != task.task_id or row.get("base_commit") != plan["base_commit"]
                or audit.get("approved") is not True or audit.get("content_sha256") != content_hash(artifact)):
            raise ValueError("observation proposal provenance audit missing or stale")
        if not isinstance(row.get("public_source_refs"), list):
            raise ValueError("observation public_source_refs must be an array")


def verify_public_sources(task, refs: list[dict], image_id: str) -> list[dict]:
    """Bind the proposer's source claims to fresh exact pinned-base reads."""
    if not refs:
        return []  # Pure runtime/public-instruction probes are reviewed as such.
    if not all(isinstance(row, dict) and isinstance(row.get("path"), str) for row in refs):
        raise ValueError("invalid public source reference")
    source = run(task, {"kind": "source", "paths": sorted({r["path"] for r in refs})}, image_id)
    if not checkout_unchanged(source):
        raise ValueError("public source checkout changed")
    for ref in refs:
        lines = source["files"][ref["path"]]["text"].splitlines(keepends=True)
        start, end = ref.get("line_start"), ref.get("line_end")
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError("invalid public source range")
        text = "".join(lines[start - 1:end])
        if ref.get("text") != text or ref.get("sha256") != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("proposed public-source text does not match pinned base")
    return refs


def collect_observation(task, check: dict, image_id: str, *, client, trace_dir: Path) -> dict:
    row = {**check, "task_id": task.task_id, "base_commit": task.repository.base_commit_hash,
           "image_id": image_id}
    try:
        refs = verify_public_sources(task, check["public_source_refs"], image_id)
        row["public_sources_verified"] = True
        request = {"kind": "observation", "argv": check["argv"], "timeout_sec": check["timeout_sec"]}
        first, second = run(task, request, image_id), run(task, request, image_id)
        row.update(first)
        second_row = {**row, **second}
        first_hash, second_hash = content_hash(visible_observation(row)), content_hash(visible_observation(second_row))
        row["repeat_check"] = {"runs": 2, "first_sha256": first_hash, "second_sha256": second_hash,
                               "passed": first_hash == second_hash and checkout_unchanged(first) and checkout_unchanged(second),
                               "second_result": second}
        if not row["repeat_check"]["passed"]:
            row["failure_kind"] = "nonreproducible_or_checkout_changed"
            return row
        row["provenance_audit"] = audit_solver_visible(
            client=client, task=task, artifact=visible_observation(row), public_sources=refs,
            trace_dir=trace_dir / check["id"])
    except (RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError, OSError) as exc:
        row.update(exit_code=None, stdout="", stderr=str(exc),
                   collection_error=type(exc).__name__, failure_kind="setup_error", elapsed_seconds=0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-image")
    parser.add_argument("--audit-model", default="gpt-5.4")
    parser.add_argument("--env-file", default=str(Path.cwd() / ".env"))
    add_model_arguments(parser)
    args = parser.parse_args()
    task = load_task(args.task_dir, runtime_image=args.runtime_image)
    plan = json.loads(args.plan.read_text())
    validate_collection_plan(plan, task)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "collection_plan.json", plan)
    image_info = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", task.runtime.docker_image], text=True))[0]

    atoms = []
    if plan.get("code_atoms"):
        source = run(task, {"kind": "source",
                            "paths": sorted({a["path"] for a in plan["code_atoms"]})},
                     image_info["Id"])
        if not checkout_unchanged(source):
            raise ValueError("code-atom source checkout changed")
        atomic_write_json(out / "base_sources.json", source)
        for atom in plan["code_atoms"]:
            raw = source["files"][atom["path"]]
            lines = raw["text"].splitlines(keepends=True)
            start, end = atom["line_start"], atom["line_end"]
            if not 1 <= start <= end <= len(lines):
                raise ValueError(f"Invalid source range: {atom}")
            text = "".join(lines[start - 1:end])
            atoms.append({**atom, "text": text,
                          "sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "source_sha256": raw["sha256"], "base_commit": source["base_commit"]})

    observations = []
    client = OpenAIJsonClient(model_config(model=args.audit_model, env_file=args.env_file,
                                          **model_options(args)))
    for check in plan["observations"]:
        result = collect_observation(task, check, image_info["Id"], client=client, trace_dir=out / "provenance")
        observations.append(result)
        atomic_write_json(out / "observations.json", observations)
        print(json.dumps({"id": check["id"], "exit_code": result["exit_code"],
                          "seconds": result["elapsed_seconds"]}), flush=True)

    usable, rejected = usable_observations(observations, task_id=task.task_id,
                                         base_commit=task.repository.base_commit_hash)
    atomic_write_json(out / "observations.all.json", observations)
    atomic_write_json(out / "observations.json", usable)
    atomic_write_json(out / "collection_feedback.json", {
        "usable": len(usable), "rejected": rejected,
        "policy": "keep real base behavior, reject setup/dependency/timeouts/checkout changes",
    })

    artifact = {"schema": "deepswe_raw_evidence_candidate_v1", "task_id": task.task_id,
                "status": "draft_not_qualified", "instruction": task.instruction,
                "image_id": image_info["Id"], "repo_digests": image_info.get("RepoDigests", []),
                "code_atoms": atoms, "observations": usable,
                "allocation_candidates": plan.get("allocation_candidates", []),
                "unallocated_background_observation_candidates":
                    plan.get("unallocated_background_observation_candidates", []),
                "allocation_notes": plan.get("allocation_notes", []),
                "audits": {"foreign_riu": "pending", "format_salience": "pending",
                           "semantic_leakage": "pending",
                           "single_hidden_test_qualification": "not_run",
                           "necessity_loo": "deferred"}}
    atomic_write_json(out / "candidate.hidden.json", artifact)


if __name__ == "__main__":
    main()
