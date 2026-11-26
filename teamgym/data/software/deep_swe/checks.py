"""Build and mechanically validate task-specific DeepSWE public-check catalogs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from teamgym.llm.construction_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)

from .prompts import load_prompt
from .checkout_integrity import CHECKOUT_HELPERS, checkout_unchanged
from .model_config import model_config
from .provenance import (audit_solver_visible, auditor_identity, content_hash,
                         PUBLIC_BASELINE_AUDIT_SYSTEM)
from .integrity import atomic_write_json, code_identity, task_identity, file_sha256


_CHECK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PUBLIC_INVENTORY = CHECKOUT_HELPERS + r'''
import json, pathlib, subprocess, sys
request = json.load(sys.stdin)
before = checkout_snapshot(request["commit"])
assert before["clean"], "Base checkout is not clean"
files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
manifest_names = {
    "pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "package.json",
    "pnpm-workspace.yaml", "go.mod", "makefile", "justfile", "tsconfig.json",
}
manifests = {}
for name in files:
    path = pathlib.Path(name)
    lower = path.name.lower()
    if (lower in manifest_names or lower.startswith("jest.config.")
            or lower.startswith("vitest.config.")):
        try:
            manifests[name] = path.read_text(errors="replace")[:50000]
        except OSError:
            pass
test_paths = [name for name in files if any(
    part.lower() in {"test", "tests", "__tests__", "spec", "specs"}
    for part in pathlib.PurePosixPath(name).parts[:-1]
) or pathlib.PurePosixPath(name).name.lower().startswith("test_")
  or pathlib.PurePosixPath(name).name.lower().endswith("_test.go")
  or ".test." in pathlib.PurePosixPath(name).name.lower()
  or ".spec." in pathlib.PurePosixPath(name).name.lower()]
after = checkout_snapshot()
assert before == after, "Inventory changed the checkout"
print(json.dumps({"repository_files": files, "public_test_paths": test_paths,
                  "build_test_manifests": manifests, "base_commit": before["head"],
                  "checkout_before": before, "checkout_after": after}))
'''
_RUN_CHECK = CHECKOUT_HELPERS + r'''
import json, subprocess, sys
r = json.load(sys.stdin)
before = checkout_snapshot(r["commit"])
assert before["clean"], "Base checkout is not clean"
try:
    p = subprocess.run(r["argv"], capture_output=True, text=True, errors="replace",
                       timeout=r["timeout_sec"])
    result = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
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
result.update(base_commit=r["commit"], resolved_base_commit=before["head"], checkout_before=before, checkout_after=after,
              tracked_checkout_unchanged=(before == after))
print(json.dumps(result))
'''


class PublicCheckError(ValueError):
    """Raised when no safe, base-passing public check can be constructed."""


def _check_artifact(check_id, argv, result):
    return {"id": check_id, "argv": argv, "stdout": result.get("stdout"),
            "stderr": result.get("stderr"), "exit_code": result.get("exit_code")}


def _validate_catalog(checks):
    if not isinstance(checks, dict) or not checks:
        raise PublicCheckError('at least one public check is required')
    for check_id, argv in checks.items():
        if not isinstance(check_id, str) or not _CHECK_ID.fullmatch(check_id):
            raise PublicCheckError(f'invalid public check id: {check_id!r}')
        if (not isinstance(argv, list) or not argv
                or not all(isinstance(x, str) and x and '\x00' not in x for x in argv)):
            raise PublicCheckError(f'invalid argv for public check {check_id!r}')


def _cache_inputs(task, checks, identity, cfg):
    """Require the caller's frozen run identity to describe current inputs."""
    if (not isinstance(identity, dict)
            or identity.get('source') != task_identity(task.source_dir)
            or identity.get('code') != code_identity()
            or identity.get('runtime_image', {}).get('id') != task.runtime.docker_image
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', task.runtime.docker_image)
            or identity.get('environment', {}).get('uv_lock_sha256') != file_sha256(
                Path(__file__).resolve().parents[4] / 'uv.lock')):
        raise PublicCheckError('public-check cache identity is not current and content-pinned')
    return {'task_id': task.task_id, 'base_commit': task.repository.base_commit_hash,
            'runtime_image_id': task.runtime.docker_image, 'checks': checks,
            'run_identity': identity, 'auditor': auditor_identity(cfg),
            'audit_system_sha256': content_hash(PUBLIC_BASELINE_AUDIT_SYSTEM)}


def _valid_cached_rows(task, checks, results, auditor):
    if not isinstance(results, dict) or set(results) != set(checks):
        return False
    for check_id, argv in checks.items():
        row = results[check_id]
        if not isinstance(row, dict):
            return False
        audit = row.get('provenance_audit')
        if (type(row.get('exit_code')) is not int or row['exit_code'] != 0
                or row.get('timed_out') is not False or not checkout_unchanged(row)
                or row.get('base_commit') != task.repository.base_commit_hash
                or row.get('resolved_base_commit') != task.repository.base_commit_hash
                or not all(isinstance(row.get(k), str) for k in ('stdout', 'stderr'))
                or not isinstance(audit, dict) or audit.get('approved') is not True
                or audit.get('status') != 'reviewed'
                or audit.get('artifact_role') != 'public_baseline_check'
                or audit.get('content_sha256') != content_hash(_check_artifact(check_id, argv, row))
                or audit.get('auditor') != auditor
                or not re.fullmatch(r'[0-9a-f]{64}', str(audit.get('audit_input_sha256', '')))
                or not all(isinstance(audit.get(k), str) and audit[k].strip()
                           for k in ('reason', 'public_grounding'))):
            return False
    return True


def write_public_check_validation_cache(path, task, checks, results, *, identity,
                                        model='gpt-5.4', env_file=None,
                                        trace_dir=None, **model_kwargs):
    """Write new-role results; legacy approvals require explicit revalidation.

    Store the resulting file digest in the run's trusted checkpoint. A cache's
    self-reported hashes alone are not authority to skip validation.
    """
    _validate_catalog(checks)
    cfg = model_config(model=model, env_file=env_file, **model_kwargs)
    inputs = _cache_inputs(task, checks, identity, cfg)
    if not _valid_cached_rows(task, checks, results, inputs['auditor']):
        raise PublicCheckError('public-check validation rows are incomplete or unbound')
    atomic_write_json(path, {'schema': 'deepswe_public_check_validation_cache_v1',
                            'inputs': inputs, 'results': results})


def _cached_validation(path, expected_sha256, task, checks, identity, cfg):
    try:
        if not isinstance(expected_sha256, str) or file_sha256(path) != expected_sha256:
            return None
        cache = json.loads(Path(path).read_text())
        inputs = _cache_inputs(task, checks, identity, cfg)
        if (set(cache) != {'schema', 'inputs', 'results'}
                or cache['schema'] != 'deepswe_public_check_validation_cache_v1'
                or cache['inputs'] != inputs
                or not _valid_cached_rows(task, checks, cache['results'], inputs['auditor'])):
            return None
        return cache['results']
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def run_base_check(task, argv: list[str], *, timeout_sec: int = 300) -> dict:
    """Run one candidate check in a fresh network-disabled base container."""

    request = {
        "commit": task.repository.base_commit_hash,
        "argv": argv,
        "timeout_sec": timeout_sec,
    }
    command = [
        "docker", "run", "--rm", "-i", "--platform", "linux/amd64",
        "--network", "none", "--cpus", str(task.runtime.cpus),
        "--memory", f"{task.runtime.memory_mb}m", "--workdir", "/app",
        "--entrypoint", "python3", task.runtime.docker_image, "-I", "-c", _RUN_CHECK,
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=timeout_sec + 60,
    )
    if completed.returncode:
        raise PublicCheckError(completed.stderr[-2000:])
    result = json.loads(completed.stdout)
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


def validate_public_checks(task, checks: dict[str, list[str]], *, model="gpt-5.4",
                           env_file=None, trace_dir: Path | None = None,
                           validation_cache_path: Path | None = None,
                           validation_identity: dict | None = None,
                           validation_cache_sha256: str | None = None,
                           **model_kwargs) -> dict[str, dict]:
    """Require every actor-visible check to pass on an untouched base checkout."""

    _validate_catalog(checks)
    cfg = model_config(model=model, env_file=env_file, **model_kwargs)
    if validation_cache_path is not None:
        cached = _cached_validation(validation_cache_path, validation_cache_sha256,
                                    task, checks, validation_identity, cfg)
        if cached is not None:
            return cached
    results = {}
    inventory = public_repository_inventory(task)
    client = OpenAIJsonClient(cfg)
    for check_id, argv in checks.items():
        result = run_base_check(task, argv)
        results[check_id] = result
        if result.get("timed_out") or result.get("exit_code") != 0 or not checkout_unchanged(result):
            raise PublicCheckError(
                f"public check {check_id!r} does not pass on base: "
                f"exit={result.get('exit_code')} stderr={result.get('stderr', '')[-400:]}"
            )
        audit = audit_solver_visible(client=client, task=task,
            artifact=_check_artifact(check_id, argv, result), artifact_role='public_baseline_check',
            public_sources=inventory, trace_dir=trace_dir or Path(".deepswe-public-check-audit"))
        result["provenance_audit"] = audit
        if not audit["approved"]:
            raise PublicCheckError(f"public check {check_id!r} provenance rejected: {audit['reason']}")
    return results


def public_repository_inventory(task) -> dict:
    """Read only base file names and public build/test manifests from the image."""

    command = [
        "docker", "run", "--rm", "-i", "--platform", "linux/amd64",
        "--network", "none", "--cpus", str(task.runtime.cpus),
        "--memory", f"{task.runtime.memory_mb}m", "--workdir", "/app",
        "--entrypoint", "python3", task.runtime.docker_image, "-I", "-c",
        _PUBLIC_INVENTORY,
    ]
    completed = subprocess.run(command, input=json.dumps({"commit": task.repository.base_commit_hash}),
                               capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise PublicCheckError(completed.stderr[-2000:])
    return json.loads(completed.stdout)


def plan_public_checks(
    *, task, model: str, env_file: str, trace_dir: Path, **model_kwargs
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Plan from public base inventory only and retain base-passing checks."""

    inventory = public_repository_inventory(task)
    request = {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "language": task.repository.language,
        "base_commit": task.repository.base_commit_hash,
        **inventory,
    }
    cfg = model_config(model=model, env_file=env_file, **model_kwargs)
    client = OpenAIJsonClient(cfg)
    for attempt in range(1, 4):
        try:
            response = client.generate_json(
                prompt_id=f"deep_swe.public_check_planner_{attempt}",
                system_prompt=load_prompt("public_check_planner_system.txt"),
                user_prompt=json.dumps(request, ensure_ascii=False), trace_dir=trace_dir)
        except ValueError as exc:
            request["json_validation_feedback"] = str(exc)
            if attempt == 3:
                raise PublicCheckError("public-check JSON validation retries exhausted") from exc
            continue
        checks: dict[str, list[str]] = {}
        raw_checks = response.get("checks") if isinstance(response, dict) else None
        for raw in raw_checks if isinstance(raw_checks, list) else []:
            if not isinstance(raw, dict):
                continue
            check_id = str(raw.get("id") or "")
            argv = raw.get("argv")
            if _CHECK_ID.fullmatch(check_id) and isinstance(argv, list) and argv:
                if all(isinstance(item, str) and item and "\x00" not in item for item in argv):
                    checks[check_id] = argv
        accepted: dict[str, list[str]] = {}
        validation: dict[str, dict] = {}
        rejected: dict[str, dict] = {}
        for check_id, argv in checks.items():
            try:
                result = run_base_check(task, argv)
            except (PublicCheckError, subprocess.TimeoutExpired) as exc:
                result = {"exit_code": None, "stderr": str(exc), "setup_error": True}
            # Some pinned Go images export GOFLAGS=-mod=mod while also activating a
            # workspace. Go rejects that combination before compiling anything.
            # Disable the workspace only for this exact base-environment conflict,
            # then expose the validated command rather than the rejected original.
            if (
                result.get("exit_code") != 0
                and argv[0] == "go"
                and "workspace mode" in str(result.get("stderr") or "")
            ):
                argv = ["env", "GOWORK=off", *argv]
                result = run_base_check(task, argv)
            if not result.get("timed_out") and result.get("exit_code") == 0 and checkout_unchanged(result):
                audit = audit_solver_visible(client=client, task=task,
                    artifact=_check_artifact(check_id, argv, result), artifact_role='public_baseline_check',
                    public_sources=inventory, trace_dir=trace_dir / "provenance")
                result["provenance_audit"] = audit
                if not audit["approved"]:
                    rejected[check_id] = result
                    continue
                accepted[check_id] = argv
                validation[check_id] = result
            else:
                rejected[check_id] = result
        trace_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(trace_dir / "rejected_checks.json", rejected)
        if accepted:
            return accepted, validation
        request["repair_feedback"] = {
            "rejected_checks": rejected,
            "instruction": "Previous commands failed on base. Choose existing public scripts/runtimes and repair their exact errors; do not repeat them or weaken checks to unconditional success.",
        }
        atomic_write_json(trace_dir / f"rejected_checks_{attempt}.json", rejected)
    raise PublicCheckError("planner produced no base-passing public checks after 3 proposals")
