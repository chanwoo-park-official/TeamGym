"""Lossless, bounded constructor evidence; never a solver instruction source."""
from __future__ import annotations

import json
from collections import Counter

from teamgym.tokenization import count_tokens
from .checkout_integrity import checkout_unchanged
from .provenance import content_hash, visible_observation


def text_parts(text: str, budget: int = 8_000) -> list[str]:
    """Split exact text into bounded parts, including pathological long lines."""
    if budget < 1:
        raise ValueError("budget must be positive")
    result = []
    while text:
        if count_tokens(text) <= budget:
            result.append(text)
            break
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(text[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        newline = text.rfind("\n", 0, end)
        if newline > end // 2:
            end = newline + 1
        part = text[:end]
        if count_tokens(part) > budget:
            raise ValueError("a single character exceeds the evidence token budget")
        result.append(part)
        text = text[end:]
    return result


def stage_materials(request: dict, part_tokens: int = 8_000) -> tuple[dict, dict]:
    """Defer large patches/logs/observations instead of copying them every call."""
    base = dict(request)
    materials = {}
    inventory = []

    def add(key, value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts = text_parts(text, part_tokens)
        for n, part in enumerate(parts, 1):
            evidence_id = f"{key}:part-{n:04d}"
            row = {"id": evidence_id, "material": key, "part": n,
                   "parts": len(parts), "tokens": count_tokens(part), "text": part}
            materials[evidence_id] = row
            inventory.append({k: v for k, v in row.items() if k != "text"})

    for key in ("reference_patch", "hidden_test_patch", "qualification_feedback",
                "instruction_only_failures"):
        if key in base:
            value = base.pop(key)
            if key == "qualification_feedback":
                base["qualification_summary"] = value.get("summary", {})
                failures = value.get("failing_tests", [])
                base["failure_inventory"] = {
                    "count": len(failures),
                    "by_bucket": dict(Counter(row.get("bucket", "unknown") for row in failures)),
                    "evidence_material": "qualification_failures",
                }
                add("qualification_failures", failures)
                value = {k: v for k, v in value.items() if k != "failing_tests"}
            add(key, value)
    observations = base.pop("runtime_observations", [])
    base["runtime_observations"] = []
    for observation in observations:
        key = observation["id"]
        add(key, observation)
        base["runtime_observations"].append({
            k: observation[k] for k in ("id", "family", "boundary", "exit_code")
            if k in observation
        })
    base["constructor_evidence_inventory"] = inventory
    return base, materials


def setup_failure(result: dict) -> str | None:
    """Distinguish probe setup errors from meaningful nonzero base behavior."""
    if result.get("collection_error") or result.get("failure_kind") == "setup_error":
        return "setup_error"
    if result.get("timed_out"):
        return "timeout"
    if result.get("tracked_checkout_unchanged") is False:
        return "tracked_checkout_changed"
    text = str(result.get("stderr", "")) + "\n" + str(result.get("stdout", ""))
    # These are dependency/launcher failures, not observations of target behavior.
    markers = ("command not found", "Cannot find package",
               "Cannot find module", "no matching package named", "offline mode",
               "found packages ", "Could not resolve", "ENOENT:", "ModuleNotFoundError:",
               "No module named ", "ImportError:", "Permission denied", "No such file or directory")
    reviewed_target = (result.get("provenance_audit") or {}).get("nonzero_is_target_behavior") is True
    if result.get("exit_code") != 0 and not reviewed_target and any(marker in text for marker in markers):
        return "dependency_or_launcher_error"
    return None


def observation_invalid_reason(row, *, task_id=None, base_commit=None) -> str | None:
    if not isinstance(row, dict):
        return "observation_must_be_object"
    for key in ("id", "task_id", "base_commit"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            return f"missing_or_invalid_{key}"
    if task_id is not None and row["task_id"] != task_id:
        return "task_identity_mismatch"
    if base_commit is not None and row["base_commit"] != base_commit:
        return "base_identity_mismatch"
    argv = row.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v and "\x00" not in v for v in argv):
        return "invalid_argv"
    if type(row.get("exit_code")) is not int or not all(isinstance(row.get(k), str) for k in ("stdout", "stderr")):
        return "missing_or_invalid_outcome"
    if row.get("timed_out") is not False:
        return "missing_or_invalid_timeout_status"
    if not checkout_unchanged(row):
        return "missing_or_changed_checkout_fingerprint"
    digest = content_hash(visible_observation(row))
    repeat = row.get("repeat_check")
    if not (isinstance(repeat, dict) and repeat.get("passed") is True
            and repeat.get("runs") == 2 and repeat.get("first_sha256") == digest
            and repeat.get("second_sha256") == digest):
        return "missing_or_unstable_repeat_check"
    audit = row.get("provenance_audit")
    if not (isinstance(audit, dict) and audit.get("approved") is True
            and audit.get("content_sha256") == digest and audit.get("status") == "reviewed"
            and row.get("public_sources_verified") is True):
        return "missing_or_rejected_provenance_audit"
    if row["exit_code"] != 0 and audit.get("nonzero_is_target_behavior") is not True:
        return "nonzero_behavior_not_reviewed"
    return None


def usable_observations(rows: list[dict], *, task_id=None, base_commit=None) -> tuple[list[dict], list[dict]]:
    if not isinstance(rows, list):
        raise ValueError("observations must be an array")
    ids = Counter(row.get("id") for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str))
    seen_content = set()
    usable, rejected = [], []
    for row in rows:
        reason = observation_invalid_reason(row, task_id=task_id, base_commit=base_commit)
        if not reason:
            reason = setup_failure(row)
        if isinstance(row, dict) and isinstance(row.get("id"), str) and ids.get(row["id"], 0) > 1:
            reason = "duplicate_observation_id"
        if not reason:
            digest = content_hash(visible_observation(row))
            if digest in seen_content:
                reason = "duplicate_observation_content"
            seen_content.add(digest)
        (rejected if reason else usable).append(
            {**(row if isinstance(row, dict) else {"invalid_row": row}), "rejection_reason": reason} if reason else row
        )
    return usable, rejected
