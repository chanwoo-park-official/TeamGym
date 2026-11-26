#!/usr/bin/env python3
"""Extract a hash-verified DeepSWE fragment pool from selected base files.

Candidate files come from ``--paths`` or the constructor-only reference-patch
footprint. The resulting pool is filtered before any solver packet is built.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from teamgym.data.software.deep_swe import load_task
from teamgym.data.software.deep_swe.fragments import (
    DEFAULT_MAX_TOKENS,
    extract_complete,
)
from teamgym.tokenization import count_tokens
from teamgym.data.software.deep_swe.stages.patch_runtime import BASE_READ, docker_python, save


def changed_paths(inventory: Path, task_id: str) -> list[str]:
    rows = json.loads(inventory.read_text())["tasks"]
    for row in rows:
        if row["task_id"] == task_id:
            return list(row["reference_footprint"]["changed_paths"])
    raise KeyError(f"{task_id} not present in {inventory}")


def discovered_paths(discovered: Path) -> list[str]:
    """Paths selected by the packaged constructor discovery stage."""
    payload = json.loads(discovered.read_text())
    return sorted(payload.get("files", {}))


def reference_changed_paths(task_dir: Path) -> list[str]:
    """Changed paths read straight from the solution patch, if present."""
    patch = task_dir / "solution" / "solution.patch"
    if not patch.exists():
        return []
    paths = []
    for line in patch.read_text(errors="replace").splitlines():
        if line.startswith("+++ b/"):
            value = line[6:].strip()
            if value and value not in paths:
                paths.append(value)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", type=Path, required=True)
    ap.add_argument("--runtime-image")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="Files to mine. Defaults to --discovered, then the reference "
                         "patch's changed paths.")
    ap.add_argument("--discovered", type=Path, default=None,
                    help="Output of the discovery stage: the file set a model chose "
                         "to read from the instruction, reference patch, and hidden test. "
                         "Preferred over the patch's own changed paths when given.")
    ap.add_argument("--inventory", type=Path, default=None,
                    help="Optional task inventory; the solution patch is read directly otherwise.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--prior-pool", type=Path, help="Preserve IDs during source expansion")
    args = ap.parse_args()

    task = load_task(args.task_dir, runtime_image=args.runtime_image)
    image = task.runtime.docker_image
    commit = task.repository.base_commit_hash
    paths = args.paths
    if not paths and args.discovered:
        paths = discovered_paths(args.discovered)
    if not paths and args.inventory:
        paths = changed_paths(args.inventory, task.task_id)
    if not paths:
        paths = reference_changed_paths(args.task_dir)
    if not paths:
        raise SystemExit(f"{task.task_id}: no candidate paths; pass --paths")

    print(json.dumps({"event": "stage1", "task_id": task.task_id, "image": image,
                      "base_commit": commit, "paths": len(paths)}), flush=True)

    fragments: list[dict] = []
    oversized: list[dict] = []
    skipped: list[dict] = []
    files: list[dict] = []
    previous = json.loads(args.prior_pool.read_text()) if args.prior_pool and args.prior_pool.exists() else {}
    old_rows = previous.get("fragments", []) + previous.get("oversized", [])
    old_ids = {(r["path"], r["line_start"], r["line_end"], r["sha256"]): r["id"] for r in old_rows}
    counters = {prefix: max([int(r["id"].split("-")[-1]) for r in old_rows
                            if r["id"].startswith(prefix + "-")] or [0]) for prefix in ("frag", "over")}

    def identify(fragment, prefix):
        key = (fragment.path, fragment.line_start, fragment.line_end, fragment.sha256)
        if key in old_ids:
            return old_ids[key]
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]:04d}"

    for path in paths:
        raw = docker_python(image, BASE_READ, {"path": path, "commit": commit})
        if raw["kind"] != "file":
            skipped.append({"path": path, "reason": f"base_{raw['kind']}"})
            print(json.dumps({"event": "file", "path": path, "kind": raw["kind"]}), flush=True)
            continue
        source = raw["text"]
        files.append({
            "path": path,
            "lines": len(source.splitlines()),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        })
        report = extract_complete(path, source, count_tokens=count_tokens, max_tokens=args.max_tokens)

        # Every fragment must reproduce exactly from the pinned base, by the same
        # 1-indexed inclusive slice the solver harness re-verifies later.
        lines = source.splitlines(keepends=True)
        for fragment in list(report.fragments) + list(report.oversized):
            actual = "".join(lines[fragment.line_start - 1:fragment.line_end])
            if actual != fragment.text or hashlib.sha256(actual.encode()).hexdigest() != fragment.sha256:
                raise AssertionError(f"fragment does not reproduce from base: {path}:{fragment.name}")

        for fragment in report.fragments:
            fragments.append({"id": identify(fragment, "frag"), **fragment.as_dict()})
        for fragment in report.oversized:
            oversized.append({"id": identify(fragment, "over"), **fragment.as_dict()})
        skipped.extend(report.skipped)
        print(json.dumps({"event": "file", "path": path, "lines": len(lines),
                          "fragments": len(report.fragments),
                          "oversized": len(report.oversized),
                          "skipped": len(report.skipped)}), flush=True)

    by_kind: dict[str, int] = {}
    for fragment in fragments:
        by_kind[fragment["kind"]] = by_kind.get(fragment["kind"], 0) + 1

    pool = {
        "extraction_policy": "lossless_context_v2",
        "schema": "deepswe_fragment_pool_v1",
        "task_id": task.task_id,
        "instruction": task.instruction,
        "image_id": image,
        "base_commit": commit,
        "max_tokens": args.max_tokens,
        "files": files,
        "fragments": fragments,
        "oversized": oversized,
        "skipped": skipped,
        "summary": {
            "files": len(files),
            "fragments": len(fragments),
            "oversized": len(oversized),
            "skipped": len(skipped),
            "by_kind": by_kind,
            "total_tokens": sum(f["tokens"] for f in fragments),
        },
        "stage": "1-2",
        "qualification": "unaudited_pool",
    }
    if not fragments and not oversized:
        raise SystemExit(f"{task.task_id}: empty fragment pool; construction cannot proceed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, pool)
    print(json.dumps({"event": "done", **pool["summary"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
