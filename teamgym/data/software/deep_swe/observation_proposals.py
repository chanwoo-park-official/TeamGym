#!/usr/bin/env python3
"""Propose base-repository runtime observations for DeepSWE construction.

Reference material guides constructor-side targeting, but actor-visible units
contain only commands and real outputs from the pinned, network-disabled base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path.cwd()
from teamgym.llm.construction_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)
from teamgym.data.software.deep_swe.prompts import load_prompt
from teamgym.data.software.deep_swe.stages.patch_runtime import save
from .candidate import selectable
from teamgym.tokenization import count_tokens
from .model_config import add_model_arguments, model_config, model_options
from .provenance import content_hash, audit_solver_visible
from .source import load_task


SYSTEM = load_prompt("observation_proposer_system.txt")


def normalize_proposals(response, *, pool: dict) -> list[dict]:
    """Reject malformed model JSON before assigning deterministic content IDs."""
    if not isinstance(response, dict) or not isinstance(response.get("observations"), list):
        raise ValueError("proposal response must contain an observations array")
    fragments = {row["id"]: row for row in selectable(pool)}
    rows, seen = [], set()
    for raw in response["observations"]:
        if not isinstance(raw, dict):
            raise ValueError("each observation proposal must be an object")
        argv = raw.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v and "\x00" not in v for v in argv):
            raise ValueError("proposal argv must be a nonempty array of nonempty strings")
        timeout = raw.get("timeout_sec", 300)
        if type(timeout) is not int or not 1 <= timeout <= 1800:
            raise ValueError("proposal timeout_sec must be an integer in [1,1800]")
        refs = raw.get("source_fragment_ids", [])
        if not isinstance(refs, list) or not all(isinstance(i, str) and i in fragments for i in refs):
            raise ValueError("source_fragment_ids must reference supplied public base fragments")
        for key in ("family", "boundary", "why_not_in_instruction"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"proposal requires nonempty {key}")
        identity = {"task_id": pool["task_id"], "base_commit": pool["base_commit"], "argv": argv}
        digest = content_hash(identity)
        if digest in seen:
            raise ValueError("duplicate observation argv")
        seen.add(digest)
        sources = [{k: fragments[i][k] for k in ("path", "line_start", "line_end", "text", "sha256")}
                   for i in dict.fromkeys(refs)]
        rows.append({**raw, **identity, "id": "observation-" + digest[:20],
                     "timeout_sec": timeout, "public_source_refs": sources})
    return rows


def fragment_inventory(pool: dict, limit: int, reference_patch: str | None = None) -> list[dict]:
    """Return bounded signatures, prioritizing reference-touched fragments."""
    from teamgym.data.software.deep_swe.attribution import (
        reference_touched_ranges,
        touches_fragment,
    )

    ranges = reference_touched_ranges(reference_patch) if reference_patch else {}
    rows = []
    for fragment in selectable(pool):
        head = fragment["text"].strip().splitlines()
        rows.append({
            "id": fragment["id"], "path": fragment["path"], "kind": fragment["kind"], "name": fragment["name"],
            "lines": f"{fragment['line_start']}-{fragment['line_end']}",
            "signature": head[0][:160] if head else "",
            "reference_touches_this": touches_fragment(fragment, ranges),
        })
    rows.sort(key=lambda r: not r["reference_touches_this"])
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--task-dir", type=Path, required=True,
                    help="Public task and private reference artifacts for independent provenance review.")
    ap.add_argument("--runtime-image")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--inventory-limit", type=int, default=120,
                    help="Signatures shown to the proposer. A large pool can push the "
                         "response past its output budget and lose every probe.")
    ap.add_argument("--reference-patch", type=Path, default=None,
                    help="The task's solution patch. Constructor-side only, same as stage "
                         "4b: it points the proposer at the touched files and behaviors so "
                         "probes aim at informative boundaries. The actor never sees it.")
    ap.add_argument("--discovered", type=Path, default=None,
                    help="Discovered code JSON; paths are listed and exact fragments are supplied within --source-tokens.")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--collection-feedback", type=Path)
    ap.add_argument("--prior-plan", type=Path)
    ap.add_argument("--source-tokens", type=int, default=40_000)
    ap.add_argument("--env-file", default=str(ROOT / ".env"))
    add_model_arguments(ap)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text())
    task = load_task(args.task_dir, runtime_image=args.runtime_image)
    if pool.get("task_id") != task.task_id or pool.get("base_commit") != task.repository.base_commit_hash:
        raise ValueError("proposal pool task/base identity mismatch")
    reference_patch = args.reference_patch.read_text() if args.reference_patch else None
    discovered_files = {}
    if args.discovered:
        discovered_files = {path: row["text"]
                            for path, row in json.loads(args.discovered.read_text())["files"].items()}
    # Prefer implementation-bearing exact fragments; never send the entire
    # discovered tree repeatedly just because it exists on disk.
    from .attribution import reference_touched_ranges, touches_fragment
    ranges = reference_touched_ranges(reference_patch or "")
    sources, source_used = [], 0
    for fragment in sorted(selectable(pool), key=lambda row: not touches_fragment(row, ranges)):
        cost = count_tokens(json.dumps(fragment, ensure_ascii=False))
        if source_used + cost <= args.source_tokens:
            sources.append(fragment)
            source_used += cost
    request = {
        "task_id": pool["task_id"],
        "instruction": pool["instruction"],
        "base_commit": pool["base_commit"],
        "files": [row["path"] for row in pool["files"]],
        "fragment_inventory": fragment_inventory(pool, args.inventory_limit, reference_patch),
        "reference_patch": reference_patch,
        "base_source_fragments": sources,
        "discovered_paths": sorted(discovered_files),
        "source_supply": {"tokens": source_used, "budget": args.source_tokens,
                          "complete_pool_fragments": len(selectable(pool)), "supplied": len(sources)},
        "probes_requested": args.count,
    }
    if args.collection_feedback:
        raw = json.loads(args.collection_feedback.read_text())
        request["probe_repair_feedback"] = {
            "usable": raw.get("usable"),
            "rejected": [{**{k: row.get(k) for k in ("id", "argv", "exit_code", "rejection_reason")},
                          "stderr_excerpt": str(row.get("stderr", ""))[-4_000:],
                          "full_feedback_path": str(args.collection_feedback)}
                         for row in raw.get("rejected", [])],
        }
        request["available_tooling"] = next((row.get("tooling_inventory") for row in raw.get("rejected", [])
                                            if row.get("tooling_inventory")), None)
    if args.prior_plan:
        request["prior_plan"] = json.loads(args.prior_plan.read_text())

    cfg = model_config(model=args.model, env_file=args.env_file,
                       max_output_tokens=32000, **model_options(args))
    client = OpenAIJsonClient(cfg)

    print(json.dumps({"event": "proposing", "task_id": pool["task_id"],
                      "model": cfg.model, "fragments": len(request["fragment_inventory"])}),
          flush=True)
    for attempt in range(3):
        try:
            response = client.generate_json(
                prompt_id=f"stage3a.propose_observations_{attempt + 1}", system_prompt=SYSTEM,
                user_prompt=json.dumps(request, ensure_ascii=False),
                trace_dir=args.output.parent / "prompts")
            proposals = normalize_proposals(response, pool=pool)
            break
        except (ValueError, TypeError) as exc:
            request["json_validation_feedback"] = str(exc)
            if attempt == 2:
                raise ValueError("observation proposal validation retries exhausted") from exc
    rejected = []
    for proposal in proposals:
        artifact = {k: proposal[k] for k in ("task_id", "base_commit", "argv")}
        proposal["proposal_audit"] = audit_solver_visible(
            client=client, task=task, artifact=artifact,
            public_sources=proposal["public_source_refs"], trace_dir=args.output.parent / "prompts" / "provenance")
        if not proposal["proposal_audit"]["approved"]:
            rejected.append(proposal)
    proposals = [row for row in proposals if row["proposal_audit"]["approved"]]

    plan = {
        "schema": "deepswe_observation_plan_v1",
        "task_id": pool["task_id"],
        "base_commit": pool["base_commit"],
        "proposed_by": cfg.model,
        "stage": "3a",
        "code_atoms": [],
        "observations": proposals,
        "rejected_proposals": rejected,
        "allocation_candidates": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, plan)
    for proposal in proposals:
        print(json.dumps({"id": proposal["id"], "family": proposal.get("family"),
                          "boundary": proposal.get("boundary", "")[:90]}), flush=True)
    print(json.dumps({"event": "done", "observations": len(proposals),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
