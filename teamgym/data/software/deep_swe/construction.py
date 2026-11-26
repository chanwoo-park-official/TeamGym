"""Construct functional DeepSWE PIUs from informed, constructor-only evidence."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from teamgym.llm.construction_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)
from teamgym.tokenization import count_tokens

from .attribution import failing_tests
from .candidate import assemble_functional_candidate, selectable
from .prompts import load_prompt
from .source import load_task
from .evidence import stage_materials, usable_observations
from .model_config import add_model_arguments, model_config, model_options
from .integrity import atomic_write_json


SCHEMA = "deepswe_functional_riu_plan_v1"


class FunctionalRIUConstructionError(ValueError):
    """Raised when constructor input or output violates the RIU contract."""


class ConstructorActionClient:
    """Use an actual function call so a read cannot simulate its own result."""

    def __init__(self, client):
        self.client = client
        self.cfg = client.cfg
        self.last_request_meta = {}

    def generate_json(self, *, prompt_id, system_prompt, user_prompt, trace_dir):
        tool = {
            "type": "function", "name": "constructor_action", "strict": False,
            "description": "Request fragment bodies OR submit an RIU plan. The harness supplies read results.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["read", "propose"]},
                "fragment_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "rius": {"type": "array", "items": {"type": "object", "properties": {
                    "functional_boundary": {"type": "string"},
                    "code_fragment_ids": {"type": "array", "items": {"type": "string"}},
                    "observation_ids": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                    "coverage": {"type": "array", "items": {"type": "string"}},
                }, "required": ["functional_boundary", "code_fragment_ids", "observation_ids", "rationale"],
                    "additionalProperties": False}},
                "uncovered": {"type": "array", "items": {"type": "object"}},
                "diagnosis": {"type": "object"},
            }, "required": ["action"], "additionalProperties": False},
        }
        body = {"model": self.cfg.model,
                "input": [{"role": "developer", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                "tools": [tool], "tool_choice": {"type": "function", "name": "constructor_action"},
                "parallel_tool_calls": False, "max_output_tokens": self.cfg.max_output_tokens,
                "reasoning": {"effort": self.cfg.reasoning_effort}}
        prefix = Path(trace_dir) / f"{prompt_id}.{uuid.uuid4().hex}"
        _write_json(Path(str(prefix) + ".request.json"), body)
        response = self.client._client.responses.create(**body).model_dump()
        self.last_request_meta = {"usage": response.get("usage"), "status": response.get("status")}
        _write_json(Path(str(prefix) + ".response.json"), response)
        self.client._validate_response_status(response)
        calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
        if len(calls) != 1 or calls[0].get("name") != "constructor_action":
            raise FunctionalRIUConstructionError("exactly one constructor_action call is required")
        try:
            result = json.loads(calls[0]["arguments"])
        except (ValueError, TypeError, KeyError) as exc:
            raise FunctionalRIUConstructionError("constructor_action arguments must be valid JSON") from exc
        if not isinstance(result, dict):
            raise FunctionalRIUConstructionError("constructor_action arguments must be a JSON object")
        _write_json(Path(str(prefix) + ".parsed.json"), result)
        return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _output_charge(client) -> tuple[int, str]:
    usage = (client.last_request_meta or {}).get("usage")
    value = usage.get("output_tokens") if isinstance(usage, dict) else None
    if type(value) is int and value >= 0:
        return value, "observed"
    return client.cfg.max_output_tokens, "missing_usage_charged_at_ceiling"


def build_constructor_request(
    *,
    task,
    pool: dict,
    observations: list[dict],
    baseline_run: Path,
    public_checks: dict[str, list[str]] | None,
) -> dict:
    """Build the complete constructor-only view for functional grouping."""

    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "reference_patch": task.artifacts.solution_patch.read_text(
            encoding="utf-8", errors="replace"
        ),
        "hidden_test_patch": task.artifacts.test_patch.read_text(
            encoding="utf-8", errors="replace"
        ),
        "hidden_test_ids": list(task.test_config.get("f2p_node_ids") or []),
        "instruction_only_failures": failing_tests(baseline_run),
        "public_checks": public_checks or {},
        "source_fragments": [
            {
                "id": row["id"],
                "path": row["path"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "kind": row["kind"],
                "name": row["name"],
                "tokens": row.get("tokens"),
                "text": row["text"],
            }
            for row in selectable(pool)
        ],
        "runtime_observations": observations,
    }


def qualification_feedback(run_dir: Path) -> dict:
    """Collect constructor-only feedback from one repeated qualification round."""

    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    attempts = []
    for result_path in sorted(run_dir.glob("baseline_rep*/result.json")):
        episode_dir = result_path.parent
        result = _read_json(result_path)
        if not episode_dir.name.removeprefix("baseline_rep").isdigit() or result.get("episode_valid") is False:
            continue
        patch_path = episode_dir / "model.patch"
        trajectory_path = episode_dir / "trajectory.json"
        attempts.append({
            "episode": episode_dir.name,
            "success": bool(result.get("success")),
            "reward": result.get("reward") or {},
            "termination": (result.get("episode") or {}).get("termination"),
            "solver_patch": patch_path.read_text(encoding="utf-8", errors="replace")
            if patch_path.is_file() else "",
            # A trajectory repeats full old/new bodies and public check output.
            # The actual final patch plus operation/check status is sufficient
            # to locate gaps, with full trajectories remaining on disk.
            "trajectory_summary": [
                {"action": (event.get("action") or {}).get("action")
                           if isinstance(event.get("action"), dict) else event.get("action"),
                 "feedback": event.get("feedback"),
                 "operation_paths": [op.get("path") for op in
                                     ((event.get("action") or {}).get("operations", [])
                                      if isinstance(event.get("action"), dict) else event.get("operations", []))]}
                for event in (_read_json(trajectory_path) if trajectory_path.is_file() else [])
            ],
            "trajectory_path": str(trajectory_path),
            "verifier_diagnostics": _verifier_diagnostics(episode_dir),
        })
    return {
        "summary": summary,
        "failing_tests": failing_tests(run_dir),
        "attempts": attempts,
    }


def _verifier_diagnostics(episode_dir: Path) -> list[str]:
    """Surface build/collection failures even when test reports are empty."""
    import re
    pattern = re.compile(r"error\[|error TS|Error:|Exception:|FAILED|Illegal instruction|Unknown option|Cannot find")
    lines = []
    for path in episode_dir.glob("verifier/**/*.txt"):
        lines.extend(line[:1_000] for line in path.read_text(errors="replace").splitlines()
                     if pattern.search(line))
    # This is explicitly a diagnostic excerpt; the raw artifacts stay intact.
    return list(dict.fromkeys(lines))[:80]


def validate_plan(response: dict, *, pool: dict, observations: list[dict]) -> dict:
    """Normalize model output and reject unknown, duplicate, or empty RIUs."""

    if not isinstance(response, dict) or not isinstance(response.get("rius"), list):
        raise FunctionalRIUConstructionError("constructor response must contain a rius array")
    if not response["rius"]:
        raise FunctionalRIUConstructionError("constructor returned no functional RIUs")
    uncovered = response.get("uncovered", [])
    if not isinstance(uncovered, list) or not all(isinstance(row, dict) for row in uncovered):
        raise FunctionalRIUConstructionError("uncovered must be an array of objects")

    known_fragments = {row["id"] for row in selectable(pool)}
    known_observations = {row["id"] for row in observations}
    seen_fragments: set[str] = set()
    seen_observations: set[str] = set()
    rius = []
    for index, raw in enumerate(response["rius"], 1):
        if not isinstance(raw, dict):
            raise FunctionalRIUConstructionError(f"RIU {index} must be an object")
        for key in ("code_fragment_ids", "observation_ids", "coverage"):
            values = raw.get(key, [])
            if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                raise FunctionalRIUConstructionError(f"RIU {index} {key} must be an array of nonempty strings")
        fragments = list(dict.fromkeys(raw.get("code_fragment_ids", [])))
        observation_ids = list(dict.fromkeys(raw.get("observation_ids", [])))
        if not fragments and not observation_ids:
            raise FunctionalRIUConstructionError(f"RIU {index} is empty")
        unknown_fragments = set(fragments) - known_fragments
        unknown_observations = set(observation_ids) - known_observations
        if unknown_fragments or unknown_observations:
            raise FunctionalRIUConstructionError(
                f"RIU {index} references unknown units: "
                f"fragments={sorted(unknown_fragments)}, "
                f"observations={sorted(unknown_observations)}"
            )
        if seen_fragments.intersection(fragments) or seen_observations.intersection(observation_ids):
            raise FunctionalRIUConstructionError(
                f"RIU {index} overlaps an earlier RIU; functional ownership must be disjoint"
            )
        boundary = raw.get("functional_boundary")
        rationale = raw.get("rationale")
        if not isinstance(boundary, str) or not isinstance(rationale, str):
            raise FunctionalRIUConstructionError(f"RIU {index} boundary and rationale must be strings")
        boundary, rationale = boundary.strip(), rationale.strip()
        if not boundary or not rationale:
            raise FunctionalRIUConstructionError(
                f"RIU {index} needs functional_boundary and rationale"
            )
        seen_fragments.update(fragments)
        seen_observations.update(observation_ids)
        rius.append({
            "riu_id": f"riu-{index:02d}",
            "functional_boundary": boundary,
            "code_fragment_ids": fragments,
            "observation_ids": observation_ids,
            "rationale": rationale,
            "coverage": list(raw.get("coverage") or []),
        })
    return {
        "schema": SCHEMA,
        "rius": rius,
        "uncovered": uncovered,
    }


def construct_in_stages(
    *, client, request: dict, system_prompt: str, pool: dict,
    observations: list[dict], trace_dir: Path, max_input_tokens: int = 180_000,
    round_source_tokens: int = 20_000, total_source_tokens: int = 80_000,
    max_read_rounds: int = 10, validation_retries: int = 2,
    total_output_tokens: int = 120_000, prior_candidate: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Let the constructor read exact pool fragments and repair invalid plans.

    No source is truncated. Over-budget reads are reported explicitly. Invalid
    proposals stay constructor-only and are never assembled into actor packets.
    """

    bodies = {row["id"]: row for row in request["source_fragments"]}
    base, materials = stage_materials(request)
    base["source_fragments"] = [
        {key: value for key, value in row.items() if key != "text"}
        for row in bodies.values()
    ]
    read: dict[str, dict] = {}
    read_materials: dict[str, dict] = {}
    events: list[dict] = []
    feedback = None
    source_used = output_used = read_rounds = errors = 0
    trace_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_prompt("functional_riu_staged_system.txt")
    system = system_prompt + "\n\n" + protocol
    for call_no in range(1, max_read_rounds + validation_retries + 3):
        payload = {
            **base, "read_source_fragments": list(read.values()),
            "read_constructor_evidence": list(read_materials.values()),
            "constructor_feedback": feedback,
            "remaining": {
                "read_rounds": (max_read_rounds - read_rounds
                                if total_output_tokens - output_used > 32_000 else 0),
                "round_source_tokens": round_source_tokens,
                "source_tokens": total_source_tokens - source_used,
                "validation_retries": validation_retries - errors,
                "output_tokens": total_output_tokens - output_used,
            },
        }
        input_tokens = count_tokens(system) + count_tokens(json.dumps(payload, ensure_ascii=False))
        if input_tokens > max_input_tokens:
            raise FunctionalRIUConstructionError(
                f"staged constructor input is {input_tokens} tokens; cap is {max_input_tokens}"
            )
        if output_used >= total_output_tokens:
            raise FunctionalRIUConstructionError("constructor output-token budget exhausted")
        client.cfg.max_output_tokens = min(32_000, total_output_tokens - output_used)
        try:
            response = client.generate_json(
                prompt_id="deep_swe.functional_riu_staged",
                system_prompt=system, user_prompt=json.dumps(payload, ensure_ascii=False),
                trace_dir=trace_dir,
            )
        except ValueError as exc:
            charged, basis = _output_charge(client)
            output_used += charged
            errors += 1
            feedback = {"kind": "response_error", "error": str(exc),
                        "instruction": "Return only one compact function call. Do not generate source text or simulate read results."}
            events.append({"call": call_no, "input_tokens": input_tokens,
                           "output_tokens_used": output_used, "usage_basis": basis,
                           "status": "invalid_response", "feedback": feedback})
            _write_json(trace_dir / "staged_events.json", events)
            if errors > validation_retries:
                raise FunctionalRIUConstructionError("invalid-response retries exhausted") from exc
            continue
        # Missing usage is conservatively charged at the per-call ceiling.
        charged, basis = _output_charge(client)
        output_used += charged
        event = {"call": call_no, "input_tokens": input_tokens,
                 "output_tokens_used": output_used, "usage_basis": basis,
                 "action": response.get("action", "propose") if isinstance(response, dict) else None}
        events.append(event)
        try:
            if not isinstance(response, dict):
                raise FunctionalRIUConstructionError("constructor response must be an object")
            if response.get("action", "propose") not in ("read", "propose"):
                raise FunctionalRIUConstructionError("constructor action must be read or propose")
            if response.get("action") == "read":
                if read_rounds >= max_read_rounds or payload["remaining"]["read_rounds"] == 0:
                    raise FunctionalRIUConstructionError("read-round budget exhausted; propose a plan")
                ids = response.get("fragment_ids") or []
                evidence_ids = response.get("evidence_ids") or []
                if not all(isinstance(values, list) and all(isinstance(i, str) for i in values)
                           for values in (ids, evidence_ids)) or not (ids or evidence_ids):
                    raise FunctionalRIUConstructionError("read requires fragment_ids or evidence_ids")
                unknown = set(ids) - bodies.keys()
                unknown |= set(evidence_ids) - materials.keys()
                if unknown:
                    raise FunctionalRIUConstructionError(f"unknown read fragments: {sorted(unknown)}")
                read_rounds += 1
                accepted, dropped, round_used = [], [], 0
                for fragment_id, material in ([(i, False) for i in dict.fromkeys(ids)]
                                               + [(i, True) for i in dict.fromkeys(evidence_ids)]):
                    destination = read_materials if material else read
                    if fragment_id in destination:
                        continue
                    row = (materials if material else bodies)[fragment_id]
                    cost = count_tokens(json.dumps(row, ensure_ascii=False))
                    if round_used + cost > round_source_tokens or source_used + cost > total_source_tokens:
                        dropped.append(fragment_id)
                        continue
                    trial = {**payload,
                             "read_source_fragments": list(read.values()) + ([] if material else [row]),
                             "read_constructor_evidence": list(read_materials.values()) + ([row] if material else [])}
                    if count_tokens(system) + count_tokens(json.dumps(trial, ensure_ascii=False)) + 2_000 > max_input_tokens:
                        dropped.append(fragment_id)
                        continue
                    destination[fragment_id] = row
                    accepted.append(fragment_id)
                    round_used += cost
                    source_used += cost
                impossible = [i for i in dropped if count_tokens(json.dumps((materials if i in materials else bodies)[i], ensure_ascii=False)) > round_source_tokens]
                feedback = {"kind": "read_result", "accepted": accepted,
                            "dropped_over_source_budget": dropped,
                            "exceeds_per_read_budget": impossible,
                            "instruction": "Dropped fragments were not supplied. Units in exceeds_per_read_budget cannot fit separately either; request other exact units or report the budget limitation."}
                event.update(feedback=feedback, source_tokens_used=source_used)
            else:
                plan = validate_plan(response, pool=pool, observations=observations)
                selected = {i for riu in plan["rius"] for i in riu["code_fragment_ids"]}
                unread = selected - read.keys()
                if unread:
                    raise FunctionalRIUConstructionError(f"selected fragments must be read first: {sorted(unread)}")
                selected_observations = {i for riu in plan["rius"] for i in riu["observation_ids"]}
                unread_observations = {row["id"] for row in materials.values()
                                       if row["material"] in selected_observations and row["id"] not in read_materials}
                if unread_observations:
                    raise FunctionalRIUConstructionError(
                        f"selected observation parts must be read first: {sorted(unread_observations)}")
                if prior_candidate is not None:
                    old_source = {a.get("pool_fragment_id") for a in prior_candidate.get("code_atoms", [])}
                    old_obs = {o["id"] for o in prior_candidate.get("observations", [])}
                    new_obs = {i for riu in plan["rius"] for i in riu["observation_ids"]}
                    if old_source == selected and old_obs == new_obs:
                        raise FunctionalRIUConstructionError("revision must change the evidence set, not only regroup it")
                event["status"] = "accepted"
                _write_json(trace_dir / "staged_events.json", events)
                return plan, response, {"peak_input_tokens": max(e["input_tokens"] for e in events),
                                        "source_tokens": source_used, "output_tokens": output_used,
                                        "read_rounds": read_rounds, "validation_errors": errors,
                                        "calls": len(events)}
        except FunctionalRIUConstructionError as exc:
            errors += 1
            feedback = {"kind": "validation_error", "error": str(exc),
                        "invalid_proposal": response,
                        "instruction": "Repair this constructor-only proposal. Do not invent evidence or add task instructions. Empty RIUs are invalid."}
            event.update(status="rejected", feedback=feedback)
            _write_json(trace_dir / "staged_events.json", events)
            if errors > validation_retries:
                raise FunctionalRIUConstructionError(f"constructor validation retries exhausted: {exc}") from exc
        _write_json(trace_dir / "staged_events.json", events)
    raise FunctionalRIUConstructionError("constructor call budget exhausted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--public-checks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--prior-candidate", type=Path)
    parser.add_argument("--qualification-run", type=Path)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--runtime-image")
    add_model_arguments(parser)
    parser.add_argument("--env-file", default=str(Path.cwd() / ".env"))
    parser.add_argument("--max-input-tokens", type=int, default=180_000)
    parser.add_argument("--round-source-tokens", type=int, default=20_000)
    parser.add_argument("--total-source-tokens", type=int, default=80_000)
    parser.add_argument("--max-read-rounds", type=int, default=10)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--total-output-tokens", type=int, default=120_000)
    args = parser.parse_args()
    if min(args.max_input_tokens, args.round_source_tokens, args.total_source_tokens,
           args.max_read_rounds) < 1 or args.validation_retries < 0:
        parser.error("token/read budgets must be positive and validation retries nonnegative")

    task = load_task(args.task_dir, runtime_image=args.runtime_image)
    pool = _read_json(args.pool)
    if pool.get("task_id") != task.task_id or pool.get("base_commit") != task.repository.base_commit_hash:
        raise FunctionalRIUConstructionError("constructor pool task/base identity mismatch")
    raw_observations = _read_json(args.observations)
    observations = (
        raw_observations.get("observations", [])
        if isinstance(raw_observations, dict)
        else raw_observations
    )
    observations, rejected_observations = usable_observations(
        observations, task_id=task.task_id, base_commit=task.repository.base_commit_hash)
    public_checks = _read_json(args.public_checks) if args.public_checks else None
    request = build_constructor_request(
        task=task,
        pool=pool,
        observations=observations,
        baseline_run=args.baseline_run,
        public_checks=public_checks,
    )
    request["rejected_probe_summary"] = [
        {"id": row.get("id"), "reason": row.get("rejection_reason")}
        for row in rejected_observations
    ]
    revising = bool(args.prior_candidate or args.qualification_run)
    prior_candidate = None
    if revising and not (args.prior_candidate and args.qualification_run):
        parser.error("--prior-candidate and --qualification-run must be supplied together")
    if revising:
        prior_candidate = _read_json(args.prior_candidate)
        request["current_riu_plan"] = (prior_candidate.get("constructor_metadata") or {}).get(
            "rius", []
        )
        request["qualification_feedback"] = qualification_feedback(args.qualification_run)
    cfg = model_config(model=args.model, env_file=args.env_file,
                       max_output_tokens=32_000, **model_options(args))
    client = ConstructorActionClient(OpenAIJsonClient(cfg))
    plan, response, staged = construct_in_stages(
        client=client, request=request, pool=pool, observations=observations,
        prior_candidate=prior_candidate,
        max_input_tokens=args.max_input_tokens,
        round_source_tokens=args.round_source_tokens,
        total_source_tokens=args.total_source_tokens,
        max_read_rounds=args.max_read_rounds,
        validation_retries=args.validation_retries,
        total_output_tokens=args.total_output_tokens,
        system_prompt=load_prompt(
            "functional_riu_revision_system.txt"
            if revising else "functional_riu_constructor_system.txt"
        ),
        trace_dir=args.output.parent / "prompts" / "functional_riu_constructor",
    )
    request_tokens = staged["peak_input_tokens"]
    if revising and prior_candidate is not None:
        old_fragments = {
            atom.get("pool_fragment_id") for atom in prior_candidate.get("code_atoms", [])
        }
        old_observations = {row["id"] for row in prior_candidate.get("observations", [])}
        new_fragments = {
            fragment_id for riu in plan["rius"] for fragment_id in riu["code_fragment_ids"]
        }
        new_observations = {
            observation_id for riu in plan["rius"] for observation_id in riu["observation_ids"]
        }
        if old_fragments == new_fragments and old_observations == new_observations:
            raise FunctionalRIUConstructionError(
                "revision only regrouped the same solver-visible evidence; "
                "a failed Single qualification requires an evidence-set change"
            )
    plan.update({
        "task_id": task.task_id,
        "constructed_by": cfg.model,
        "constructor_input_tokens": request_tokens,
        "staged_construction": staged,
        "revision": revising,
        "diagnosis": response.get("diagnosis") if revising else None,
    })
    candidate = assemble_functional_candidate(
        pool,
        rius=plan["rius"],
        observations=observations,
        uncovered=plan["uncovered"],
    )
    candidate["constructor_metadata"].update({
        "model": cfg.model,
        "input_tokens": request_tokens,
        "plan_schema": SCHEMA,
        "revision": revising,
        "diagnosis": plan.get("diagnosis"),
        "staged_construction": staged,
    })
    _write_json(args.plan_output or args.output.with_name("functional_riu_plan.json"), plan)
    _write_json(args.output, candidate)
    print(json.dumps({
        "event": "done",
        "task_id": task.task_id,
        "rius": len(plan["rius"]),
        "fragments": len(candidate["code_atoms"]),
        "observations": len(candidate["observations"]),
        "constructor_input_tokens": request_tokens,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
