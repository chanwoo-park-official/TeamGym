"""Repartition frozen FinLongDocQA tasks and rerun packet leakage audits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from teamgym.data.finance.finlongdocqa.partition import (
    repartition_common_datapoint,
)
from teamgym.data.finance.finlongdocqa.pipeline import (
    run_packet_leakage_repair,
    run_initial_view_audit, run_packet_salience_audit,
)
from .packet_budget import materialize_context_cap, certify_materialized_views, PAPER_CONTEXT_BUDGET
from .evaluation import evaluator_datapoint, capture_model_observations
from .provenance import digest, file_digest, read_commits, REPO_ROOT
from teamgym.data.finance.finlongdocqa.runtime import (
    append_jsonl,
    build_client,
    read_jsonl,
    write_json,
    write_jsonl,
)
from teamgym.data.finance.finlongdocqa.schema import (
    validate_common_datapoint,
)
from teamgym.core.information_allocation import (
    PAPER_ALLOCATION_MODES,
)


ROOT = Path(os.environ.get("COLLABGYM_WORKSPACE", Path.cwd())).expanduser().resolve()
DEFAULT_OUTPUT = ROOT / "artifacts/generated/finance/finlongdocqa_allocations"
ALTERNATIVE_MODES = tuple(
    mode for mode in PAPER_ALLOCATION_MODES if mode != "one_each"
)


def _client_args(output: Path, overrides=None) -> SimpleNamespace:
    values = dict(
        provider="openai_compatible",
        model="openai.gpt-5.5",
        base_url="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
        endpoint="/responses",
        api_key_env="AWS_BEARER_TOKEN_BEDROCK",
        env_file=str(ROOT / ".env"),
        region="us-east-1",
        temperature=None,
        reasoning_effort="medium",
        max_output_tokens=8000,
        timeout_sec=300,
        request_retry_attempts=2,
        output_root=str(output),
    )
    if overrides is not None:
        for key in ("model", "env_file", "provider", "base_url", "api_key_env", "max_output_tokens", "reasoning_effort"):
            if getattr(overrides, key, None) is not None: values[key] = getattr(overrides, key)
    return SimpleNamespace(**values)


def _latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        instance_id = str(row.get("instance_id") or "")
        if instance_id:
            latest[instance_id] = row
    return latest


def _public_task(datapoint: dict[str, Any]) -> dict[str, Any]:
    task = datapoint.get("task_description") or {}
    return {
        "title": task.get("title"),
        "report_context": task.get("report_context"),
        "question": task.get("objective"),
        "response_requirements": [
            str((task.get("answer_format") or {}).get("instruction") or "")
        ],
        'source_unit_conventions': (datapoint.get('initial_common_information') or {}).get('source_unit_conventions', []),
    }


def _packet_words(datapoint: dict[str, Any]) -> dict[str, int]:
    return {
        str(row.get("participant_id") or ""): int(
            row.get("private_word_count") or 0
        )
        for row in datapoint.get("participants") or []
        if isinstance(row, dict)
    }


def _eligible(mode: str, rui_count: int) -> bool:
    return (
        mode not in {"overlap", "common_anchor"}
        or rui_count >= 3
    )


def _run_mode(
    *,
    mode: str,
    rows: list[dict[str, Any]],
    input_path: Path,
    output_root: Path,
    allocation_seed: int,
    audit_workers: int,
    packet_repair_rounds: int,
    packet_audit_max_chars: int,
    max_items: int,
    client_options=None,
    participant_context_token_budget: int = PAPER_CONTEXT_BUDGET,
) -> dict[str, Any]:
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    result_path = mode_dir / "repartition_results.jsonl"
    common_path = mode_dir / "common_datapoints.jsonl"
    summary_path = mode_dir / "summary.json"
    latest = _latest_results(result_path)
    common_by_id = {
        str(row.get("instance_id") or ""): row
        for row in read_jsonl(common_path)
        if row.get("instance_id")
    }
    eligible_rows = [
        row
        for row in rows
        if _eligible(
            mode,
            len(row.get("reference_information_units") or []),
        )
    ]
    if max_items > 0:
        eligible_rows = eligible_rows[:max_items]
    client = build_client(
        _client_args(mode_dir, client_options), mode_dir / "_llm_requests.jsonl"
    )

    for position, base in enumerate(eligible_rows, start=1):
        instance_id = str(base.get("instance_id") or "")
        if (
            (latest.get(instance_id) or {}).get("status") == "accepted"
            and instance_id in common_by_id
        ):
            print(
                json.dumps(
                    {
                        "event": "frozen_repartition_skip",
                        "mode": mode,
                        "position": position,
                        "total": len(eligible_rows),
                        "instance_id": instance_id,
                    }
                ),
                flush=True,
            )
            continue

        try:
            local_rows, salience_rows = [], []
            common = repartition_common_datapoint(
                base,
                allocation_mode=mode,
                allocation_seed=allocation_seed,
            )
            materialize_context_cap(common, token_budget=participant_context_token_budget)
            common, audits, removed_ids, errors = run_packet_leakage_repair(
                client=client,
                datapoint=common,
                public_task=_public_task(base),
                trace_root=mode_dir / "_llm_traces" / instance_id,
                max_workers=audit_workers,
                max_rounds=packet_repair_rounds,
                max_prompt_chars=packet_audit_max_chars,
            )
            if not errors:
                certify_materialized_views(common, token_budget=participant_context_token_budget)
                salience_rows, salience_errors = run_packet_salience_audit(client=client, datapoint=common,
                    trace_root=mode_dir / "_llm_traces" / instance_id / "salience", max_workers=audit_workers,
                    context_token_budget=participant_context_token_budget)
                errors.extend(salience_errors)
            if not errors:
                task = {"public_task": _public_task(base), "rubric": base["dataset_specific"]["free_response_rubric"],
                        "oracle": {"reference_answer": base["oracle"].get("answer_text") or base["oracle"].get("canonical_response")}}
                local_rows, local_errors = run_initial_view_audit(client=client, datapoint=common, task=task,
                    trace_root=mode_dir / "_llm_traces" / instance_id / "initial_view", max_workers=audit_workers,
                    context_token_budget=participant_context_token_budget)
                errors.extend(local_errors)
            for key in ('task_description', 'reference_information_units', 'oracle'):
                if common[key] != base[key]: errors.append(f'Frozen {key} changed during redistribution')
            common['generation_metadata'].update(initial_view_audits=local_rows, packet_salience_audits=salience_rows,
                rematerialized_from_instance_id=instance_id)
            if not errors:
                common['instance_id'] = f'{instance_id}__{mode}__seed{allocation_seed}'
                before = capture_model_observations(common, budget=participant_context_token_budget)
                common = evaluator_datapoint(common)
                if capture_model_observations(common, budget=participant_context_token_budget) != before:
                    errors.append('Evaluator export changed actual model observations')
            validation = validate_common_datapoint(common)
            common["format_validation"] = validation
            if not validation.get("valid"):
                errors = [
                    *errors,
                    f"structural validation failed: {validation}",
                ]
        except Exception as exc:  # noqa: BLE001
            common = None
            audits = []
            removed_ids = set()
            errors = [f"{type(exc).__name__}: {exc}"]

        result = {
            "schema": "finlongdocqa_frozen_repartition_result_v1",
            "mode": mode,
            "instance_id": instance_id,
            "rui_count": len(base.get("reference_information_units") or []),
            "status": "accepted" if not errors else "rejected",
            "errors": errors,
            "removed_leaking_context_unit_ids": sorted(removed_ids),
            "packet_words": _packet_words(common or {}),
            "audit_count": len(audits),
            "initial_view_audits": local_rows,
            "salience_audits": salience_rows,
        }
        append_jsonl(result_path, result)
        latest[instance_id] = result
        print(
            json.dumps(
                {
                    "event": "frozen_repartition_done",
                    "position": position,
                    "total": len(eligible_rows),
                    **result,
                }
            ),
            flush=True,
        )
        if errors or common is None:
            continue
        common_by_id[instance_id] = common
        write_jsonl(
            common_path,
            [
                common_by_id[str(row.get("instance_id") or "")]
                for row in eligible_rows
                if str(row.get("instance_id") or "") in common_by_id
            ],
        )

    accepted_ids = {
        instance_id
        for instance_id, result in latest.items()
        if result.get("status") == "accepted"
        and instance_id in common_by_id
    }
    rejected_ids = {
        instance_id
        for instance_id, result in latest.items()
        if result.get("status") == "rejected"
        and instance_id not in accepted_ids
    }
    eligible_ids = [
        str(row.get("instance_id") or "") for row in eligible_rows
    ]
    summary = {
        "schema": "finlongdocqa_frozen_repartition_summary_v1",
        "mode": mode,
        "source_input": str(input_path),
        "eligible_count": len(eligible_ids),
        "accepted_count": sum(
            instance_id in accepted_ids for instance_id in eligible_ids
        ),
        "rejected_count": sum(
            instance_id in rejected_ids for instance_id in eligible_ids
        ),
        "pending_count": sum(
            instance_id not in accepted_ids | rejected_ids
            for instance_id in eligible_ids
        ),
        "common_datapoints": str(common_path),
        "results": str(result_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--mode",
        action="append",
        choices=PAPER_ALLOCATION_MODES,
        default=None,
    )
    parser.add_argument("--allocation-seed", type=int, default=20260801)
    parser.add_argument("--audit-workers", type=int, default=4)
    parser.add_argument("--packet-repair-rounds", type=int, default=6)
    parser.add_argument("--packet-audit-max-chars", type=int, default=500_000)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument('--model', '--audit-model', dest='model', default='openai.gpt-5.5')
    parser.add_argument('--env-file', default=str(ROOT / '.env'))
    parser.add_argument('--base-url', default='https://bedrock-mantle.us-east-2.api.aws/openai/v1')
    parser.add_argument('--provider', choices=['openai_compatible','bedrock_anthropic'], default='openai_compatible')
    parser.add_argument('--api-key-env', default='AWS_BEARER_TOKEN_BEDROCK')
    parser.add_argument('--reasoning-effort', default='medium')
    parser.add_argument('--max-output-tokens', type=int, default=12000)
    parser.add_argument('--participant-context-token-budget', type=int, default=PAPER_CONTEXT_BUDGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError('Use a new empty output directory; stale allocation audits cannot be resumed')
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No frozen tasks found in {input_path}")
    from .source import PARSER_VERSION
    if any(row.get('source', {}).get('parser_version') != PARSER_VERSION for row in rows):
        raise ValueError('Re-index and reconstruct with the current source-unit-context parser before redistribution')
    producer = json.loads((input_path.parent / 'run_manifest.json').read_text())
    commits, invalid = read_commits(input_path.parent, producer['fingerprint'], producer['selected_report_ids'])
    accepted = {digest(c['common']) for c in commits if c['result'].get('status') == 'accepted'}
    if invalid or any(digest(row) not in accepted for row in rows):
        raise ValueError('Redistribution input is not an unchanged accepted producer commit')
    if not 0 < args.participant_context_token_budget <= PAPER_CONTEXT_BUDGET:
        raise ValueError('Finance participant budget must be between 1 and 40000')
    frozen_input = file_digest(input_path)
    frozen_code = {str(p): file_digest(p) for p in (REPO_ROOT / 'teamgym').rglob('*.py')}
    modes = tuple(args.mode or ALTERNATIVE_MODES)
    manifest = {
        "schema": "finlongdocqa_frozen_repartition_manifest_v1",
        "source_input": str(input_path),
        "source_input_sha256": frozen_input,
        "audit_model": args.model, "participant_context_token_budget": args.participant_context_token_budget,
        "status": "running", "code_sha256": frozen_code,
        "source_count": len(rows),
        "modes": list(modes),
        "allocation_seed": args.allocation_seed,
        "audit_workers": args.audit_workers,
        "packet_repair_rounds": args.packet_repair_rounds,
        "packet_audit_max_chars": args.packet_audit_max_chars,
        "frozen_fields": [
            "task_description",
            "reference_information_units",
            "oracle",
        ],
        "base_context_policy": (
            "reuse the already-sanitized balanced one_each context universe"
        ),
    }
    write_json(output_root / "manifest.json", manifest)
    summaries = [
        _run_mode(
            mode=mode,
            rows=rows,
            input_path=input_path,
            output_root=output_root,
            allocation_seed=args.allocation_seed,
            audit_workers=args.audit_workers,
            packet_repair_rounds=args.packet_repair_rounds,
            packet_audit_max_chars=args.packet_audit_max_chars,
            max_items=args.max_items,
            client_options=args, participant_context_token_budget=args.participant_context_token_budget,
        )
        for mode in modes
    ]
    write_json(
        output_root / "summary.json",
        {
            "schema": "finlongdocqa_frozen_repartition_all_modes_summary_v1",
            "modes": summaries,
        },
    )
    if file_digest(input_path) != frozen_input or {str(p): file_digest(p) for p in (REPO_ROOT / 'teamgym').rglob('*.py')} != frozen_code:
        raise RuntimeError('Finance source or code changed during allocation auditing')
    manifest.update(status='completed', outputs={str(p.relative_to(output_root)):file_digest(p)
        for mode in modes for p in (output_root/mode).glob('*.jsonl')})
    write_json(output_root/'manifest.json', manifest)


if __name__ == "__main__":
    main()
