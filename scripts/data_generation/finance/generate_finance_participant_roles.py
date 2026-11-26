#!/usr/bin/env python3
"""Generate audited public participant roles for FinDVer and FinLongDocQA.

The input JSONL is never modified. By default the script processes one record
per dataset as a safe smoke test. Pass ``--all-problem-instances`` for a full
run. Outputs include an audited feature sidecar, role previews, a derived JSONL,
prompt traces, request metadata, errors, and a report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.core.io import (  # noqa: E402
    prompt_text,
    render_template,
    write_json,
    write_jsonl,
)
from teamgym.core.participant_roles import (  # noqa: E402
    duplicate_role_names,
    reference_information_units,
    riu_text_sha256,
)
from teamgym.data.long_context_reasoning.finance_participant_roles import (  # noqa: E402
    FINANCE_DATASETS,
    PROMPT_VERSION,
    apply_public_roles,
    audit_information_features,
    finance_dataset,
    finance_information_feature_input,
    generate_information_features,
    problem_instance_id,
    public_problem_description,
    valid_cached_feature_rows,
)
from teamgym.llm import (  # noqa: E402
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=FINANCE_DATASETS,
        default=list(FINANCE_DATASETS),
    )
    parser.add_argument(
        "--per-dataset-limit",
        type=int,
        default=1,
        help="Records to process per dataset unless --all-problem-instances is set.",
    )
    parser.add_argument("--all-problem-instances", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--audit-attempts", type=int, default=4)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.per_dataset_limit < 1:
        raise ValueError("--per-dataset-limit must be positive.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    input_path = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_root.mkdir(parents=True, exist_ok=True)

    selected, scan_report = select_datapoints(
        input_path,
        datasets=set(args.datasets),
        per_dataset_limit=(
            None if args.all_problem_instances else args.per_dataset_limit
        ),
    )
    missing = sorted(set(args.datasets) - {dataset for dataset, _ in selected})
    if missing:
        raise ValueError(f"Input contains no selected records for {missing}")
    sharded = {
        key: row
        for index, (key, row) in enumerate(sorted(selected.items()))
        if index % args.num_shards == args.shard_index
    }
    if not sharded:
        raise ValueError(f"Shard {args.shard_index} contains no selected records.")

    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            credentials_file=args.credentials_file,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=4,
            request_log_path=str(output_root / "llm_requests.jsonl"),
            log_prompts=True,
            log_responses=True,
        )
    )
    feature_system = prompt_text("legal/common/information_feature_system.txt")
    feature_user = prompt_text("legal/common/information_feature_user.txt")
    audit_system = prompt_text("legal/common/information_feature_audit_system.txt")
    audit_user = prompt_text("legal/common/information_feature_audit_user.txt")

    cached = load_cached_features(output_root / "information_features.jsonl")
    sidecar_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    augmented_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed = 0
    reused = 0

    for (dataset, instance_id), row in sharded.items():
        try:
            units = reference_information_units(row)
            rui_ids = [str(unit["rui_id"]) for unit in units]
            cached_rows = valid_cached_feature_rows(
                cached.get((dataset, instance_id), []),
                units=units,
            )
            if cached_rows:
                features = {
                    str(item["rui_id"]): str(item["information_feature"])
                    for item in cached_rows
                }
                group_sidecar = cached_rows
                reused += 1
            else:
                feature_payload = {
                    "public_problem_description": public_problem_description(row),
                    "rius": [
                        finance_information_feature_input(unit) for unit in units
                    ],
                }
                features, generation_requests = generate_information_features(
                    client=client,
                    dataset=dataset,
                    instance_id=instance_id,
                    system_prompt=feature_system,
                    user_prompt=render_template(feature_user, feature_payload),
                    expected_rui_ids=rui_ids,
                    trace_root=(
                        output_root
                        / "traces"
                        / dataset
                        / instance_id
                        / "generation"
                    ),
                    attempts=args.generation_attempts,
                )
                features, audits, audit_requests = audit_information_features(
                    client=client,
                    dataset=dataset,
                    instance_id=instance_id,
                    system_prompt=audit_system,
                    user_template=audit_user,
                    datapoint=row,
                    units=units,
                    features=features,
                    trace_root=(
                        output_root / "traces" / dataset / instance_id / "audit"
                    ),
                    attempts=args.audit_attempts,
                )
                unsafe = {
                    rui_id: audit
                    for rui_id, audit in audits.items()
                    if audit.get("safe") is not True
                }
                if unsafe:
                    raise ValueError(
                        f"Leakage audit did not accept all features: {unsafe}"
                    )
                units_by_id = {str(unit["rui_id"]): unit for unit in units}
                group_sidecar = [
                    {
                        "schema": "finance_riu_information_feature_sidecar_v1",
                        "prompt_version": PROMPT_VERSION,
                        "dataset": dataset,
                        "problem_instance_id": instance_id,
                        "rui_id": rui_id,
                        "riu_text_sha256": riu_text_sha256(units_by_id[rui_id]),
                        "information_feature": features[rui_id],
                        "leakage_audit": audits[rui_id],
                        "generation_requests": generation_requests,
                        "audit_requests": audit_requests,
                    }
                    for rui_id in rui_ids
                ]

            augmented, preview = apply_public_roles(
                row,
                features_by_rui_id=features,
            )
            duplicate_names = duplicate_role_names(preview)
            if duplicate_names:
                raise ValueError(f"Duplicate proposed role names: {duplicate_names}")
            sidecar_rows.extend(group_sidecar)
            augmented_rows.append(augmented)
            preview_rows.append(
                {
                    "schema": "finance_participant_role_preview_v1",
                    "dataset": dataset,
                    "problem_instance_id": instance_id,
                    "all_information_features_passed_leakage_audit": True,
                    "participants": preview,
                    "duplicate_proposed_role_names": duplicate_names,
                }
            )
            completed += 1
            print(
                f"completed {dataset} {instance_id}: {len(units)} RIUs",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            error = {
                "dataset": dataset,
                "problem_instance_id": instance_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            errors.append(error)
            print(
                f"error {dataset} {instance_id}: {error['error_message']}",
                flush=True,
            )
            write_outputs(
                output_root,
                args=args,
                input_path=input_path,
                scan_report=scan_report,
                selected_count=len(selected),
                shard_count=len(sharded),
                completed=completed,
                reused=reused,
                sidecar_rows=sidecar_rows,
                preview_rows=preview_rows,
                augmented_rows=augmented_rows,
                errors=errors,
            )
            if not args.continue_on_error:
                raise
        write_outputs(
            output_root,
            args=args,
            input_path=input_path,
            scan_report=scan_report,
            selected_count=len(selected),
            shard_count=len(sharded),
            completed=completed,
            reused=reused,
            sidecar_rows=sidecar_rows,
            preview_rows=preview_rows,
            augmented_rows=augmented_rows,
            errors=errors,
        )

    report = build_report(
        args=args,
        input_path=input_path,
        scan_report=scan_report,
        selected_count=len(selected),
        shard_count=len(sharded),
        completed=completed,
        reused=reused,
        sidecar_rows=sidecar_rows,
        preview_rows=preview_rows,
        augmented_rows=augmented_rows,
        errors=errors,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def select_datapoints(
    path: Path,
    *,
    datasets: set[str],
    per_dataset_limit: int | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    selected_counts: Counter[str] = Counter()
    scanned: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                dataset = finance_dataset(row)
            except ValueError:
                scanned["unsupported_rows"] += 1
                continue
            scanned[f"seen_{dataset}"] += 1
            if dataset not in datasets:
                continue
            if (
                per_dataset_limit is not None
                and selected_counts[dataset] >= per_dataset_limit
            ):
                continue
            instance_id = problem_instance_id(row)
            key = (dataset, instance_id)
            if key in selected:
                raise ValueError(
                    f"{path}:{line_number}: duplicate finance record {key}"
                )
            selected[key] = row
            selected_counts[dataset] += 1
            scanned[f"selected_{dataset}"] += 1
    return selected, dict(sorted(scanned.items()))


def load_cached_features(
    path: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return groups
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[
                (
                    str(row.get("dataset") or ""),
                    str(row.get("problem_instance_id") or ""),
                )
            ].append(row)
    return groups


def build_report(
    *,
    args: argparse.Namespace,
    input_path: Path,
    scan_report: dict[str, int],
    selected_count: int,
    shard_count: int,
    completed: int,
    reused: int,
    sidecar_rows: list[dict[str, Any]],
    preview_rows: list[dict[str, Any]],
    augmented_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "finance_participant_role_generation_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "input": str(input_path),
        "target_datasets": list(args.datasets),
        "per_dataset_limit": (
            None if args.all_problem_instances else args.per_dataset_limit
        ),
        "all_problem_instances": bool(args.all_problem_instances),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "model": args.model,
        "provider": args.provider,
        "selected_problem_instances": selected_count,
        "problem_instances_in_shard": shard_count,
        "completed_problem_instances": completed,
        "reused_problem_instances": reused,
        "feature_count": len(sidecar_rows),
        "role_preview_count": len(preview_rows),
        "derived_row_count": len(augmented_rows),
        "error_count": len(errors),
        "input_scan": scan_report,
    }


def write_outputs(
    output_root: Path,
    **report_kwargs: Any,
) -> None:
    sidecar_rows = report_kwargs["sidecar_rows"]
    preview_rows = report_kwargs["preview_rows"]
    augmented_rows = report_kwargs["augmented_rows"]
    errors = report_kwargs["errors"]
    write_jsonl(output_root / "information_features.jsonl", sidecar_rows)
    write_jsonl(output_root / "participant_role_previews.jsonl", preview_rows)
    write_jsonl(output_root / "role_augmented.jsonl", augmented_rows)
    write_jsonl(output_root / "errors.jsonl", errors)
    write_json(
        output_root / "report.json",
        build_report(**report_kwargs),
    )


if __name__ == "__main__":
    main()
