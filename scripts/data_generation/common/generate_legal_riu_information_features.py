#!/usr/bin/env python3
"""Generate model-produced PIU information features and participant role names.

The process targets eCFR Casebook, MSLR, Multi-LexSum, and SARA. CUAD and MAUD
are explicitly excluded because their existing construction already produces
participant roles. The script never modifies its input and writes only a
feature sidecar, role-name previews, traces, and a report under --output-root.
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

from teamgym.core.io import (
    prompt_text,
    render_template,
    write_json,
    write_jsonl,
)
from teamgym.core.participant_roles import (
    allocation_mode,
    duplicate_role_names,
    normalize_information_features,
    normalize_leakage_audit,
    reference_information_units,
    riu_text_sha256,
    role_name_preview,
    validate_feature,
)
from teamgym.data.legal.participant_role_augmentation import (
    EXCLUDED_SOURCE_CORPORA,
    PROMPT_VERSION,
    TARGET_SOURCE_CORPORA,
    ExcludedSourceCorpusError,
    canonical_problem_instance_id,
    information_feature_input,
    public_problem_description,
    reference_decision,
    source_corpus,
)
from teamgym.llm import (
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
        "--corpora",
        nargs="+",
        choices=TARGET_SOURCE_CORPORA,
        default=list(TARGET_SOURCE_CORPORA),
    )
    parser.add_argument(
        "--per-corpus-limit",
        type=int,
        default=1,
        help="Canonical problem instances to test per source corpus (default: 1).",
    )
    parser.add_argument(
        "--all-problem-instances",
        action="store_true",
        help="Process every canonical problem instance in the selected source corpora.",
    )
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
    parser.add_argument("--generation-attempts", type=int, default=2)
    parser.add_argument("--audit-attempts", type=int, default=4)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.per_corpus_limit < 1:
        raise ValueError("--per-corpus-limit must be positive.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    input_path = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_groups, scan_report = select_problem_instances(
        input_path,
        corpora=set(args.corpora),
        per_corpus_limit=None if args.all_problem_instances else args.per_corpus_limit,
    )
    missing_corpora = sorted(set(args.corpora) - {corpus for corpus, _ in all_groups})
    if missing_corpora:
        raise ValueError(f"Input contains no selected problem instances for {missing_corpora}")
    groups = {
        key: variants
        for index, (key, variants) in enumerate(sorted(all_groups.items()))
        if index % args.num_shards == args.shard_index
    }
    if not groups:
        raise ValueError(f"Shard {args.shard_index} contains no selected problem instances.")

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

    cached_groups = load_cached_feature_groups(output_root / "information_features.jsonl")
    sidecar_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    completed = 0
    reused = 0
    for (corpus, problem_id), variants in groups.items():
        representative = variants[0]
        try:
            ensure_matching_rius(variants)
            units = reference_information_units(representative)
            rui_ids = [str(unit["rui_id"]) for unit in units]
            cached = valid_cached_features(
                cached_groups.get((corpus, problem_id), []),
                units=units,
            )
            if cached:
                features = {
                    str(row["rui_id"]): str(row["information_feature"])
                    for row in cached
                }
                sidecar_rows.extend(cached)
                for variant in variants:
                    preview = role_name_preview(variant, features_by_rui_id=features)
                    preview_rows.append(
                        {
                            "schema": "legal_participant_role_preview_v1",
                            "source_corpus": corpus,
                            "problem_instance_id": problem_id,
                            "instance_id": variant.get("instance_id"),
                            "riu_distribution": allocation_mode(variant),
                            "all_information_features_passed_leakage_audit": True,
                            "participants": preview,
                            "duplicate_proposed_role_names": duplicate_role_names(preview),
                        }
                    )
                completed += 1
                reused += 1
                print(
                    f"reused {corpus} {problem_id}: {len(units)} RIUs",
                    flush=True,
                )
                continue
            generation_payload = {
                "public_problem_description": public_problem_description(representative),
                "rius": [information_feature_input(unit) for unit in units],
            }
            features, generation_metas = generate_information_features(
                client=client,
                corpus=corpus,
                problem_id=problem_id,
                system_prompt=feature_system,
                user_prompt=render_template(feature_user, generation_payload),
                expected_rui_ids=rui_ids,
                trace_root=output_root / "traces" / corpus / safe_id(problem_id) / "generation",
                attempts=max(1, args.generation_attempts),
            )

            features, audits, audit_metas = audit_information_features(
                client=client,
                corpus=corpus,
                problem_id=problem_id,
                system_prompt=audit_system,
                user_template=audit_user,
                representative=representative,
                units=units,
                features=features,
                trace_root=output_root / "traces" / corpus / safe_id(problem_id) / "audit",
                attempts=max(1, args.audit_attempts),
            )

            units_by_id = {str(unit["rui_id"]): unit for unit in units}
            for rui_id in rui_ids:
                unit = units_by_id[rui_id]
                sidecar_rows.append(
                    {
                        "schema": "legal_riu_information_feature_sidecar_v1",
                        "prompt_version": PROMPT_VERSION,
                        "source_corpus": corpus,
                        "problem_instance_id": problem_id,
                        "rui_id": rui_id,
                        "riu_text_sha256": riu_text_sha256(unit),
                        "information_feature": features[rui_id],
                        "leakage_audit": audits[rui_id],
                        "generation_requests": generation_metas,
                        "audit_requests": audit_metas,
                    }
                )

            all_safe = all(audit["safe"] for audit in audits.values())
            for variant in variants:
                preview = role_name_preview(variant, features_by_rui_id=features)
                preview_rows.append(
                    {
                        "schema": "legal_participant_role_preview_v1",
                        "source_corpus": corpus,
                        "problem_instance_id": problem_id,
                        "instance_id": variant.get("instance_id"),
                        "riu_distribution": allocation_mode(variant),
                        "all_information_features_passed_leakage_audit": all_safe,
                        "participants": preview,
                        "duplicate_proposed_role_names": duplicate_role_names(preview),
                    }
                )
            completed += 1
            print(
                f"completed {corpus} {problem_id}: {len(units)} RIUs; "
                f"safe={sum(audit['safe'] for audit in audits.values())}/{len(audits)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            error = {
                "source_corpus": corpus,
                "problem_instance_id": problem_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            errors.append(error)
            print(f"error {corpus} {problem_id}: {error['error_message']}", flush=True)
            if not args.continue_on_error:
                write_outputs(
                    output_root,
                    sidecar_rows=sidecar_rows,
                    preview_rows=preview_rows,
                    errors=errors,
                    report=build_report(
                        args=args,
                        input_path=input_path,
                        scan_report=scan_report,
                        groups=groups,
                        total_selected_problem_instances=len(all_groups),
                        completed=completed,
                        reused=reused,
                        sidecar_rows=sidecar_rows,
                        preview_rows=preview_rows,
                        errors=errors,
                    ),
                )
                raise

        write_outputs(
            output_root,
            sidecar_rows=sidecar_rows,
            preview_rows=preview_rows,
            errors=errors,
            report=build_report(
                args=args,
                input_path=input_path,
                scan_report=scan_report,
                groups=groups,
                total_selected_problem_instances=len(all_groups),
                completed=completed,
                reused=reused,
                sidecar_rows=sidecar_rows,
                preview_rows=preview_rows,
                errors=errors,
            ),
        )

    report = build_report(
        args=args,
        input_path=input_path,
        scan_report=scan_report,
        groups=groups,
        total_selected_problem_instances=len(all_groups),
        completed=completed,
        reused=reused,
        sidecar_rows=sidecar_rows,
        preview_rows=preview_rows,
        errors=errors,
    )
    write_outputs(
        output_root,
        sidecar_rows=sidecar_rows,
        preview_rows=preview_rows,
        errors=errors,
        report=report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def select_problem_instances(
    path: Path,
    *,
    corpora: set[str],
    per_corpus_limit: int | None,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    selected_ids: dict[str, list[str]] = defaultdict(list)
    scanned = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                corpus = source_corpus(row)
            except ExcludedSourceCorpusError:
                raw = str(row.get("dataset") or "").lower()
                excluded = "cuad" if "cuad" in raw else "maud"
                scanned[f"excluded_{excluded}"] += 1
                continue
            except ValueError:
                scanned["unsupported"] += 1
                continue
            scanned[f"seen_{corpus}"] += 1
            if corpus not in corpora:
                continue
            problem_id = canonical_problem_instance_id(row)
            if problem_id not in selected_ids[corpus]:
                if per_corpus_limit is not None and len(selected_ids[corpus]) >= per_corpus_limit:
                    continue
                selected_ids[corpus].append(problem_id)
            if problem_id in selected_ids[corpus]:
                groups.setdefault((corpus, problem_id), []).append(row)
                scanned[f"selected_variants_{corpus}"] += 1
    return groups, dict(sorted(scanned.items()))


def load_cached_feature_groups(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return groups
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("source_corpus") or ""),
                str(row.get("problem_instance_id") or ""),
            )
            groups[key].append(row)
    return groups


def valid_cached_features(
    rows: list[dict[str, Any]],
    *,
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not rows or any(row.get("prompt_version") != PROMPT_VERSION for row in rows):
        return []
    by_id = {str(row.get("rui_id") or ""): row for row in rows}
    expected_ids = {str(unit["rui_id"]) for unit in units}
    if set(by_id) != expected_ids:
        return []
    for unit in units:
        rui_id = str(unit["rui_id"])
        row = by_id[rui_id]
        if row.get("riu_text_sha256") != riu_text_sha256(unit):
            return []
        audit = row.get("leakage_audit")
        if not isinstance(audit, dict) or audit.get("safe") is not True:
            return []
        try:
            validate_feature(str(row.get("information_feature") or ""), rui_id=rui_id)
        except ValueError:
            return []
    return [by_id[rui_id] for rui_id in sorted(expected_ids)]


def generate_information_features(
    *,
    client: OpenAIJsonClient,
    corpus: str,
    problem_id: str,
    system_prompt: str,
    user_prompt: str,
    expected_rui_ids: list[str],
    trace_root: Path,
    attempts: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    previous_response: dict[str, Any] | None = None
    validation_error: str | None = None
    request_metas: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        prompt = user_prompt
        if previous_response is not None:
            prompt += "\n\nThe previous response failed validation. Return a complete corrected response for every RIU."
            prompt += "\nDo not preserve a feature that caused the validation error."
            prompt += "\nPrevious response:\n" + json.dumps(previous_response, indent=2, ensure_ascii=False)
            prompt += f"\nValidation error:\n{validation_error}"
        response = client.generate_json(
            prompt_id=(
                f"{corpus}.{safe_id(problem_id)}.information_features.attempt_{attempt}"
            ),
            system_prompt=system_prompt,
            user_prompt=prompt,
            trace_dir=trace_root / f"attempt_{attempt}",
        )
        request_metas.append(dict(client.last_request_meta))
        try:
            return (
                normalize_information_features(
                    response,
                    expected_rui_ids=expected_rui_ids,
                ),
                request_metas,
            )
        except ValueError as exc:
            previous_response = response
            validation_error = str(exc)
            if attempt == attempts:
                raise
    raise RuntimeError("Information-feature generation ended without a result.")


def ensure_matching_rius(variants: list[dict[str, Any]]) -> None:
    expected: dict[str, str] | None = None
    for variant in variants:
        actual = {
            str(unit["rui_id"]): riu_text_sha256(unit)
            for unit in reference_information_units(variant)
        }
        if expected is None:
            expected = actual
        elif actual != expected:
            raise ValueError(
                f"RIUs differ across variants of {canonical_problem_instance_id(variant)}"
            )


def audit_information_features(
    *,
    client: OpenAIJsonClient,
    corpus: str,
    problem_id: str,
    system_prompt: str,
    user_template: str,
    representative: dict[str, Any],
    units: list[dict[str, Any]],
    features: dict[str, str],
    trace_root: Path,
    attempts: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    expected_rui_ids = [str(unit["rui_id"]) for unit in units]
    units_by_id = {str(unit["rui_id"]): unit for unit in units}
    current_features = dict(features)
    request_metas: list[dict[str, Any]] = []
    accepted_audits: dict[str, dict[str, Any]] = {}
    pending_rui_ids = list(expected_rui_ids)
    final_unsafe_audits: dict[str, dict[str, Any]] = {}
    for attempt in range(1, attempts + 1):
        payload = {
            "public_problem_description": public_problem_description(representative),
            "reference_decision": reference_decision(representative),
            "rius_and_features": [
                {
                    **information_feature_input(units_by_id[rui_id]),
                    "proposed_information_feature": current_features[rui_id],
                }
                for rui_id in pending_rui_ids
            ],
        }
        response = client.generate_json(
            prompt_id=(
                f"{corpus}.{safe_id(problem_id)}.information_feature_audit.attempt_{attempt}"
            ),
            system_prompt=system_prompt,
            user_prompt=render_template(user_template, payload),
            trace_dir=trace_root / f"attempt_{attempt}",
        )
        request_metas.append(dict(client.last_request_meta))
        attempt_audits = normalize_leakage_audit(
            response,
            expected_rui_ids=pending_rui_ids,
        )
        unsafe = [
            rui_id for rui_id, audit in attempt_audits.items() if not audit["safe"]
        ]
        accepted_audits.update(
            {
                rui_id: audit
                for rui_id, audit in attempt_audits.items()
                if audit["safe"]
            }
        )
        final_unsafe_audits = {
            rui_id: attempt_audits[rui_id] for rui_id in unsafe
        }
        if not unsafe:
            return current_features, accepted_audits, request_metas
        if attempt == attempts:
            return (
                current_features,
                {**accepted_audits, **final_unsafe_audits},
                request_metas,
            )
        for rui_id in unsafe:
            suggestion = attempt_audits[rui_id]["suggested_information_feature"]
            validate_feature(suggestion, rui_id=rui_id)
            current_features[rui_id] = suggestion
        pending_rui_ids = unsafe
    return current_features, {**accepted_audits, **final_unsafe_audits}, request_metas


def safe_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned[:160] or "problem_instance"


def build_report(
    *,
    args: argparse.Namespace,
    input_path: Path,
    scan_report: dict[str, Any],
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    total_selected_problem_instances: int,
    completed: int,
    reused: int,
    sidecar_rows: list[dict[str, Any]],
    preview_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus_features = Counter(row["source_corpus"] for row in sidecar_rows)
    unsafe = [row for row in sidecar_rows if not row["leakage_audit"]["safe"]]
    duplicate_previews = [row for row in preview_rows if row["duplicate_proposed_role_names"]]
    return {
        "schema": "legal_participant_role_augmentation_generation_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "input": str(input_path),
        "target_source_corpora": list(args.corpora),
        "explicitly_excluded_source_corpora": list(EXCLUDED_SOURCE_CORPORA),
        "per_corpus_limit": None if args.all_problem_instances else args.per_corpus_limit,
        "all_problem_instances": args.all_problem_instances,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "model": args.model,
        "provider": args.provider,
        "total_selected_problem_instances": total_selected_problem_instances,
        "selected_problem_instances_in_shard": len(groups),
        "completed_problem_instances": completed,
        "reused_problem_instances": reused,
        "feature_count": len(sidecar_rows),
        "feature_counts_by_source_corpus": dict(sorted(corpus_features.items())),
        "unsafe_feature_count": len(unsafe),
        "role_preview_count": len(preview_rows),
        "role_previews_with_duplicate_names": len(duplicate_previews),
        "error_count": len(errors),
        "input_scan": scan_report,
    }


def write_outputs(
    output_root: Path,
    *,
    sidecar_rows: list[dict[str, Any]],
    preview_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    write_jsonl(output_root / "information_features.jsonl", sidecar_rows)
    write_jsonl(output_root / "role_name_previews.jsonl", preview_rows)
    write_jsonl(output_root / "errors.jsonl", errors)
    write_json(output_root / "report.json", report)


if __name__ == "__main__":
    main()
