"""Build common datapoints from long-context reasoning seed specs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from teamgym.core.information_allocation import (
    assign_reference_units,
    owners_by_reference_unit,
)
from teamgym.core.io import estimate_tokens, read_jsonl, write_json, write_jsonl
from teamgym.data.long_context_reasoning.schema import (
    build_long_context_common_datapoint,
    decision_set_for_seed,
    normalize_answer,
)

PARTICIPANT_NAMES = [
    f"participant_{chr(ord('a') + index)}" for index in range(26)
]


def build_from_seed_specs(
    *,
    seed_specs: list[dict[str, Any]],
    output_root: str | Path,
    allocation_mode: str,
    target_context_tokens: int,
    min_participants: int,
    seed: int,
) -> dict[str, Any]:
    """Build common datapoints and write artifacts."""

    rng = random.Random(seed)
    datapoints = []
    skipped = []
    for index, spec in enumerate(seed_specs):
        try:
            datapoints.append(
                build_one_datapoint(
                    spec=spec,
                    index=index,
                    allocation_mode=allocation_mode,
                    target_context_tokens=target_context_tokens,
                    min_participants=min_participants,
                    rng=rng,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch reports useful.
            skipped.append(
                {
                    "instance_id": spec.get("instance_id", f"row_{index}"),
                    "error": str(exc),
                }
            )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "common_datapoints.jsonl", datapoints)
    write_json(output / "common_datapoints.pretty.json", datapoints)

    report = {
        "count": len(datapoints),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "allocation_mode": allocation_mode,
        "target_context_tokens": target_context_tokens,
        "min_participants": min_participants,
        "answers": Counter(dp["oracle"]["answer"] for dp in datapoints),
        "source_families": Counter(dp["generation_metadata"].get("source_family", "unknown") for dp in datapoints),
        "output_root": str(output),
    }
    write_json(output / "build_report.json", _jsonable(report))
    return report


def build_one_datapoint(
    *,
    spec: dict[str, Any],
    index: int,
    allocation_mode: str,
    target_context_tokens: int,
    min_participants: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Build one schema-compatible common datapoint from a seed spec."""

    spec = dict(spec)
    spec.setdefault("instance_id", f"long_context_reasoning_{index:05d}")
    spec["answer"] = normalize_answer(spec["answer"], decision_set=decision_set_for_seed(spec))
    required_rui_ids = [str(rui_id) for rui_id in spec["required_rui_ids"]]
    distribution_rui_ids = [
        str(rui_id)
        for rui_id in spec.get("distribution_rui_ids", required_rui_ids)
    ]
    if len(required_rui_ids) < 2:
        raise ValueError("required_rui_ids must contain at least two RUIs")

    source_text = _resolve_source_text(spec)
    ruis = _normalize_rius(spec)
    _validate_required_rius(required_rui_ids, ruis)
    _validate_required_rius(distribution_rui_ids, ruis)
    if not set(required_rui_ids) <= set(distribution_rui_ids):
        raise ValueError("distribution_rui_ids must include every required RIU")

    participant_count = max(min_participants, len(required_rui_ids))
    if participant_count > len(PARTICIPANT_NAMES):
        raise ValueError(f"Need {participant_count} participants but only {len(PARTICIPANT_NAMES)} names are configured")

    allocation = assign_reference_units(
        distribution_rui_ids,
        mode=allocation_mode,
        min_participants=participant_count,
        rng=rng,
        smallest_sufficient_sets=[required_rui_ids],
    )
    assignment = _rename_assignment(allocation.private_assignment)
    common_rui_ids = list(allocation.common_rui_ids)
    effective_assignment = {
        participant_id: list(dict.fromkeys([*private_rui_ids, *common_rui_ids]))
        for participant_id, private_rui_ids in assignment.items()
    }
    owners_by_rui = owners_by_reference_unit(effective_assignment)
    participants, private_context_units = _build_participant_payloads(
        spec=spec,
        source_text=source_text,
        ruis=ruis,
        assignment=effective_assignment,
        target_context_tokens=target_context_tokens,
    )
    reference_information_units = _build_reference_information_units(
        spec=spec,
        ruis=ruis,
        owners_by_rui=owners_by_rui,
    )

    return build_long_context_common_datapoint(
        seed=spec,
        participants=participants,
        private_context_units=private_context_units,
        reference_information_units=reference_information_units,
        metadata={
            "instance_id": spec["instance_id"],
            "source_id": spec.get("source_id", spec["instance_id"]),
            "source_title": spec.get("source_title", ""),
            "allocation_mode": allocation.mode,
            "requested_allocation_mode": allocation_mode,
            "private_information_allocation": {
                **allocation.to_metadata(),
                "mapped_private_assignment": assignment,
            },
            "common_rui_ids": common_rui_ids,
            "target_context_tokens": target_context_tokens,
            "actual_private_context_tokens": {
                unit["assigned_to"][0]: estimate_tokens(unit["text"]) for unit in private_context_units
            },
            "case_fact_count": len(spec.get("case_facts", [])),
            "rui_count": len(ruis),
        },
    )


def _resolve_source_text(spec: dict[str, Any]) -> str:
    source_text = spec.get("source_text")
    if source_text:
        return str(source_text)
    source_path = spec.get("source_path")
    if source_path:
        return Path(source_path).read_text(encoding="utf-8")
    raise ValueError("Seed spec must include source_text or source_path")


def _normalize_rius(spec: dict[str, Any]) -> list[dict[str, Any]]:
    ruis = []
    for index, raw in enumerate(spec.get("rius", [])):
        rui = dict(raw)
        rui.setdefault("rui_id", f"rui_{index + 1}")
        rui["rui_id"] = str(rui["rui_id"])
        rui.setdefault("source_span", rui.get("text", ""))
        rui.setdefault("normalized_claim", rui.get("summary") or rui.get("source_span", ""))
        rui.setdefault("rui_type", "rule_or_fact")
        ruis.append(rui)

    for fact in spec.get("case_facts", []):
        fact_id = str(fact.get("fact_id") or f"case_fact_{len(ruis) + 1}")
        if any(rui["rui_id"] == fact_id for rui in ruis):
            continue
        fact_text = str(fact.get("text", ""))
        ruis.append(
            {
                "rui_id": fact_id,
                "source_span": fact_text,
                "normalized_claim": fact.get("normalized_claim", fact_text),
                "rui_type": "case_fact",
            }
        )

    if not ruis:
        raise ValueError("Seed spec must include rius or case_facts")
    return ruis


def _rename_assignment(assignment: dict[str, list[str]]) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for index, source_participant_id in enumerate(sorted(assignment)):
        if index >= len(PARTICIPANT_NAMES):
            raise ValueError(f"Too many participants to map: {sorted(assignment)}")
        mapped[PARTICIPANT_NAMES[index]] = list(assignment[source_participant_id])
    return mapped


def _validate_required_rius(required_rui_ids: list[str], ruis: list[dict[str, Any]]) -> None:
    available = {rui["rui_id"] for rui in ruis}
    missing = [rui_id for rui_id in required_rui_ids if rui_id not in available]
    if missing:
        raise ValueError(f"required_rui_ids missing from rius/case_facts: {missing}")


def _build_participant_payloads(
    *,
    spec: dict[str, Any],
    source_text: str,
    ruis: list[dict[str, Any]],
    assignment: dict[str, list[str]],
    target_context_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    participants = []
    private_context_units = []
    rui_by_id = {rui["rui_id"]: rui for rui in ruis}

    for participant_id in sorted(assignment):
        owned_rui_ids = [rui_id for rui_id in assignment[participant_id] if rui_id in rui_by_id]
        owned_rius = [rui_by_id[rui_id] for rui_id in owned_rui_ids]
        context_text = _participant_context_text(
            source_text=source_text,
            all_rius=ruis,
            owned_rius=owned_rius,
            question=spec["question"],
            target_context_tokens=target_context_tokens,
        )
        context_unit_id = f"{participant_id}_long_private_context"
        participants.append(
            {
                "participant_id": participant_id,
                "role_name": spec.get("participant_role_name", "Document analyst"),
                "private_context_unit_ids": [context_unit_id],
                "owned_rui_ids": owned_rui_ids,
                "metadata": {
                    "private_context_tokens": estimate_tokens(context_text),
                    "owned_rui_count": len(owned_rui_ids),
                },
            }
        )
        private_context_units.append(
            {
                "context_unit_id": context_unit_id,
                "assigned_to": [participant_id],
                "text": context_text,
                "hidden_contains_rui_ids": owned_rui_ids,
                "metadata": {
                    "context_kind": "long_private_document_excerpt",
                    "source_id": spec.get("source_id", spec["instance_id"]),
                },
            }
        )
    return participants, private_context_units


def _participant_context_text(
    *,
    source_text: str,
    all_rius: list[dict[str, Any]],
    owned_rius: list[dict[str, Any]],
    question: str,
    target_context_tokens: int,
) -> str:
    owned_ids = {rui["rui_id"] for rui in owned_rius}
    masked_source = source_text
    nonowned_rius = [rui for rui in all_rius if rui["rui_id"] not in owned_ids]
    for rui in sorted(nonowned_rius, key=lambda item: len(str(item.get("source_span", ""))), reverse=True):
        span = str(rui.get("source_span", ""))
        if span and span in masked_source:
            masked_source = masked_source.replace(span, "[passage assigned to another participant]")

    windows = []
    if owned_rius:
        per_rui_chars = max(1000, int(target_context_tokens * 3.2 / max(1, len(owned_rius))))
        for rui in owned_rius:
            span = str(rui.get("source_span", ""))
            atomic_text = str(rui.get("normalized_claim") or span).strip()
            idx = masked_source.find(span) if span else -1
            if idx < 0:
                windows.append(
                    "\n".join(
                        [
                            "[Document proposition]",
                            atomic_text,
                        ]
                    )
                )
                continue
            start = max(0, idx - per_rui_chars // 2)
            end = min(len(masked_source), idx + len(span) + per_rui_chars // 2)
            windows.append(
                "\n".join(
                    [
                        masked_source[start:end].strip(),
                        "",
                        "[Document proposition]",
                        atomic_text,
                    ]
                )
            )

    distractors = _dedupe_preserve_order(
        _distractor_windows(masked_source, target_context_tokens=target_context_tokens)
    )
    body = "\n\n[...]\n\n".join([window for window in [*windows, *distractors] if window])
    context = (
        f"Task question: {question}\n\n"
        "Private document excerpt. Share useful public information from this excerpt; "
        "you are not responsible for making the final decision.\n\n"
        f"{body}"
    ).strip()
    return _trim_to_token_budget(context, target_context_tokens)


def _distractor_windows(source_text: str, *, target_context_tokens: int) -> list[str]:
    if not source_text:
        return []
    window_chars = max(1200, int(target_context_tokens * 1.5))
    starts = [0, max(0, len(source_text) // 2 - window_chars // 2), max(0, len(source_text) - window_chars)]
    windows = []
    for start in starts:
        windows.append(source_text[start : start + window_chars].strip())
    return windows


def _trim_to_token_budget(text: str, target_tokens: int) -> str:
    if estimate_tokens(text) <= target_tokens:
        return text
    target_chars = max(1200, int(target_tokens * 4.1))
    trimmed = text[:target_chars].rstrip()
    while estimate_tokens(trimmed) > target_tokens and len(trimmed) > 1200:
        trimmed = trimmed[: int(len(trimmed) * 0.9)].rstrip()
    return trimmed + "\n\n[excerpt truncated]"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        key = re.sub(r"\s+", " ", item).strip()[:500]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_reference_information_units(
    *,
    spec: dict[str, Any],
    ruis: list[dict[str, Any]],
    owners_by_rui: dict[str, list[str]],
) -> list[dict[str, Any]]:
    output = []
    for rui in ruis:
        rui_id = rui["rui_id"]
        output.append(
            {
                "rui_id": rui_id,
                "text": rui.get("normalized_claim", rui.get("source_span", "")),
                "raw_text": rui.get("source_span", ""),
                "information_feature": rui.get("normalized_claim", rui.get("source_span", "")),
                "summary": rui.get("normalized_claim", rui.get("source_span", "")),
                "rui_type": rui.get("rui_type", "rule_or_fact"),
                "source_ref": rui.get("source_ref", spec.get("source_id", spec["instance_id"])),
                "owned_by": owners_by_rui.get(rui_id, []),
                "metadata": {
                    key: value
                    for key, value in rui.items()
                    if key
                    not in {
                        "rui_id",
                        "source_span",
                        "normalized_claim",
                        "rui_type",
                        "source_ref",
                    }
                },
            }
        )
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonable(inner) for inner in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-spec", required=True, help="JSONL file containing long-context seed specs.")
    parser.add_argument("--output-root", required=True, help="Directory for common_datapoints.jsonl and report files.")
    parser.add_argument("--allocation-mode", default="one_each", help="PIU allocation mode from core.information_allocation.")
    parser.add_argument("--target-context-tokens", type=int, default=12000)
    parser.add_argument("--min-participants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_specs = read_jsonl(args.seed_spec)
    report = build_from_seed_specs(
        seed_specs=seed_specs,
        output_root=args.output_root,
        allocation_mode=args.allocation_mode,
        target_context_tokens=args.target_context_tokens,
        min_participants=args.min_participants,
        seed=args.seed,
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
