"""Decision-option label permutation helpers for legal datapoints."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import random
from typing import Any

from teamgym.core.schema import validate_common_datapoint


PERMUTATION_SCHEMA_VERSION = "collabgym_fc_decision_option_permutation_v1"


def permute_common_datapoint_decisions(
    datapoint: dict[str, Any],
    *,
    seed_material: str,
    target_answer: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of ``datapoint`` with decision labels permuted.

    The option texts and option-keyed metadata move to new labels; the oracle
    answer and sufficient-set answer fields are remapped accordingly.
    """

    out = deepcopy(datapoint)
    decision_set = [str(item) for item in out.get("decision_set") or [] if str(item)]
    definitions = out.get("decision_option_definitions")
    if not decision_set or not isinstance(definitions, dict):
        return out, _empty_metadata(seed_material)

    old_answer = str((out.get("oracle") or {}).get("answer") or "")
    old_to_new = decision_label_permutation(
        decision_set,
        seed_material=seed_material,
        old_answer=old_answer,
        target_answer=target_answer,
    )
    new_to_old = {new: old for old, new in old_to_new.items()}

    out["decision_set"] = list(decision_set)
    out["decision_option_definitions"] = {
        new_label: str(definitions.get(new_to_old[new_label], ""))
        for new_label in decision_set
    }
    oracle = out.get("oracle")
    if isinstance(oracle, dict):
        _remap_answer_fields(oracle, old_to_new)
    dataset_specific = out.get("dataset_specific")
    if isinstance(dataset_specific, dict):
        _remap_option_keyed_dict(dataset_specific, "decision_option_metadata", decision_set, old_to_new)
    initial_common = out.get("initial_common_information")
    if isinstance(initial_common, dict):
        if isinstance(initial_common.get("decision_option_definitions"), dict):
            initial_common["decision_option_definitions"] = dict(out["decision_option_definitions"])
        _remap_option_keyed_dict(initial_common, "decision_option_metadata", decision_set, old_to_new)

    metadata = _metadata(
        seed_material=seed_material,
        old_decision_set=decision_set,
        old_answer=old_answer,
        old_to_new=old_to_new,
    )
    generation_metadata = out.setdefault("generation_metadata", {})
    if isinstance(generation_metadata, dict):
        generation_metadata["decision_option_permutation"] = metadata
    out["format_validation"] = validate_common_datapoint(out)
    return out, metadata


def decision_label_permutation(
    decision_set: list[str],
    *,
    seed_material: str,
    old_answer: str | None = None,
    target_answer: str | None = None,
) -> dict[str, str]:
    labels = list(dict.fromkeys(str(item) for item in decision_set if str(item)))
    rng = random.Random(_stable_seed(seed_material))
    shuffled = list(labels)
    rng.shuffle(shuffled)
    old_to_new = dict(zip(labels, shuffled, strict=True))
    if old_answer and target_answer and old_answer in labels and target_answer in labels:
        current_target_owner = next((old for old, new in old_to_new.items() if new == target_answer), None)
        if current_target_owner is not None and current_target_owner != old_answer:
            old_to_new[current_target_owner] = old_to_new[old_answer]
        old_to_new[old_answer] = target_answer
    return old_to_new


def apply_permutation_to_public_common(
    public_common: dict[str, Any],
    metadata: dict[str, Any],
    *,
    source_common_datapoint: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(public_common)
    decision_set = source_common_datapoint.get("decision_set")
    definitions = source_common_datapoint.get("decision_option_definitions")
    if isinstance(decision_set, list):
        out["decision_set"] = list(decision_set)
    if isinstance(definitions, dict):
        out["decision_option_definitions"] = dict(definitions)

    old_to_new = metadata.get("old_to_new") if isinstance(metadata.get("old_to_new"), dict) else {}
    if old_to_new:
        _remap_option_keyed_dict(out, "decision_option_metadata", list(source_common_datapoint.get("decision_set") or []), old_to_new)
    out["decision_option_permutation"] = {
        "schema_version": metadata.get("schema_version"),
        "old_to_new": metadata.get("old_to_new"),
        "new_to_old": metadata.get("new_to_old"),
    }
    return out


def _remap_answer_fields(value: dict[str, Any], old_to_new: dict[str, str]) -> None:
    answer = value.get("answer")
    if isinstance(answer, str) and answer in old_to_new:
        value["answer"] = old_to_new[answer]
    sufficient_sets = value.get("smallest_sufficient_rui_sets")
    if isinstance(sufficient_sets, list):
        for row in sufficient_sets:
            if isinstance(row, dict):
                row_answer = row.get("answer")
                if isinstance(row_answer, str) and row_answer in old_to_new:
                    row["answer"] = old_to_new[row_answer]


def _remap_option_keyed_dict(
    payload: dict[str, Any],
    key: str,
    decision_set: list[str],
    old_to_new: dict[str, str],
) -> None:
    keyed = payload.get(key)
    if not isinstance(keyed, dict):
        return
    new_to_old = {new: old for old, new in old_to_new.items()}
    payload[key] = {
        new_label: deepcopy(keyed.get(new_to_old[new_label]))
        for new_label in decision_set
        if new_label in new_to_old and new_to_old[new_label] in keyed
    }


def _metadata(
    *,
    seed_material: str,
    old_decision_set: list[str],
    old_answer: str,
    old_to_new: dict[str, str],
) -> dict[str, Any]:
    new_to_old = {new: old for old, new in old_to_new.items()}
    return {
        "schema_version": PERMUTATION_SCHEMA_VERSION,
        "seed_material": seed_material,
        "old_decision_set": list(old_decision_set),
        "new_decision_set": list(old_decision_set),
        "old_answer": old_answer,
        "new_answer": old_to_new.get(old_answer, old_answer),
        "old_to_new": dict(old_to_new),
        "new_to_old": new_to_old,
    }


def _empty_metadata(seed_material: str) -> dict[str, Any]:
    return {
        "schema_version": PERMUTATION_SCHEMA_VERSION,
        "seed_material": seed_material,
        "old_decision_set": [],
        "new_decision_set": [],
        "old_answer": None,
        "new_answer": None,
        "old_to_new": {},
        "new_to_old": {},
    }


def _stable_seed(seed_material: str) -> int:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
