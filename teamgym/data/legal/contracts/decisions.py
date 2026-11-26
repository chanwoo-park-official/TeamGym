"""Finite-choice decision helpers shared by CUAD and MAUD."""

from __future__ import annotations

from typing import Any

from .records import letter_id as _letter_id


DECISION_RELATIONS = {
    "supports",
    "rules_out",
    "weakens",
    "distinguishes",
    "required_for",
}


def validate_decision_set_balance(public_task: dict[str, Any], answer: str) -> list[str]:
    decision_set = [str(item) for item in public_task.get("decision_set", [])]
    if answer not in decision_set or len(decision_set) < 2:
        return []
    definitions = public_task.get("decision_option_definitions")
    definitions = definitions if isinstance(definitions, dict) else {}
    answer_text = _option_text(answer, definitions)
    distractor_texts = [_option_text(item, definitions) for item in decision_set if item != answer]
    errors: list[str] = []

    answer_value_components = _option_value_component_score(answer)
    distractor_value_components = [_option_value_component_score(item) for item in decision_set if item != answer]
    if answer_value_components >= 2:
        required_comparable_values = 2 if len(decision_set) >= 4 else 1
        comparable_values = sum(1 for score in distractor_value_components if score >= answer_value_components)
        if comparable_values < required_comparable_values:
            errors.append("decision_set is too obvious: correct option label is the only comparably composite label.")

    answer_components = _option_component_score(answer_text)
    distractor_components = [_option_component_score(text) for text in distractor_texts]
    if answer_components >= 2 and max(distractor_components, default=0) < answer_components:
        errors.append("decision_set is too obvious: correct option is the only comparably composite option.")

    answer_word_count = len(answer_text.split())
    distractor_word_counts = [len(text.split()) for text in distractor_texts]
    if distractor_word_counts and answer_word_count > max(distractor_word_counts) * 1.6 + 4:
        errors.append("decision_set is too obvious: correct option is much more detailed than every distractor.")

    comparable = sum(1 for count in distractor_word_counts if count >= max(1, int(answer_word_count * 0.65)))
    required_comparable = 2 if len(decision_set) >= 4 else 1
    if comparable < required_comparable:
        errors.append("decision_set needs more plausible distractors with similar specificity to the correct option.")

    answer_value = answer.lower()
    if _has_all_or_both_marker(answer_value):
        distractor_markers = sum(1 for item in decision_set if item != answer and _has_all_or_both_marker(item.lower()))
        distractor_composites = sum(1 for score in distractor_components if score >= 2)
        if distractor_markers == 0 and distractor_composites == 0:
            errors.append("decision_set is too obvious: answer uses both/all while no distractor has similar structure.")
    return errors


def validate_option_value_names(decision_values: set[str]) -> list[str]:
    errors: list[str] = []
    banned_tokens = {
        "plus",
        "and",
        "with",
        "both",
        "all",
        "only",
        "mutual",
        "reciprocal",
    }
    for value in sorted(decision_values):
        lowered = value.lower().replace("-", "_")
        tokens = {token for token in lowered.split("_") if token}
        banned = sorted(tokens & banned_tokens)
        if banned:
            errors.append(
                f"{value}: decision_set.value should not expose component structure via tokens {banned}."
            )
    return errors


def _option_text(value: str, definitions: dict[str, Any]) -> str:
    return " ".join(str(value).replace("_", " ").split()) + " " + " ".join(str(definitions.get(value) or "").split())


def _option_component_score(text: str) -> int:
    lowered = text.lower().replace("_", " ")
    score = 1
    markers = [
        " and ",
        " plus ",
        " with ",
        " both ",
        " all ",
        "combined",
        "including",
        "together",
        " as well as ",
    ]
    score += sum(1 for marker in markers if marker in f" {lowered} ")
    score += lowered.count(",")
    return score


def _option_value_component_score(value: str) -> int:
    lowered = value.lower().replace("-", "_")
    score = 1
    composite_markers = [
        "_and_",
        "_with_",
        "_plus_",
        "_combined_",
        "_together_",
        "both_",
        "_both_",
        "all_",
        "_all_",
        "mutual",
        "reciprocal",
        "bilateral",
        "two_sided",
        "dual",
    ]
    score += sum(1 for marker in composite_markers if marker in lowered)
    return score


def _has_all_or_both_marker(value: str) -> bool:
    tokens = set(value.replace("-", "_").split("_"))
    return (
        "both" in tokens
        or "all" in tokens
        or "mutual" in tokens
        or "reciprocal" in tokens
        or "bilateral" in tokens
        or value.startswith("all_")
    )


def normalize_smallest_sufficient_sets(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        unit_ids = [str(item) for item in row.get("unit_ids", []) if str(item)]
        if not unit_ids:
            continue
        copy = dict(row)
        copy["unit_ids"] = unit_ids
        out.append(copy)
    return out


def normalize_decision_set(plan: dict[str, Any]) -> dict[str, Any]:
    public_task = plan.get("public_task")
    if not isinstance(public_task, dict):
        return plan
    decision_set = public_task.get("decision_set")
    definitions: dict[str, str] = {}
    if isinstance(decision_set, list) and decision_set and all(isinstance(item, dict) for item in decision_set):
        values: list[str] = []
        for item in decision_set:
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            values.append(value)
            definitions[value] = str(item.get("definition") or "")
    elif isinstance(decision_set, list):
        values = [str(item).strip() for item in decision_set if str(item).strip()]
        raw_definitions = public_task.get("decision_option_definitions")
        if isinstance(raw_definitions, dict):
            definitions = {str(key): str(value) for key, value in raw_definitions.items()}
    else:
        return plan

    value_map = {
        value: _letter_id("decision", idx)
        for idx, value in enumerate(values, start=1)
    }
    public_task["decision_set"] = [value_map[value] for value in values]
    public_task["decision_option_definitions"] = {
        value_map[value]: definitions.get(value, "")
        for value in values
    }
    public_task["source_decision_value_map"] = value_map

    oracle = plan.get("oracle")
    if isinstance(oracle, dict):
        answer = str(oracle.get("answer") or "")
        if answer in value_map:
            oracle["answer"] = value_map[answer]

    units = plan.get("evidence_units") if isinstance(plan.get("evidence_units"), list) else []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        relevance = unit.get("decision_relevance")
        if not isinstance(relevance, list):
            continue
        for row in relevance:
            if isinstance(row, dict):
                decision = str(row.get("decision") or "")
                if decision in value_map:
                    row["decision"] = value_map[decision]
    return plan


def validate_decision_relevance(unit_id: str, relevance: Any, decision_values: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(relevance, list) or not relevance:
        return [f"{unit_id}: decision_relevance is required."]
    for idx, row in enumerate(relevance, start=1):
        if not isinstance(row, dict):
            errors.append(f"{unit_id}: decision_relevance[{idx}] must be an object.")
            continue
        if str(row.get("decision") or "") not in decision_values:
            errors.append(f"{unit_id}: decision_relevance[{idx}].decision must be in decision_set.")
        if str(row.get("relation") or "") not in DECISION_RELATIONS:
            errors.append(f"{unit_id}: decision_relevance[{idx}].relation is invalid.")
        if not str(row.get("reason") or "").strip():
            errors.append(f"{unit_id}: decision_relevance[{idx}].reason is required.")
    return errors
