"""Reusable short-answer grading rules, prompts, and judgment normalization."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

from teamgym.llm.openai_client import OpenAIJsonClient


# This also versions cache keys for normalized written judgments.
JUDGE_SCHEMA = "long_context_short_answer_judgment_v2"


SYSTEM_PROMPT = """You are a strict evaluator for a distributed reasoning benchmark.
Judge whether the candidate written answer reaches the oracle conclusion and covers the required decision steps.
Do not require exact wording. Do require the substantive rule/fact checks, calculations, exceptions, and final action.
Do not reward a bare label or vague answer that does not explain the decision path.
If the judge rubric says require_all_steps_for_pass=true, set passed=true only when the candidate has the correct
conclusion, covers every required decision step, and has no disqualifying error.
When deterministic_numeric_evaluation is present, use its passed value for final-amount correctness. A candidate
within the recorded tolerance must not fail merely because it states cents while the oracle stores whole dollars.
Use the provided reference information units as the evidence record. Do not mark a candidate claim unsupported if
it is supported by the task, canonical response, required steps, or reference information units, even when the
canonical response is less specific. Mark claims unsupported only when they are absent from the provided evidence
or contradict it.
Return exactly one JSON object and no extra text."""


MAX_JUDGE_RUI_CHARS = 8000


NUMERIC_ORACLE_TOLERANCE = Decimal("1")


_CURRENCY_AMOUNT_RE = re.compile(
    r"\$\s*(?P<parenthesized>\()?\s*(?P<sign>[-+]?)\s*"
    r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\)?"
)


_NUMERIC_ORACLE_RE = re.compile(
    r"^\s*\$?\s*(?P<parenthesized>\()?\s*(?P<sign>[-+]?)\s*"
    r"(?P<amount>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\)?\s*$"
)


_FINAL_AMOUNT_PREFIX_RE = re.compile(
    r"(?ix)\b(?:"
    r"owes?|"
    r"final\s+(?:federal\s+)?(?:tax|liability)|"
    r"total\s+(?:federal\s+)?(?:tax|liability)|"
    r"(?:tax|liability)\s+(?:is|equals|of|due|owed|payable)"
    r")\b[^$]{0,100}"
    + _CURRENCY_AMOUNT_RE.pattern
)


_FINAL_AMOUNT_CUE_RE = re.compile(
    r"(?ix)\b(?:"
    r"owes?|owed|final|"
    r"total\s+(?:federal\s+)?(?:tax|liabilit(?:y|ies))|"
    r"liabilit(?:y|ies)|tax\s+(?:due|payable)|"
    r"taxable\s+income|standard\s+deduction|"
    r"personal[-\s]+exemption(?:\s+deduction)?|"
    r"employer\s+excise\s+tax|individual\s+income\s+tax"
    r")\b"
)


def judge_one_answer(
    *,
    client: OpenAIJsonClient,
    datapoint: dict[str, Any],
    answer_row: dict[str, Any],
    trace_dir: str | Path | None,
    index: int,
) -> dict[str, Any]:
    pass_policy = pass_policy_for_datapoint(datapoint)
    require_all_steps = pass_policy["require_all_steps_for_pass"]
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    task_description = datapoint.get("task_description") if isinstance(datapoint.get("task_description"), dict) else {}
    dataset_specific = datapoint.get("dataset_specific") if isinstance(datapoint.get("dataset_specific"), dict) else {}
    required_steps = oracle.get("required_decision_steps") or oracle.get("reasoning_steps") or dataset_specific.get("reasoning_steps") or []
    candidate_answer = answer_row.get("answer_text") or answer_row.get("written_answer") or answer_row.get("rationale") or ""
    numeric_evaluation = deterministic_numeric_evaluation(datapoint, str(candidate_answer))
    payload = {
        "instance_id": datapoint.get("instance_id"),
        "question": (
            task_description.get("question")
            or task_description.get("objective")
        ),
        "answer_type": task_description.get("answer_type"),
        "canonical_response": (
            oracle.get("canonical_response")
            or oracle.get("short_answer")
            or oracle.get("answer_text")
            or dataset_specific.get("canonical_response")
        ),
        "required_decision_steps": required_steps,
        "reference_information_units": judge_reference_information_units(datapoint),
        "judge_rubric": oracle.get("judge_rubric") or dataset_specific.get("judge_rubric") or {},
        "pass_policy": pass_policy,
        "disqualifying_errors": oracle.get("disqualifying_errors") or dataset_specific.get("disqualifying_errors") or [],
        "candidate_answer": candidate_answer,
        "deterministic_numeric_evaluation": numeric_evaluation,
        "candidate_metadata": {
            key: value
            for key, value in answer_row.items()
            if key not in {"answer_text", "written_answer", "rationale"}
        },
        "response_format": {
            "passed": "boolean",
            "score": "number from 0 to 1",
            "conclusion_correct": "boolean",
            "covered_step_count": "integer",
            "total_step_count": "integer",
            "step_scores": [
                {
                    "step": "step id or number",
                    "covered": "boolean",
                    "score": "number from 0 to 1",
                    "reason": "brief explanation",
                }
            ],
            "missing_steps": ["required step claims that are missing"],
            "incorrect_claims": ["substantive wrong claims in the candidate answer"],
            "disqualifying_errors_found": ["matched disqualifying errors, if any"],
            "reason": "brief overall explanation",
        },
    }
    prompt_id = f"{datapoint.get('instance_id')}.short_answer_judge.{index:04d}"
    response = client.generate_json(
        prompt_id=prompt_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, indent=2, ensure_ascii=False),
        trace_dir=trace_dir,
    )
    normalized = normalize_judgment(
        response,
        required_steps=required_steps,
        require_all_steps_for_pass=require_all_steps,
        numeric_evaluation=numeric_evaluation,
        numeric_final_step_ids=numeric_final_step_ids(required_steps),
    )
    if require_all_steps:
        # An inconsistent judge response is a grading error, not evidence that
        # the submitted answer was wrong. Batch callers record these as errors.
        fatal = [error for error in normalized['judgment_consistency_errors']
                 if error != 'aggregate_total_disagrees_with_required_steps'
                 or normalized['covered_step_count'] == normalized['total_step_count']]
        if fatal:
            raise ValueError('Inconsistent strict judgment: ' + ', '.join(fatal))
    return {
        "schema": JUDGE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_id": datapoint.get("instance_id"),
        "dataset": datapoint.get("dataset"),
        "answer_source": answer_row.get("answer_source") or answer_row.get("source") or "candidate",
        "communication_structure": answer_row.get("communication_structure"),
        "candidate_answer": payload["candidate_answer"],
        "canonical_response": payload["canonical_response"],
        "judgment": normalized,
        "request_meta": dict(client.last_request_meta),
        "answer_row": answer_row,
    }


def judge_reference_information_units(datapoint: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact RIU evidence for short-answer judging."""

    required_ids = required_rui_ids_for_judge(datapoint)
    units = datapoint.get("reference_information_units")
    if not isinstance(units, list):
        units = []

    payload_units: list[dict[str, Any]] = []
    for index, raw in enumerate(units, start=1):
        if not isinstance(raw, dict):
            continue
        rui_id = str(raw.get("rui_id") or raw.get("unit_id") or f"rui_{index:02d}")
        text = str(raw.get("text") or raw.get("source_span") or "")
        if len(text) > MAX_JUDGE_RUI_CHARS:
            text = text[:MAX_JUDGE_RUI_CHARS].rstrip() + "\n[truncated]"
        source_ref = raw.get("source_ref") if isinstance(raw.get("source_ref"), (dict, str)) else None
        payload_units.append(
            {
                "rui_id": rui_id,
                "text": text,
                "summary": raw.get("summary") or raw.get("normalized_claim") or raw.get("information_feature") or "",
                "source_ref": source_ref,
                "required_for_answer": rui_id in required_ids if required_ids else None,
            }
        )
    return payload_units


def required_rui_ids_for_judge(datapoint: dict[str, Any]) -> set[str]:
    """Infer which RIUs are intended as answer-supporting evidence."""

    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    dataset_specific = datapoint.get("dataset_specific") if isinstance(datapoint.get("dataset_specific"), dict) else {}
    ids: set[str] = set()

    for source in (oracle, dataset_specific):
        raw_ids = source.get("required_rui_ids")
        if isinstance(raw_ids, list):
            ids.update(str(item) for item in raw_ids if str(item))
        sets = source.get("smallest_sufficient_rui_sets")
        if isinstance(sets, list):
            for row in sets:
                if not isinstance(row, dict):
                    continue
                unit_ids = row.get("rui_ids") or row.get("unit_ids")
                if isinstance(unit_ids, list):
                    ids.update(str(item) for item in unit_ids if str(item))

    for step in oracle.get("required_decision_steps") or oracle.get("reasoning_steps") or dataset_specific.get("reasoning_steps") or []:
        if not isinstance(step, dict):
            continue
        uses = step.get("uses") or step.get("required_rui_ids") or step.get("rui_ids")
        if isinstance(uses, list):
            ids.update(str(item) for item in uses if str(item))
    return ids


def pass_policy_for_datapoint(datapoint: dict[str, Any]) -> dict[str, Any]:
    require_all = requires_all_steps_for_pass(datapoint)
    return {
        "require_all_steps_for_pass": require_all,
        "passed_definition": (
            "passed is true only when conclusion_correct is true, every required_decision_step is covered, "
            "and disqualifying_errors_found is empty"
            if require_all
            else "passed is the judge's holistic correctness decision under the rubric"
        ),
    }


def requires_all_steps_for_pass(datapoint: dict[str, Any]) -> bool:
    require_all = False
    for field in ("oracle", "dataset_specific"):
        source = datapoint.get(field)
        if not isinstance(source, dict):
            continue
        rubric = source.get("judge_rubric")
        if rubric is not None and not isinstance(rubric, dict):
            raise ValueError(f"{field}.judge_rubric must be a JSON object")
        if rubric is not None and "require_all_steps_for_pass" in rubric:
            flag = rubric["require_all_steps_for_pass"]
            if not isinstance(flag, bool):
                raise ValueError(f"{field}.judge_rubric.require_all_steps_for_pass must be a JSON boolean")
            require_all = require_all or flag
        require_all = require_all or source.get("pass_policy") == "all_required_decision_steps"
    return require_all


def deterministic_numeric_evaluation(datapoint: dict[str, Any], candidate_answer: str) -> dict[str, Any] | None:
    """Compare a numeric short-answer oracle with the candidate's final currency amount."""

    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    oracle_amount = _parse_single_numeric_oracle(oracle.get("short_answer"))
    if oracle_amount is None:
        return None

    candidate_amount, extraction_strategy = _extract_final_currency_amount(candidate_answer)
    if candidate_amount is None:
        return {
            "applicable": True,
            "oracle_amount": _decimal_text(oracle_amount),
            "candidate_amount": None,
            "candidate_amount_strategy": extraction_strategy,
            "absolute_error": None,
            "tolerance_dollars": _decimal_text(NUMERIC_ORACLE_TOLERANCE),
            "passed": False,
            "comparison": "absolute_error < tolerance_dollars",
        }

    absolute_error = abs(candidate_amount - oracle_amount)
    return {
        "applicable": True,
        "oracle_amount": _decimal_text(oracle_amount),
        "candidate_amount": _decimal_text(candidate_amount),
        "candidate_amount_strategy": extraction_strategy,
        "absolute_error": _decimal_text(absolute_error),
        "tolerance_dollars": _decimal_text(NUMERIC_ORACLE_TOLERANCE),
        "passed": absolute_error < NUMERIC_ORACLE_TOLERANCE,
        "comparison": "absolute_error < tolerance_dollars",
    }


def _extract_final_currency_amount(candidate_answer: str) -> tuple[Decimal | None, str]:
    explicit = [
        amount
        for match in _FINAL_AMOUNT_PREFIX_RE.finditer(candidate_answer)
        if (amount := _decimal_from_match(match)) is not None
    ]
    if explicit:
        return explicit[0], "explicit_final_amount_cue"

    for sentence in re.split(r"(?<=[.!?])\s+", candidate_answer):
        if not _FINAL_AMOUNT_CUE_RE.search(sentence):
            continue
        amounts = [
            amount
            for match in _CURRENCY_AMOUNT_RE.finditer(sentence)
            if (amount := _decimal_from_match(match)) is not None
        ]
        if amounts:
            return amounts[-1], "final_amount_cue_sentence"

    amounts = [
        amount
        for match in _CURRENCY_AMOUNT_RE.finditer(candidate_answer)
        if (amount := _decimal_from_match(match)) is not None
    ]
    return (amounts[-1], "last_currency_amount") if amounts else (None, "no_currency_amount")


def numeric_final_step_ids(required_steps: list[Any]) -> set[str]:
    ids: set[str] = set()
    for index, step in enumerate(required_steps, start=1):
        if not isinstance(step, dict):
            continue
        text = f"{step.get('claim') or ''} {step.get('judge_focus') or ''}".lower()
        if "final" in text and ("amount" in text or "liability" in text or "owed" in text):
            ids.add(str(step.get("step", index)))
    return ids


def normalize_judgment(
    response: dict[str, Any],
    *,
    required_steps: list[Any],
    require_all_steps_for_pass: bool = False,
    numeric_evaluation: dict[str, Any] | None = None,
    numeric_final_step_ids: set[str] | None = None,
) -> dict[str, Any]:
    score = normalize_judge_score(response.get("score"))
    step_scores = response.get("step_scores") if isinstance(response.get("step_scores"), list) else []
    normalized_steps = []
    consistency_errors: list[str] = []
    def step_key(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and math.isfinite(value):
            return str(int(value)) if value.is_integer() else str(value)
        return None

    expected_ids, canonical_values, aliases = [], {}, {}
    for index, step in enumerate(required_steps, start=1):
        value = step.get('step', index) if isinstance(step, dict) else index
        canonical = step_key(value)
        if canonical is None:
            consistency_errors.append('invalid_required_step_id')
            canonical, value = str(index), index
        expected_ids.append(canonical)
        canonical_values[canonical] = value
        for alias in ([value, step.get('step_id'), step.get('claim_id'), step.get('id')]
                      if isinstance(step, dict) else [value]):
            key = step_key(alias)
            if key is not None:
                aliases.setdefault(key, set()).add(canonical)
    # A positional number is a fallback only; it cannot shadow an explicitly
    # displayed stable identifier belonging to another required step.
    for index, canonical in enumerate(expected_ids, start=1):
        aliases.setdefault(str(index), {canonical})
    if len(set(expected_ids)) != len(expected_ids):
        consistency_errors.append("duplicate_required_step_id")
    if require_all_steps_for_pass and not expected_ids:
        consistency_errors.append('strict_policy_missing_required_steps')
    seen_ids: set[str] = set()
    for index, row in enumerate(step_scores, start=1):
        if not isinstance(row, dict):
            consistency_errors.append("invalid_step_score_row")
            continue
        reported_id = step_key(row.get('step', index))
        matches = aliases.get(reported_id, set())
        if len(matches) > 1:
            consistency_errors.append('ambiguous_judged_step_id')
        step_id = next(iter(matches)) if len(matches) == 1 else reported_id
        if step_id in seen_ids:
            consistency_errors.append("duplicate_judged_step_id")
        seen_ids.add(step_id)
        if expected_ids and step_id not in expected_ids:
            consistency_errors.append("unknown_judged_step_id")
        if type(row.get("covered")) is not bool:
            consistency_errors.append("non_boolean_step_coverage")
        normalized_steps.append(
            {
                "step": canonical_values.get(step_id, row.get('step', index)),
                "covered": row.get("covered") is True,
                "score": normalize_judge_score(row.get("score")),
                "reason": str(row.get("reason") or ""),
            }
        )
    original_covered = len({str(row['step']) for row in normalized_steps if row['covered']
                            and (not expected_ids or str(row['step']) in expected_ids)})
    # The model's aggregate cannot override contradictory per-step evidence.
    for key in ('covered_step_count', 'total_step_count'):
        if key in response and (type(response[key]) is not int or response[key] < 0):
            consistency_errors.append(f'invalid_{key}')
    if type(response.get('covered_step_count')) is int and response['covered_step_count'] != original_covered:
        consistency_errors.append('aggregate_coverage_disagrees_with_steps')
    if expected_ids and type(response.get('total_step_count')) is int \
            and response['total_step_count'] != len(expected_ids):
        consistency_errors.append('aggregate_total_disagrees_with_required_steps')
    for key in ('passed', 'conclusion_correct'):
        if key in response and type(response[key]) is not bool:
            consistency_errors.append(f'non_boolean_{key}')
    numeric_override = bool(numeric_evaluation and numeric_evaluation.get("passed"))
    final_step_ids = numeric_final_step_ids or set()
    if numeric_override and final_step_ids:
        for row in normalized_steps:
            if str(row.get("step")) in final_step_ids and not row["covered"]:
                row["covered"] = True
                row["score"] = max(float(row["score"]), 1.0)
                row["reason"] = "Final amount passes the deterministic one-dollar tolerance."

    response_total = response.get('total_step_count') if type(response.get('total_step_count')) is int else 0
    total = len(expected_ids) if expected_ids else max(response_total, len(normalized_steps))
    if total <= 0:
        total = response_total
    covered = len({str(row['step']) for row in normalized_steps if row['covered']
                   and (not expected_ids or str(row['step']) in expected_ids)})
    covered = min(total, covered)
    holistic_passed = response.get('passed') is True
    conclusion_correct = response.get('conclusion_correct') is True
    incorrect_claims = _string_list(response.get("incorrect_claims"))
    disqualifying_errors = _string_list(response.get("disqualifying_errors_found"))
    missing_steps = _string_list(response.get("missing_steps"))
    if numeric_override:
        conclusion_correct = True
        incorrect_claims = [item for item in incorrect_claims if not _is_numeric_amount_only_issue(item)]
        disqualifying_errors = [item for item in disqualifying_errors if not _is_numeric_amount_only_issue(item)]
        missing_steps = [item for item in missing_steps if not _is_numeric_final_step(item)]
        covered_rate = covered / total if total > 0 else 0.0
        if not holistic_passed and covered_rate >= 0.5:
            holistic_passed = not incorrect_claims and not disqualifying_errors and not missing_steps
    strict_passed = (holistic_passed and conclusion_correct and bool(expected_ids)
                     and total > 0 and covered == total
                     and not disqualifying_errors and not incorrect_claims and not missing_steps
                     and not consistency_errors)
    passed = strict_passed if require_all_steps_for_pass else holistic_passed
    return {
        "passed": passed,
        "holistic_passed": holistic_passed,
        "strict_all_step_passed": strict_passed,
        "pass_policy": "all_required_decision_steps" if require_all_steps_for_pass else "holistic",
        "score": score,
        "conclusion_correct": conclusion_correct,
        "numeric_evaluation": numeric_evaluation,
        "numeric_override_applied": numeric_override,
        "covered_step_count": covered,
        "total_step_count": total,
        "step_scores": normalized_steps,
        "missing_steps": missing_steps,
        "incorrect_claims": incorrect_claims,
        "disqualifying_errors_found": disqualifying_errors,
        "reason": str(response.get("reason") or ""),
        "raw_response": response,
        "judgment_consistency_errors": sorted(set(consistency_errors)),
    }


def strict_judgment_passed(judgment: dict[str, Any]) -> bool:
    """Read strict success from a normalized judgment, including older records."""
    strict = judgment.get("strict_all_step_passed")
    if isinstance(strict, bool):
        return strict
    # Early v2 records included consistency errors but did not store strict
    # success separately. For strict rubrics, their overall grade is authoritative.
    if "judgment_consistency_errors" in judgment:
        if judgment.get("pass_policy") == "all_required_decision_steps":
            return judgment.get("passed") is True
        if (
            judgment.get("holistic_passed") is not True
            or judgment.get("judgment_consistency_errors")
            or judgment.get("missing_steps")
            or judgment.get("incorrect_claims")
        ):
            return False
    total = int(judgment.get("total_step_count") or 0)
    covered = int(judgment.get("covered_step_count") or 0)
    return (
        bool(judgment.get("conclusion_correct"))
        and total > 0
        and covered == total
        and not bool(judgment.get("disqualifying_errors_found"))
    )


def _parse_single_numeric_oracle(value: Any) -> Decimal | None:
    if not isinstance(value, (str, int, float, Decimal)):
        return None
    match = _NUMERIC_ORACLE_RE.fullmatch(str(value))
    return _decimal_from_match(match) if match else None


def _decimal_from_match(match: re.Match[str]) -> Decimal | None:
    try:
        amount = Decimal(match.group("amount").replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None
    if match.group("sign") == "-" or match.group("parenthesized"):
        amount = -amount
    return amount


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _is_numeric_amount_only_issue(value: str) -> bool:
    text = value.lower()
    if any(
        phrase in text
        for phrase in (
            "requested result is incorrect",
            "requested result is not correct",
            "requested result does not match",
        )
    ):
        return True
    amount_terms = ("amount", "dollar", "round", "rounding", "numeric", "liability")
    substantive_terms = (
        "deduction",
        "dependent",
        "employment",
        "exemption",
        "filing",
        "income",
        "rate",
        "status",
        "wage",
        "wrong taxpayer",
        "wrong tax year",
    )
    return any(term in text for term in amount_terms) and not any(term in text for term in substantive_terms)


def _is_numeric_final_step(value: str) -> bool:
    text = value.lower()
    return "final" in text and ("amount" in text or "liability" in text or "owed" in text)


def normalize_judge_score(value: Any) -> float:
    """Bound a finite score to the supported reward range."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        raise ValueError("Judge score must be finite")
    return max(0.0, min(1.0, score))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []
