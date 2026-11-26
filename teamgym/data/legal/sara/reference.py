"""Reference execution and source-grounding helpers for SARA construction."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import re
import subprocess
from typing import Any


def execute_reference(
    row: dict[str, Any],
    *,
    work_dir: Path,
    swipl_image: Path,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    """Execute an audited reference while recording successful user predicates."""

    work_dir.mkdir(parents=True, exist_ok=True)
    program_path = work_dir / "traced_reference.pl"
    program_path.write_text(_trace_prefix() + str(row.get("reference_prolog") or ""), encoding="utf-8")
    completed = subprocess.run(
        [
            "apptainer",
            "exec",
            "--bind",
            f"{work_dir.resolve()}:{work_dir.resolve()}",
            str(swipl_image.resolve()),
            "swipl",
            "-q",
            "-f",
            str(program_path.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    (work_dir / "reference.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (work_dir / "reference.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    reported = extract_reported_total(completed.stdout)
    gold = decimal_amount(row.get("label"))
    rounded = round_dollars(reported) if reported is not None else None
    trace = _filtered_trace(completed.stderr)
    issues: list[str] = []
    if completed.returncode != 0:
        issues.append(f"Reference program exited with status {completed.returncode}")
    if reported is None:
        issues.append("Reference program did not print a Tax result")
    if gold is None or rounded != gold:
        issues.append(f"Rounded reference result {rounded} does not equal gold {gold}")
    if not trace:
        issues.append("Reference program produced no successful proof trace")
    return {
        "accepted": not issues,
        "issues": issues,
        "returncode": completed.returncode,
        "reported_total": str(reported) if reported is not None else None,
        "rounded_total": str(rounded) if rounded is not None else None,
        "gold_total": str(gold) if gold is not None else None,
        "trace": trace,
        "trace_goal_count": len(trace),
    }


def source_payload(row: dict[str, Any]) -> dict[str, Any]:
    case_id = str(row.get("id") or "unknown_case")
    text = str(row.get("text") or "").strip()
    question = str(row.get("question") or "").strip()
    statutes = str(row.get("statutes") or "").strip()
    return {
        "case_id": case_id,
        "case_facts": text,
        "original_question": question,
        "statutes": statutes,
        "natural_source": f"Case facts:\n{text}\n\nOriginal question:\n{question}\n\nSupplied statutes:\n{statutes}",
        "gold_answer": str(row.get("label")),
    }


def extract_reported_total(stdout: str) -> Decimal | None:
    matches = re.findall(r"Tax result:\s*\$?\s*(-?[0-9]+(?:\.[0-9]+)?)", stdout, flags=re.I)
    return decimal_amount(matches[-1]) if matches else None


def decimal_amount(value: Any) -> Decimal | None:
    text = re.sub(r"[^0-9.\-]", "", str(value or ""))
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def round_dollars(value: Decimal) -> Decimal:
    return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _filtered_trace(stderr: str, *, max_goals: int = 180) -> list[str]:
    goals: list[str] = []
    seen: set[str] = set()
    predicate_counts: dict[str, int] = {}
    for line in stderr.splitlines():
        if not line.startswith("PROOF_EXIT\t"):
            continue
        goal = line.split("\t", 1)[1].strip()
        if not goal or goal in seen:
            continue
        predicate_match = re.match(r"(?:[a-z][a-z0-9_]*:)?([a-z][a-z0-9_]*)\(", goal)
        predicate = predicate_match.group(1) if predicate_match else "unknown"
        if predicate in {
            "findall",
            "length",
            "sum_list",
            "sum_amounts_",
            "year_of_date",
            "ceil_div",
            "min_amount",
            "max_amount",
        }:
            continue
        if predicate_counts.get(predicate, 0) >= 12:
            continue
        seen.add(goal)
        predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        goals.append(goal)
    if len(goals) <= max_goals:
        return goals
    # Preserve both early premises and late aggregate conclusions.
    head_count = max_goals // 2
    return goals[:head_count] + goals[-(max_goals - head_count) :]


def _trace_prefix() -> str:
    return r'''
:- multifile user:prolog_trace_interception/4.
user:prolog_trace_interception(Port, Frame, _Choice, continue) :-
    ( Port == exit,
      catch(prolog_frame_attribute(Frame, clause, Ref), _, fail),
      catch(clause_property(Ref, file(_)), _, fail),
      catch(prolog_frame_attribute(Frame, goal, Goal), _, fail)
    -> format(user_error, 'PROOF_EXIT\t', []),
       write_term(user_error, Goal, [quoted(true), max_depth(20)]),
       nl(user_error)
    ; true
    ).
:- trace.
'''


def snap_span(span: str, source_text: str) -> str | None:
    if not span:
        return None
    if span in source_text:
        return span
    candidate_words = span.replace("…", " ... ").split()
    anchors: list[tuple[int, int, int]] = []
    for width in range(min(10, len(candidate_words)), 3, -1):
        for offset in range(0, len(candidate_words) - width + 1):
            phrase = " ".join(candidate_words[offset : offset + width]).strip(".,;:()[]\"")
            if "..." in phrase or not phrase:
                continue
            position = source_text.find(phrase)
            if position >= 0:
                anchors.append((offset, position, position + len(phrase)))
        if len(anchors) >= 2:
            break
    if not anchors:
        return None
    anchors.sort()
    start = anchors[0][1]
    end = max(anchor[2] for anchor in anchors if anchor[1] >= start)
    snapped = source_text[start:end].strip()
    return snapped if 1 <= len(snapped.split()) <= 180 else None
