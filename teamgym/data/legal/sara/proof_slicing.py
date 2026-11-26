"""Grounded proof slicing and premise requirements for SARA construction."""

from __future__ import annotations

from typing import Any

from .prolog import (
    goal_signature,
    instantiate_prolog_term,
    parse_goal,
    parse_prolog_clauses,
    parse_prolog_term,
    unify_prolog_terms,
)
from .reference import normalize_text


DETERMINISTIC_INTERMEDIATE_PREDICATES = {
    "age",
    "append",
    "date_year",
    "dependent_list",
    "length",
    "return_taxpayer",
    "sum_list",
    "sumlist",
    "total_cash_wages_agri",
    "within_two_years",
    "year_of",
}


def select_target_proof_slice(
    execution: dict[str, Any],
    options: list[dict[str, Any]],
    reference_program: str,
) -> dict[str, Any] | None:
    """Select a high-priority target whose proof fits the atomic RIU budget."""

    candidates: list[dict[str, Any]] = []
    for option in options:
        proof_slice = target_proof_slice(execution, option, reference_program)
        predicates = {
            parsed[0]
            for goal in proof_slice["successful_trace_goals"]
            if (parsed := parse_goal(goal)) is not None
        }
        if (
            4 <= len(proof_slice["successful_trace_goals"]) <= 18
            and len(predicates) >= 3
        ):
            candidates.append(proof_slice)
    if not candidates:
        return None
    within_budget = [
        row for row in candidates if 6 <= len(row["successful_trace_goals"]) <= 12
    ]
    if within_budget:
        return max(
            within_budget,
            key=lambda row: integer_value(row["target"].get("priority")),
        )
    return min(
        candidates,
        key=lambda row: (
            abs(len(row["successful_trace_goals"]) - 9),
            -integer_value(row["target"].get("priority")),
        ),
    )


def target_proof_slice(
    execution: dict[str, Any],
    target: dict[str, Any],
    reference_program: str,
) -> dict[str, Any]:
    """Build a grounded proof envelope around one target goal.

    The execution trace is flat, so the slicer reconstructs parent-child links by
    unifying grounded trace goals with clause heads and bodies.  It also retains
    successful facts that make a competing, differently valued clause fail.
    """

    clauses = parse_prolog_clauses(reference_program)
    trace = [str(goal) for goal in execution.get("trace", [])]
    end = min(integer_value(target.get("trace_position")) + 1, len(trace))
    prefix = trace[:end]
    selected_goal_indices: set[int] = set()
    supporting_clause_indices: set[int] = set()
    competing_clause_indices: set[int] = set()
    exception_goal_indices: set[int] = set()
    required_negative_conditions: set[str] = set()
    resolving: set[int] = set()

    def resolve(goal_index: int) -> None:
        if goal_index in selected_goal_indices or goal_index in resolving:
            return
        resolving.add(goal_index)
        selected_goal_indices.add(goal_index)
        goal_term = parse_prolog_term(prefix[goal_index])
        signature = goal_signature(prefix[goal_index])
        if goal_term is None or signature is None:
            resolving.remove(goal_index)
            return

        matches: list[tuple[int, list[tuple[int, dict[str, str]]], dict[str, str]]] = []
        for clause_index, clause in enumerate(clauses):
            if clause["head_signature"] != signature:
                continue
            bindings: dict[str, str] = {}
            if not unify_prolog_terms(clause.get("head_term"), goal_term, bindings):
                continue
            body_matches = grounded_body_matches(
                clause,
                prefix,
                before=goal_index,
                initial_bindings=bindings,
                include_negated=False,
            )
            matches.append((clause_index, body_matches, bindings))

        if matches:
            clause_index, body_matches, selected_bindings = max(
                matches,
                key=lambda row: (
                    len(row[1]),
                    bool(clauses[row[0]].get("body")),
                    -row[0],
                ),
            )
            supporting_clause_indices.add(clause_index)
            for child_index, _ in body_matches:
                resolve(child_index)
            for call in clauses[clause_index].get("body_calls", []):
                if not call.get("negated"):
                    continue
                instantiated = instantiate_prolog_term(
                    call.get("term"), selected_bindings
                )
                if instantiated:
                    required_negative_conditions.add(f"not:{instantiated}")
                for nested_index, nested in enumerate(clauses):
                    if nested["head_signature"] == call.get("signature"):
                        supporting_clause_indices.add(nested_index)

        # A differently valued clause can still determine the answer when a
        # positive fact defeats one of its negated conditions. Preserve only
        # those blockers; positive premises that merely enable the competing
        # clause are not independently necessary to rule it out.
        target_args = goal_term.get("args", [])
        for clause_index, clause in enumerate(clauses):
            if clause["head_signature"] != signature or clause_index in supporting_clause_indices:
                continue
            head = clause.get("head_term")
            if not same_input_arguments(head, goal_term):
                continue
            full_bindings: dict[str, str] = {}
            if unify_prolog_terms(head, goal_term, full_bindings):
                continue
            branch_matches = grounded_body_matches(
                clause,
                prefix,
                before=goal_index,
                initial_bindings=bindings_from_input_arguments(head, target_args),
                include_negated=True,
            )
            blockers = [
                child_index
                for child_index, call in branch_matches
                if call.get("negated")
            ]
            if not blockers:
                continue
            competing_clause_indices.add(clause_index)
            for child_index in blockers:
                resolve(child_index)
                exception_goal_indices.add(child_index)

        resolving.remove(goal_index)

    target_goal = str(target["goal"])
    target_index = next(
        (index for index in range(len(prefix) - 1, -1, -1) if prefix[index] == target_goal),
        len(prefix) - 1,
    )
    if target_index >= 0:
        resolve(target_index)
    sliced_goals = [prefix[index] for index in sorted(selected_goal_indices)]
    for condition in sorted(required_negative_conditions):
        if condition not in sliced_goals:
            sliced_goals.append(condition)
    if target_goal not in sliced_goals:
        sliced_goals.append(target_goal)

    clause_indices = supporting_clause_indices | competing_clause_indices
    relevant_clauses = [
        clause["text"] for index, clause in enumerate(clauses) if index in clause_indices
    ]
    closure = {
        signature
        for index, clause in enumerate(clauses)
        if index in clause_indices
        for signature in [clause.get("head_signature"), *clause.get("body_signatures", [])]
        if signature
    }
    supporting_goals = [
        prefix[index]
        for index in sorted(selected_goal_indices - exception_goal_indices)
    ]
    exception_goals = [prefix[index] for index in sorted(exception_goal_indices)]
    exception_goals.extend(sorted(required_negative_conditions))
    supporting_clauses = [
        clause["text"]
        for index, clause in enumerate(clauses)
        if index in supporting_clause_indices
    ]
    competing_clauses = [
        clause["text"]
        for index, clause in enumerate(clauses)
        if index in competing_clause_indices
    ]
    target_signature = goal_signature(target_goal)
    construction_requirements = proof_construction_requirements(
        sliced_goals,
        relevant_clauses,
        exception_goals,
    )
    return {
        "schema": "sara_target_proof_envelope_v3",
        "slice_strategy": "grounded_backward_with_exception_blockers",
        "target": dict(target),
        "target_signature": target_signature,
        "predicate_closure": sorted(closure),
        "successful_trace_goals": sliced_goals,
        "formal_clauses": relevant_clauses,
        "formal_reference_program": "\n\n".join(f"{clause}." for clause in relevant_clauses),
        "proof_roles": {
            "supporting_goals": supporting_goals,
            "exception_blocker_goals": exception_goals,
            "required_negative_conditions": sorted(required_negative_conditions),
            "supporting_clauses": supporting_clauses,
            "competing_clauses": competing_clauses,
        },
        "construction_requirements": construction_requirements,
        "allowed_oracle_bases": [
            *[f"goal:{goal}" for goal in sliced_goals],
            *[f"clause:{clause}" for clause in relevant_clauses],
        ],
    }


def proof_construction_requirements(
    goals: list[str],
    formal_clauses: list[str],
    exception_goals: list[str],
) -> dict[str, Any]:
    """Classify the fixed proof into task-facing premises and derivation-only operations."""

    clauses = parse_prolog_clauses("\n\n".join(f"{clause}." for clause in formal_clauses))
    normalized_exceptions = {normalize_text(goal) for goal in exception_goals}
    leaves: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    deterministic: list[str] = []
    blockers: list[str] = []
    for goal in goals:
        if normalize_text(goal) in normalized_exceptions or goal.startswith("not:"):
            blockers.append(goal)
            continue
        if is_deterministic_intermediate_goal(goal, clauses):
            deterministic.append(goal)
            continue
        if goal_has_deriving_clause(goal, clauses):
            rules.append(
                {
                    "goal": goal,
                    "formal_clauses": matching_derivation_clauses(goal, clauses),
                }
            )
            continue
        leaves.append({"goal": goal})
    return {
        "required_leaf_goals": leaves,
        "required_rule_goals": rules,
        "required_exception_blockers": list(dict.fromkeys(blockers)),
        "deterministic_derivation_goals": deterministic,
        "estimated_atomic_riu_count": len(leaves) + len(rules) + len(set(blockers)),
    }


def matching_derivation_clauses(
    goal: str, clauses: list[dict[str, Any]]
) -> list[str]:
    """Return grounded-compatible, non-fact clauses that can derive a proof goal."""

    goal_term = parse_prolog_term(goal)
    signature = goal_signature(goal)
    if goal_term is None or signature is None:
        return []
    matches = []
    for clause in clauses:
        if clause.get("head_signature") != signature or not str(clause.get("body") or "").strip():
            continue
        if unify_prolog_terms(clause.get("head_term"), goal_term, {}):
            matches.append(str(clause.get("text") or ""))
    return matches


def grounded_body_matches(
    clause: dict[str, Any],
    trace: list[str],
    *,
    before: int,
    initial_bindings: dict[str, str],
    include_negated: bool,
) -> list[tuple[int, dict[str, Any]]]:
    bindings = dict(initial_bindings)
    matches: list[tuple[int, dict[str, Any]]] = []
    for call in clause.get("body_calls", []):
        if call.get("negated") and not include_negated:
            continue
        signature = call.get("signature")
        if not signature:
            continue
        matched: list[tuple[int, dict[str, str]]] = []
        for index in range(before - 1, -1, -1):
            if goal_signature(trace[index]) != signature:
                continue
            candidate_bindings = dict(bindings)
            if unify_prolog_terms(
                call.get("term"),
                parse_prolog_term(trace[index]),
                candidate_bindings,
            ):
                matched.append((index, candidate_bindings))
                if not call.get("collection"):
                    break
        if matched:
            if call.get("collection"):
                matches.extend((index, call) for index, _ in reversed(matched))
            else:
                index, candidate_bindings = matched[0]
                bindings = candidate_bindings
                matches.append((index, call))
    return matches


def same_input_arguments(
    clause_head: dict[str, Any] | None,
    target: dict[str, Any],
) -> bool:
    if clause_head is None or clause_head.get("functor") != target.get("functor"):
        return False
    head_args = clause_head.get("args", [])
    target_args = target.get("args", [])
    if len(head_args) != len(target_args) or len(head_args) < 2:
        return False
    bindings: dict[str, str] = {}
    return all(
        unify_prolog_terms(left, right, bindings)
        for left, right in zip(head_args[:-1], target_args[:-1])
    )


def bindings_from_input_arguments(
    clause_head: dict[str, Any] | None,
    target_args: list[dict[str, Any]],
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    if clause_head is None:
        return bindings
    for left, right in zip(clause_head.get("args", [])[:-1], target_args[:-1]):
        unify_prolog_terms(left, right, bindings)
    return bindings


def is_deterministic_intermediate_goal(
    goal: str, clauses: list[dict[str, Any]]
) -> bool:
    """Identify derived bookkeeping that should remain in reasoning steps."""

    parsed = parse_goal(goal)
    return bool(
        parsed
        and parsed[0] in DETERMINISTIC_INTERMEDIATE_PREDICATES
        and goal_has_deriving_clause(goal, clauses)
    )


def goal_requires_explicit_rule_riu(
    goal: str, clauses: list[dict[str, Any]]
) -> bool:
    """Require RIU coverage for derived domain rules, but not arithmetic internals."""

    return goal_has_deriving_clause(goal, clauses) and not is_deterministic_intermediate_goal(
        goal, clauses
    )


def integer_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def goal_has_deriving_clause(goal: str, clauses: list[dict[str, Any]]) -> bool:
    goal_term = parse_prolog_term(goal)
    signature = goal_signature(goal)
    if goal_term is None or signature is None:
        return False
    for clause in clauses:
        if clause.get("head_signature") != signature or not str(clause.get("body") or "").strip():
            continue
        if unify_prolog_terms(clause.get("head_term"), goal_term, {}):
            return True
    return False
