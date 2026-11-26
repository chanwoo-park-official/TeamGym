"""Construct source-complete SARA tasks around executable intermediate subproofs."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any

from teamgym.data.legal.sara.reference import (
    snap_span,
    decimal_amount,
    normalize_text,
    source_payload,
)
from teamgym.llm.openai_client import OpenAIJsonClient

# Preserve the original imports used by construction stages and downstream tools.
from .prolog import (
    canonical_prolog_atom as canonical_prolog_atom,
    clean_prolog_atom as clean_prolog_atom,
    extract_body_calls as extract_body_calls,
    extract_body_signatures as extract_body_signatures,
    goal_signature as goal_signature,
    instantiate_prolog_term as instantiate_prolog_term,
    is_prolog_variable as is_prolog_variable,
    matching_delimiter as matching_delimiter,
    parse_goal as parse_goal,
    parse_prolog_clauses as parse_prolog_clauses,
    parse_prolog_list as parse_prolog_list,
    parse_prolog_term as parse_prolog_term,
    serialize_prolog_term as serialize_prolog_term,
    split_list_tail as split_list_tail,
    split_prolog_args as split_prolog_args,
    split_prolog_clauses as split_prolog_clauses,
    split_prolog_rule as split_prolog_rule,
    strip_prolog_comments as strip_prolog_comments,
    term_signature as term_signature,
    unify_prolog_terms as unify_prolog_terms,
)
from .proof_slicing import (
    DETERMINISTIC_INTERMEDIATE_PREDICATES as DETERMINISTIC_INTERMEDIATE_PREDICATES,
    bindings_from_input_arguments as bindings_from_input_arguments,
    goal_has_deriving_clause as goal_has_deriving_clause,
    goal_requires_explicit_rule_riu as goal_requires_explicit_rule_riu,
    grounded_body_matches as grounded_body_matches,
    integer_value as integer_value,
    is_deterministic_intermediate_goal as is_deterministic_intermediate_goal,
    matching_derivation_clauses as matching_derivation_clauses,
    proof_construction_requirements as proof_construction_requirements,
    same_input_arguments as same_input_arguments,
    select_target_proof_slice as select_target_proof_slice,
    target_proof_slice as target_proof_slice,
)


TARGET_SLICED_POLICY = "target_sliced"
GENERATION_POLICIES = {TARGET_SLICED_POLICY}

TARGET_SPECS: dict[str, dict[str, Any]] = {
    "income_tax": {"label": "individual income tax", "kind": "money", "priority": 100},
    "section1_tax": {"label": "individual income tax", "kind": "money", "priority": 99},
    "tax_from_brackets": {"label": "individual income tax", "kind": "money", "priority": 98},
    "taxable_income": {"label": "taxable income", "kind": "money", "priority": 95},
    "employer_tax": {"label": "employer excise tax", "kind": "money", "priority": 92},
    "personal_exemption_deduction": {
        "label": "personal-exemption deduction",
        "kind": "money",
        "priority": 86,
    },
    "personal_exemptions": {
        "label": "personal-exemption deduction",
        "kind": "money",
        "priority": 85,
    },
    "standard_deduction": {"label": "standard deduction", "kind": "money", "priority": 82},
    "adjusted_gross_income": {
        "label": "adjusted gross income",
        "kind": "money",
        "priority": 78,
    },
    "gross_income": {"label": "gross income", "kind": "money", "priority": 74},
    "filing_status": {"label": "filing status", "kind": "category", "priority": 70},
}


BLIND_SOLVER_PROMPT = """Solve the legal calculation using only the supplied question and RIUs. Apply only the
provided rules. Return the requested answer and a concise step-by-step derivation as one JSON object. Do not
guess missing legal facts and do not use outside tax law."""


TARGET_SLICED_CONSTRUCTION_PROMPT = """Verbalize one SARA legal calculation task from the supplied fixed
target proof envelope. Target selection, grounded supporting goals, exception-blocking goals, and admissible
formal clauses have already been computed programmatically. Do not select another target or introduce a
sibling calculation. A source fact may clarify why a supplied rule applies even when the formal program
encodes the resulting status directly; map it to the status or rule application it supports.

The fixed envelope contains construction_requirements. Treat it as a mandatory coverage checklist, not as
solver-visible text. Every required_leaf_goal needs an atomic factual RIU. Every required_rule_goal needs an
ordinary-language rule RIU whose supports_goals includes that goal; never substitute the already-computed
result of the goal for its governing rule. Every required_exception_blocker needs its own explicit blocker RIU.
Goals listed under deterministic_derivation_goals belong only in the reasoning steps. Before returning, audit
the RIUs against every checklist entry and ensure that no checklist item is missing, duplicated, or represented
only by a derived intermediate. One RIU may cover multiple goals only when they are inseparable parts of the
same statutory branch; do not merge otherwise independent premises to fit the RIU limit.

Return 6-12 reference information units (RIUs) and a 4-10 step derivation. Each RIU must express exactly one
independently removable premise used by the proof envelope. Assign one reasoning_role:
- case_input: a factual value or relationship used in the calculation;
- applicability_condition: a fact establishing that a legal rule or status applies;
- exception_blocker: a fact needed to defeat an exception or competing branch;
- legal_rule: a classification or entitlement rule; or
- calculation_rule: a threshold, schedule, rate, cap, or arithmetic composition rule.

An explicit-source RIU must quote one exact contiguous source span, interpreted in its ordinary surrounding
context: phrases such as "their son" can establish the stated family relationship. An oracle-derived RIU must
express one implicit condition or bridge rule and cite exact entries from allowed_oracle_bases. Split premises
when they can be independently removed, but keep the conditions and numeric constant of one usable statutory
branch together. A general rule may contain the same number as the final answer; that is necessary input, not
answer leakage. Leakage means directly asserting the selected instance's final conclusion before applying the
rules. Text copied from formal_clauses is never an explicit-source quotation: verbalize it as oracle_derived and
cite the corresponding clause entry.

Do not overstate explicit evidence. Marriage alone does not establish that a joint return could have been
filed; parenthood, residence, or zero income alone does not establish dependent status; and maintaining a home
does not by itself establish furnishing more than half its cost. When the proof envelope supplies one of these
conditions as a formal base fact but the source does not fully state it, create a separate oracle-derived
applicability_condition with that exact goal or clause as its basis. Do not map a weaker explicit fact directly
to the stronger condition.

Do not duplicate a conclusion as an RIU when other RIUs and a supplied rule derive it. For example, when
marriage, separate filing, and the filing-status rule establish married-filing-separately status, put that
status in a reasoning step rather than adding a second oracle-derived status RIU. A grounded formal fact is a
valid RIU only when it is a leaf condition in this envelope, not when it is the output of a retained rule.
The subject and year already named in the question are fixed query metadata, not RIUs. Do not restate either
one as a premise. Include the legal rule that connects supplied conditions to any non-arithmetic classification
used by the proof; the solver must not be expected to invent a filing-status or entitlement rule.
Write every task-facing claim in ordinary legal language. Never expose formal predicate names, executable call
syntax, underscore-delimited category names, or construction terms such as program, proof, oracle, or trace.
Each RIU must also be self-contained when read independently; do not refer to "those conditions," "the above
facts," or another RIU's unstated content. State either an input premise or a reusable rule, not a causal
explanation that repeats an input and its derived conclusion in one RIU.

Use exception-blocking goals when the proof envelope supplies them. A premise that rules out a zero-deduction
exception or another competing outcome is outcome-critical even though it is absent from the final positive
clause. Cover leaf facts and rule applications, but put deterministic operations such as date conversion, age
subtraction, list construction or summation, counting, multiplication, and subtraction in reasoning steps
rather than separate RIUs. Do not create an RIU that merely verbalizes one of these operations.
Every entry in required_negative_conditions must have its own oracle-derived exception_blocker RIU. Do not
infer a negative condition merely because no positive fact was supplied, and do not bundle two independently
removable negative conditions into one RIU.

Classify every RIU as outcome_critical when omitting it changes or prevents the target answer, or as
derivation_active_masked when it changes an intermediate result on the fixed proof but a downstream numerical
operation masks the final effect. Potential relevance in a different hypothetical case is insufficient. Use
every RIU in the derivation and return one JSON object only."""


TARGET_SLICED_VERIFIER_PROMPT = """Verify a verbalized SARA task against its programmatically constructed
target proof envelope and the natural-language source. Judge semantic support, not literal identity with one
Prolog trace. The envelope distinguishes grounded supporting goals from facts that defeat exceptions. A
source-grounded fact may establish the applicability of a formal status or rule even when the reference program
encodes that status directly. Do not call such a fact off-path merely because it is not a body literal in the
selected positive clause.

Apply these rules carefully:
1. A fact that makes a negated condition fail and thereby defeats an exception or competing branch is
   outcome-critical, not off-path. A positive premise that merely helps enable that competing clause is not
   automatically necessary: reject it when another retained blocker already defeats the clause by itself.
2. A general statutory branch may state its threshold, base amount, or rate even when that number equals the
   selected answer. This is not leakage. Reject leakage only when an RIU directly states the selected person's
   final requested result without requiring the supplied derivation.
3. Deterministic date conversion, list summation, counting, multiplication, subtraction, and similar computed
   goals belong in reasoning steps. Do not require a separate RIU for every internal execution-trace goal.
4. Interpret quoted language in context. For example, "Alice and Bob have a son, Charlie" entails that Charlie
   is Alice's child. Do not impose exact-string entailment.
5. A source-grounded, legally valid derivation may refine a formal base fact. Accept it when the stated facts and
   supplied rule establish the same intermediate result; reject it only when a required condition is missing.
   Marriage alone does not establish eligibility to file jointly; parenthood, residence, or zero income alone
   does not establish dependency; and maintaining a home alone does not establish furnishing over half its
   cost. Require the remaining conditions or a separate oracle-derived formal-base RIU.
6. Do not require the task to answer the source dataset's original total-tax question. The selected target and
   programmatically supplied question intentionally define a legal subproblem.
7. An oracle-derived leaf fact with an exact goal or fact-clause basis is a valid task-specific condition even
   when the natural source does not state it. Source completeness may be false without being a rejection under
   this policy. However, reject an oracle RIU that merely duplicates an intermediate already derivable from
   other RIUs and a retained rule.
8. Every required_negative_conditions entry must be explicitly materialized by an atomic exception_blocker
   RIU. Silence or the mere absence of a positive RIU is not enough under the open-world removal test.
9. The subject and year stated in the question are fixed query metadata. Reject an RIU that only repeats that
   metadata. Also reject a derivation that omits a non-arithmetic branching rule, such as the rule connecting
   marital and exception conditions to a filing status.
10. Task-facing RIUs must use ordinary legal language. Reject formal predicate-call syntax, underscore-delimited
    category names, construction terminology, and RIUs that merely verbalize deterministic bookkeeping such as
    extracting a year, computing age, or constructing and summing a list.
11. When two RIUs support the same formal goal, verify that neither repeats or substantially contains the
    other's factual content. A broad claim and a narrower restatement of one fact inside it are not atomic.
12. Audit construction_requirements directly. Every required leaf, rule transition, and exception blocker must
    be represented by an atomic RIU with the appropriate role, while deterministic_derivation_goals must not be
    promoted to RIUs. A formula that merely reports a derived result does not satisfy a required rule transition.

Reject a genuinely unrelated sibling objective, an unsupported factual addition, a wrong argument assignment,
an incorrect or incomplete legal calculation, direct instance-level answer leakage, or an RIU that combines
independently removable premises. Validate reasoning_role: case_input supplies a used fact;
applicability_condition establishes why a rule applies; exception_blocker resolves a competing outcome;
legal_rule supplies a classification or entitlement rule; and calculation_rule supplies a usable numeric rule.

For each RIU, test its role in the fixed proof. Label it outcome_critical if omission changes the target answer
or makes the answer or applicable branch indeterminate under an open-world reading of the remaining RIUs.
Do not assume an omitted condition is false. Label it derivation_active_masked if omission changes an
intermediate legal classification, applicable branch, or amount while a specific downstream threshold, floor,
cap, or rate masks the final numerical effect. Label it off_path only if the same result and legally complete
derivation remain available from stated remaining premises, or if it serves an unrelated objective.

Explicit-source claims must be entailed by their quoted spans. Oracle-derived claims must be established by
their exact allowed basis. Exact formal base facts and closed-world conditions are valid oracle bases unless
contradicted by the source. The full RIU set must determine the target, and the outcome-critical subset must
constitute the answer-determining sufficient set. Return one JSON object only."""


TARGET_SLICED_ADJUDICATOR_PROMPT = """Independently adjudicate a rejected SARA target-sliced candidate.
The prior verification is evidence, not authority. Recheck every criterion in the candidate against the source
and proof envelope. Overturn a rejection only when the complete candidate is valid.

Specifically guard against these known false-positive patterns: calling an applicability fact off-path merely
because the formal program hard-codes the resulting status; calling a fact that defeats a competing exception
off-path; calling a general statutory amount or rate answer leakage; requiring an RIU for deterministic
arithmetic trace internals; or refusing ordinary contextual entailment such as "their child." Conversely, do
not excuse a wrong person or year, an invented condition, a missing legal prerequisite, a bundled premise, a
genuinely unrelated objective, or a direct assertion of the selected instance's final result. Evaluate RIU
necessity under an open-world reading: after removing a premise, its content is unknown rather than false.
The selected target intentionally defines a subproblem and need not match the source dataset's original
total-tax question. Treat an exact formal base fact as a valid oracle-derived task condition even when source
completeness is false; do not demand natural-language support for that fact. Still reject a derived intermediate
duplicated as an RIU when the remaining RIUs and a retained rule already establish it. An explicit election or
applicability fact remains necessary when it determines whether a supplied deduction or rule can legally be
used, even if the formal program omits that guard.
Require one atomic exception_blocker RIU for each required_negative_conditions entry in the proof envelope;
never accept silence as proof of a negative condition.
Do not retain a positive premise merely because it helps enable a competing clause when another supplied fact
already defeats that clause. Treat the subject and year in the question as query metadata rather than RIUs,
require every non-arithmetic rule used by the proof, and reject task-facing implementation language or a
separate RIU for deterministic bookkeeping.
Return the same verification schema as requested and explain every remaining blocking issue."""


def execution_is_usable(execution: dict[str, Any]) -> bool:
    """Allow label-inconsistent programs while requiring a successful executable trace."""

    blocking = [
        issue
        for issue in execution.get("issues", [])
        if not str(issue).startswith("Rounded reference result")
    ]
    return not blocking and bool(execution.get("trace"))


def target_options(execution: dict[str, Any], source: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    """Extract stable intermediate targets from successful reference goals."""

    taxpayer, year = taxpayer_and_year(source)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for position, goal in enumerate(execution.get("trace", [])):
        parsed = parse_goal(str(goal))
        if parsed is None:
            continue
        predicate, args = parsed
        spec = TARGET_SPECS.get(predicate)
        if spec is None or not args:
            continue
        answer = clean_prolog_atom(args[-1])
        if spec["kind"] == "money" and decimal_amount(answer) is None:
            continue
        key = (predicate, answer)
        if key in seen:
            continue
        seen.add(key)
        first_argument = clean_prolog_atom(args[0])
        subject = (
            taxpayer
            if predicate in {"section1_tax", "tax_from_brackets"}
            or re.fullmatch(r"(?:19|20)\d{2}", first_argument)
            else first_argument
        )
        target_year = next((clean_prolog_atom(arg) for arg in args[1:-1] if re.fullmatch(r"(?:19|20)\d{2}", clean_prolog_atom(arg))), year)
        candidates.append(
            {
                "predicate": predicate,
                "goal": goal,
                "answer": answer,
                "answer_kind": spec["kind"],
                "label": spec["label"],
                "subject": subject or taxpayer,
                "year": target_year or year,
                "trace_position": position,
                "priority": int(spec["priority"]),
            }
        )
    candidates.sort(
        key=lambda row: (
            row["priority"],
            row["trace_position"],
            decimal_amount(row["answer"]) not in {None, Decimal("0")},
        ),
        reverse=True,
    )
    options = candidates[:limit]
    for index, option in enumerate(options, start=1):
        option["target_id"] = f"target_{index:02d}"
        option["question"] = target_question(option)
    return options


def generate_case(
    client: OpenAIJsonClient,
    row: dict[str, Any],
    execution: dict[str, Any],
    *,
    trace_dir: Path,
    policy: str = TARGET_SLICED_POLICY,
) -> dict[str, Any]:
    validate_policy(policy)
    source = source_payload(row)
    full_reference = str(row.get("reference_prolog") or "")
    source["construction_reference"] = full_reference
    options = target_options(execution, source)
    case_id = source["case_id"]
    if not execution_is_usable(execution) or not options:
        return {
            "schema": "sara_subproof_generation_record_v2",
            "generation_policy": policy,
            "case_id": case_id,
            "accepted": False,
            "source": source,
            "oracle_execution": execution,
            "target_options": options,
            "failed_stage": "target_extraction",
            "seed_spec": None,
        }

    proof_slice: dict[str, Any] | None = None
    target_execution = execution
    proof_slice = select_target_proof_slice(execution, options, full_reference)
    if proof_slice is None:
        screened_options = []
        screening = {
            "schema": "sara_deterministic_target_screen_v1",
            "accepted": False,
            "reason": "No deterministic target proof slice could be formed.",
        }
    else:
        screened_options = [proof_slice["target"]]
        source["construction_reference"] = proof_slice["formal_reference_program"]
        source["target_proof_slice"] = proof_slice
        target_execution = {
            **execution,
            "trace": proof_slice["successful_trace_goals"],
            "trace_goal_count": len(proof_slice["successful_trace_goals"]),
        }
        screening = {
            "schema": "sara_deterministic_target_screen_v1",
            "accepted": True,
            "target_id": proof_slice["target"]["target_id"],
            "trace_goal_count": len(proof_slice["successful_trace_goals"]),
            "predicate_closure": proof_slice["predicate_closure"],
        }
    if not screened_options:
        return {
            "schema": "sara_subproof_generation_record_v2",
            "generation_policy": policy,
            "case_id": case_id,
            "accepted": False,
            "source": source,
            "oracle_execution": execution,
            "target_options": options,
            "target_screening": screening,
            "failed_stage": "target_screening",
            "seed_spec": None,
        }
    candidate = construct_candidate(
        client,
        source,
        target_execution,
        screened_options,
        screening,
        trace_dir=trace_dir / "construction",
        policy=policy,
    )
    attempts: list[dict[str, Any]] = []
    maximum_attempts = 3
    for attempt_number in range(1, maximum_attempts + 1):
        evaluation = evaluate_candidate_attempt(
            client,
            source,
            target_execution,
            candidate,
            screened_options,
            trace_dir=trace_dir,
            suffix="initial" if attempt_number == 1 else f"repair_{attempt_number - 1}",
            policy=policy,
        )
        evaluation["attempt"] = attempt_number
        attempts.append(evaluation)
        if candidate_passes(
            evaluation["structural_audit"],
            evaluation["blind_all_riu_audit"],
            evaluation["semantic_verification"],
            candidate,
            policy=policy,
            require_blind=False,
        ):
            break
        if attempt_number == maximum_attempts:
            break
        candidate = repair_candidate(
            client,
            source,
            target_execution,
            screened_options,
            screening,
            candidate,
            evaluation,
            trace_dir=trace_dir / f"repair_{attempt_number}",
            repair_round=attempt_number,
            policy=policy,
        )

    final_attempt = attempts[-1]
    candidate = final_attempt["candidate"]
    structural = final_attempt["structural_audit"]
    blind_solution = final_attempt["blind_all_riu_solution"]
    blind_audit = final_attempt["blind_all_riu_audit"]
    verification = final_attempt["semantic_verification"]
    accepted = candidate_passes(
        structural,
        blind_audit,
        verification,
        candidate,
        policy=policy,
        require_blind=False,
    )
    selected = selected_target(candidate, screened_options)
    return {
        "schema": "sara_subproof_generation_record_v2",
        "generation_policy": policy,
        "case_id": case_id,
        "accepted": accepted,
        "source": source,
        "oracle_execution": execution,
        "target_execution": target_execution,
        "target_proof_slice": proof_slice,
        "target_options": screened_options,
        "all_target_options": options,
        "target_screening": screening,
        "selected_target": selected,
        "candidate": candidate,
        "structural_audit": structural,
        "blind_all_riu_solution": blind_solution,
        "blind_all_riu_audit": blind_audit,
        "semantic_verification": verification,
        "semantic_primary_verification": final_attempt.get("semantic_primary_verification"),
        "semantic_adjudication": final_attempt.get("semantic_adjudication"),
        "repair_attempt": final_attempt if len(attempts) > 1 else None,
        "generation_attempts": attempts,
        "seed_spec": (
            candidate_to_seed(source, candidate, selected, policy=policy)
            if accepted and selected
            else None
        ),
    }


def evaluate_candidate_attempt(
    client: OpenAIJsonClient,
    source: dict[str, Any],
    execution: dict[str, Any],
    candidate: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    trace_dir: Path,
    suffix: str,
    policy: str,
) -> dict[str, Any]:
    """Run deterministic and model checks for one construction attempt."""

    prepare_candidate(candidate, source, options, policy=policy)
    structural = structural_audit(candidate, source, execution, options, policy=policy)
    if not structural.get('accepted'):
        return {
            "candidate": candidate,
            "structural_audit": structural,
            "blind_all_riu_solution": {},
            "blind_all_riu_audit": {
                "accepted": False,
                "issues": ["Blind solve deferred until deterministic structure passes."],
            },
            "semantic_primary_verification": None,
            "semantic_verification": {
                "accepted": False,
                "issues": ["Semantic verification deferred until deterministic structure passes."],
            },
            "semantic_adjudication": None,
        }

    blind_solution, blind_audit = solve_from_rius(
        client,
        source,
        candidate,
        options,
        trace_dir=trace_dir / "blind",
        suffix=suffix,
    )
    primary_verification = verify_candidate(
        client,
        source,
        execution,
        candidate,
        structural,
        blind_solution,
        blind_audit,
        options,
        trace_dir=trace_dir / "verification",
        suffix=suffix,
        policy=policy,
    )
    verification = primary_verification
    adjudication: dict[str, Any] | None = None
    if (
        structural.get('accepted') and (not candidate_passes(structural, blind_audit, verification, candidate, policy=policy, require_blind=False))
    ):
        adjudication = adjudicate_candidate(
            client,
            source,
            execution,
            candidate,
            structural,
            blind_solution,
            blind_audit,
            options,
            verification,
            trace_dir=trace_dir / "adjudication",
            suffix=suffix,
        )
        if candidate_passes(
            structural,
            blind_audit,
            adjudication,
            candidate,
            policy=policy,
            require_blind=False,
        ):
            verification = adjudication
    return {
        "candidate": candidate,
        "structural_audit": structural,
        "blind_all_riu_solution": blind_solution,
        "blind_all_riu_audit": blind_audit,
        "semantic_primary_verification": primary_verification,
        "semantic_verification": verification,
        "semantic_adjudication": adjudication,
    }


def repair_candidate(
    client: OpenAIJsonClient,
    source: dict[str, Any],
    execution: dict[str, Any],
    options: list[dict[str, Any]],
    screening: dict[str, Any],
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    trace_dir: Path,
    repair_round: int,
    policy: str,
) -> dict[str, Any]:
    """Request a complete replacement candidate using the latest audit evidence."""

    structural = evaluation["structural_audit"]
    blind_solution = evaluation["blind_all_riu_solution"]
    blind_audit = evaluation["blind_all_riu_audit"]
    verification = evaluation["semantic_verification"]
    return client.generate_json(
        prompt_id=f"{source['case_id']}.subproof_repair_{repair_round}",
        system_prompt=construction_prompt(policy),
        user_prompt=json.dumps(
            {
                "source": source_for_model(source),
                "target_options": options,
                "target_screening": screening,
                "successful_trace_goals": execution["trace"],
                "formal_reference_program": construction_reference(source, policy),
                "fixed_target_proof_slice": source.get("target_proof_slice"),
                "construction_requirements": (
                    (source.get("target_proof_slice") or {}).get("construction_requirements")
                ),
                "previous_candidate": candidate,
                "structural_issues": structural.get("issues", []),
                "blind_solution": blind_solution,
                "blind_issues": blind_audit.get("issues", []),
                "verifier_issues": verification.get("issues", []),
                "riu_criticality": verification.get("riu_criticality", []),
                "instruction": (
                    "Return a complete replacement that resolves every listed issue and covers every "
                    "construction requirement. The target and proof slice are fixed. Do not preserve a "
                    "previous RIU merely to minimize edits."
                ),
                "required_schema": candidate_schema(policy),
            },
            ensure_ascii=False,
        ),
        trace_dir=trace_dir,
    )


def construct_candidate(
    client: OpenAIJsonClient,
    source: dict[str, Any],
    execution: dict[str, Any],
    options: list[dict[str, Any]],
    screening: dict[str, Any],
    *,
    trace_dir: Path,
    policy: str = TARGET_SLICED_POLICY,
) -> dict[str, Any]:
    return client.generate_json(
        prompt_id=f"{source['case_id']}.subproof_construct",
        system_prompt=construction_prompt(policy),
        user_prompt=json.dumps(
            {
                "source": source_for_model(source),
                "target_options": options,
                "target_screening": screening,
                "successful_trace_goals": execution["trace"],
                "formal_reference_program": construction_reference(source, policy),
                "fixed_target_proof_slice": source.get("target_proof_slice"),
                "required_schema": candidate_schema(policy),
            },
            ensure_ascii=False,
        ),
        trace_dir=trace_dir,
    )


def solve_from_rius(
    client: OpenAIJsonClient,
    source: dict[str, Any],
    candidate: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    trace_dir: Path,
    suffix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = selected_target(candidate, options)
    solution = client.generate_json(
        prompt_id=f"{source['case_id']}.subproof_blind_{suffix}",
        system_prompt=BLIND_SOLVER_PROMPT,
        user_prompt=json.dumps(
            {
                "question": candidate.get("question"),
                "rius": solver_rius(candidate),
                "required_output": {"answer": "requested answer", "derivation": ["reasoning step"]},
            },
            ensure_ascii=False,
        ),
        trace_dir=trace_dir,
    )
    return solution, blind_solution_audit(solution, target)


def verify_candidate(
    client: OpenAIJsonClient,
    source: dict[str, Any],
    execution: dict[str, Any],
    candidate: dict[str, Any],
    structural: dict[str, Any],
    blind_solution: dict[str, Any],
    blind_audit: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    trace_dir: Path,
    suffix: str,
    policy: str = TARGET_SLICED_POLICY,
) -> dict[str, Any]:
    return client.generate_json(
        prompt_id=f"{source['case_id']}.subproof_verify_{suffix}",
        system_prompt=verifier_prompt(policy),
        user_prompt=json.dumps(
            {
                "source": source_for_model(source),
                "selected_target": selected_target(candidate, options),
                "successful_trace_goals": execution["trace"],
                "formal_reference_program": construction_reference(source, policy),
                "candidate": candidate,
                "programmatic_audit": structural,
                "blind_solution": blind_solution,
                "blind_audit": blind_audit,
                "required_output": verification_output_schema(),
                "fixed_target_proof_slice": source.get("target_proof_slice"),
            },
            ensure_ascii=False,
        ),
        trace_dir=trace_dir,
    )


def adjudicate_candidate(
    client: OpenAIJsonClient,
    source: dict[str, Any],
    execution: dict[str, Any],
    candidate: dict[str, Any],
    structural: dict[str, Any],
    blind_solution: dict[str, Any],
    blind_audit: dict[str, Any],
    options: list[dict[str, Any]],
    prior_verification: dict[str, Any],
    *,
    trace_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    return client.generate_json(
        prompt_id=f"{source['case_id']}.subproof_adjudicate_{suffix}",
        system_prompt=TARGET_SLICED_ADJUDICATOR_PROMPT,
        user_prompt=json.dumps(
            {
                "source": source_for_model(source),
                "selected_target": selected_target(candidate, options),
                "candidate": candidate,
                "fixed_target_proof_slice": source.get("target_proof_slice"),
                "successful_trace_goals": execution["trace"],
                "formal_reference_program": construction_reference(
                    source, TARGET_SLICED_POLICY
                ),
                "programmatic_audit": structural,
                "blind_solution": blind_solution,
                "blind_audit": blind_audit,
                "prior_verification": prior_verification,
                "required_output": verification_output_schema(),
            },
            ensure_ascii=False,
        ),
        trace_dir=trace_dir,
    )


def verification_output_schema() -> dict[str, Any]:
    return {
        "accepted": "boolean",
        "source_complete": "boolean",
        "oracle_consistent": "boolean",
        "collectively_sufficient": "boolean",
        "atomic_rius": "boolean",
        "reasoning_roles_valid": "boolean",
        "applicability_complete": "boolean",
        "exception_resolution_complete": "boolean",
        "target_unique": "boolean",
        "proof_correct": "boolean",
        "issues": ["specific issue"],
        "riu_criticality": [
            {
                "rui_id": "riu_01",
                "removal_effect": (
                    "indeterminate|different_branch|different_amount|"
                    "masked_intermediate_change|still_solvable"
                ),
                "criticality_class": (
                    "outcome_critical|derivation_active_masked|off_path"
                ),
                "explanation": "brief explanation",
                "counterfactual_witness": {
                    "alternative_assignment": "condition possible after omitting this RIU",
                    "resulting_answer_or_branch": "different result, branch, or indeterminate",
                    "why_remaining_rius_allow_it": "brief explanation",
                },
            }
        ],
    }


def candidate_passes(
    structural: dict[str, Any],
    blind_audit: dict[str, Any],
    verification: dict[str, Any],
    candidate: dict[str, Any],
    *,
    policy: str = TARGET_SLICED_POLICY,
    require_blind: bool = True,
) -> bool:
    if not structural.get("accepted") or (
        require_blind and not blind_audit.get("accepted")
    ):
        return False
    if verification.get("accepted") is not True:
        return False
    if not all((verification.get(field) is True for field in ('oracle_consistent', 'collectively_sufficient', 'atomic_rius', 'target_unique', 'proof_correct'))):
        return False
    riu_ids = {
        str(row.get("rui_id"))
        for row in candidate.get("rius", [])
        if isinstance(row, dict) and row.get("rui_id")
    }
    criticality = verification.get("riu_criticality")
    if not isinstance(criticality, list):
        return False
    rows = {
        str(row.get("rui_id")): row
        for row in criticality
        if isinstance(row, dict)
    }
    if set(rows) != riu_ids:
        return False
    allowed_effects = {
        "indeterminate",
        "different_branch",
        "different_amount",
        "masked_intermediate_change",
    }
    if any(str(row.get("removal_effect")) not in allowed_effects for row in rows.values()):
        return False
    if not all(valid_counterfactual_witness(row) for row in rows.values()):
        return False
    if not all(
        verification.get(field) is True
        for field in (
            "reasoning_roles_valid",
            "applicability_complete",
            "exception_resolution_complete",
        )
    ):
        return False
    candidate_classes = {
        str(row.get("rui_id")): str(row.get("criticality_class") or "")
        for row in candidate.get("rius", [])
        if isinstance(row, dict)
    }
    verified_classes = {
        rui_id: str(row.get("criticality_class") or "")
        for rui_id, row in rows.items()
    }
    allowed_classes = {"outcome_critical", "derivation_active_masked"}
    if any(value not in allowed_classes for value in verified_classes.values()):
        return False
    for rui_id, criticality_class in verified_classes.items():
        effect = str(rows[rui_id].get("removal_effect") or "")
        if (criticality_class == "derivation_active_masked") != (
            effect == "masked_intermediate_change"
        ):
            return False
    if candidate_classes != verified_classes:
        return False
    if sum(value == "outcome_critical" for value in verified_classes.values()) < 2:
        return False
    return True


def structural_audit(
    candidate: dict[str, Any],
    source: dict[str, Any],
    execution: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    policy: str = TARGET_SLICED_POLICY,
) -> dict[str, Any]:
    issues: list[str] = []
    target = selected_target(candidate, options)
    if target is None:
        issues.append("selected_target_id is not one of the executable target options")
    elif candidate.get("question") != target["question"]:
        issues.append("Question does not exactly match the selected executable target")

    rius = candidate.get("rius") if isinstance(candidate.get("rius"), list) else []
    steps = candidate.get("reasoning_steps") if isinstance(candidate.get("reasoning_steps"), list) else []
    maximum_rius = (
        12
    )
    if not 6 <= len(rius) <= maximum_rius:
        issues.append(f"Expected 6-{maximum_rius} RIUs, found {len(rius)}")
    if not 4 <= len(steps) <= 10:
        issues.append(f"Expected 4-10 reasoning steps, found {len(steps)}")
    ids = [str(row.get("rui_id") or "") for row in rius if isinstance(row, dict)]
    if len(ids) != len(rius) or any(not value for value in ids) or len(ids) != len(set(ids)):
        issues.append("RIU IDs must be nonempty and unique")

    normalized_source = normalize_text(source["natural_source"])
    trace_goals = {normalize_text(goal) for goal in execution.get("trace", [])}
    normalized_reference = normalize_text(source.get("construction_reference"))
    reference_clauses = parse_prolog_clauses(
        str(source.get("construction_reference") or "")
    )
    for riu in rius:
        if not isinstance(riu, dict):
            issues.append("Every RIU must be an object")
            continue
        riu_id = str(riu.get("rui_id") or "")
        span = str(riu.get("source_span") or "").strip()
        provenance = str(riu.get("provenance_type") or "explicit_source")
        if provenance == 'explicit_source':
            if not span or normalize_text(span) not in normalized_source:
                issues.append(f"RIU {riu_id} source span is not grounded")
        elif provenance == "oracle_derived":
            basis = riu.get("oracle_basis") if isinstance(riu.get("oracle_basis"), list) else []
            if not basis:
                issues.append(f"RIU {riu_id} has no oracle basis")
            for goal in basis:
                basis_text = str(goal)
                if basis_text.startswith(('goal:', 'clause:')):
                    basis_text = basis_text.split(":", 1)[1]
                normalized_basis = normalize_text(basis_text)
                if not normalized_basis or (
                    normalized_basis not in trace_goals
                    and normalized_basis not in normalized_reference
                ):
                    issues.append(
                        f"RIU {riu_id} oracle basis is absent from the trace and formal reference"
                    )
                    break
            if (
                str(riu.get('reasoning_role') or '') in {'case_input', 'applicability_condition'} and any((str(item).startswith('goal:') and goal_has_deriving_clause(str(item).split(':', 1)[1], reference_clauses) for item in basis))
            ):
                issues.append(
                    f"RIU {riu_id} materializes a derived intermediate instead of a leaf premise"
                )
            if not span or normalize_text(span) != normalize_text(riu.get("normalized_claim")):
                issues.append(f"RIU {riu_id} oracle-derived span must equal its task-facing claim")
        else:
            issues.append(f"RIU {riu_id} has invalid provenance type {provenance!r}")
        if len(span.split()) > 200:
            issues.append(f"RIU {riu_id} exceeds 200 source words")
        claim = str(riu.get("normalized_claim") or "")
        if not claim or len(claim.split()) > 90:
            issues.append(f"RIU {riu_id} claim must contain 1-90 words")
        language_issue = task_facing_text_issue(claim)
        if language_issue:
            issues.append(f"RIU {riu_id} {language_issue}")
        if target and only_restates_target_metadata(claim, target):
            issues.append(
                f"RIU {riu_id} only restates subject or year metadata already fixed by the question"
            )
        supported = riu.get("supports_goals") if isinstance(riu.get("supports_goals"), list) else []
        if not supported:
            issues.append(f"RIU {riu_id} has no trace-goal mapping")
        for goal in supported:
            supported_goal = str(goal)
            if supported_goal.startswith('goal:'):
                supported_goal = supported_goal.split(":", 1)[1]
            if normalize_text(supported_goal) not in trace_goals:
                issues.append(f"RIU {riu_id} maps to a goal absent from the successful trace")
                break
            if provenance == "explicit_source":
                support_issue = explicit_support_overstatement(claim, supported_goal)
                if support_issue:
                    issues.append(f"RIU {riu_id} {support_issue}")
                    break
        supported_goal_texts = [str(goal).removeprefix("goal:") for goal in supported]
        if (
            supported_goal_texts and all((is_deterministic_intermediate_goal(goal, reference_clauses) for goal in supported_goal_texts))
        ):
            issues.append(
                f"RIU {riu_id} verbalizes only deterministic bookkeeping that belongs in the derivation"
            )
        criticality_class = str(riu.get("criticality_class") or "")
        if criticality_class not in {
            "outcome_critical",
            "derivation_active_masked",
        }:
            issues.append(f"RIU {riu_id} has invalid criticality class")
        reasoning_role = str(riu.get("reasoning_role") or "")
        if reasoning_role not in {
            "case_input",
            "applicability_condition",
            "exception_blocker",
            "legal_rule",
            "calculation_rule",
        }:
            issues.append(f"RIU {riu_id} has invalid reasoning role")
        if target and target["answer_kind"] == "money":
            answer_value = decimal_amount(target["answer"])
            if (
                answer_value is not None
                and answer_value in numeric_values(claim)
                and directly_states_target(claim, target)
            ):
                issues.append(f"RIU {riu_id} directly states the selected target result")
        elif (
            target
            and canonical_category(target["answer"]) in canonical_category(claim)
            and directly_states_target(claim, target)
        ):
            issues.append(f"RIU {riu_id} directly states the selected target result")

    for index, left in enumerate(rius):
        if not isinstance(left, dict):
            continue
        left_goals = {
            normalize_text(str(goal).removeprefix("goal:"))
            for goal in left.get("supports_goals", [])
        }
        for right in rius[index + 1 :]:
            if not isinstance(right, dict):
                continue
            right_goals = {
                normalize_text(str(goal).removeprefix("goal:"))
                for goal in right.get("supports_goals", [])
            }
            if left_goals & right_goals and claims_substantially_overlap(
                str(left.get("normalized_claim") or ""),
                str(right.get("normalized_claim") or ""),
            ):
                issues.append(
                    f"RIUs {left.get('rui_id')} and {right.get('rui_id')} substantially repeat "
                    "the same-goal factual content"
                )

    supported_goals = {
        normalize_text(str(goal).removeprefix("goal:"))
        for riu in rius
        if isinstance(riu, dict)
        for goal in riu.get("supports_goals", [])
    }
    construction_requirements = (
        (source.get("target_proof_slice") or {}).get("construction_requirements") or {}
    )
    for requirement in construction_requirements.get("required_leaf_goals", []):
        goal_text = str(requirement.get("goal") or "")
        if normalize_text(goal_text) not in supported_goals:
            issues.append(f"No RIU supplies the required leaf premise {goal_text}")
    for requirement in construction_requirements.get("required_rule_goals", []):
        goal_text = str(requirement.get("goal") or "")
        matching = [
            riu
            for riu in rius
            if isinstance(riu, dict)
            and normalize_text(goal_text)
            in {
                normalize_text(str(goal).removeprefix("goal:"))
                for goal in riu.get("supports_goals", [])
            }
        ]
        if matching and not any(
            riu.get("reasoning_role") in {"legal_rule", "calculation_rule"}
            for riu in matching
        ):
            issues.append(
                f"The RIU covering derived goal {goal_text} is not labeled as a legal or calculation rule"
            )
    for goal in execution.get("trace", []):
        goal_text = str(goal)
        if goal_text.startswith("not:") or normalize_text(goal_text) in supported_goals:
            continue
        if goal_requires_explicit_rule_riu(goal_text, reference_clauses):
            issues.append(
                f"No RIU supplies the non-arithmetic rule needed to derive {goal_text}"
            )

    required_positive_blockers = {
        normalize_text(value)
        for value in construction_requirements.get("required_exception_blockers", [])
        if not str(value).startswith("not:")
    }
    for condition in sorted(required_positive_blockers):
        matching = [
            riu
            for riu in rius
            if isinstance(riu, dict)
            and condition
            in {
                normalize_text(str(goal).removeprefix("goal:"))
                for goal in riu.get("supports_goals", [])
            }
        ]
        if not matching:
            issues.append(
                f"Required exception blocker {condition!r} has no RIU"
            )
        elif any(riu.get("reasoning_role") != "exception_blocker" for riu in matching):
            issues.append(
                f"Required exception blocker {condition!r} is not labeled as an exception blocker"
            )

    required_negatives = {
        normalize_text(value)
        for value in (
            (source.get("target_proof_slice") or {})
            .get("proof_roles", {})
            .get("required_negative_conditions", [])
        )
    }
    for condition in sorted(required_negatives):
        matching = [
            riu
            for riu in rius
            if isinstance(riu, dict)
            and condition
            in {
                normalize_text(str(goal).removeprefix("goal:"))
                for goal in riu.get("supports_goals", [])
            }
        ]
        if not matching:
            issues.append(
                f"Required negative condition {condition!r} has no exception-blocker RIU"
            )
        elif any(riu.get("reasoning_role") != "exception_blocker" for riu in matching):
            issues.append(
                f"Required negative condition {condition!r} is not labeled as an exception blocker"
            )
    for riu in rius:
        if not isinstance(riu, dict) or riu.get("reasoning_role") != "exception_blocker":
            continue
        covered = {
            normalize_text(str(goal).removeprefix("goal:"))
            for goal in riu.get("supports_goals", [])
        } & required_negatives
        if len(covered) > 1:
            issues.append(
                f"RIU {riu.get('rui_id')} bundles multiple required negative conditions"
            )

    used = {
        str(rui_id)
        for step in steps
        if isinstance(step, dict)
        for rui_id in step.get("uses", [])
    }
    if set(ids) - used:
        issues.append(f"Unused RIUs: {sorted(set(ids) - used)}")
    if used - set(ids):
        issues.append(f"Unknown RIUs in steps: {sorted(used - set(ids))}")

    solution = candidate.get("structured_solution")
    if not isinstance(solution, dict):
        issues.append("Missing structured solution")
    elif target:
        answer_audit = answer_matches(solution.get("target_answer"), target)
        if not answer_audit:
            issues.append(
                f"Structured target answer {solution.get('target_answer')} does not match {target['answer']}"
            )
        derivation = solution.get("derivation")
        if not isinstance(derivation, list) or len(derivation) < 4:
            issues.append("Structured solution requires at least four derivation steps")
    return {"accepted": not issues, "issues": issues, "riu_count": len(rius), "step_count": len(steps)}


def blind_solution_audit(solution: dict[str, Any], target: dict[str, Any] | None) -> dict[str, Any]:
    issues: list[str] = []
    if target is None:
        return {"accepted": False, "issues": ["No selected executable target"]}
    if not answer_matches(solution.get("answer"), target):
        issues.append(f"Blind answer {solution.get('answer')} does not match oracle {target['answer']}")
    derivation = solution.get("derivation")
    if not isinstance(derivation, list) or len(derivation) < 3:
        issues.append("Blind solution requires at least three derivation steps")
    return {"accepted": not issues, "issues": issues}


def prepare_candidate(
    candidate: dict[str, Any],
    source: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    policy: str = TARGET_SLICED_POLICY,
) -> None:
    target = selected_target(candidate, options)
    if target:
        candidate["question"] = target["question"]
    normalized = normalize_text(source["natural_source"])
    normalized_reference = normalize_text(source.get("construction_reference"))
    proof_roles = (source.get("target_proof_slice") or {}).get("proof_roles") or {}
    exception_goals = {
        normalize_text(goal)
        for goal in proof_roles.get("exception_blocker_goals", [])
    }
    role_aliases = {
        "case_fact": "case_input",
        "fact": "case_input",
        "input": "case_input",
        "applicability": "applicability_condition",
        "branch_blocker": "exception_blocker",
        "rule": "legal_rule",
        "numeric_rule": "calculation_rule",
        "arithmetic_rule": "calculation_rule",
    }
    for riu in candidate.get("rius", []):
        if not isinstance(riu, dict):
            continue
        role = str(riu.get("reasoning_role") or "").strip().lower()
        role = role_aliases.get(role, role)
        supported = {
            normalize_text(str(goal).removeprefix("goal:"))
            for goal in riu.get("supports_goals", [])
        }
        if supported & exception_goals:
            role = "exception_blocker"
        riu["reasoning_role"] = role
        span_text = str(riu.get("source_span") or "").strip()
        if (
            riu.get('provenance_type') == 'explicit_source' and span_text and (normalize_text(span_text) not in normalized) and (normalize_text(span_text) in normalized_reference)
        ):
            riu["provenance_type"] = "oracle_derived"
            basis = riu.get("oracle_basis") if isinstance(riu.get("oracle_basis"), list) else []
            riu["oracle_basis"] = list(dict.fromkeys([*basis, f"clause:{span_text}"]))
        if (
            riu.get('provenance_type') == 'oracle_derived'
        ):
            riu["source_span"] = str(riu.get("normalized_claim") or "").strip()
            continue
        snapped = snap_span(normalize_text(riu.get("source_span")), normalized)
        if snapped:
            riu["source_span"] = snapped


def selected_target(
    candidate: dict[str, Any], options: list[dict[str, Any]]
) -> dict[str, Any] | None:
    target_id = str(candidate.get("selected_target_id") or "")
    return next((row for row in options if row["target_id"] == target_id), None)


def candidate_to_seed(
    source: dict[str, Any],
    candidate: dict[str, Any],
    target: dict[str, Any],
    *,
    policy: str = TARGET_SLICED_POLICY,
) -> dict[str, Any]:
    rius = [seed_riu(row) for row in candidate["rius"]]
    all_ids = [str(row["rui_id"]) for row in rius]
    required_ids = (
        [
            str(row["rui_id"])
            for row in rius
            if row.get("criticality_class") == "outcome_critical"
        ]
    )
    masked_ids = [
        str(row["rui_id"])
        for row in rius
        if row.get("criticality_class") == "derivation_active_masked"
    ]
    steps = [
        {
            "step": index,
            "uses": [str(value) for value in row.get("uses", [])],
            "claim": str(row.get("claim") or ""),
            "judge_focus": str(row.get("judge_focus") or row.get("claim") or ""),
        }
        for index, row in enumerate(candidate["reasoning_steps"], start=1)
    ]
    answer = str(target["answer"])
    required_steps = []
    for row in steps:
        critical_uses = [rui_id for rui_id in row["uses"] if rui_id in required_ids]
        if critical_uses:
            required_steps.append({**row, "uses": critical_uses})
    return {
        "instance_id": f"sara_subproof_{source['case_id']}_{target['predicate']}",
        "source_family": (
            "sara_target_sliced_subproof"
        ),
        "source_id": source["case_id"],
        "source_title": f"SARA subproof case {source['case_id']}",
        "answer_type": "short_answer_judged",
        "answer": "free_response",
        "decision_set": ["free_response"],
        "decision_option_definitions": {"free_response": "Submit the requested legal calculation."},
        "question": candidate["question"],
        "response_format_note": "State the requested result and a concise derivation.",
        "short_answer": answer,
        "canonical_response": f"{target['label'].capitalize()}: {answer}. " + " ".join(candidate["structured_solution"]["derivation"]),
        "source_text": task_source_text(source, candidate, policy=policy),
        "rius": rius,
        "required_rui_ids": required_ids,
        "distribution_rui_ids": all_ids,
        "outcome_critical_rui_ids": required_ids,
        "derivation_active_masked_rui_ids": masked_ids,
        "reasoning_steps": steps,
        "required_decision_steps": required_steps,
        "why_sufficient": (
            "The outcome-critical premises determine the selected executable target; masked premises remain "
            "active in the faithful derivation but are not required for final-answer correctness."
        ),
        "structured_answer_label": {
            "schema": "sara_subproof_answer_v1",
            "case_id": source["case_id"],
            "target_predicate": target["predicate"],
            "target_goal": target["goal"],
            "answer_kind": target["answer_kind"],
            "answer": answer,
            "derivation": candidate["structured_solution"]["derivation"],
        },
        "judge_rubric": {
            "passing_score": 1.0,
            "criteria": [
                {"name": "answer", "weight": 0.6, "description": "States the exact requested result."},
                {"name": "derivation", "weight": 0.4, "description": "Correctly applies the supplied premises."},
            ],
        },
        "disqualifying_errors": [
            "The requested result is incorrect.",
            "The derivation relies on a legal fact or rule not supplied in the task.",
        ],
        "primary_answer_field": "written_answer",
        "fallback_answer_fields": ["rationale", "answer"],
        "dataset_variant": (
            "sara_target_sliced_subproof_v7"
        ),
    }


def candidate_schema(policy: str = TARGET_SLICED_POLICY) -> dict[str, Any]:
    validate_policy(policy)
    riu_schema = {
        "rui_id": "riu_01",
        "source_span": "exact contiguous source span",
        "normalized_claim": "one source-grounded premise",
        "rui_type": "case_fact|legal_rule|threshold|rate_rule",
        "supports_goals": ["exact goal copied from successful_trace_goals"],
    }
    riu_schema.update(
        {
            "provenance_type": "explicit_source|oracle_derived",
            "source_span": (
                "exact source span for explicit_source; normalized_claim for oracle_derived"
            ),
            "oracle_basis": [
                "empty for explicit_source; exact trace goals or reference clauses for oracle_derived"
            ],
        }
    )
    riu_schema["criticality_class"] = (
        "outcome_critical|derivation_active_masked"
    )
    riu_schema["reasoning_role"] = (
        "case_input|applicability_condition|exception_blocker|legal_rule|calculation_rule"
    )
    return {
        "selected_target_id": "one target_id from target_options",
        "question": "programmatically replaced with the selected target question",
        "rius": [riu_schema],
        "reasoning_steps": [
            {"uses": ["riu_01"], "claim": "derived step", "judge_focus": "required conclusion"}
        ],
        "structured_solution": {
            "target_answer": "exact selected target answer",
            "derivation": ["four or more explicit legal or arithmetic steps with RIU IDs"],
        },
    }


def construction_prompt(policy: str) -> str:
    validate_policy(policy)
    return TARGET_SLICED_CONSTRUCTION_PROMPT


def verifier_prompt(policy: str) -> str:
    validate_policy(policy)
    return TARGET_SLICED_VERIFIER_PROMPT


def construction_reference(source: dict[str, Any], policy: str) -> str | None:
    return str(source.get("construction_reference") or "")


def validate_policy(policy: str) -> None:
    if policy not in GENERATION_POLICIES:
        raise ValueError(f"Unknown SARA generation policy: {policy}")


def valid_counterfactual_witness(criticality: dict[str, Any]) -> bool:
    witness = criticality.get("counterfactual_witness")
    if not isinstance(witness, dict):
        return False
    return all(
        str(witness.get(field) or "").strip()
        for field in (
            "alternative_assignment",
            "resulting_answer_or_branch",
            "why_remaining_rius_allow_it",
        )
    )


def solver_rius(candidate: dict[str, Any]) -> list[dict[str, str]]:
    """Expose task-facing RIU content without construction-only trace metadata."""

    return [
        {
            "rui_id": str(row.get("rui_id") or ""),
            "source_span": str(row.get("source_span") or ""),
            "normalized_claim": str(row.get("normalized_claim") or ""),
            "rui_type": str(row.get("rui_type") or "rule_or_fact"),
            "reasoning_role": str(row.get("reasoning_role") or ""),
        }
        for row in candidate.get("rius", [])
        if isinstance(row, dict)
    ]


def seed_riu(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("oracle_basis", None)
    return result


def task_source_text(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    policy: str,
) -> str:
    text = str(source["natural_source"])
    implicit = [
        str(row.get("source_span") or "").strip()
        for row in candidate.get("rius", [])
        if isinstance(row, dict) and row.get("provenance_type") == "oracle_derived"
    ]
    if not implicit:
        return text
    conditions = "\n".join(f"- {value}" for value in implicit if value)
    return f"{text}\n\nTask-specific case conditions:\n{conditions}"


def source_for_model(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": source["case_id"],
        "case_facts": source["case_facts"],
        "original_question": source["original_question"],
        "statutes": source["statutes"],
    }


def target_question(target: dict[str, Any]) -> str:
    raw_subject = str(target.get("subject") or "the taxpayer")
    year = str(target.get("year") or "the stated year")
    label = target["label"]
    joint_return = joint_return_subject(raw_subject)
    action = "determine" if target["answer_kind"] == "category" else "calculate"
    if joint_return:
        requested = f"{action} the {label} for {joint_return} for {year}"
    else:
        subject = human_name(raw_subject)
        requested = f"{action} {subject}'s {label} for {year}"
    return (
        f"Using only the supplied facts and rules, {requested}. State the result and give a concise derivation "
        "showing how the applicable conditions and calculations determine it."
    )


def taxpayer_and_year(source: dict[str, Any]) -> tuple[str, str]:
    question = str(source.get("original_question") or "")
    name = re.search(r"how much tax does\s+([A-Za-z]+)", question, flags=re.I)
    year = re.search(r"\b(?:19|20)\d{2}\b", question)
    return (name.group(1) if name else "the taxpayer", year.group(0) if year else "the stated year")


def human_name(value: str) -> str:
    text = clean_prolog_atom(value).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "the taxpayer"


def joint_return_subject(value: str) -> str | None:
    """Render internal joint-return identifiers as task-facing entities."""

    match = re.fullmatch(r"r_([a-z][a-z0-9]*)_([a-z][a-z0-9]*)_(?:19|20)\d{2}", clean_prolog_atom(value))
    if not match:
        return None
    first, second = (human_name(part) for part in match.groups())
    return f"{first} and {second}'s joint return"


def only_restates_target_metadata(claim: str, target: dict[str, Any]) -> bool:
    """Identify premise-shaped restatements of the subject or target year."""

    normalized = normalize_text(claim).strip(" .,:;!?")
    year = re.escape(str(target.get("year") or ""))
    if year and re.fullmatch(
        rf"(?:the )?(?:relevant )?(?:tax )?year (?:is|was) {year}", normalized
    ):
        return True
    subject = normalize_text(human_name(str(target.get("subject") or "")))
    return bool(
        subject
        and re.fullmatch(
            rf"(?:the )?(?:relevant )?(?:taxpayer|subject|person) (?:is|was) {re.escape(subject)}",
            normalized,
        )
    )


def task_facing_text_issue(value: str) -> str | None:
    """Reject construction syntax and terminology from task-facing prose."""

    text = str(value or "")
    if re.search(r"\b[a-z][a-z0-9_]*\s*\([^)]*\)", text):
        return "uses formal predicate-call syntax instead of ordinary legal language"
    if re.search(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b", text):
        return "uses an underscore-delimited formal identifier"
    if re.search(r"\b(?:oracle|predicate|program|prolog|proof|trace)\b", text, re.I):
        return "uses construction terminology instead of ordinary legal language"
    if re.search(
        r"\b(?:those|these|above|aforementioned|foregoing)\s+"
        r"(?:amounts|conditions|facts|inputs|premises|rules)\b",
        text,
        re.I,
    ):
        return "is not self-contained because it refers anaphorically to other premises"
    if re.match(r"\s*because\b", text, re.I):
        return "bundles a supporting premise and its derived conclusion as a causal explanation"
    return None


def claims_substantially_overlap(left: str, right: str) -> bool:
    """Catch a narrower RIU that repeats content already bundled in a peer."""

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "were",
        "where",
        "with",
    }
    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold())) - stopwords
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold())) - stopwords
    smaller = min(len(left_tokens), len(right_tokens))
    return bool(
        smaller >= 3 and len(left_tokens & right_tokens) / smaller >= 0.8
    )


def answer_matches(observed: Any, target: dict[str, Any]) -> bool:
    if target["answer_kind"] == "money":
        right = decimal_amount(target["answer"])
        if right is None:
            return False
        return any(abs(value - right) <= Decimal("0.01") for value in numeric_values(observed))
    return canonical_category(observed) == canonical_category(target["answer"])


def canonical_category(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def directly_states_target(claim: str, target: dict[str, Any]) -> bool:
    """Distinguish an instance conclusion from a generally applicable amount rule."""

    normalized = normalize_text(claim)
    subject = normalize_text(human_name(str(target.get("subject") or "")))
    year = str(target.get("year") or "").strip()
    label_tokens = [
        token
        for token in normalize_text(target.get("label")).split()
        if token not in {"individual", "personal", "the"}
    ]
    subject_present = bool(subject and subject != "the taxpayer" and subject in normalized)
    year_present = bool(year and year in normalized)
    label_present = bool(label_tokens and all(token in normalized for token in label_tokens))
    conclusion_language = bool(
        re.search(r"\b(?:is|was|equals?|amount(?:s)? to|owes?|has)\b", normalized)
    )
    return subject_present and year_present and label_present and conclusion_language


def explicit_support_overstatement(claim: str, goal: str) -> str | None:
    """Catch recurring cases where a weak source fact is mapped to a stronger condition."""

    parsed = parse_goal(goal)
    if parsed is None:
        return None
    predicate = parsed[0]
    normalized = normalize_text(claim)
    if predicate == "could_have_filed_joint_return" and not (
        "joint" in normalized and ("file" in normalized or "return" in normalized)
    ):
        return "overstates its source as establishing joint-return eligibility"
    if predicate == "furnishes_half_cost" and not (
        "half" in normalized and ("cost" in normalized or "furnish" in normalized)
    ):
        return "overstates household maintenance as establishing the half-cost condition"
    return None


def numeric_values(value: Any) -> list[Decimal]:
    values = []
    for match in re.findall(r"(?<![A-Za-z0-9])-?\$?\s*\d[\d,]*(?:\.\d+)?", str(value or "")):
        parsed = decimal_amount(match)
        if parsed is not None:
            values.append(parsed)
    return values
