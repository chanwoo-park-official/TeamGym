from decimal import Decimal

from teamgym.data.legal.sara.reference import extract_reported_total, round_dollars

from teamgym.data.legal.sara.subproof_generation import (
    TARGET_SLICED_POLICY,
    answer_matches,
    blind_solution_audit as subproof_blind_solution_audit,
    candidate_to_seed,
    candidate_passes,
    claims_substantially_overlap,
    directly_states_target,
    explicit_support_overstatement,
    goal_has_deriving_clause,
    goal_requires_explicit_rule_riu,
    is_deterministic_intermediate_goal,
    numeric_values,
    only_restates_target_metadata,
    parse_goal,
    parse_prolog_clauses,
    proof_construction_requirements,
    prepare_candidate,
    structural_audit as subproof_structural_audit,
    target_proof_slice,
    target_options,
    task_facing_text_issue,
)


def test_reference_uses_nearest_dollar_rounding() -> None:
    assert extract_reported_total("Tax result: 8581.70\n") == Decimal("8581.70")
    assert round_dollars(Decimal("8581.50")) == Decimal("8582")
    assert round_dollars(Decimal("8581.49")) == Decimal("8581")


def test_subproof_target_extraction_prefers_dependent_numeric_targets() -> None:
    execution = {
        "trace": [
            'filing_status("Alice",2017,surviving_spouse)',
            'standard_deduction("Alice",2017,6000)',
            'taxable_income("Alice",2017,226422)',
            'income_tax("Alice",2017,67040.42)',
        ]
    }
    source = {"original_question": "How much tax does Alice have to pay in 2017?"}

    options = target_options(execution, source)

    assert options[0]["predicate"] == "income_tax"
    assert options[0]["answer"] == "67040.42"
    assert "Alice's individual income tax for 2017" in options[0]["question"]


def test_subproof_target_extraction_uses_taxpayer_when_goal_starts_with_year() -> None:
    execution = {"trace": ["taxable_income(2017,22600)"]}
    source = {"original_question": "How much tax does Alice have to pay in 2017?"}

    options = target_options(execution, source)

    assert options[0]["subject"] == "Alice"
    assert "Alice's taxable income for 2017" in options[0]["question"]


def test_subproof_goal_parser_preserves_nested_arguments() -> None:
    assert parse_goal('user:wage("Alice","Bob",2017,5012,agricultural,cash)') == (
        "wage",
        ['"Alice"', '"Bob"', "2017", "5012", "agricultural", "cash"],
    )


def test_target_proof_slice_excludes_sibling_tax_objective() -> None:
    reference = """
    gross_income(alice, 2017, 10000).
    income_tax(P, Y, Tax) :- gross_income(P, Y, Income), Tax is Income * 0.1.
    paid_wages(alice, 2017, 7000).
    futa_tax(P, Y, Tax) :- paid_wages(P, Y, Wages), Tax is Wages * 0.06.
    tax(P, Y, Total) :- income_tax(P, Y, IncomeTax), futa_tax(P, Y, Futa),
                        Total is IncomeTax + Futa.
    """
    execution = {
        "trace": [
            "gross_income(alice,2017,10000)",
            "income_tax(alice,2017,1000)",
            "paid_wages(alice,2017,7000)",
            "futa_tax(alice,2017,420)",
            "tax(alice,2017,1420)",
        ]
    }
    target = {
        "goal": "income_tax(alice,2017,1000)",
        "trace_position": 1,
        "target_id": "target_01",
    }

    proof_slice = target_proof_slice(execution, target, reference)

    assert proof_slice["successful_trace_goals"] == [
        "gross_income(alice,2017,10000)",
        "income_tax(alice,2017,1000)",
    ]
    assert "futa_tax/3" not in proof_slice["predicate_closure"]
    assert "paid_wages/3" not in proof_slice["predicate_closure"]
    assert parse_goal("date_leq(date(2015,1,24),date(2019,5,5))") == (
        "date_leq",
        ["date(2015,1,24)", "date(2019,5,5)"],
    )


def test_proof_construction_requirements_separate_inputs_rules_and_arithmetic() -> None:
    requirements = proof_construction_requirements(
        [
            "gross_income(alice,2017,10000)",
            "sum_list([10000],10000)",
            "taxable_income(alice,2017,7000)",
            "not:itemizes(alice,2017)",
        ],
        [
            "gross_income(alice,2017,10000)",
            "sum_list(Values,Total) :- sumlist(Values,Total)",
            "taxable_income(P,Y,Amount) :- gross_income(P,Y,Gross), Amount is Gross - 3000",
        ],
        ["not:itemizes(alice,2017)"],
    )

    assert requirements["required_leaf_goals"] == [
        {"goal": "gross_income(alice,2017,10000)"}
    ]
    assert [row["goal"] for row in requirements["required_rule_goals"]] == [
        "taxable_income(alice,2017,7000)"
    ]
    assert requirements["deterministic_derivation_goals"] == [
        "sum_list([10000],10000)"
    ]
    assert requirements["required_exception_blockers"] == [
        "not:itemizes(alice,2017)"
    ]


def test_target_proof_slice_grounds_repeated_predicates_to_the_needed_person() -> None:
    reference = """
    gross_income("Alice",2017,33200).
    gross_income("Bob",2017,0).
    dependent_of(Dep,Taxpayer,Year) :- gross_income(Dep,Year,GI), GI =:= 0.
    deduction(Taxpayer,Year,4400) :- dependent_of("Bob",Taxpayer,Year).
    """
    execution = {
        "trace": [
            'gross_income("Alice",2017,33200)',
            'gross_income("Bob",2017,0)',
            'dependent_of("Bob","Alice",2017)',
            'deduction("Alice",2017,4400)',
        ]
    }
    target = {
        "goal": 'deduction("Alice",2017,4400)',
        "trace_position": 3,
        "target_id": "target_01",
    }

    proof_slice = target_proof_slice(execution, target, reference)

    assert 'gross_income("Bob",2017,0)' in proof_slice["successful_trace_goals"]
    assert 'gross_income("Alice",2017,33200)' not in proof_slice["successful_trace_goals"]


def test_target_proof_slice_preserves_facts_that_defeat_an_exception() -> None:
    reference = r"""
    married("Alice","Bob").
    filing_separate("Alice",2017).
    filing_separate("Bob",2017).
    takes_standard_deduction("Bob",2017).
    spouse(P,S) :- married(P,S).
    filing_status(P,Y,married_separate) :- filing_separate(P,Y), spouse(P,_).
    basic_standard_deduction(2017,married_separate,3000).
    standard_deduction(P,Y,0) :-
        filing_status(P,Y,married_separate), spouse(P,S),
        filing_separate(S,Y), \+ takes_standard_deduction(S,Y), !.
    standard_deduction(P,Y,Amount) :-
        filing_status(P,Y,married_separate),
        basic_standard_deduction(Y,married_separate,Amount).
    """
    execution = {
        "trace": [
            'filing_separate("Alice",2017)',
            'married("Alice","Bob")',
            'spouse("Alice","Bob")',
            'filing_status("Alice",2017,married_separate)',
            'filing_separate("Bob",2017)',
            'takes_standard_deduction("Bob",2017)',
            'basic_standard_deduction(2017,married_separate,3000)',
            'standard_deduction("Alice",2017,3000)',
        ]
    }
    target = {
        "goal": 'standard_deduction("Alice",2017,3000)',
        "trace_position": 7,
        "target_id": "target_01",
    }

    proof_slice = target_proof_slice(execution, target, reference)

    blockers = proof_slice["proof_roles"]["exception_blocker_goals"]
    assert blockers == ['takes_standard_deduction("Bob",2017)']
    assert 'filing_separate("Bob",2017)' not in proof_slice["successful_trace_goals"]


def test_joint_return_target_question_hides_internal_identifier() -> None:
    execution = {
        "trace": ["personal_exemption_deduction(r_alice_bob_2015,4000)"]
    }
    source = {"original_question": "How much tax does Alice have to pay in 2015?"}

    option = target_options(execution, source)[0]

    assert "R alice bob" not in option["question"]
    assert "Alice and Bob's joint return" in option["question"]


def test_target_year_is_query_metadata_not_an_riu() -> None:
    target = {"subject": "Alice", "year": "2014"}

    assert only_restates_target_metadata("The relevant tax year is 2014.", target)
    assert not only_restates_target_metadata(
        "Alice's spouse died in 2014.", target
    )


def test_deterministic_intermediate_stays_in_reasoning_steps() -> None:
    clauses = parse_prolog_clauses(
        "birth_year(bob,2014). age(P,Y,A) :- birth_year(P,B), A is Y-B."
    )

    assert is_deterministic_intermediate_goal("age(bob,2017,3)", clauses)
    assert not goal_requires_explicit_rule_riu("age(bob,2017,3)", clauses)
    assert not is_deterministic_intermediate_goal("birth_year(bob,2014)", clauses)


def test_task_facing_riu_text_rejects_formal_syntax_and_jargon() -> None:
    assert task_facing_text_issue("maintains_household_for_dependent(bob,2015).")
    assert task_facing_text_issue("The proof program treats Bob as unmarried.")
    assert task_facing_text_issue("Bob has joint_surviving status.")
    assert task_facing_text_issue("Those conditions make Bob a surviving spouse.")
    assert task_facing_text_issue(
        "Because Alice and Bob were married, Bob is Alice's spouse."
    )
    assert task_facing_text_issue("Bob maintained a household in 2015.") is None


def test_same_goal_claim_overlap_catches_narrow_restatement() -> None:
    assert claims_substantially_overlap(
        "Bob maintained the home where he and Charlie lived in 2015.",
        "Charlie lived with Bob in 2015.",
    )
    assert not claims_substantially_overlap(
        "Alice died in 2014.",
        "Alice was Bob's spouse.",
    )


def test_target_proof_slice_materializes_successful_negative_conditions() -> None:
    reference = r"""
    not_married(alice,2020).
    surviving_spouse(_,_) :- fail.
    filing_status(P,Y,unmarried) :- not_married(P,Y), \+ surviving_spouse(P,Y).
    """
    execution = {
        "trace": [
            "not_married(alice,2020)",
            "filing_status(alice,2020,unmarried)",
        ]
    }
    target = {
        "goal": "filing_status(alice,2020,unmarried)",
        "trace_position": 1,
        "target_id": "target_01",
    }

    proof_slice = target_proof_slice(execution, target, reference)

    assert proof_slice["proof_roles"]["required_negative_conditions"] == [
        "not:surviving_spouse(alice,2020)"
    ]
    assert "not:surviving_spouse(alice,2020)" in proof_slice["successful_trace_goals"]


def test_general_numeric_rule_is_not_direct_target_leakage() -> None:
    target = {
        "subject": "Alice",
        "year": "2020",
        "label": "standard deduction",
    }

    assert not directly_states_target(
        "For 2018 through 2025, an unmarried filer has a basic standard deduction of $12,000.",
        target,
    )
    assert directly_states_target(
        "Alice's standard deduction for 2020 is $12,000.",
        target,
    )


def test_explicit_fact_cannot_overstate_a_stronger_formal_condition() -> None:
    assert explicit_support_overstatement(
        "Alice was Bob's spouse.",
        "could_have_filed_joint_return(bob,2014)",
    )
    assert explicit_support_overstatement(
        "Alice maintained the home.",
        "furnishes_half_cost(alice,2017)",
    )
    assert not explicit_support_overstatement(
        "Alice and Bob filed a joint return for 2014.",
        "could_have_filed_joint_return(bob,2014)",
    )


def test_oracle_leaf_distinguishes_formal_fact_from_derived_intermediate() -> None:
    clauses = parse_prolog_clauses(
        "not_married(alice,2020). "
        "filing_status(P,Y,unmarried) :- not_married(P,Y)."
    )

    assert not goal_has_deriving_clause("not_married(alice,2020)", clauses)
    assert goal_has_deriving_clause("filing_status(alice,2020,unmarried)", clauses)


def test_subproof_answer_matching_accepts_money_and_normalized_category() -> None:
    money = {"answer_kind": "money", "answer": "67040.42"}
    category = {"answer_kind": "category", "answer": "married_joint"}

    assert answer_matches("$67,040.42", money)
    assert answer_matches("Alice's individual income tax for 2019 is $67,040.42.", money)
    assert answer_matches("Married joint", category)
    assert not answer_matches("67041", money)


def test_subproof_numeric_parser_does_not_confuse_zero_with_a_year() -> None:
    assert numeric_values("Alice takes the standard deduction for 2019.") == [Decimal("2019")]
    assert numeric_values("Alice's 2019 income tax is $0.") == [Decimal("2019"), Decimal("0")]


def test_subproof_blind_audit_requires_answer_and_derivation() -> None:
    target = {"answer_kind": "money", "answer": "67040.42"}

    accepted = subproof_blind_solution_audit(
        {"answer": "67040.42", "derivation": ["one", "two", "three"]}, target
    )
    rejected = subproof_blind_solution_audit(
        {"answer": "67040.42", "derivation": ["only one"]}, target
    )

    assert accepted == {"accepted": True, "issues": []}
    assert rejected["accepted"] is False


def test_target_sliced_generation_can_defer_a_single_blind_failure() -> None:
    candidate = {"rius": [{"rui_id": "riu_01"}, {"rui_id": "riu_02"}]}
    witness = {
        "alternative_assignment": "the premise is unknown",
        "resulting_answer_or_branch": "the result is indeterminate",
        "why_remaining_rius_allow_it": "no remaining RIU supplies the premise",
    }
    verification = {
        "accepted": True,
        "oracle_consistent": True,
        "collectively_sufficient": True,
        "atomic_rius": True,
        "reasoning_roles_valid": True,
        "applicability_complete": True,
        "exception_resolution_complete": True,
        "target_unique": True,
        "proof_correct": True,
        "riu_criticality": [
            {
                "rui_id": "riu_01",
                "removal_effect": "indeterminate",
                "criticality_class": "outcome_critical",
                "counterfactual_witness": witness,
            },
            {
                "rui_id": "riu_02",
                "removal_effect": "different_amount",
                "criticality_class": "outcome_critical",
                "counterfactual_witness": witness,
            },
        ],
    }
    for riu in candidate["rius"]:
        riu["criticality_class"] = "outcome_critical"

    assert not candidate_passes(
        {"accepted": True},
        {"accepted": False},
        verification,
        candidate,
        policy=TARGET_SLICED_POLICY,
    )
    assert candidate_passes(
        {"accepted": True},
        {"accepted": False},
        verification,
        candidate,
        policy=TARGET_SLICED_POLICY,
        require_blind=False,
    )


def test_target_sliced_acceptance_distinguishes_masked_from_off_path() -> None:
    candidate = {
        "rius": [
            {"rui_id": "riu_01", "criticality_class": "outcome_critical"},
            {"rui_id": "riu_02", "criticality_class": "outcome_critical"},
            {"rui_id": "riu_03", "criticality_class": "derivation_active_masked"},
        ]
    }
    witness = {
        "alternative_assignment": "the premise is omitted",
        "resulting_answer_or_branch": "a different result remains possible",
        "why_remaining_rius_allow_it": "no other RIU supplies the premise",
    }
    verification = {
        "accepted": True,
        "oracle_consistent": True,
        "collectively_sufficient": True,
        "atomic_rius": True,
        "reasoning_roles_valid": True,
        "applicability_complete": True,
        "exception_resolution_complete": True,
        "target_unique": True,
        "proof_correct": True,
        "riu_criticality": [
            {
                "rui_id": "riu_01",
                "removal_effect": "indeterminate",
                "criticality_class": "outcome_critical",
                "counterfactual_witness": witness,
            },
            {
                "rui_id": "riu_02",
                "removal_effect": "different_amount",
                "criticality_class": "outcome_critical",
                "counterfactual_witness": witness,
            },
            {
                "rui_id": "riu_03",
                "removal_effect": "masked_intermediate_change",
                "criticality_class": "derivation_active_masked",
                "counterfactual_witness": witness,
            },
        ],
    }

    assert candidate_passes(
        {"accepted": True},
        {"accepted": True},
        verification,
        candidate,
        policy=TARGET_SLICED_POLICY,
    )
    verification["riu_criticality"][2]["criticality_class"] = "off_path"
    verification["riu_criticality"][2]["removal_effect"] = "still_solvable"
    assert not candidate_passes(
        {"accepted": True},
        {"accepted": True},
        verification,
        candidate,
        policy=TARGET_SLICED_POLICY,
    )


def test_target_sliced_seed_uses_only_outcome_critical_rius_as_sufficient() -> None:
    candidate = {
        "question": "Calculate the result.",
        "rius": [
            {"rui_id": "riu_01", "criticality_class": "outcome_critical"},
            {"rui_id": "riu_02", "criticality_class": "outcome_critical"},
            {"rui_id": "riu_03", "criticality_class": "derivation_active_masked"},
        ],
        "reasoning_steps": [
            {"uses": ["riu_01", "riu_02"], "claim": "critical step"},
            {"uses": ["riu_03"], "claim": "masked step"},
        ],
        "structured_solution": {"derivation": ["one", "two", "three", "four"]},
    }
    source = {"case_id": "case", "natural_source": "source"}
    target = {
        "predicate": "income_tax",
        "label": "individual income tax",
        "goal": "income_tax(alice,2017,100)",
        "answer_kind": "money",
        "answer": "100",
    }

    seed = candidate_to_seed(
        source, candidate, target, policy=TARGET_SLICED_POLICY
    )

    assert seed["required_rui_ids"] == ["riu_01", "riu_02"]
    assert seed["distribution_rui_ids"] == ["riu_01", "riu_02", "riu_03"]
    assert seed["derivation_active_masked_rui_ids"] == ["riu_03"]
    assert len(seed["required_decision_steps"]) == 1


def test_target_sliced_structural_audit_accepts_prefixed_goal_ids() -> None:
    target = {
        "target_id": "target_01",
        "question": "Calculate the result.",
        "answer_kind": "money",
        "answer": "100",
    }
    candidate = {
        "selected_target_id": "target_01",
        "question": target["question"],
        "rius": [
            {
                "rui_id": f"riu_{index:02d}",
                "source_span": "A supplied premise.",
                "normalized_claim": "A supplied premise.",
                "provenance_type": "explicit_source",
                "supports_goals": ["goal:result(alice,2017,100)"],
                "criticality_class": "outcome_critical",
                "reasoning_role": "case_input",
            }
            for index in range(1, 7)
        ],
        "reasoning_steps": [
            {
                "uses": [f"riu_{index:02d}" for index in range(1, 7)],
                "claim": f"step {step}",
            }
            for step in range(1, 5)
        ],
        "structured_solution": {
            "target_answer": "100",
            "derivation": ["one", "two", "three", "four"],
        },
    }
    audit = subproof_structural_audit(
        candidate,
        {
            "natural_source": "A supplied premise.",
            "construction_reference": "result(alice,2017,100).",
        },
        {"trace": ["result(alice,2017,100)"]},
        [target],
        policy=TARGET_SLICED_POLICY,
    )

    assert audit["accepted"] is True


def test_target_sliced_preparation_converts_formal_clause_provenance() -> None:
    clause = "result(P, Y, Value) :- input(P, Y, Value)"
    candidate = {
        "selected_target_id": "target_01",
        "rius": [
            {
                "rui_id": "riu_01",
                "source_span": clause,
                "normalized_claim": "The result equals the supplied input.",
                "provenance_type": "explicit_source",
                "oracle_basis": [],
            }
        ],
    }
    prepare_candidate(
        candidate,
        {
            "natural_source": "An unrelated natural-language source.",
            "construction_reference": clause,
        },
        [{"target_id": "target_01", "question": "Calculate the result."}],
        policy=TARGET_SLICED_POLICY,
    )

    riu = candidate["rius"][0]
    assert riu["provenance_type"] == "oracle_derived"
    assert riu["source_span"] == riu["normalized_claim"]
    assert riu["oracle_basis"] == [f"clause:{clause}"]
