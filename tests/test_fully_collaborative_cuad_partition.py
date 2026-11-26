from __future__ import annotations

import random

from teamgym.data.legal.cuad.partition import (
    Chunk,
    RUIRange,
    _balanced_participant_for_neutral_chunk,
    _best_participant_for_chunk,
    _retain_one_smallest_sufficient_set,
    build_full_document_partition,
)


def test_partition_retains_only_one_audited_smallest_sufficient_set() -> None:
    ranges = [
        RUIRange(
            unit_id=f"span_{index:02d}",
            source_span_index=index,
            clean_text=text,
            raw_start=index * 10,
            raw_end=index * 10 + len(text),
            raw_text=text,
            legal_feature=f"feature_{index}",
            summary=f"summary_{index}",
            section_key="section",
        )
        for index, text in enumerate(("Alpha", "Beta", "Gamma"), start=1)
    ]
    trial = {
        "audit": {
            "smallest_sufficient_span_sets": [
                {"unit_ids": ["span_01", "span_02"]},
                {"unit_ids": ["span_02", "span_03"]},
            ]
        }
    }

    retained_ranges, retained_sets = _retain_one_smallest_sufficient_set(trial, ranges)

    assert [rui.unit_id for rui in retained_ranges] == ["span_01", "span_02"]
    assert retained_sets == [["span_01", "span_02"]]


def test_full_document_partition_respects_rui_split_and_coverage() -> None:
    spans = [
        "Alpha must maintain insurance coverage with a five million dollar limit.",
        "Alpha must provide cancellation notice at least thirty days in advance.",
        "Beta may audit books and records once per calendar year.",
        "Audit costs shift if an underpayment exceeds five percent.",
    ]
    context = (
        "1. Insurance\n"
        f"{spans[0]}\n\n"
        "This paragraph is related insurance background without critical terms.\n\n"
        "2. Notice\n"
        f"{spans[1]}\n\n"
        "General notice routing language appears here.\n\n"
        "3. Audit Scope\n"
        f"{spans[2]}\n\n"
        "General audit cooperation language appears here.\n\n"
        "4. Audit Costs\n"
        f"{spans[3]}\n\n"
        "Final boilerplate sentence closes the contract."
    )
    starts = [context.index(span) for span in spans]
    candidate = {
        "source_id": "synthetic__insurance",
        "contract_index": 1,
        "contract_title": "Synthetic Contract",
        "contract_text": context,
        "positive_spans": [
            {"span_index": idx, "text": span, "answer_start": starts[idx - 1]}
            for idx, span in enumerate(spans, start=1)
        ],
    }
    trial = {
        "instance_id": "trial_001",
        "candidate_index": 1,
        "source_id": "synthetic__insurance",
        "label_family": "Insurance",
        "selection": {
            "selected_span_basis": [
                {
                    "unit_id": f"span_{idx:02d}",
                    "source_span_index": idx,
                    "text": span,
                    "legal_feature": feature,
                }
                for idx, (span, feature) in enumerate(
                    zip(
                        spans,
                        [
                            "insurance limit",
                            "cancellation notice",
                            "audit scope",
                            "audit cost shifting",
                        ],
                    ),
                    start=1,
                )
            ]
        },
        "plan": {
            "public_task": {
                "objective": "Pick the matching profile.",
                "decision_set": ["decision_a", "decision_b"],
                "decision_option_definitions": {
                    "decision_a": "Insurance, cancellation notice, audit scope, and audit cost-shifting terms are all present.",
                    "decision_b": "Only insurance and cancellation notice terms are present.",
                },
            },
            "evidence_units": [
                {"unit_id": f"span_{idx:02d}", "summary": feature}
                for idx, feature in enumerate(
                    ["insurance limit", "cancellation notice", "audit scope", "audit cost shifting"],
                    start=1,
                )
            ],
            "oracle": {"answer": "decision_a"},
        },
        "audit": {
            "smallest_sufficient_span_sets": [
                {"unit_ids": ["span_01", "span_02", "span_03", "span_04"]}
            ]
        },
    }

    partition = build_full_document_partition(
        trial,
        candidate,
        rng=random.Random(3),
        seed=3,
        strategy="one_each",
        role_style="feature",
        target_words=18,
        min_words=5,
        soft_max_words=30,
        hard_max_words=45,
        min_participants=4,
    )

    assert partition["validation"]["valid"] is True
    assert partition["dataset"] == "cuad"
    assert partition["common_datapoint"]["dataset"] == "cuad"
    assert partition["common_datapoint"]["format_validation"]["valid"] is True
    assert partition["common_datapoint"]["decision_set"] == ["decision_a", "decision_b"]
    assert [row["participant_id"] for row in partition["common_datapoint"]["participant_directory"]] == [
        "participant_a",
        "participant_b",
        "participant_c",
        "participant_d",
    ]
    assert partition["common_datapoint"]["generation_metadata"]["no_synthetic_fallback"] is True
    assert len(partition["common_datapoint"]["private_context_units"]) == len(partition["document_units"])
    assert len(partition["participants"]) == 4
    assert partition["document_units"][0]["start"] == 0
    assert partition["document_units"][-1]["end"] == len(context)
    participant_rui_sets = partition["validation"]["metrics"]["participant_rui_sets"]
    assert all(len(values) == 1 for values in participant_rui_sets.values())


def test_full_document_partition_balances_neutral_chunks_for_extra_participants() -> None:
    spans = [
        "Provider may use Client trademarks only for marketing approved by Client.",
        "Provider may display Client website screenshots as examples of completed work.",
    ]
    context = (
        "Opening recital identifies the parties and the service relationship.\n\n"
        "The agreement describes ordinary website development services and project scheduling.\n\n"
        f"{spans[0]}\n\n"
        "Payment timing, invoice review, and late payment mechanics appear in this section.\n\n"
        "The parties include routine confidentiality language unrelated to brand usage.\n\n"
        f"{spans[1]}\n\n"
        "Termination notice and transition support language appears near the end.\n\n"
        "Signature blocks and administrative contact details close the agreement."
    )
    starts = [context.index(span) for span in spans]
    candidate = {
        "source_id": "synthetic__license",
        "contract_index": 2,
        "contract_title": "Synthetic License Contract",
        "contract_text": context,
        "positive_spans": [
            {"span_index": idx, "text": span, "answer_start": starts[idx - 1]}
            for idx, span in enumerate(spans, start=1)
        ],
    }
    trial = {
        "instance_id": "trial_balanced_extra_participant",
        "candidate_index": 2,
        "source_id": "synthetic__license",
        "label_family": "License Grant",
        "selection": {
            "selected_span_basis": [
                {
                    "unit_id": "span_01",
                    "source_span_index": 1,
                    "text": spans[0],
                    "legal_feature": "trademark marketing license",
                },
                {
                    "unit_id": "span_02",
                    "source_span_index": 2,
                    "text": spans[1],
                    "legal_feature": "portfolio display license",
                },
            ]
        },
        "plan": {
            "public_task": {
                "objective": "Pick the matching permission profile.",
                "decision_set": ["decision_a", "decision_b"],
                "decision_option_definitions": {
                    "decision_a": "Both trademark marketing and portfolio display permissions are present.",
                    "decision_b": "Only trademark marketing permission is present.",
                },
            },
            "evidence_units": [
                {"unit_id": "span_01", "summary": "Trademark marketing permission."},
                {"unit_id": "span_02", "summary": "Portfolio display permission."},
            ],
            "oracle": {"answer": "decision_a"},
        },
        "audit": {
            "smallest_sufficient_span_sets": [{"unit_ids": ["span_01", "span_02"]}]
        },
    }

    partition = build_full_document_partition(
        trial,
        candidate,
        rng=random.Random(11),
        seed=11,
        strategy="one_each",
        role_style="feature",
        target_words=11,
        min_words=5,
        soft_max_words=18,
        hard_max_words=28,
        min_participants=3,
    )

    counts = {
        participant["participant_id"]: len(participant.get("private_doc_unit_ids") or [])
        for participant in partition["participants"]
    }

    assert partition["validation"]["valid"] is True
    assert counts["p3"] > 0
    assert max(counts.values()) - min(counts.values()) <= 1
    assert partition["validation"]["metrics"]["participant_doc_unit_count_gap"] <= 1
    assert any(
        check["name"] == "every_participant_has_private_context" and check["passed"]
        for check in partition["validation"]["checks"]
    )


def test_assignment_score_prefers_matching_reviewer() -> None:
    participants = [
        {
            "participant_id": "p1",
            "role_terms": ["insurance", "coverage", "limit"],
            "private_word_count": 0,
        },
        {
            "participant_id": "p2",
            "role_terms": ["audit", "records", "cost"],
            "private_word_count": 0,
        },
    ]
    chunk = Chunk(
        start=0,
        end=64,
        text="The audit records may be reviewed and audit costs may shift.",
        word_count=11,
        contains_rui_ids=(),
    )

    assigned, scores = _best_participant_for_chunk(chunk, participants, target_words=100)

    assert assigned == ["p2"]
    assert scores["p2"]["final"] > scores["p1"]["final"]
    assert scores["p2"]["role_overlap"] > scores["p1"]["role_overlap"]


def test_riu_related_neutral_chunk_avoids_related_rui_owner() -> None:
    participants = [
        {
            "participant_id": "p1",
            "owned_rui_ids": ["span_01"],
            "role_terms": ["insurance", "limit"],
            "owned_rui_feature_terms": ["insurance", "limit"],
            "owned_rui_centers": [120],
            "owned_rui_section_keys": ["insurance"],
            "private_doc_unit_ids": [],
            "private_word_count": 0,
        },
        {
            "participant_id": "p2",
            "owned_rui_ids": [],
            "role_terms": ["background"],
            "owned_rui_feature_terms": [],
            "owned_rui_centers": [],
            "owned_rui_section_keys": [],
            "private_doc_unit_ids": [],
            "private_word_count": 0,
        },
        {
            "participant_id": "p3",
            "owned_rui_ids": [],
            "role_terms": ["operations"],
            "owned_rui_feature_terms": [],
            "owned_rui_centers": [],
            "owned_rui_section_keys": [],
            "private_doc_unit_ids": [],
            "private_word_count": 0,
        },
    ]
    rui = RUIRange(
        unit_id="span_01",
        source_span_index=1,
        clean_text="Provider must maintain insurance coverage with a five million dollar limit.",
        raw_start=100,
        raw_end=160,
        raw_text="Provider must maintain insurance coverage with a five million dollar limit.",
        legal_feature="insurance limit",
        summary="Insurance coverage limit.",
        section_key="insurance",
    )
    chunk = Chunk(
        start=170,
        end=240,
        text="This insurance background explains coverage administration and limit documentation.",
        word_count=9,
        contains_rui_ids=(),
        section_key="insurance",
    )

    assigned, scores = _balanced_participant_for_neutral_chunk(
        chunk,
        participants,
        rui_ranges=[rui],
        rui_assignment={"p1": ["span_01"], "p2": [], "p3": []},
        target_words=80,
        rng=random.Random(5),
    )

    assert assigned
    assert assigned[0] != "p1"
    assert scores["_assignment_context"]["related_rui_ids"] == ["span_01"]
    assert scores["p1"]["owns_related_rui"] == 1.0
    assert scores["p1"]["candidate_after_safety_filter"] == 0.0
