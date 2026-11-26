from argparse import Namespace

from teamgym.data.legal.option_exclusion_verifier import (
    exclusion_payload,
    filtered_datapoints,
    is_leakage_safe,
    score_exclusion_audit,
    summarize,
)


def _datapoint():
    return {
        "dataset": "unit",
        "instance_id": "option_exclusion_case",
        "task_description": "Choose the matching profile.",
        "decision_set": ["decision_a", "decision_b", "decision_c"],
        "decision_option_definitions": {
            "decision_a": "Requires termination for convenience.",
            "decision_b": "Requires termination for breach.",
            "decision_c": "Requires no termination right.",
        },
        "oracle": {"answer": "decision_b"},
        "initial_common_information": "Only public shell information.",
    }


def test_score_marks_single_participant_oracle_reduction():
    context_units = [{"context_unit_id": "du_1", "text": "The clause covers termination for breach only."}]
    audit = {
        "prompt_id": "prompt",
        "response": {
            "excluded_options": ["decision_a", "decision_c"],
            "remaining_options": ["decision_b"],
            "confidence": 0.9,
            "option_assessments": [
                {
                    "option": "decision_a",
                    "status": "excluded",
                    "context_unit_ids": ["du_1"],
                    "quotes": ["termination for breach only"],
                    "reason": "Convenience termination is contradicted.",
                }
            ],
            "reason": "Only the breach profile remains possible.",
        },
    }

    scored = score_exclusion_audit(
        _datapoint(),
        participant_id="participant_a",
        audit=audit,
        context_units=context_units,
        confidence_threshold=0.7,
    )

    assert scored["excluded_options"] == ["decision_a", "decision_c"]
    assert scored["remaining_options_from_exclusions"] == ["decision_b"]
    assert scored["reduces_to_oracle"] is True
    assert scored["single_option_remaining"] is True
    assert scored["quote_hit_count"] == 1


def test_score_does_not_reduce_when_multiple_options_remain():
    audit = {
        "prompt_id": "prompt",
        "response": {
            "excluded_options": ["decision_a"],
            "remaining_options": ["decision_b", "decision_c"],
            "confidence": 0.9,
        },
    }

    scored = score_exclusion_audit(
        _datapoint(),
        participant_id="participant_a",
        audit=audit,
        context_units=[],
        confidence_threshold=0.7,
    )

    assert scored["remaining_options_from_exclusions"] == ["decision_b", "decision_c"]
    assert scored["reduces_to_oracle"] is False
    assert scored["single_option_remaining"] is False


def test_score_marks_wrong_single_option_as_keepable_signal():
    audit = {
        "prompt_id": "prompt",
        "response": {
            "excluded_options": ["decision_a", "decision_b"],
            "remaining_options": ["decision_c"],
            "confidence": 0.9,
        },
    }

    scored = score_exclusion_audit(
        _datapoint(),
        participant_id="participant_a",
        audit=audit,
        context_units=[],
        confidence_threshold=0.7,
    )

    assert scored["single_option_remaining"] is True
    assert scored["single_option_is_oracle"] is False
    assert scored["single_option_wrong"] is True
    assert scored["reduces_to_oracle"] is False


def test_score_does_not_reduce_below_confidence_threshold():
    audit = {
        "prompt_id": "prompt",
        "response": {
            "excluded_options": ["decision_a", "decision_c"],
            "remaining_options": ["decision_b"],
            "confidence": 0.4,
        },
    }

    scored = score_exclusion_audit(
        _datapoint(),
        participant_id="participant_a",
        audit=audit,
        context_units=[],
        confidence_threshold=0.7,
    )

    assert scored["remaining_options_from_exclusions"] == ["decision_b"]
    assert scored["reduces_to_oracle"] is False
    assert scored["single_option_remaining"] is True


def test_exclusion_payload_hides_riu_metadata():
    payload = exclusion_payload(
        _datapoint(),
        participant_id="participant_a",
        context_units=[
            {
                "context_unit_id": "du_1",
                "text": "Private text.",
                "hidden_contains_rui_ids": ["riu_1"],
            }
        ],
        max_unit_chars=1000,
        max_participant_context_chars=1000,
    )

    unit = payload["private_context_units"][0]
    assert unit == {"context_unit_id": "du_1", "text": "Private text."}


def test_leakage_safe_filter_reads_generation_metadata():
    datapoint = {
        "generation_metadata": {
            "participant_leakage_deletion_funnel": {
                "converged": True,
                "excluded": False,
            }
        }
    }

    assert is_leakage_safe(datapoint) is True


def test_summary_keeps_wrong_single_option_datapoints_under_filter():
    reports = [
        {
            "dataset": "unit",
            "has_single_participant_oracle_reduction": True,
            "has_any_single_option_reduction": True,
            "keep_under_solo_oracle_filter": False,
            "has_only_wrong_single_option_reductions": False,
        },
        {
            "dataset": "unit",
            "has_single_participant_oracle_reduction": False,
            "has_any_single_option_reduction": True,
            "keep_under_solo_oracle_filter": True,
            "has_only_wrong_single_option_reductions": True,
        },
    ]
    audit_rows = [
        {
            "dataset": "unit",
            "reduces_to_oracle": True,
            "single_option_remaining": True,
            "single_option_is_oracle": True,
            "single_option_wrong": False,
        },
        {
            "dataset": "unit",
            "reduces_to_oracle": False,
            "single_option_remaining": True,
            "single_option_is_oracle": False,
            "single_option_wrong": True,
        },
    ]

    summary = summarize(
        reports,
        audit_rows,
        Namespace(
            input_root="input.jsonl",
            limit=2,
            seed=1,
            shuffle=False,
            dataset=["unit"],
            require_leakage_safe=True,
            workers=1,
            confidence_threshold=0.7,
            model="unit-model",
            max_participant_context_chars=1000,
            max_unit_chars=1000,
        ),
        selected_count=2,
    )

    assert summary["aggregate"]["datapoints_kept_under_solo_oracle_filter"] == 1
    assert summary["aggregate"]["datapoints_dropped_solo_oracle_reducible"] == 1
    assert summary["aggregate"]["datapoints_with_only_wrong_single_option_reductions"] == 1
    assert summary["by_dataset"]["unit"]["only_wrong_single_option_datapoints"] == 1


def test_filtered_datapoints_keeps_misleading_single_option_rows():
    datapoints = [
        {"instance_id": "drop_me", "generation_metadata": {"method": "unit"}},
        {"instance_id": "keep_misleading", "generation_metadata": {"method": "unit"}},
    ]
    reports = [
        {
            "instance_id": "drop_me",
            "keep_under_solo_oracle_filter": False,
            "filter_decision": "drop_solo_oracle_reducible",
            "has_single_participant_oracle_reduction": True,
            "has_any_single_option_reduction": True,
        },
        {
            "instance_id": "keep_misleading",
            "keep_under_solo_oracle_filter": True,
            "filter_decision": "keep_misleading_single_option",
            "has_single_participant_oracle_reduction": False,
            "has_any_single_option_reduction": True,
            "has_only_wrong_single_option_reductions": True,
        },
    ]

    kept, dropped, misleading = filtered_datapoints(datapoints, reports)

    assert [row["instance_id"] for row in kept] == ["keep_misleading"]
    assert [row["instance_id"] for row in dropped] == ["drop_me"]
    assert [row["instance_id"] for row in misleading] == ["keep_misleading"]
    funnel = kept[0]["generation_metadata"]["participant_option_exclusion_funnel"]
    assert funnel["filter_decision"] == "keep_misleading_single_option"
    assert funnel["keep_under_solo_oracle_filter"] is True
