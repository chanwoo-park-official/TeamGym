from teamgym.data.legal.leakage_deletion_funnel import audit_payload, score_audit


def _datapoint():
    return {
        "dataset": "unit",
        "instance_id": "leakage_case",
        "task_description": "Choose the correct decision.",
        "decision_set": ["decision_a", "decision_b"],
        "decision_option_definitions": {
            "decision_a": "The wrong profile.",
            "decision_b": "The correct profile.",
        },
        "oracle": {"answer": "decision_b"},
        "reference_information_units": [
            {
                "rui_id": "rui_1",
                "text": "Required collaborative fact.",
                "information_feature": "required fact",
            }
        ],
    }


def test_riu_bearing_leak_excludes_instead_of_deleting():
    context_units = [
        {
            "context_unit_id": "du_riu",
            "text": "Required collaborative fact.",
            "hidden_contains_rui_ids": ["rui_1"],
        }
    ]
    audit = {
        "round": 1,
        "prompt_id": "prompt",
        "response": {
            "answerable_from_this_context": True,
            "selected_answer": "decision_b",
            "confidence": 0.95,
            "decisive_context_unit_ids": ["du_riu"],
            "decisive_quotes": ["Required collaborative fact."],
            "reason": "The RIU-bearing unit decides the answer.",
        },
    }

    scored = score_audit(
        _datapoint(),
        participant_id="participant_a",
        audit=audit,
        context_units=context_units,
        confidence_threshold=0.7,
    )

    assert scored["leak_found"] is True
    assert scored["structural_riu_leak"] is True
    assert scored["recommended_action"] == "exclude_datapoint"
    assert scored["decisive_rui_ids"] == ["rui_1"]
    assert scored["delete_context_unit_ids"] == []


def test_non_riu_leak_is_deletable():
    context_units = [
        {
            "context_unit_id": "du_extra",
            "text": "Extra non-RIU text that leaks the answer.",
            "hidden_contains_rui_ids": [],
        }
    ]
    audit = {
        "round": 1,
        "prompt_id": "prompt",
        "response": {
            "answerable_from_this_context": True,
            "selected_answer": "decision_b",
            "confidence": 0.95,
            "decisive_context_unit_ids": ["du_extra"],
            "decisive_quotes": ["leaks the answer"],
            "reason": "The non-RIU unit decides the answer.",
        },
    }

    scored = score_audit(
        _datapoint(),
        participant_id="participant_a",
        audit=audit,
        context_units=context_units,
        confidence_threshold=0.7,
    )

    assert scored["leak_found"] is True
    assert scored["structural_riu_leak"] is False
    assert scored["recommended_action"] == "delete_non_riu_context"
    assert scored["delete_context_unit_ids"] == ["du_extra"]


def test_audit_payload_marks_riu_bearing_context_units():
    payload = audit_payload(
        _datapoint(),
        participant_id="participant_a",
        context_units=[
            {
                "context_unit_id": "du_riu",
                "text": "Required collaborative fact.",
                "hidden_contains_rui_ids": ["rui_1"],
            }
        ],
        max_unit_chars=1000,
        max_participant_context_chars=1000,
    )

    unit = payload["private_context_units"][0]
    assert unit["contains_rui_ids"] == ["rui_1"]
    assert unit["contains_rui_units"][0]["information_feature"] == "required fact"
    assert unit["is_riu_bearing"] is True
