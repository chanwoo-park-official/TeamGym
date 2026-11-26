from __future__ import annotations

import json

from teamgym.scoring.short_answers import (
    deterministic_numeric_evaluation,
    judge_one_answer,
)


class CapturingJudgeClient:
    def __init__(self) -> None:
        self.last_request_meta = {"provider": "fake", "model": "fake"}
        self.payload: dict | None = None

    def generate_json(self, *, user_prompt: str, **_: object) -> dict:
        self.payload = json.loads(user_prompt)
        assert "reference_information_units" in self.payload
        assert "Person B became aware of Person A's gambling problem 5 months ago." in json.dumps(
            self.payload["reference_information_units"]
        )
        return {
            "passed": True,
            "score": 1.0,
            "conclusion_correct": True,
            "covered_step_count": 1,
            "total_step_count": 1,
            "step_scores": [{"step": 1, "covered": True, "score": 1.0, "reason": "Supported by RIU."}],
            "missing_steps": [],
            "incorrect_claims": [],
            "disqualifying_errors_found": [],
            "reason": "The candidate is supported by the RIUs.",
        }


def test_short_answer_judge_payload_includes_reference_information_units() -> None:
    datapoint = {
        "instance_id": "short_answer_case",
        "dataset": "long_context_reasoning",
        "task_description": {
            "question": "Can Person B file, and when does the right expire?",
            "answer_type": "short_answer_judged",
        },
        "reference_information_units": [
            {
                "rui_id": "rui_case_background",
                "text": "Person B became aware of Person A's gambling problem 5 months ago.",
                "summary": "Awareness timing.",
            }
        ],
        "oracle": {
            "canonical_response": "Person B may file within six months from awareness.",
            "required_decision_steps": [
                {"step": 1, "uses": ["rui_case_background"], "claim": "Apply the awareness timing."}
            ],
        },
    }
    answer_row = {
        "instance_id": "short_answer_case",
        "answer_text": "Person B learned of the problem five months ago, so roughly one month remains.",
    }
    client = CapturingJudgeClient()

    judgment = judge_one_answer(
        client=client,
        datapoint=datapoint,
        answer_row=answer_row,
        trace_dir=None,
        index=1,
    )

    assert judgment["judgment"]["passed"] is True
    assert client.payload is not None
    assert client.payload["reference_information_units"][0]["required_for_answer"] is True


def test_short_answer_judge_can_require_all_steps_for_pass() -> None:
    datapoint = {
        "instance_id": "strict_short_answer_case",
        "dataset": "long_context_reasoning",
        "task_description": {
            "question": "State the answer and every necessary premise.",
            "answer_type": "short_answer_judged",
        },
        "reference_information_units": [
            {
                "rui_id": "rui_case_background",
                "text": "Person B became aware of Person A's gambling problem 5 months ago.",
                "summary": "Awareness timing.",
            }
        ],
        "oracle": {
            "canonical_response": "Person B may file within six months from awareness.",
            "required_decision_steps": [
                {"step": 1, "uses": ["rui_case_background"], "claim": "Apply the awareness timing."},
                {"step": 2, "uses": ["rui_case_background"], "claim": "State when the right expires."},
            ],
            "judge_rubric": {"require_all_steps_for_pass": True},
        },
    }
    answer_row = {
        "instance_id": "strict_short_answer_case",
        "answer_text": "Person B learned of the problem five months ago.",
    }
    client = CapturingJudgeClient()

    judgment = judge_one_answer(
        client=client,
        datapoint=datapoint,
        answer_row=answer_row,
        trace_dir=None,
        index=1,
    )

    assert judgment["judgment"]["holistic_passed"] is True
    assert judgment["judgment"]["passed"] is False
    assert judgment["judgment"]["pass_policy"] == "all_required_decision_steps"
    assert client.payload is not None
    assert client.payload["pass_policy"]["require_all_steps_for_pass"] is True


class NumericMismatchJudgeClient:
    def __init__(self) -> None:
        self.last_request_meta = {"provider": "fake", "model": "fake"}
        self.payload: dict | None = None

    def generate_json(self, *, user_prompt: str, **_: object) -> dict:
        self.payload = json.loads(user_prompt)
        return {
            "passed": False,
            "score": 0.9,
            "conclusion_correct": False,
            "covered_step_count": 1,
            "total_step_count": 2,
            "step_scores": [
                {"step": 1, "covered": True, "score": 1.0, "reason": "Calculation is supported."},
                {"step": 2, "covered": False, "score": 0.0, "reason": "Amount differs from oracle."},
            ],
            "missing_steps": ["State the final tax-liability conclusion: Alice owes $45946."],
            "incorrect_claims": [],
            "disqualifying_errors_found": ["The requested result is incorrect."],
            "reason": "The cents do not exactly match the whole-dollar oracle.",
        }


class LowScoreNumericMismatchJudgeClient(NumericMismatchJudgeClient):
    def generate_json(self, *, user_prompt: str, **kwargs: object) -> dict:
        response = super().generate_json(user_prompt=user_prompt, **kwargs)
        response["score"] = 0.65
        return response


def _numeric_sara_datapoint() -> dict:
    return {
        "instance_id": "sara_numeric_case",
        "dataset": "long_context_reasoning",
        "task_description": {
            "question": "Calculate Alice's federal tax liability.",
            "answer_type": "short_answer_judged",
        },
        "oracle": {
            "short_answer": "$45946",
            "canonical_response": "Alice owes $45946.",
            "required_decision_steps": [
                {"step": 1, "claim": "Apply the statutory calculation."},
                {"step": 2, "claim": "State the final tax-liability amount Alice owes."},
            ],
            "judge_rubric": {"passing_score": 0.75, "require_all_steps_for_pass": True},
        },
    }


def test_numeric_oracle_uses_one_dollar_tolerance() -> None:
    datapoint = _numeric_sara_datapoint()

    within_tolerance = deterministic_numeric_evaluation(
        datapoint,
        "The calculation gives $45,946.94 in total federal tax owed.",
    )
    outside_tolerance = deterministic_numeric_evaluation(
        datapoint,
        "The calculation gives $45,947.00 in total federal tax owed.",
    )

    assert within_tolerance is not None
    assert within_tolerance["passed"] is True
    assert within_tolerance["absolute_error"] == "0.94"
    assert outside_tolerance is not None
    assert outside_tolerance["passed"] is False


def test_numeric_oracle_prefers_explicit_final_amount_over_later_incidental_amount() -> None:
    datapoint = _numeric_sara_datapoint()
    result = deterministic_numeric_evaluation(
        datapoint,
        "Alice owes $45,946.94 in federal tax. A later payment of $3,200 does not alter the result.",
    )

    assert result is not None
    assert result["candidate_amount"] == "45946.94"
    assert result["candidate_amount_strategy"] == "explicit_final_amount_cue"
    assert result["passed"] is True


def test_numeric_oracle_handles_amount_before_total_tax_cue() -> None:
    datapoint = _numeric_sara_datapoint()
    result = deterministic_numeric_evaluation(
        datapoint,
        "The statutory calculation gives $45,946.94 in total federal tax owed.",
    )

    assert result is not None
    assert result["candidate_amount"] == "45946.94"
    assert result["candidate_amount_strategy"] == "final_amount_cue_sentence"
    assert result["passed"] is True


def test_numeric_oracle_recognizes_taxable_income_as_requested_result() -> None:
    datapoint = _numeric_sara_datapoint()
    datapoint["oracle"]["short_answer"] = "89580"
    result = deterministic_numeric_evaluation(
        datapoint,
        "Bob's taxable income is $89,580. Total income is $113,580 and the standard deduction is $24,000.",
    )

    assert result is not None
    assert result["candidate_amount"] == "89580"
    assert result["candidate_amount_strategy"] == "final_amount_cue_sentence"
    assert result["passed"] is True


def test_numeric_oracle_does_not_treat_total_income_as_final_result() -> None:
    datapoint = _numeric_sara_datapoint()
    datapoint["oracle"]["short_answer"] = "89580"
    result = deterministic_numeric_evaluation(
        datapoint,
        "Total income is $113,580. Taxable income is $113,580 - $24,000 = $89,580.",
    )

    assert result is not None
    assert result["candidate_amount"] == "89580"
    assert result["candidate_amount_strategy"] == "final_amount_cue_sentence"
    assert result["passed"] is True


def test_numeric_tolerance_overrides_amount_only_llm_failure() -> None:
    datapoint = _numeric_sara_datapoint()
    answer_row = {
        "instance_id": "sara_numeric_case",
        "answer_text": "The statutory calculation gives total tax owed of $45,946.94.",
    }
    client = NumericMismatchJudgeClient()

    judgment = judge_one_answer(
        client=client,
        datapoint=datapoint,
        answer_row=answer_row,
        trace_dir=None,
        index=1,
    )

    assert client.payload is not None
    assert client.payload["deterministic_numeric_evaluation"]["passed"] is True
    assert judgment["judgment"]["numeric_override_applied"] is True
    assert judgment["judgment"]["conclusion_correct"] is True
    assert judgment["judgment"]["holistic_passed"] is True
    assert judgment["judgment"]["passed"] is True


def test_numeric_override_is_not_blocked_by_amount_penalized_llm_score() -> None:
    datapoint = _numeric_sara_datapoint()
    answer_row = {
        "instance_id": "sara_numeric_case",
        "answer_text": "Alice owes $45,946.94; a separate $3,200 payment does not change the result.",
    }
    client = LowScoreNumericMismatchJudgeClient()

    judgment = judge_one_answer(
        client=client,
        datapoint=datapoint,
        answer_row=answer_row,
        trace_dir=None,
        index=1,
    )

    assert judgment["judgment"]["numeric_override_applied"] is True
    assert judgment["judgment"]["holistic_passed"] is True
    assert judgment["judgment"]["passed"] is True
