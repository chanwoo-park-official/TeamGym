from random import Random

from teamgym.data.legal.mslr.generation import make_rui
from teamgym.data.long_context_reasoning.pipeline import structural_leakage_report
from teamgym.data.long_context_reasoning.seed_builder import build_one_datapoint


def test_mslr_structural_leakage_checks_the_actual_excerpts():
    excerpts = [
        "The board approved the acquisition on Monday.",
        "The employee learned of the decision on Tuesday.",
        "The employee bought shares on Wednesday.",
    ]
    pius = [
        make_rui(f"rui_{index:02d}", source_span=excerpt, source_section="case_description")
        for index, excerpt in enumerate(excerpts, start=1)
    ]
    datapoint = build_one_datapoint(
        spec={
            "source_text": "\n\n".join(excerpts),
            "question": "Summarize the sequence of events.",
            "answer_type": "short_answer_judged",
            "answer": "free_response",
            "decision_set": ["free_response"],
            "decision_option_definitions": {"free_response": "Summarize the events."},
            "canonical_response": " ".join(excerpts),
            "rius": pius,
            "required_rui_ids": [piu["rui_id"] for piu in pius],
        },
        index=0,
        allocation_mode="one_each",
        target_context_tokens=512,
        min_participants=3,
        rng=Random(17),
    )

    _, report, _ = structural_leakage_report(datapoint)

    assert [piu["text"] for piu in datapoint["reference_information_units"]] == excerpts
    assert not report["excluded"], report["exclusion_reasons"]
