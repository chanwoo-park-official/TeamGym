from copy import deepcopy

from teamgym.data.medical.evidencebench import boundary
from tests.medical.evidencebench.test_independent_pipeline import _rubric, _candidates


def test_alternative_support_is_judged_without_condition_or_declared_source_set(monkeypatch, tmp_path):
    original = _rubric(); before = deepcopy(original)
    checked = []
    def call(**kw):
        if kw["stage"].startswith("boundary_solver"):
            return {"answer": "Complete answer", "evidence_used": [], "unresolved_requirements": []}, []
        variables = kw["variables"]
        assert not {"CONDITION_JSON", "MATERIAL_MAPPING_JSON", "APPLY_RECORDED_SUPPORT_LOGIC"} & variables.keys()
        claim = variables["RUBRIC_JSON"]["required_claims"][0]
        assert not {"support_logic", "supporting_candidate_ids", "candidate_roles"} & claim.keys()
        available = variables["AVAILABLE_MATERIAL_JSON"]
        supported = bool(available)
        row = {"content_complete": supported, "grounded_complete": supported,
               "claim_results": [{"claim_id": "claim_01", "content_correct": supported,
                   "grounded_in_available_material": supported, "satisfied": supported,
                   "evidence_numbers": [available[0]["evidence_number"]] if available else []}],
               "unsupported_claims": [], "critical_errors": []}
        errors = kw["validator"](row)
        checked.append(errors)
        return row, errors
    monkeypatch.setattr(boundary, "structured_call", call)
    result = boundary.run_boundaries(instance_id="fixture", task_rubric={
        "public_task": {"question": "Q"}, "rubric": original, "canonical_answer": "A"},
        selected_candidates=_candidates()[1:], client=None, cache=None,
        trace_root=tmp_path, attempts=1, seed=1)
    assert checked and not any(checked)
    assert not result["passed"]
    assert all(not case["outcome"]["condition_ok"] for case in result["cases"] if case["kind"] == "loo")
    assert original == before
