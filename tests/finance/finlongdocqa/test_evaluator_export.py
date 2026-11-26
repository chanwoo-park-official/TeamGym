"""Generic evaluator metadata must preserve the producer's full claim contract."""
from argparse import Namespace
from copy import deepcopy
import json
from zipfile import ZipFile

import pytest

from scripts.data_generation.finance.finlongdocqa.assemble_producer_bundle import assemble_producer_bundle
from teamgym.data.finance.finlongdocqa.evaluation import (
    capture_grader_payload, capture_model_observations, evaluator_datapoint, write_release,
)
from teamgym.data.finance.finlongdocqa.packet_budget import PAPER_CONTEXT_BUDGET, certify_materialized_views
from teamgym.data.finance.finlongdocqa.pipeline import build_block_index
from teamgym.data.finance.finlongdocqa.provenance import (
    commit_report, establish_run, file_digest, rebuild_exports, run_identity, source_identity,
)
from teamgym.data.finance.finlongdocqa.source import group_qas_by_report, load_qa_rows
from teamgym.scoring.short_answers import judge_one_answer

REPORT_TEXT = "# Page 1\n\nRevenue was 120 million.\n\n# Page 2\n\nIncome was 30 million."


def fixture():
    from teamgym.data.finance.finlongdocqa.partition import build_partitioned_datapoint
    from teamgym.data.finance.finlongdocqa.source import index_report
    from teamgym.data.finance.finlongdocqa.validation import hydrate_task
    indexed = index_report("X_2024", REPORT_TEXT)
    task = {"source_qa_ids_used": ["q"], "public_task": {"title": "Review", "question": "State revenue and income.", "response_requirements": ["Give both values."]},
        "evidence_units": [{"unit_id": f"riu_{i:02d}", "source_block_id": b["block_id"]} for i, b in enumerate(indexed["blocks"], 1)],
        "rubric": {"rubric_version": "test", "required_claims": [
            {"claim_id": "c1", "description": "State revenue", "expected_answer_element": "120 million", "supporting_unit_ids": ["riu_01"], "grading_rule": "Require value and unit"},
            {"claim_id": "c2", "description": "State income", "expected_answer_element": "30 million", "supporting_unit_ids": ["riu_02"], "grading_rule": "Require value and unit"}],
            "prohibited_errors": ["Do not reverse the two values."]},
        "oracle": {"reference_answer": "Revenue 120 million; income 30 million."}}
    task = hydrate_task(task, indexed["blocks"])
    return build_partitioned_datapoint(preserve_rui_order=True, report_id="X_2024", company="X", year="2024", report_source={}, report_index=indexed,
        task=task, qa_mappings=[], solver_audit={})


def test_actual_grader_gets_all_native_claims_links_errors_and_strict_policy():
    raw = fixture()
    before = deepcopy(raw)
    final = evaluator_datapoint(raw)
    payload = capture_grader_payload(final, raw["oracle"]["answer_text"])["payload"]
    native = raw["dataset_specific"]["free_response_rubric"]
    assert payload["judge_rubric"]["required_claims"] == native["required_claims"]
    assert payload["disqualifying_errors"] == native["prohibited_errors"]
    assert [step["uses"] for step in payload["required_decision_steps"]] == [["riu_01"], ["riu_02"]]
    assert payload["canonical_response"] == raw["oracle"]["answer_text"]
    assert payload["pass_policy"]["require_all_steps_for_pass"] is True
    assert raw == before
    assert capture_model_observations(raw) == capture_model_observations(final)


def test_actual_generic_grader_rejects_partial_even_if_provider_passed_flag_is_true():
    row = evaluator_datapoint(fixture())
    class Client:
        last_request_meta = {"offline_contract_test": True}
        def __init__(self, covered): self.covered = covered
        def generate_json(self, **kwargs):
            import json
            payload = json.loads(kwargs["user_prompt"])
            assert len(payload["required_decision_steps"]) == 2
            assert payload["pass_policy"]["require_all_steps_for_pass"] is True
            return {"passed": True, "score": 1 if self.covered == 2 else .5,
                "conclusion_correct": True, "total_step_count": 2, "covered_step_count": self.covered,
                "step_scores": [{"step": i, "covered": i <= self.covered, "score": int(i <= self.covered)} for i in (1, 2)],
                "missing_steps": [] if self.covered == 2 else ["income"], "incorrect_claims": [],
                "disqualifying_errors_found": [], "reason": "Offline positive/partial controls."}
    correct = judge_one_answer(client=Client(2), datapoint=row,
        answer_row={"answer_text": "Revenue120million; income30million"}, trace_dir=None, index=1)
    partial = judge_one_answer(client=Client(1), datapoint=row,
        answer_row={"answer_text": "Revenue120million"}, trace_dir=None, index=2)
    assert correct["judgment"]["passed"] is True
    assert partial["judgment"]["passed"] is False
    assert partial["judgment"]["holistic_passed"] is True


def test_completed_run_can_be_bundled_and_exported_offline(tmp_path):
    report_id = "X_2024"
    run = tmp_path / "run"
    qa = tmp_path / "qa.jsonl"
    qa.write_text(json.dumps({
        "id": "q", "company": "X", "year": "2024", "question": "State revenue and income.",
        "answer": "120 and 30 million", "page_numbers": [1, 2],
    }) + "\n")
    reports = tmp_path / "reports.zip"
    with ZipFile(reports, "w") as archive:
        archive.writestr("reports/X/2024.md", REPORT_TEXT)
    args = Namespace(qa_path=qa, reports_zip=reports, output_root=run, report_ids=report_id)
    assert build_block_index(args)["report_count"] == 1
    index = json.loads((run / "report_block_indexes.jsonl").read_text())
    sources = source_identity(args, dict(group_qas_by_report(load_qa_rows(qa))), {report_id: index}, [report_id])
    identity = run_identity(args, sources, [report_id])
    establish_run(run, identity)
    raw = fixture()
    certify_materialized_views(raw)
    units = [row["rui_id"] for row in raw["reference_information_units"]]
    participants = [row["participant_id"] for row in raw["participants"]]
    result = {
        "status": "accepted", "report_id": report_id, "errors": [],
        "task": {"evidence_units": [{"unit_id": unit} for unit in units]},
        "solver_audit": {
            "accept": True,
            "full_evidence": {"judge_result": {"overall_complete_and_correct": True}},
            "no_evidence": {"judge_result": {"overall_complete_and_correct": False}},
            "leave_one_out": [{"omitted_unit_id": unit, "broken_claim_ids": ["criterion"]} for unit in units],
        },
        "initial_view_audits": [
            {"mode": mode, "participant_id": pid, "context_token_budget": PAPER_CONTEXT_BUDGET,
             "judge_result": {"overall_complete_and_correct": False}}
            for mode in ("multi_agent_full_context", "multi_agent_piu_only") for pid in participants
        ],
        "packet_salience_audits": [
            {"participant_id": pid, "errors": [], "audit": {
                "leaks_constructor_metadata": False, "reveals_answer_in_identity": False,
                "marks_private_piu_as_important": False,
            }} for pid in participants
        ],
        "packet_leakage_audits": [
            {"round": 0, "participant_id": pid, "audit": {"foreign_rui_results": []}}
            for pid in participants
        ],
    }
    commit = commit_report(run, identity["fingerprint"], report_id, 1, result, raw)
    rebuild_exports(run, [commit])

    bundle = tmp_path / "producer"
    receipt = assemble_producer_bundle(run_dir=run, qa_path=qa, reports_zip=reports, output_dir=bundle)
    assert receipt["producer_fingerprint"] == identity["fingerprint"]
    assert (bundle / "generated/run_manifest.json").read_bytes() == (run / "run_manifest.json").read_bytes()
    capture = json.loads((bundle / "consumer_inputs.json").read_text())
    assert capture["inputs"] == capture_model_observations(raw)

    output = tmp_path / "evaluation"
    released = write_release(bundle, output)
    assert released["model_observations_byte_equal"]
    accepted_path = output / "common_datapoints.jsonl"
    assert released["outputs"]["common_datapoints.jsonl"] == file_digest(accepted_path)
    saved = json.loads(accepted_path.read_text())
    assert saved["schema_version"] == "teamgym_release_datapoint_v1"
    assert saved["oracle"]["judge_rubric"]["require_all_steps_for_pass"]

    qa.write_text(qa.read_text() + "\n")
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="producer source changed"):
        assemble_producer_bundle(run_dir=run, qa_path=qa, reports_zip=reports, output_dir=invalid)
    assert not invalid.exists()
