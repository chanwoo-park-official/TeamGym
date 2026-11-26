"""Synthetic completed audits for export tests; these are not live model results."""
import json
from pathlib import Path

from teamgym.data.medical.certification import certify_export


def _write(path, value):
    Path(path).write_text(json.dumps(value))


def boundary_fixture(candidate_ids):
    return {"passed": True, "cases": [
        {"case_id": cid, "outcome": {"condition_ok": True}, "errors": []}
        for cid in ["all", "none", *(f"without_{cid}" for cid in candidate_ids)]]}


def seal_fixture(root, domain):
    module = __import__(f"teamgym.data.medical.{domain}.execution", fromlist=["export_artifacts"])
    name = "summary.json" if domain == "evidencebench" else "run_report.json"
    payload = json.loads((root / name).read_text())
    payload["export_certification"] = certify_export(payload, module.export_artifacts(root))
    _write(root / name, payload)


def complete_fixture(root, domain, *, token_budget=40000):
    """Supply explicit passing synthetic audits, then freeze exactly this fixture."""
    _write(root / "run_manifest.json", {"fixture": True})
    if domain == "mlcr":
        from teamgym.data.medical.mlcr.evaluation import consumer_full_context_materials
        report = json.loads((root / "run_report.json").read_text())
        task, partition = report["task"], report["partition"]
        report["execution_identity"] = {"participant_context_token_budget": token_budget}
        partition["participant_token_budget"] = token_budget
        report["grounding_audit"] = {"passed": True, "reference_answer_fully_grounded": True,
            "all_units_materially_necessary": True, "claim_audits": [
                {"claim_id": c["claim_id"], "grounded": True, "scope_preserved": True}
                for c in task["rubric"]["required_claims"]]}
        report["boundary_audits"] = boundary_fixture([u["candidate_id"] for u in task["evidence_units"]])
        report["packet_leakage_audit"] = {"passed": True}
        material = consumer_full_context_materials(partition, token_budget=token_budget)
        report["participant_final_answer_audit"] = {"passed": True, "participants": [
            {"participant_id": pid, "passed": True, "errors": [],
             "participant_context_token_budget": token_budget,
             "display_materials": [{"text": text}]}
            for pid, text in material.items()]}
        _write(root / "run_report.json", report)
    else:
        from teamgym.data.medical.evidencebench.evaluation import consumer_full_context_materials
        from teamgym.data.medical.evidencebench.execution import export_artifacts
        task = json.loads((root / "02_task_rubric.json").read_text())
        units = json.loads((root / "05_final_rius.hidden.json").read_text())
        partition = json.loads((root / "06_participant_packets.hidden.json").read_text())
        partition["participant_token_budget"] = token_budget
        _write(root / "06_participant_packets.hidden.json", partition)
        task["selected_candidate_ids"] = [u["candidate_id"] for u in units]
        task["candidate_dispositions"] = [{"candidate_id": cid, "included": True}
                                          for cid in task["selected_candidate_ids"]]
        _write(root / "02_task_rubric.json", task)
        audit = {key: True for key in ("valid", "question_natural", "question_answerable_from_selected_material",
            "rubric_grounded", "canonical_answer_grounded", "selected_candidates_relevant",
            "question_not_artificially_expanded", "no_construction_metadata_leakage", "medical_scope_preserved")}
        audit["claim_results"] = [{"claim_id": c["claim_id"], "grounded": True, "support_logic_valid": True,
            "supporting_candidate_ids": c["supporting_candidate_ids"]} for c in task["rubric"]["required_claims"]]
        audit["candidate_results"] = [{"candidate_id": cid, "included_as_selected": True,
            "question_relevant": True, "direct_support": True, "necessary_under_support_logic": True,
            "mapped_claim_ids": [c["claim_id"] for c in task["rubric"]["required_claims"]
                                 if cid in c["supporting_candidate_ids"]]} for cid in task["selected_candidate_ids"]]
        _write(root / "03_task_audit.json", audit)
        _write(root / "04_boundaries.json", boundary_fixture(task["selected_candidate_ids"]))
        _write(root / "07_leakage_audit.json", {"passed": True})
        material = consumer_full_context_materials(partition, token_budget=token_budget)
        _write(root / "09_participant_final_answer_audit.json", {"passed": True, "participants": [
            {"participant_id": pid, "passed": True, "errors": [],
             "participant_context_token_budget": token_budget,
             "display_material": [{"text": text}]} for pid, text in material.items()]})
        summary = json.loads((root / "summary.json").read_text())
        summary["execution_identity"] = {"participant_context_token_budget": token_budget}
        _write(root / "summary.json", summary)
        for path in export_artifacts(root).values():
            if not path.exists(): _write(path, {"fixture": True})
    seal_fixture(root, domain)
