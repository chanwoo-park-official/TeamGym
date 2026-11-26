"""Reject incomplete audits and edits after the last live certification."""
import json
import pytest

from tests.medical.test_grading_contract import export_fixture
from tests.medical.certified_fixtures import seal_fixture, complete_fixture


def exporter(root, domain):
    if domain == "mlcr":
        from teamgym.data.medical.mlcr.evaluation import build_evaluation_datapoint
        return build_evaluation_datapoint(root / "run_report.json")
    from teamgym.data.medical.evidencebench.evaluation import build_evaluation_datapoint
    return build_evaluation_datapoint(root)


@pytest.mark.parametrize("domain", ["evidencebench", "mlcr"])
@pytest.mark.parametrize("change", ["missing", "empty", "partial", "duplicate", "failed", "string_false", "budget"])
def test_export_rejects_missing_incomplete_or_failed_participant_audits(tmp_path, domain, change):
    export_fixture(tmp_path, domain)
    root = tmp_path / domain
    path = root / ("run_report.json" if domain == "mlcr" else "09_participant_final_answer_audit.json")
    value = json.loads(path.read_text())
    audit = value.get("participant_final_answer_audit") if domain == "mlcr" else value
    if change == "missing":
        if domain == "mlcr": value.pop("participant_final_answer_audit")
        else: path.unlink()
    elif change == "empty": audit["participants"] = []
    elif change == "partial": audit["participants"].pop()
    elif change == "duplicate": audit["participants"][1] = audit["participants"][0]
    elif change == "failed": audit["passed"] = False
    elif change == "string_false": audit["passed"] = "false"
    elif change == "budget": audit["participants"][0]["participant_context_token_budget"] = 999
    if change != "missing" or domain == "mlcr":
        path.write_text(json.dumps(value))
        # Even an internally consistent receipt must not override failed audits.
        seal_fixture(root, domain)
    with pytest.raises(ValueError): exporter(root, domain)


@pytest.mark.parametrize("domain", ["evidencebench", "mlcr"])
@pytest.mark.parametrize("change", ["task", "packet", "manifest", "missing_receipt"])
def test_export_rejects_stale_certification(tmp_path, domain, change):
    export_fixture(tmp_path, domain)
    root = tmp_path / domain
    if domain == "mlcr":
        path = root / "run_report.json"
        value = json.loads(path.read_text())
        if change == "task": value["task"]["public_task"]["question"] = "Changed task"
        elif change == "packet": value["partition"]["contexts"][0]["text"] = "Changed text"
        elif change == "missing_receipt": value.pop("export_certification")
        else: (root / "run_manifest.json").write_text("{}")
    else:
        name = {"task": "02_task_rubric.json", "packet": "06_participant_packets.hidden.json",
                "manifest": "run_manifest.json", "missing_receipt": "summary.json"}[change]
        path = root / name; value = json.loads(path.read_text())
        if change == "task": value["public_task"]["question"] = "Changed task"
        elif change == "packet": value["contexts"][0]["text"] = "Changed text"
        elif change == "missing_receipt": value.pop("export_certification")
        else: value["changed"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="certif"): exporter(root, domain)


@pytest.mark.parametrize("domain", ["evidencebench", "mlcr"])
def test_configured_budget_is_preserved_in_export(tmp_path, domain):
    export_fixture(tmp_path, domain)
    complete_fixture(tmp_path / domain, domain, token_budget=1234)
    row = exporter(tmp_path / domain, domain)
    assert row["dataset_specific"]["audited_participant_context_token_budget"] == 1234


@pytest.mark.parametrize("field", ["grounding_audit", "boundary_audits", "packet_leakage_audit"])
@pytest.mark.parametrize("passed", [False, "false", "true", 1])
def test_mlcr_requires_boolean_true_construction_audits(tmp_path, field, passed):
    export_fixture(tmp_path, "mlcr")
    root = tmp_path / "mlcr"; path = root / "run_report.json"
    report = json.loads(path.read_text()); report[field]["passed"] = passed
    path.write_text(json.dumps(report)); seal_fixture(root, "mlcr")
    with pytest.raises(ValueError, match="passed"): exporter(root, "mlcr")


@pytest.mark.parametrize("domain", ["evidencebench", "mlcr"])
def test_success_flag_cannot_hide_missing_loo_case(tmp_path, domain):
    export_fixture(tmp_path, domain); root = tmp_path / domain
    path = root / ("run_report.json" if domain == "mlcr" else "04_boundaries.json")
    value = json.loads(path.read_text())
    audit = value["boundary_audits"] if domain == "mlcr" else value
    audit["cases"].pop(); path.write_text(json.dumps(value)); seal_fixture(root, domain)
    with pytest.raises(ValueError, match="ALL/NONE/LOO"): exporter(root, domain)
