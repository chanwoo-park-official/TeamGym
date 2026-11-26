from types import SimpleNamespace
import pytest


def test_evidencebench_cli_accepts_independent_audit_model_and_budgets(monkeypatch):
    import sys
    from teamgym.data.medical.evidencebench.pipeline import _parse_args
    monkeypatch.setattr(sys, "argv", ["run", "--source-path", "source", "--output-root", "out",
        "--model", "constructor", "--audit-model", "auditor", "--participant-token-budget", "12345",
        "--max-input-tokens", "23456"])
    args = _parse_args()
    assert (args.model, args.verification_model, args.participant_token_budget, args.max_input_tokens) == (
        "constructor", "auditor", 12345, 23456)


def test_mlcr_cli_forwards_model_and_budget_options(monkeypatch, tmp_path):
    import sys
    from teamgym.data.medical.mlcr import cli
    captured = {}
    monkeypatch.setattr(sys, "argv", ["run", "--source-dir", str(tmp_path), "--case-id", "case",
        "--output-root", str(tmp_path), "--model", "constructor", "--audit-model", "auditor",
        "--participant-token-budget", "12345", "--max-input-tokens", "23456"])
    monkeypatch.setattr(cli, "load_inspectable", lambda _: ([], []))
    monkeypatch.setattr(cli, "build_case_indexes", lambda *_: [{"case_uuid": "case"}])
    monkeypatch.setattr(cli, "OpenAIJsonClient", lambda cfg: SimpleNamespace(cfg=cfg))
    def run(**kw): captured.update(kw); return {"status": "fixture_not_accepted"}
    monkeypatch.setattr(cli, "run_case", run)
    cli.main()
    assert captured["generation_client"].cfg.model == "constructor"
    assert captured["verification_client"].cfg.model == "auditor"
    assert captured["participant_token_budget"] == 12345
    assert captured["max_input_tokens"] == 23456


@pytest.mark.parametrize("domain", ["evidencebench", "mlcr"])
def test_input_budget_preflight_prevents_a_model_request(tmp_path, domain):
    module = __import__(f"teamgym.data.medical.{domain}.llm", fromlist=["structured_call"])
    class Cache:
        max_input_tokens = 1
        def call(self, **kw): raise AssertionError("over-budget request must not reach the model")
    variables = {"PUBLIC_TASK_JSON": {"question": "Q"}, "AVAILABLE_MATERIAL_JSON": [], "DISPLAY_MATERIALS_JSON": []}
    key = "instance_id" if domain == "evidencebench" else "case_id"
    _, errors = module.structured_call(client=None, cache=Cache(), trace_root=tmp_path,
        **{key: "fixture"}, stage="test", user_prompt_name="04_boundary_solver_user.txt",
        variables=variables, validator=lambda _: [], attempts=1)
    assert "input budget exceeded" in errors[0]
