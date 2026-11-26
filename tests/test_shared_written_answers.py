from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.scoring.short_answers import SYSTEM_PROMPT
from teamgym.scoring import written_answers as scoring
from teamgym.rl import written_reward as legacy_scoring


def test_saved_nonfinite_raw_score_cannot_be_rewarded():
    judgment = {"judgment": {"score": 1.0, "raw_response": {"score": "NaN"}}}
    with pytest.raises(ValueError, match="Judge score must be finite"):
        scoring.holistic_score(judgment)


def _datapoint() -> dict:
    return {
        "instance_id": "written_case",
        "dataset": "cuad",
        "task_description": {
            "question": "Is the project within budget? Explain the cost and limit.",
            "answer_type": "short_answer_judged",
        },
        "oracle": {
            "canonical_response": "The project costs 40 dollars against a 50-dollar limit.",
            "required_decision_steps": [
                {"step": "cost", "uses": ["r1"], "claim": "The project costs 40 dollars."},
                {"step": "limit", "uses": ["r2"], "claim": "The limit is 50 dollars."},
            ],
            "judge_rubric": {"require_all_steps_for_pass": True},
            "disqualifying_errors": ["Claiming the project exceeds the limit."],
        },
        "reference_information_units": [
            {"rui_id": "r1", "text": "Cost: 40 dollars.\u2028Signed estimate."},
            {"rui_id": "r2", "text": "Authorized limit: 50 dollars."},
        ],
    }


def _episode() -> dict:
    return {
        "final_decision": {"written_answer": "  Cost is 40 dollars, below the 50-dollar limit.  "},
        "communication_structure": {"name": "coordinator_select_one__coordinator_decision"},
    }


@pytest.fixture
def clients(monkeypatch):
    created = []

    class CapturingClient:
        def __init__(self, config):
            self.config = config
            self.calls = []
            self.last_request_meta = {"provider": "mock", "model": "mock-judge"}
            created.append(self)

        def generate_json(self, **request):
            self.calls.append(deepcopy(request))
            return {
                "passed": True, "score": 0.63, "conclusion_correct": True,
                "covered_step_count": 2, "total_step_count": 2,
                "step_scores": [
                    {"step": "cost", "covered": True, "score": 1.0},
                    {"step": "limit", "covered": True, "score": 1.0},
                ],
                "missing_steps": [], "incorrect_claims": [],
                "disqualifying_errors_found": [], "reason": "Mocked evidence assessment.",
            }

    monkeypatch.setattr(scoring, "OpenAIJsonClient", CapturingClient)
    return created


def test_previous_scorer_path_exports_the_shared_implementation() -> None:
    for name in (
        "CACHE_SCHEMA", "JUDGE_SCHEMA", "WrittenJudgeConfig", "_cache_path",
        "is_written_answer_task", "written_answer_from_episode", "holistic_score",
        "judge_written_episode",
    ):
        assert getattr(legacy_scoring, name) is getattr(scoring, name)


def test_shared_scorer_imports_without_the_rl_stack() -> None:
    subprocess.run(
        [
            sys.executable, "-c",
            "import sys; import teamgym.scoring.written_answers; "
            "blocked = ('teamgym.data', "
            "'teamgym.interaction', 'teamgym.rl', 'rllm', 'verl', 'torch'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )


def test_judge_prompts_and_written_cache_are_shared_across_import_paths(tmp_path, clients) -> None:
    row, episode = _datapoint(), _episode()
    config = scoring.WrittenJudgeConfig(cache_dir=str(tmp_path))
    result = scoring.judge_written_episode(datapoint=row, episode=episode, config=config)
    assert result["cache_hit"] is False
    assert scoring.holistic_score(result) == 0.63
    assert len(clients) == 1
    request = clients[0].calls[0]
    assert request["system_prompt"] == SYSTEM_PROMPT
    assert request["prompt_id"] == "written_case.short_answer_judge.0001"
    assert request["trace_dir"] is None
    payload = json.loads(request["user_prompt"])
    assert payload["candidate_answer"] == episode["final_decision"]["written_answer"].strip()
    assert payload["canonical_response"] == row["oracle"]["canonical_response"]
    assert payload["required_decision_steps"] == row["oracle"]["required_decision_steps"]
    assert payload["judge_rubric"] == row["oracle"]["judge_rubric"]
    assert payload["reference_information_units"][0]["text"] == row["reference_information_units"][0]["text"]
    assert payload["candidate_metadata"]["answer_source"] == "teamgym_rl_rollout"
    options = asdict(clients[0].config)
    assert options["temperature"] == 0
    assert options["request_log_path"] is None
    assert options["log_prompts"] is False and options["log_responses"] is False
    path = scoring._cache_path(datapoint=row, answer_text=payload["candidate_answer"], config=config)
    saved = path.read_bytes()
    assert result["schema"] == "long_context_short_answer_judgment_v2"
    assert json.loads(saved)["cache_schema"] == "teamgym_written_reward_cache_v2"
    assert json.loads(saved)["judge_schema"] == result["schema"]
    assert list(tmp_path.glob("*.tmp")) == []
    hit = legacy_scoring.judge_written_episode(datapoint=row, episode=episode, config=config)
    assert hit == {**result, "cache_hit": True}
    assert len(clients) == 1 and len(clients[0].calls) == 1
    assert path.read_bytes() == saved


def test_existing_cache_can_be_read_without_creating_a_client(tmp_path, monkeypatch) -> None:
    def forbidden_client(_config):
        pytest.fail("A cache hit must not initialize an API client")

    monkeypatch.setattr(scoring, "OpenAIJsonClient", forbidden_client)
    row, episode = _datapoint(), _episode()
    config = scoring.WrittenJudgeConfig(cache_dir=str(tmp_path))
    path = scoring._cache_path(datapoint=row, answer_text=scoring.written_answer_from_episode(episode), config=config)
    saved = {"schema": scoring.JUDGE_SCHEMA, "judgment": {"score": 0.42}, "candidate_answer": "cached answer"}
    path.write_text(json.dumps({"cache_schema": scoring.CACHE_SCHEMA, "judge_schema": scoring.JUDGE_SCHEMA, "judgment": saved}))
    assert scoring.judge_written_episode(datapoint=row, episode=episode, config=config) == {**saved, "cache_hit": True}


def test_v1_judgment_cache_is_isolated_from_current_grading(tmp_path, clients) -> None:
    row, episode = _datapoint(), _episode()
    config = scoring.WrittenJudgeConfig(cache_dir=str(tmp_path))
    # This filename was generated for this fixture by the pre-integration scorer.
    legacy_path = tmp_path / "1bc57ce7eafc59f448c6e52375701d576e29ba1a2be57ec42d46da283cfe5764.json"
    legacy_entry = json.dumps({
        "cache_schema": "teamgym_written_reward_cache_v1",
        "judge_schema": "long_context_short_answer_judgment_v1",
        "judgment": {"schema": "long_context_short_answer_judgment_v1", "judgment": {"score": 1.0}},
    })
    legacy_path.write_text(legacy_entry)

    result = scoring.judge_written_episode(datapoint=row, episode=episode, config=config)

    assert result["cache_hit"] is False and len(clients) == 1
    assert scoring.holistic_score(result) == 0.63
    assert legacy_path.read_text() == legacy_entry
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize("invalid", [
    "not JSON",
    json.dumps({"cache_schema": "old-version", "judge_schema": scoring.JUDGE_SCHEMA,
                "judgment": {"schema": scoring.JUDGE_SCHEMA}}),
    json.dumps({"cache_schema": scoring.CACHE_SCHEMA, "judge_schema": scoring.JUDGE_SCHEMA,
                "judgment": []}),
    json.dumps({"cache_schema": scoring.CACHE_SCHEMA, "judge_schema": "long_context_short_answer_judgment_v1",
                "judgment": {"schema": scoring.JUDGE_SCHEMA}}),
    json.dumps({"cache_schema": scoring.CACHE_SCHEMA, "judge_schema": scoring.JUDGE_SCHEMA,
                "judgment": {"schema": "long_context_short_answer_judgment_v1"}}),
])
def test_invalid_cache_entries_are_rejudged(tmp_path, clients, invalid) -> None:
    row, episode = _datapoint(), _episode()
    config = scoring.WrittenJudgeConfig(cache_dir=str(tmp_path))
    path = scoring._cache_path(datapoint=row, answer_text=scoring.written_answer_from_episode(episode), config=config)
    path.write_text(invalid)
    result = scoring.judge_written_episode(datapoint=row, episode=episode, config=config)
    assert result["cache_hit"] is False and len(clients) == 1
    assert json.loads(path.read_text())["judgment"] == result


def test_cache_keys_distinguish_answers_rubrics_and_models(tmp_path) -> None:
    row = _datapoint()
    config = scoring.WrittenJudgeConfig(cache_dir=str(tmp_path))
    key = scoring._cache_path(datapoint=row, answer_text="first answer", config=config)
    assert key == legacy_scoring._cache_path(datapoint=deepcopy(row), answer_text="first answer", config=config)
    assert key != scoring._cache_path(datapoint=row, answer_text="different answer", config=config)
    assert key != scoring._cache_path(datapoint=row, answer_text="first answer", config=replace(config, model="another-model"))
    changed = deepcopy(row)
    changed["oracle"]["judge_rubric"]["require_all_steps_for_pass"] = False
    assert key != scoring._cache_path(datapoint=changed, answer_text="first answer", config=config)
    assert scoring._cache_path(datapoint=row, answer_text="first answer", config=replace(config, cache_dir=None)) is None


def test_judge_failures_propagate_without_caching(tmp_path, monkeypatch) -> None:
    def failed_client(_config):
        raise RuntimeError("mocked judge failure")

    monkeypatch.setattr(scoring, "OpenAIJsonClient", failed_client)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="mocked judge failure"):
        scoring.judge_written_episode(datapoint=_datapoint(), episode=_episode(), config=scoring.WrittenJudgeConfig(cache_dir=str(cache)))
    assert not cache.exists()


def test_empty_answers_fail_before_client_initialization(tmp_path, clients) -> None:
    with pytest.raises(ValueError, match="no non-empty final answer"):
        scoring.judge_written_episode(datapoint=_datapoint(), episode={"final_decision": {"written_answer": "  "}}, config=scoring.WrittenJudgeConfig(cache_dir=str(tmp_path)))
    assert clients == [] and list(tmp_path.iterdir()) == []


def test_task_detection_answer_extraction_and_score_bounds() -> None:
    assert scoring.is_written_answer_task(_datapoint())
    assert scoring.is_written_answer_task({"decision_set": ["free_response"]})
    assert not scoring.is_written_answer_task({"decision_set": ["yes", "no"]})
    assert scoring.written_answer_from_episode({"final_decision": {"rationale": " fallback "}}) == "fallback"
    assert scoring.written_answer_from_episode(None) == ""
    for raw, expected in ((0.63, 0.63), (2, 1), (-1, 0), ("invalid", 0), (None, 0)):
        assert scoring.holistic_score({"judgment": {"score": raw}}) == expected
    assert scoring.holistic_score(None) == 0
