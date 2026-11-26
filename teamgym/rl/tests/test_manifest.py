from __future__ import annotations

from teamgym.rl.training.manifest import _optional_env, _optional_env_int


def test_optional_environment_provenance(monkeypatch) -> None:
    monkeypatch.setenv("TEAMGYM_TEST_STRING", "four-gpu")
    monkeypatch.setenv("TEAMGYM_TEST_INTEGER", "4")

    assert _optional_env("TEAMGYM_TEST_STRING") == "four-gpu"
    assert _optional_env_int("TEAMGYM_TEST_INTEGER") == 4
    assert _optional_env("TEAMGYM_TEST_MISSING") is None


def test_optional_environment_integer_rejects_non_integer(monkeypatch) -> None:
    monkeypatch.setenv("TEAMGYM_TEST_INTEGER", "not-an-integer")
    assert _optional_env_int("TEAMGYM_TEST_INTEGER") is None
