from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.core.participant_roles import (
    normalize_information_features,
    private_rui_assignment,
    role_name_from_information_features,
)
from teamgym.data.legal.participant_role_augmentation import ExcludedSourceCorpusError, source_corpus


@pytest.mark.parametrize(("module", "blocked_domain"), [
    ("teamgym.core.participant_roles", "teamgym.data"),
    ("teamgym.data.long_context_reasoning.finance_participant_roles",
     "teamgym.data.legal"),
])
def test_role_helpers_import_without_legal_or_optional_dependencies(module: str, blocked_domain: str) -> None:
    code = (
        "import importlib, sys; "
        f"importlib.import_module({module!r}); "
        f"blocked = ({blocked_domain!r}, 'rllm', 'verl', 'torch', 'pandas', 'openai', 'tiktoken'); "
        "assert not [name for name in sys.modules "
        "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_cuad_and_maud_are_explicitly_excluded() -> None:
    with pytest.raises(ExcludedSourceCorpusError):
        source_corpus({"dataset": "cuad"})
    with pytest.raises(ExcludedSourceCorpusError):
        source_corpus({"dataset": "maud"})


def test_target_source_corpora_are_detected() -> None:
    assert source_corpus({"source": {"source_family": "ecfr_casebook20"}}) == "casebook"
    assert source_corpus({"source": {"source_family": "mslr_insider_trading_source_excerpt"}}) == "mslr"
    assert source_corpus({"source": {"source_family": "Multi-LexSum"}}) == "multilexsum"
    assert source_corpus({"source": {"source_family": "sara_target_sliced_subproof"}}) == "sara"


def test_information_feature_response_requires_exact_riu_coverage() -> None:
    response = {
        "information_features": [
            {"rui_id": "rui_01", "information_feature": "party knowledge"},
            {"rui_id": "rui_02", "information_feature": "trading timing"},
        ]
    }
    assert normalize_information_features(
        response,
        expected_rui_ids=["rui_01", "rui_02"],
    ) == {"rui_01": "party knowledge", "rui_02": "trading timing"}

    with pytest.raises(ValueError, match="coverage mismatch"):
        normalize_information_features(response, expected_rui_ids=["rui_01", "rui_03"])


def test_information_feature_rejects_private_values_and_conclusions() -> None:
    for feature in ("income $75,845", "purchase exceeds threshold", "approved transaction"):
        with pytest.raises(ValueError):
            normalize_information_features(
                {
                    "information_features": [
                        {"rui_id": "rui_01", "information_feature": feature}
                    ]
                },
                expected_rui_ids=["rui_01"],
            )


def test_common_riu_is_excluded_from_role_assignment() -> None:
    row = {
        "instance_id": "example__partition_common_anchor",
        "dataset_specific": {"partition_mode": "common_anchor"},
        "participants": [
            {
                "participant_id": "participant_a",
                "owned_rui_ids": ["private_a", "common"],
            },
            {
                "participant_id": "participant_b",
                "owned_rui_ids": ["private_b", "common"],
            },
            {
                "participant_id": "participant_c",
                "owned_rui_ids": ["common"],
            },
        ],
    }
    assert private_rui_assignment(row) == {
        "participant_a": ["private_a"],
        "participant_b": ["private_b"],
        "participant_c": [],
    }


def test_role_names_follow_cuad_maud_rendering() -> None:
    assert role_name_from_information_features(["party knowledge"], 1) == "party knowledge reviewer"
    assert (
        role_name_from_information_features(
            ["filing-status rule", "income amount"],
            2,
        )
        == "filing status and income amount reviewer"
    )
    assert role_name_from_information_features([], 3) == "context reviewer 3"
