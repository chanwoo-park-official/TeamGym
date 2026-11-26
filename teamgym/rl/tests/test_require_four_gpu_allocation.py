from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).parents[1] / "require_four_gpu_allocation.sh"


@pytest.mark.parametrize(
    ("detected", "attempt", "expected"),
    [
        (4, 1, "run"),
        (4, 5, "run"),
        (2, 1, "requeue"),
        (3, 4, "requeue"),
        (2, 5, "fail"),
        (3, 5, "fail"),
        (1, 5, "fail"),
    ],
)
def test_four_gpu_attempt_action(
    detected: int, attempt: int, expected: str
) -> None:
    command = f"""
set -euo pipefail
source {SCRIPT!s}
teamgym_four_gpu_attempt_action {detected} {attempt} 5
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected
