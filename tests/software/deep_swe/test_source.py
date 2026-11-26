
from tests._paths import REPO_ROOT
from pathlib import Path

import pytest

from teamgym.data.software.deep_swe import (
    DeepSWEValidationError,
    iter_task_dirs,
    load_task,
)


REPOSITORY_ROOT = REPO_ROOT
DATASET_ROOT = REPOSITORY_ROOT / ".external_datasets" / "deep-swe"


requires_source = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="the external DeepSWE checkout is intentionally not stored in this repository",
)


@requires_source
def test_loads_koota_task_and_keeps_evaluation_material_out_of_public_payload() -> None:
    task = load_task(DATASET_ROOT / "tasks" / "koota-composite-trait-aspects")

    assert task.task_id == "koota-composite-trait-aspects"
    assert task.repository.repository == "pmndrs/koota"
    assert task.repository.repository_url == "https://github.com/pmndrs/koota"
    assert task.repository.base_commit_hash == "9c434858b2b522002f8c5eb4a554fa8836a7cf3c"
    assert task.runtime.docker_image
    assert task.test_config["base_commit"] == task.repository.base_commit_hash

    public = task.public_payload()
    assert "instruction" in public
    assert "solution_patch" not in public
    assert "test_patch" not in public
    assert "f2p_node_ids" not in public
    assert "p2p_node_ids" not in public
    assert "grader" not in public


@requires_source
def test_parses_all_available_deepswe_tasks() -> None:
    task_dirs = list(iter_task_dirs(DATASET_ROOT))
    tasks = [load_task(path) for path in task_dirs]

    assert tasks
    assert len({task.task_id for task in tasks}) == len(task_dirs)
    assert all(task.repository.base_commit_hash == task.test_config["base_commit"] for task in tasks)


def test_rejects_task_id_mismatch(tmp_path: Path) -> None:
    from tests.software.deep_swe.test_reliability import make_task
    original = make_task(tmp_path)
    task_dir = original.with_name("wrong-name")
    original.rename(task_dir)

    with pytest.raises(DeepSWEValidationError, match="Task identity mismatch"):
        load_task(task_dir)
