
from tests._paths import REPO_ROOT
import os
from pathlib import Path

import pytest

from teamgym.data.software.swe_atlas.source import (
    iter_qa_task_dirs,
    load_qa_task,
)


REPOSITORY_ROOT = REPO_ROOT
DATASET_ROOT = Path(os.environ.get(
    "SWE_ATLAS_DATASET_ROOT", str(REPOSITORY_ROOT / ".external_datasets" / "SWE-Atlas"),
)).expanduser()


requires_source_checkout = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="the external SWE-Atlas checkout is intentionally not stored in this repository",
)


@requires_source_checkout
def test_loads_qa_task_without_leaking_hidden_evaluation_fields() -> None:
    task = load_qa_task(next(iter_qa_task_dirs(DATASET_ROOT)))

    public = task.public_payload()
    assert "question" in public
    assert "rubrics" not in public
    assert "reference_answer" not in public
    assert public["task_id"] == task.source_dir.name
    assert public["base_commit"] == task.repository.base_commit


@requires_source_checkout
def test_all_qa_tasks_have_unique_identities_and_valid_rubrics() -> None:
    task_dirs = list(iter_qa_task_dirs(DATASET_ROOT))
    tasks = [load_qa_task(path) for path in task_dirs]
    rubrics = [rubric for task in tasks for rubric in task.evaluation.rubrics]

    assert tasks
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert all(task.task_id == task.source_dir.name for task in tasks)
    assert all(task.runtime.docker_image and task.repository.base_commit for task in tasks)
    assert all(
        len({rubric.rubric_id for rubric in task.evaluation.rubrics})
        == len(task.evaluation.rubrics)
        for task in tasks
    )
    assert all(rubric.polarity in {"positive", "negative"} for rubric in rubrics)


def test_discovers_current_and_historical_qa_directory_names(tmp_path: Path) -> None:
    root = tmp_path / "data" / "qa"
    for name in ("qa-current", "task-historical", "tw-unrelated"):
        (root / name).mkdir(parents=True)
    (root / "qa-file").touch()

    assert [path.name for path in iter_qa_task_dirs(tmp_path)] == [
        "qa-current", "task-historical",
    ]
