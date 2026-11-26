"""Read original SWE-Atlas QA tasks without constructing derived evidence."""

from __future__ import annotations

import json
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from typing import Any, Iterator

from .schema import (
    EvaluationSpec,
    PublicTask,
    RepositorySpec,
    Rubric,
    RuntimeSpec,
    SourceValidationError,
    SWEAtlasQATask,
)


REQUIRED_FILES = (
    "task.toml",
    "instruction.md",
    "solution/answer.txt",
    "tests/prompt.txt",
    "tests/rubrics.json",
    "tests/system_prompt.txt",
    "tests/user_prompt_template.txt",
)


def iter_qa_task_dirs(dataset_root: str | Path) -> Iterator[Path]:
    """Yield QA task directories in stable task-id order."""

    root = Path(dataset_root).expanduser().resolve()
    if root.name != "qa" and (root / "data" / "qa").is_dir():
        root = root / "data" / "qa"
    if not root.is_dir():
        raise SourceValidationError(f"SWE-Atlas QA root does not exist: {root}")
    task_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith(("task-", "qa-"))
    )
    if not task_dirs:
        raise SourceValidationError(f"No QA task directories found under {root}")
    yield from task_dirs


def load_qa_task(task_dir: str | Path) -> SWEAtlasQATask:
    """Load one immutable source task and validate its cross-file identity."""

    root = Path(task_dir).expanduser().resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise SourceValidationError(f"{root.name}: missing required files: {missing}")

    config = tomllib.loads(_read_text(root / "task.toml"))
    task_id = str(_required(config, "task", "name")).rsplit("/", 1)[-1]
    if task_id != root.name:
        raise SourceValidationError(
            f"Task identity mismatch: directory={root.name!r}, task.name={task_id!r}"
        )

    metadata = _required_mapping(config, "metadata")
    environment = _required_mapping(config, "environment")
    verifier = _required_mapping(config, "verifier")
    agent = _required_mapping(config, "agent")
    rubrics = _load_rubrics(root / "tests" / "rubrics.json")

    return SWEAtlasQATask(
        task_id=task_id,
        source_dir=root,
        repository=RepositorySpec(
            repository=str(_required(metadata, "repository")),
            base_commit=str(_required(metadata, "base_commit")),
            language=str(_required(metadata, "language")),
            category=str(_required(metadata, "category")),
        ),
        runtime=RuntimeSpec(
            docker_image=str(_required(environment, "docker_image")),
            allow_internet=bool(environment.get("allow_internet", False)),
            cpus=int(_required(environment, "cpus")),
            memory_mb=int(_required(environment, "memory_mb")),
            storage_mb=int(_required(environment, "storage_mb")),
            build_timeout_sec=float(_required(environment, "build_timeout_sec")),
            verifier_timeout_sec=float(_required(verifier, "timeout_sec")),
            agent_timeout_sec=float(_required(agent, "timeout_sec")),
        ),
        public=PublicTask(
            question=_read_nonempty(root / "tests" / "prompt.txt"),
            instruction=_read_nonempty(root / "instruction.md"),
        ),
        evaluation=EvaluationSpec(
            rubrics=rubrics,
            reference_answer=_read_nonempty(root / "solution" / "answer.txt"),
            judge_system_prompt=_read_nonempty(root / "tests" / "system_prompt.txt"),
            judge_user_prompt_template=_read_nonempty(
                root / "tests" / "user_prompt_template.txt"
            ),
        ),
    )


def _load_rubrics(path: Path) -> tuple[Rubric, ...]:
    raw = json.loads(_read_text(path))
    if not isinstance(raw, list) or not raw:
        raise SourceValidationError(f"Rubrics must be a non-empty list: {path}")
    result: list[Rubric] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise SourceValidationError(f"Rubric {index} is not an object: {path}")
        rubric_id = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        annotations = row.get("annotations") or {}
        if not isinstance(annotations, dict):
            raise SourceValidationError(f"Rubric {rubric_id!r} has invalid annotations")
        rubric_type = str(annotations.get("type") or "").casefold()
        if rubric_type.startswith("positive"):
            polarity = "positive"
        elif rubric_type.startswith("negative"):
            polarity = "negative"
        else:
            raise SourceValidationError(f"Rubric {rubric_id!r} has unknown type {rubric_type!r}")
        if not rubric_id or rubric_id in seen or not title:
            raise SourceValidationError(
                f"Missing or duplicate rubric identity at index {index}: {rubric_id!r}"
            )
        seen.add(rubric_id)
        result.append(
            Rubric(
                rubric_id=rubric_id,
                title=title,
                polarity=polarity,
                importance=str(annotations.get("importance") or row.get("importance") or ""),
                annotations=dict(annotations),
            )
        )
    return tuple(result)


def _required(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise SourceValidationError(f"Missing required task.toml field: {'.'.join(keys)}")
        value = value[key]
    if value is None or value == "":
        raise SourceValidationError(f"Empty required task.toml field: {'.'.join(keys)}")
    return value


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise SourceValidationError(f"task.toml field {key!r} must be a table")
    return value


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_nonempty(path: Path) -> str:
    text = _read_text(path).strip()
    if not text:
        raise SourceValidationError(f"Required source text is empty: {path}")
    return text
