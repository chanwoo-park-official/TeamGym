"""Parse DataCurve DeepSWE source tasks without exposing evaluation data."""

from __future__ import annotations

import json
from pathlib import Path
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from typing import Any, Iterator

from .schema import (
    DeepSWEArtifacts,
    DeepSWERepositorySpec,
    DeepSWERuntimeSpec,
    DeepSWETask,
    DeepSWEValidationError,
)


REQUIRED_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/Dockerfile",
    "tests/config.json",
    "tests/grader.py",
    "tests/test.patch",
    "tests/test.sh",
    "solution/solution.patch",
    "solution/solve.sh",
)


def iter_task_dirs(dataset_root: str | Path) -> Iterator[Path]:
    """Yield DeepSWE task directories in stable task-id order.

    Accepted inputs are the corpus root (containing ``tasks``), the tasks
    directory itself, or a directory containing a single task directory.
    """

    root = Path(dataset_root).expanduser().resolve()
    if (root / "tasks").is_dir():
        root = root / "tasks"
    elif (root / "data" / "tasks").is_dir():
        root = root / "data" / "tasks"
    if not root.is_dir():
        raise DeepSWEValidationError(f"DeepSWE task root does not exist: {root}")

    task_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "task.toml").is_file()
    )
    if not task_dirs:
        raise DeepSWEValidationError(f"No DeepSWE task directories found under {root}")
    yield from task_dirs


def load_task(task_dir: str | Path, *, runtime_image: str | None = None) -> DeepSWETask:
    """Load and validate one DeepSWE task from its source checkout."""

    root = Path(task_dir).expanduser().resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise DeepSWEValidationError(f"{root.name}: missing required files: {missing}")

    try:
        config = tomllib.loads(_read_text(root / "task.toml"))
    except tomllib.TOMLDecodeError as exc:
        raise DeepSWEValidationError(f"{root.name}: invalid task.toml: {exc}") from exc

    schema_version = str(_required(config, "schema_version"))
    if schema_version != "1.3":
        raise DeepSWEValidationError(f"{root.name}: unsupported schema_version {schema_version!r}")
    task_section = _required_mapping(config, "task")
    task_name = str(_required(task_section, "name"))
    task_id = task_name.rsplit("/", 1)[-1]
    if task_id != root.name:
        raise DeepSWEValidationError(
            f"Task identity mismatch: directory={root.name!r}, task.name={task_name!r}"
        )

    metadata = _required_mapping(config, "metadata")
    metadata_task_id = str(_required(metadata, "task_id"))
    if metadata_task_id != task_id:
        raise DeepSWEValidationError(
            f"{task_id}: metadata.task_id={metadata_task_id!r} does not match task id"
        )

    environment = _required_mapping(config, "environment")
    verifier = _required_mapping(config, "verifier")
    _required_mapping(verifier, "environment")
    agent = _required_mapping(config, "agent")

    test_config = _load_json(root / "tests" / "config.json")
    _validate_test_config(task_id, test_config)
    base_commit = _required(metadata, "base_commit_hash")
    if not isinstance(base_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", base_commit):
        raise DeepSWEValidationError(f"{task_id}: base_commit_hash must be a Git commit hash (7-40 hexadecimal characters)")
    if base_commit != test_config["base_commit"]:
        raise DeepSWEValidationError(f"{task_id}: metadata.base_commit_hash and tests/config.json base_commit disagree")

    return DeepSWETask(
        task_id=task_id,
        source_dir=root,
        schema_version=schema_version,
        artifacts_declared=tuple(_string_list(config.get("artifacts"), "artifacts")),
        repository=DeepSWERepositorySpec(
            repository_url=str(_required(metadata, "repository_url")),
            base_commit_hash=base_commit,
            language=str(_required(metadata, "language")),
            category=str(_required(metadata, "category")),
            task_id=metadata_task_id,
            display_title=str(_required(metadata, "display_title")),
            display_description=str(_required(metadata, "display_description")),
        ),
        runtime=DeepSWERuntimeSpec(
            docker_image=runtime_image or str(_required(environment, "docker_image")),
            network_mode=str(_required(verifier, "network_mode")),
            verifier_environment_mode=str(_required(verifier, "environment_mode")),
            os=str(_required(environment, "os")),
            cpus=_positive_int(environment, "cpus"),
            memory_mb=_positive_int(environment, "memory_mb"),
            storage_mb=_positive_int(environment, "storage_mb"),
            gpus=_nonnegative_int(environment, "gpus"),
            build_timeout_sec=_positive_float(environment, "build_timeout_sec"),
            verifier_timeout_sec=_positive_float(verifier, "timeout_sec"),
            agent_timeout_sec=_positive_float(agent, "timeout_sec"),
        ),
        instruction=_read_nonempty(root / "instruction.md"),
        artifacts=DeepSWEArtifacts(
            environment_dockerfile=root / "environment" / "Dockerfile",
            tests_dockerfile=root / "tests" / "Dockerfile",
            test_config=root / "tests" / "config.json",
            test_grader=root / "tests" / "grader.py",
            test_patch=root / "tests" / "test.patch",
            test_script=root / "tests" / "test.sh",
            solution_patch=root / "solution" / "solution.patch",
            solve_script=root / "solution" / "solve.sh",
            pre_artifacts_script=_optional_file(root / "pre_artifacts.sh"),
        ),
        test_config=test_config,
    )


def _validate_test_config(task_id: str, config: dict[str, Any]) -> None:
    """Validate the stable verifier contract without interpreting test IDs."""

    for key in ("base_commit", "f2p_node_ids", "p2p_node_ids", "grade"):
        if key not in config:
            raise DeepSWEValidationError(f"{task_id}: tests/config.json lacks {key!r}")
    if not isinstance(config["base_commit"], str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", config["base_commit"]):
        raise DeepSWEValidationError(f"{task_id}: base_commit must be a Git commit hash (7-40 hexadecimal characters)")
    for key in ("f2p_node_ids", "p2p_node_ids"):
        values = config[key]
        if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
            raise DeepSWEValidationError(f"{task_id}: {key} must be an array of nonempty strings")
        if len(values) != len(set(values)):
            raise DeepSWEValidationError(f"{task_id}: duplicate test ids in {key}")
    if not config["f2p_node_ids"]:
        raise DeepSWEValidationError(f"{task_id}: f2p_node_ids must not be empty")
    if set(config["f2p_node_ids"]) & set(config["p2p_node_ids"]):
        raise DeepSWEValidationError(f"{task_id}: f2p and p2p test ids overlap")
    if not isinstance(config["grade"], dict):
        raise DeepSWEValidationError(f"{task_id}: grade must be an object")


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] is None or mapping[key] == "":
        raise DeepSWEValidationError(f"Missing required task.toml field: {key}")
    return mapping[key]


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise DeepSWEValidationError(f"task.toml field {key!r} must be a table")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DeepSWEValidationError(f"task.toml field {field!r} must be an array of strings")
    return list(value)


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = int(_required(mapping, key))
    if value <= 0:
        raise DeepSWEValidationError(f"task.toml field {key!r} must be positive")
    return value


def _nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = int(_required(mapping, key))
    if value < 0:
        raise DeepSWEValidationError(f"task.toml field {key!r} must be non-negative")
    return value


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    value = float(_required(mapping, key))
    if value <= 0:
        raise DeepSWEValidationError(f"task.toml field {key!r} must be positive")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise DeepSWEValidationError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeepSWEValidationError(f"Expected JSON object: {path}")
    return value


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_nonempty(path: Path) -> str:
    text = _read_text(path).strip()
    if not text:
        raise DeepSWEValidationError(f"Required source text is empty: {path}")
    return text
