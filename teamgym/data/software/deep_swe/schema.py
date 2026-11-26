"""Immutable source-level schema for DataCurve DeepSWE tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DeepSWEValidationError(ValueError):
    """Raised when a DeepSWE task is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class DeepSWERepositorySpec:
    repository_url: str
    base_commit_hash: str
    language: str
    category: str
    task_id: str
    display_title: str
    display_description: str

    @property
    def repository(self) -> str:
        """Return the ``owner/name`` repository slug from its GitHub URL."""

        marker = "github.com/"
        if marker in self.repository_url:
            return self.repository_url.split(marker, 1)[1].rstrip("/")
        return self.repository_url.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class DeepSWERuntimeSpec:
    docker_image: str
    network_mode: str
    verifier_environment_mode: str
    os: str
    cpus: int
    memory_mb: int
    storage_mb: int
    gpus: int
    build_timeout_sec: float
    verifier_timeout_sec: float
    agent_timeout_sec: float


@dataclass(frozen=True)
class DeepSWEArtifacts:
    """Paths for task materials, with evaluation materials kept separate."""

    environment_dockerfile: Path
    tests_dockerfile: Path
    test_config: Path
    test_grader: Path
    test_patch: Path
    test_script: Path
    solution_patch: Path
    solve_script: Path
    pre_artifacts_script: Path | None


@dataclass(frozen=True)
class DeepSWETask:
    """One parsed DeepSWE task.

    ``instruction`` is solver-visible.  The artifact paths are retained for
    construction/evaluation code, but ``public_payload`` deliberately does
    not expose any solution or hidden-test material.
    """

    task_id: str
    source_dir: Path
    schema_version: str
    artifacts_declared: tuple[str, ...]
    repository: DeepSWERepositorySpec
    runtime: DeepSWERuntimeSpec
    instruction: str
    artifacts: DeepSWEArtifacts
    test_config: dict[str, Any]

    def public_payload(self) -> dict[str, Any]:
        """Return only task metadata and the public problem statement."""

        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "repository": self.repository.repository,
            "repository_url": self.repository.repository_url,
            "base_commit": self.repository.base_commit_hash,
            "language": self.repository.language,
            "category": self.repository.category,
            "display_title": self.repository.display_title,
            "runtime": {
                "docker_image": self.runtime.docker_image,
                "network_mode": self.runtime.network_mode,
                "os": self.runtime.os,
                "cpus": self.runtime.cpus,
                "memory_mb": self.runtime.memory_mb,
                "storage_mb": self.runtime.storage_mb,
            },
        }
