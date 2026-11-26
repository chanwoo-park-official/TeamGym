"""Lossless source-level schema for SWE-Atlas QA tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


RubricPolarity = Literal["positive", "negative"]


class SourceValidationError(ValueError):
    """Raised when an original SWE-Atlas task is incomplete or malformed."""


@dataclass(frozen=True)
class Rubric:
    rubric_id: str
    title: str
    polarity: RubricPolarity
    importance: str
    annotations: dict[str, Any]


@dataclass(frozen=True)
class RepositorySpec:
    repository: str
    base_commit: str
    language: str
    category: str


@dataclass(frozen=True)
class RuntimeSpec:
    docker_image: str
    allow_internet: bool
    cpus: int
    memory_mb: int
    storage_mb: int
    build_timeout_sec: float
    verifier_timeout_sec: float
    agent_timeout_sec: float


@dataclass(frozen=True)
class PublicTask:
    """Material that may be shown to benchmark solvers."""

    question: str
    instruction: str


@dataclass(frozen=True)
class EvaluationSpec:
    """Hidden source material used for construction and final evaluation."""

    rubrics: tuple[Rubric, ...]
    reference_answer: str
    judge_system_prompt: str
    judge_user_prompt_template: str


@dataclass(frozen=True)
class SWEAtlasQATask:
    task_id: str
    source_dir: Path
    repository: RepositorySpec
    runtime: RuntimeSpec
    public: PublicTask
    evaluation: EvaluationSpec

    def public_payload(self) -> dict[str, Any]:
        """Return a solver-safe view with no rubric or reference-answer fields."""

        return {
            "task_id": self.task_id,
            "question": self.public.question,
            "repository": self.repository.repository,
            "base_commit": self.repository.base_commit,
            "language": self.repository.language,
            "category": self.repository.category,
        }
