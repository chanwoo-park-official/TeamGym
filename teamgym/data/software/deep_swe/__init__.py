"""Source ingestion for the DataCurve DeepSWE task corpus.

This package intentionally stops at immutable task parsing.  RIU construction,
workspace execution, and verifier evaluation are downstream stages.
"""

from .schema import (
    DeepSWEArtifacts,
    DeepSWERepositorySpec,
    DeepSWERuntimeSpec,
    DeepSWETask,
    DeepSWEValidationError,
)
from .source import iter_task_dirs, load_task

__all__ = [
    "DeepSWEArtifacts",
    "DeepSWERepositorySpec",
    "DeepSWERuntimeSpec",
    "DeepSWETask",
    "DeepSWEValidationError",
    "iter_task_dirs",
    "load_task",
]
