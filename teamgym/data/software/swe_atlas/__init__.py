"""SWE-Atlas QA source adapter.

The package intentionally starts with source ingestion only. Evidence, RIUs,
partitions, and experiment inputs are separate downstream stages.
"""

from .source import iter_qa_task_dirs, load_qa_task

__all__ = ["iter_qa_task_dirs", "load_qa_task"]
