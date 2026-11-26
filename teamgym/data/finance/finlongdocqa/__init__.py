"""FinLongDocQA block indexing, task generation, auditing, and partitioning."""

from __future__ import annotations

from teamgym.data.finance.finlongdocqa.partition import (
    build_partitioned_datapoint,
    repartition_common_datapoint,
)
from teamgym.data.finance.finlongdocqa.source import (
    index_report,
)

__all__ = [
    "build_partitioned_datapoint",
    "repartition_common_datapoint",
    "index_report",
]
