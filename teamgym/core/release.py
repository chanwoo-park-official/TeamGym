"""Write accepted tasks in the shared TeamGym dataset format."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from teamgym.core.dataset import extract_common_datapoint
from teamgym.core.io import write_jsonl
from teamgym.core.schema import to_release_datapoint


FULL_CONTEXT = "multi_agent_full_context"
TEAM_PIU_ONLY = "multi_agent_piu_only"

_DISTRIBUTION_NAMES = {
    "one_each": "One PIU per participant",
    "paired": "Paired PIUs",
    "overlap": "Pairwise PIU overlap",
    "redundant_player": "Extra participant without a PIU",
    "common_anchor": "One common PIU",
    "bottleneck": "All-but-one PIU concentration",
}


def _distribution_key(row: dict[str, Any]) -> str | None:
    if "distribution_key" in row:
        return row["distribution_key"]
    specific = row.get("dataset_specific") or {}
    generation = row.get("generation_metadata") or {}
    allocation = (
        specific.get("private_information_allocation")
        or generation.get("private_information_allocation")
        or {}
    )
    return (
        (row.get("_cross") or {}).get("partition_mode")
        or specific.get("partition_mode")
        or specific.get("pilot_partition_mode")
        or specific.get("allocation_mode")
        or generation.get("allocation_mode")
        or allocation.get("mode")
        or generation.get("allocation_policy")
    )


def _evidence_condition(row: dict[str, Any]) -> str:
    if row.get("evidence_condition"):
        return row["evidence_condition"]
    metadata = row.get("generation_metadata") or {}
    specific = row.get("dataset_specific") or {}
    if specific.get("context_variant") == "rui_only_partition" or (
        metadata.get("interaction_context_variant") == "rui_only_private_context"
    ):
        return TEAM_PIU_ONLY
    return FULL_CONTEXT


def _release_row(record: dict[str, Any], evidence_condition: str | None) -> dict[str, Any]:
    row = extract_common_datapoint(record)
    if row is None:
        raise ValueError(f"{record.get('instance_id')}: expected a TeamGym task")
    public = to_release_datapoint(row)
    public.pop("_input_path", None)
    public.pop("_input_line_number", None)
    public["evidence_condition"] = evidence_condition or _evidence_condition(row)
    distribution = _distribution_key(row)
    public["distribution_key"] = distribution
    public["distribution_name"] = row.get("distribution_name", _DISTRIBUTION_NAMES.get(distribution))
    public.setdefault("generation_metadata", {})
    release_metadata = public.setdefault("release_metadata", {})
    # SWE Atlas keeps verifier records under release metadata in the hosted dataset.
    registry = public.pop("private_experiment_registry", None)
    if registry is not None:
        release_metadata["private_experiment_registry"] = registry
    return public


def write_release_datapoints(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    evidence_condition: str | None = None,
) -> None:
    """Atomically write accepted rows, unwrapping construction records when needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / "rows.tmp"
        write_jsonl(temporary, (_release_row(record, evidence_condition) for record in rows))
        temporary.replace(destination)
