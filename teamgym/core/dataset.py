"""Read TeamGym task records for construction, evaluation, and training."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import json
from itertools import islice
from pathlib import Path
from typing import Any

from teamgym.core.schema import (
    COMMON_DATAPOINT_SCHEMA_VERSION,
    RELEASE_DATAPOINT_SCHEMA_VERSION,
    normalize_release_datapoint,
)


def discover_common_datapoint_files(input_root: str | Path) -> list[Path]:
    root = Path(input_root).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Input path not found: {root}")
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def iter_common_datapoints(
    paths: str | Path | Iterable[str | Path],
    *,
    on_skipped_row: Callable[[Path, int], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield normalized rows and optionally report unsupported nonblank records."""
    if isinstance(paths, (str, Path)):
        path_list = discover_common_datapoint_files(paths)
    else:
        path_list = [Path(path).expanduser() for path in paths]
    for path in path_list:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                datapoint = extract_common_datapoint(row)
                if datapoint is None:
                    if on_skipped_row is not None:
                        on_skipped_row(path, line_number)
                    continue
                datapoint = dict(datapoint)
                datapoint.setdefault("_input_path", str(path))
                datapoint.setdefault("_input_line_number", line_number)
                yield datapoint


def load_common_datapoints(paths: str | Path | Iterable[str | Path], *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = iter_common_datapoints(paths)
    return list(rows if limit is None else islice(rows, max(0, int(limit))))


def extract_common_datapoint(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    for datapoint in (row, row.get("common_datapoint")):
        if not isinstance(datapoint, dict):
            continue
        schema = datapoint.get("schema_version")
        if schema == COMMON_DATAPOINT_SCHEMA_VERSION:
            return datapoint
        if schema == RELEASE_DATAPOINT_SCHEMA_VERSION:
            return normalize_release_datapoint(datapoint)
    return None


def datapoint_variant(row: dict[str, Any]) -> dict[str, str]:
    return {
        key: value for key in ("evidence_condition", "distribution_key")
        if isinstance(value := row.get(key), str) and value
    }
