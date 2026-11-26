"""Load normalized fully collaborative datapoints from generated legal outputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import json
from itertools import islice
import os
from pathlib import Path
from typing import Any

from teamgym.core.schema import (
    COMMON_DATAPOINT_SCHEMA_VERSION,
    SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS,
)


_DEFAULT_LEGAL_ROOT = Path(os.environ.get(
    "TEAMGYM_LEGAL_DATA_ROOT", str(Path(os.environ.get("TEAMGYM_DATA_ROOT", "data")) / "legal"),
)).expanduser()
DEFAULT_LEGAL_DATASET_PATH = Path(os.environ.get(
    "TEAMGYM_LEGAL_DATASET", str(_DEFAULT_LEGAL_ROOT / "collections/all_task_allocations_1607.jsonl"),
)).expanduser()


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
    if row.get("schema_version") in SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS:
        return row
    common = row.get("common_datapoint")
    if isinstance(common, dict) and common.get("schema_version") in SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS:
        return common
    return None
