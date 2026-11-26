"""Read and write artifacts handed from contract generation to partitioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def split_generation_byproducts(
    accepted: list[dict[str, Any]],
    *,
    min_accept_size: int,
    save_generation_byproducts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    byproduct_filename = f"generation_byproducts_minaccept{min_accept_size}.jsonl"
    accepted_to_write: list[dict[str, Any]] = []
    byproducts: list[dict[str, Any]] = []
    for item in accepted:
        row = dict(item)
        byproduct = row.pop("_generation_byproduct", None)
        if save_generation_byproducts and isinstance(byproduct, dict):
            byproducts.append(byproduct)
            row["generation_byproduct_ref"] = {
                "path": byproduct_filename,
                "record_id": byproduct.get("instance_id"),
            }
        accepted_to_write.append(row)
    return accepted_to_write, byproducts


def read_generation_trials(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {source}")
        return [row for row in payload if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.split("\n"):
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows
