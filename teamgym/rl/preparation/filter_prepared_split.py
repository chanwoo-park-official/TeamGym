#!/usr/bin/env python3
"""Create a deterministic dataset-filtered split beside prepared TeamGym data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(
    root: Path,
    split: str,
    rows: list[dict[str, Any]],
    *,
    verl_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    jsonl_path = root / f"{split}.jsonl"
    parquet_path = root / f"{split}.parquet"
    verl_path = root / f"{split}_verl.parquet"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    if verl_rows is None:
        verl_rows = pd.DataFrame(
            [
                {
                    "prompt": [{"role": "user", "content": "TeamGym workflow task"}],
                    "reward_model": {"style": "rule", "ground_truth": None},
                    "extra_info": row,
                }
                for row in rows
            ]
        )
    verl_rows.to_parquet(verl_path, index=False)
    return {
        "jsonl": str(jsonl_path),
        "jsonl_sha256": file_sha256(jsonl_path),
        "parquet": str(parquet_path),
        "parquet_sha256": file_sha256(parquet_path),
        "verl_parquet": str(verl_path),
        "verl_parquet_sha256": file_sha256(verl_path),
    }


def main() -> None:
    args = parse_args()
    root = args.prepared_dir.expanduser().resolve()
    source_path = root / f"{args.source_split}.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    requested_dataset = args.dataset.strip().casefold()
    with source_path.open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]
    selected = [
        index for index, row in enumerate(source_rows)
        if str(row.get("dataset") or "").strip().casefold() == requested_dataset
    ]
    if not selected:
        raise ValueError(
            f"No {requested_dataset!r} rows in {source_path}"
        )
    rows = [source_rows[index] for index in selected]
    source_verl_path = root / f"{args.source_split}_verl.parquet"
    verl_rows = None
    if source_verl_path.is_file():
        source_verl = pd.read_parquet(source_verl_path)
        if len(source_verl) != len(source_rows) or "extra_info" not in source_verl:
            raise ValueError(f"Source JSONL and VERL rows do not match for {args.source_split}")
        verl_rows = source_verl.iloc[selected].copy()
        for row, extra in zip(rows, verl_rows["extra_info"]):
            if not isinstance(extra, dict) or any(
                extra.get(field) != row.get(field)
                for field in ("dataset", "instance_id", "datapoint_sha256")
            ):
                raise ValueError(f"Source JSONL and VERL rows do not match for {args.source_split}")
        verl_rows["extra_info"] = rows
    outputs = write_split(root, args.output_split, rows, verl_rows=verl_rows)
    audit_path = root / f"{args.output_split}_filter_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "teamgym_filtered_prepared_split_v1",
                "source_split": args.source_split,
                "source_jsonl": str(source_path),
                "source_jsonl_sha256": file_sha256(source_path),
                "output_split": args.output_split,
                "filter": {"dataset": requested_dataset},
                "row_count": len(rows),
                "outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_split": args.output_split,
                "dataset": requested_dataset,
                "row_count": len(rows),
                "audit": str(audit_path),
            }
        )
    )


if __name__ == "__main__":
    main()
