#!/usr/bin/env python
"""Prepare memory-light Multi-LexSum shard roots.

The original Multi-LexSum ``sources.json`` is large enough that many parallel
workers cannot each load the full file. This script applies the same source
eligibility filter used by the dataset builder, shuffles the eligible cases,
and writes shard-local ``train/dev/test.json`` plus a reduced ``sources.json``
containing only documents referenced by that shard.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

from teamgym.core.io import estimate_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--scan-per-shard", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_eligible_metadata(input_root=input_root)
    random.Random(args.seed).shuffle(rows)

    shard_rows: list[list[dict[str, Any]]] = []
    for shard_index in range(args.num_shards):
        start = shard_index * args.scan_per_shard
        stop = start + args.scan_per_shard
        shard_rows.append(rows[start:stop])

    needed_doc_ids = sorted(
        {
            str(doc_id)
            for shard in shard_rows
            for row in shard
            for doc_id in row.get("case_documents", [])
        }
    )
    print(f"Loading full sources.json once for {len(needed_doc_ids)} referenced docs...", flush=True)
    with (input_root / "sources.json").open(encoding="utf-8") as handle:
        sources = json.load(handle, parse_constant=lambda _constant: None)

    manifest: dict[str, Any] = {
        "schema": "multilexsum_source_shard_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "num_shards": args.num_shards,
        "scan_per_shard": args.scan_per_shard,
        "eligible_row_count": len(rows),
        "referenced_doc_count": len(needed_doc_ids),
        "shards": [],
    }

    for shard_index, rows_for_shard in enumerate(shard_rows):
        shard_name = f"shard_{shard_index:02d}"
        shard_root = output_root / shard_name
        shard_root.mkdir(parents=True, exist_ok=True)
        by_split = {"train": [], "dev": [], "test": []}
        shard_doc_ids: set[str] = set()
        for row in rows_for_shard:
            split = str(row.get("_split") or "train")
            clean = {key: value for key, value in row.items() if key != "_split"}
            by_split.setdefault(split, []).append(clean)
            shard_doc_ids.update(str(doc_id) for doc_id in row.get("case_documents", []))

        for split in ["train", "dev", "test"]:
            write_jsonl(shard_root / f"{split}.json", by_split.get(split, []))

        shard_sources = {doc_id: sources[doc_id] for doc_id in sorted(shard_doc_ids) if doc_id in sources}
        with (shard_root / "sources.json").open("w", encoding="utf-8") as handle:
            json.dump(shard_sources, handle, ensure_ascii=False)

        manifest["shards"].append(
            {
                "shard": shard_name,
                "root": str(shard_root),
                "row_count": len(rows_for_shard),
                "source_doc_count": len(shard_sources),
                "metadata_counts": {split: len(by_split.get(split, [])) for split in ["train", "dev", "test"]},
            }
        )
        print(f"{shard_name}: rows={len(rows_for_shard)} docs={len(shard_sources)}", flush=True)

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2)[:5000])


def load_eligible_metadata(*, input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "dev", "test"]:
        with (input_root / f"{split}.json").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_split"] = split
                if (
                    3 <= len(row.get("case_documents") or []) <= 18
                    and 250 <= estimate_tokens(row.get("summary/long") or "") <= 1800
                ):
                    rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
