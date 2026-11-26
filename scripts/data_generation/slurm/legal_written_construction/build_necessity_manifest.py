#!/usr/bin/env python3
"""Build a TSV task manifest for leave-one-PIU-out semantic checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.core.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-root", required=True)
    args = parser.parse_args()
    root = Path(args.variants_root).expanduser().resolve()
    variants = read_jsonl(root / "common_datapoints.jsonl")
    metadata = read_jsonl(root / "ablation_manifest.jsonl")
    if len(variants) != len(metadata):
        raise ValueError("Ablation rows and manifest rows have different lengths")

    output = root / "necessity_manifest.tsv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for variant_index, (variant, meta) in enumerate(zip(variants, metadata, strict=True)):
            writer.writerow(
                [
                    variant_index,
                    int(meta["source_row_index"]),
                    str(meta.get("source_dataset") or variant.get("dataset") or "unknown"),
                    str(meta.get("source_partition_mode") or "unknown"),
                    str(meta.get("source_instance_id") or ""),
                    str(meta.get("ablated_instance_id") or variant.get("instance_id") or ""),
                    str(meta.get("missing_rui_id") or ""),
                ]
            )
    print(f"Wrote {len(variants)} necessity tasks to {output}")


if __name__ == "__main__":
    main()
