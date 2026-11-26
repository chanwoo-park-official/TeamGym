#!/usr/bin/env python3
"""Build the validation-task manifest for the 0712 dataset run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = ("mslr", "casebook", "multilexsum")
MODES = ("one_each", "paired", "overlap", "redundant_player", "common_anchor", "bottleneck")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    rows: list[tuple[str, str, int, str]] = []

    for dataset in DATASETS:
        for mode in MODES:
            if dataset == "multilexsum":
                source = run_root / "build" / dataset / mode / "common_datapoints.jsonl"
            else:
                source = (
                    run_root
                    / "build"
                    / dataset
                    / mode
                    / "03_option_exclusion"
                    / "solo_oracle_kept_common_datapoints.jsonl"
                )
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open(encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
            rows.extend((dataset, mode, index, str(source)) for index in range(row_count))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validation tasks to {output}")


if __name__ == "__main__":
    main()
