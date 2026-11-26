#!/usr/bin/env python3
"""Prepare leak-closed TeamGym train/validation/test splits."""

from __future__ import annotations

import argparse
import json

from ..data import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Normalized common-datapoint JSONL or directory."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-heldout-input", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--split-group",
        choices=("canonical_task", "source_group"),
        default="canonical_task",
        help="Group by task or require explicit source_group_id values for document-level splitting.",
    )
    parser.add_argument("--interleave-datasets", action="store_true")
    args = parser.parse_args()
    result = prepare_data(
        input_paths=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        paper_heldout_input=args.paper_heldout_input,
        limit=args.limit,
        split_group=args.split_group,
        interleave_datasets=args.interleave_datasets,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
