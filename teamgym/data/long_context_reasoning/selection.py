#!/usr/bin/env python3
"""Write one zero-indexed JSONL row to a standalone JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from teamgym.core.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"row index {args.index} outside [0, {len(rows)}) for {args.input}")
    write_jsonl(args.output, [rows[args.index]])


if __name__ == "__main__":
    main()
