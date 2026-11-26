"""Canonical command dispatcher for the MultiLexSum construction stages."""

from __future__ import annotations

import argparse
import sys

from teamgym.data.legal.multilexsum import generation, grounding, packets, short_answer, source


STAGES = {
    "prepare-sources": source.main,
    "prepare-packets": packets.prepare_main,
    "generate": generation.main,
    "short-answer": short_answer.main,
    "expand-packets": packets.expand_main,
    "audit-grounding": grounding.main,
    "select-grounded": grounding.select_main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"{sys.argv[0]} {args.stage}", *sys.argv[2:]]
    STAGES[args.stage]()


__all__ = ["main"]
