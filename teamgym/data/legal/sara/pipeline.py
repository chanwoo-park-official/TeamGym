"""Run a stage of the paper's final SARA construction workflow."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys


STAGES = {
    "prepare": "preparation",
    "generate": "worker",
    "refresh": "refresh",
    "merge": "merge",
    "qualify": "qualification",
    "finalize": "finalize",
}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES, help="Construction stage to run.")
    selected = parser.parse_args(arguments[:1])
    stage = import_module(f".{STAGES[selected.stage]}", package=__package__)
    stage.main(arguments[1:])


if __name__ == "__main__":
    main()
