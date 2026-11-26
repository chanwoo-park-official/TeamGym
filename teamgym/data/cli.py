"""Run a registered domain-specific dataset-construction pipeline."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys

from teamgym.data.registry import DOMAIN_LABELS, PIPELINES, PIPELINE_SPECS


def _list_pipelines() -> None:
    for domain, label in DOMAIN_LABELS.items():
        pipelines = [pipeline for pipeline in PIPELINE_SPECS if pipeline.domain == domain]
        print(f"{label} ({domain}):")
        if not pipelines:
            print("  No construction commands available in this checkout.")
        for pipeline in pipelines:
            aliases = f" (aliases: {', '.join(pipeline.aliases)})" if pipeline.aliases else ""
            print(f"  {pipeline.qualified_name}{aliases}")


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Paper domains: " + ", ".join(f"{label} ({domain})" for domain, label in DOMAIN_LABELS.items())
        + ". Use --list to see available construction commands.",
    )
    parser.add_argument("dataset", nargs="?", choices=PIPELINES, help="Domain/dataset name or a legacy alias.")
    parser.add_argument("--list", dest="list_pipelines", action="store_true", help="List domains and available construction commands.")
    selected = parser.parse_args(arguments[:1])
    if selected.list_pipelines:
        if len(arguments) != 1:
            parser.error("--list does not accept construction arguments")
        _list_pipelines()
        return
    if selected.dataset is None:
        parser.error("a dataset is required; use --list to see available commands")
    pipeline = import_module(PIPELINES[selected.dataset])
    previous_argv = sys.argv
    try:
        sys.argv = [f"{previous_argv[0]} {selected.dataset}", *arguments[1:]]
        pipeline.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
