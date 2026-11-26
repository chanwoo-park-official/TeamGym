#!/usr/bin/env python3
"""Run a stage of the retained SARA construction workflow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.data.legal.sara.worker import main


if __name__ == "__main__":
    main()
