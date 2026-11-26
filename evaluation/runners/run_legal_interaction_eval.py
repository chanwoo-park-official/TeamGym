#!/usr/bin/env python3
"""Compatibility wrapper for the fully collaborative legal evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.interaction.run_evaluation import main


if __name__ == "__main__":
    main()
