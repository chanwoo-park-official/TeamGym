#!/usr/bin/env python
"""CLI wrapper for long-context RUI visualization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.data.long_context_reasoning.visualize import main  # noqa: E402


if __name__ == "__main__":
    main()
