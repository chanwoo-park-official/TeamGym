#!/usr/bin/env python
"""Audit whether canonical answers are fully supported by their RIUs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401
from teamgym.data.long_context_reasoning.support_audit import main


if __name__ == "__main__":
    main()
