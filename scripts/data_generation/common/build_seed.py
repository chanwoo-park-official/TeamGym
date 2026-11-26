#!/usr/bin/env python
"""Build a long-context seed from a dataset specification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401
from teamgym.data.long_context_reasoning.seed_builder import main


if __name__ == "__main__":
    main()
