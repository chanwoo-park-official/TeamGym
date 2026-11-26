#!/usr/bin/env python
"""Create a validated RIU-only information partition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401
from teamgym.data.long_context_reasoning.repartition import main


if __name__ == "__main__":
    main()
