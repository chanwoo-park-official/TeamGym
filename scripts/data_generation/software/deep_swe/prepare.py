#!/usr/bin/env python3
"""Download DeepSWE and prepare the selected task's official Docker verifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from teamgym.data.software.deep_swe.prepare import main  # noqa: E402


if __name__ == '__main__':
    main()
