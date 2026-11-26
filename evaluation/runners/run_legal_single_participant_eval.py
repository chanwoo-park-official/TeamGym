#!/usr/bin/env python3
"""Compatibility wrapper for participant-private solo legal evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401

from teamgym.interaction.run_single_participant_evaluation import main


if __name__ == "__main__":
    main()
