"""Compatibility entry point for MAUD generation."""

import sys

from .maud import generation


if __name__ == "__main__":
    generation.main()
else:
    sys.modules[__name__] = generation
