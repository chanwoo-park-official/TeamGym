"""Compatibility entry point for CUAD generation."""

import sys

from .cuad import generation


if __name__ == "__main__":
    generation.main()
else:
    sys.modules[__name__] = generation
