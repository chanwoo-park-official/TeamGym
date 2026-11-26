"""Compatibility entry point for CUAD partitioning."""

import sys

from .cuad import partition


if __name__ == "__main__":
    partition.main()
else:
    sys.modules[__name__] = partition
