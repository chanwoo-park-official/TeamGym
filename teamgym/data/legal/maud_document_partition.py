"""Compatibility entry point for MAUD partitioning."""

import sys

from .maud import partition


if __name__ == "__main__":
    partition.main()
else:
    sys.modules[__name__] = partition
