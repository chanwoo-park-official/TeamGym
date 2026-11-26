"""Run CUAD generation or partition accepted CUAD trials."""

import sys


__all__ = ["main"]


def main() -> None:
    if sys.argv[1:2] == ["partition"]:
        from .partition import main as partition

        partition(sys.argv[2:], prog=f"{sys.argv[0]} partition")
    else:
        from .generation import main as generate

        generate()
