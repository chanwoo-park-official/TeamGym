"""Compatibility import for MAUD evidence matching."""

import sys

from .maud import evidence


sys.modules[__name__] = evidence
