"""Exact, shared tokenization for dataset and interaction budgets."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


TOKENIZER = "cl100k_base"


@lru_cache(maxsize=1)
def encoding() -> Any:
    """Return the required tokenizer or fail before a budget is enforced."""

    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "Token-budget enforcement requires tiktoken; approximate word or "
            "character counting is not allowed."
        ) from exc
    return tiktoken.get_encoding(TOKENIZER)


def encode_tokens(text: str) -> list[int]:
    """Encode arbitrary source text without interpreting special-token literals."""

    return encoding().encode_ordinary(str(text))


def count_tokens(text: str) -> int:
    return len(encode_tokens(str(text)))
