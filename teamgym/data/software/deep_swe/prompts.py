"""Prompt loading for the DeepSWE construction and solver pipeline."""

from __future__ import annotations

from pathlib import Path


PROMPT_DIR = (
    Path(__file__).resolve().parents[3] / "prompts" / "software" / "deep_swe"
)


def load_prompt(name: str) -> str:
    """Load a checked-in DeepSWE prompt by filename."""

    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"DeepSWE prompt does not exist: {path}")
    return path.read_text(encoding="utf-8").strip()
