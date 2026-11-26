"""Shared helpers for deterministic LoRA adapter IDs and agent name handling."""

from __future__ import annotations

import zlib


def stable_lora_int_id(name: str) -> int:
    """Return a stable 31-bit positive integer for a LoRA adapter name."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def get_agent_names(config) -> list[str]:
    if config is None or not hasattr(config, "rllm"):
        raise ValueError("Missing config.rllm for multi-agent LoRA setup")
    names = getattr(config.rllm, "multi_agent_names", None)
    if not names:
        raise ValueError("config.rllm.multi_agent_names must be set for multi-agent LoRA")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate agent names in config.rllm.multi_agent_names: {names}")
    return list(names)


def build_lora_name_map(names: list[str]) -> dict[str, int]:
    return {name: stable_lora_int_id(name) for name in names}