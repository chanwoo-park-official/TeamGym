"""Checkpoint validation and CUAD-best retention for long TeamGym runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable


_STEP_RE = re.compile(r"global_step_([0-9]+)")


def checkpoint_step(path: Path) -> int | None:
    match = _STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def verify_model_checkpoint(
    checkpoint_dir: Path,
    *,
    world_size: int,
    two_policy: bool,
) -> dict[str, Any]:
    """Fail unless every expected FSDP model shard is present and nonempty."""

    step = checkpoint_step(checkpoint_dir)
    if step is None:
        raise ValueError(f"Not a global_step_N directory: {checkpoint_dir}")
    if world_size < 1:
        raise ValueError("world_size must be positive")

    actor_root = checkpoint_dir / "actor"
    role_dirs = (
        {
            "coordinator": actor_root / "coordinator",
            "participant": actor_root / "participant",
        }
        if two_policy
        else {"actor": actor_root}
    )
    sizes: dict[str, int] = {}
    for role, role_dir in role_dirs.items():
        expected = {
            f"model_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        }
        actual = {path.name for path in role_dir.glob("model_world_size_*_rank_*.pt")}
        if actual != expected:
            raise RuntimeError(
                f"Incomplete {role} checkpoint at {role_dir}: "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )
        for name in sorted(expected):
            path = role_dir / name
            size = path.stat().st_size
            if size <= 0:
                raise RuntimeError(f"Empty checkpoint shard: {path}")
            sizes[f"{role}/{name}"] = size

    data_path = checkpoint_dir / "data.pt"
    if not data_path.is_file() or data_path.stat().st_size <= 0:
        raise RuntimeError(f"Missing dataloader state: {data_path}")
    return {
        "step": step,
        "two_policy": two_policy,
        "world_size": world_size,
        "model_shards": sizes,
        "data_pt_bytes": data_path.stat().st_size,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prune_model_checkpoints(root: Path, keep_steps: Iterable[int]) -> list[Path]:
    """Remove only exact global_step_N children not named in ``keep_steps``."""

    root = root.resolve()
    keep = {int(step) for step in keep_steps}
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for path in sorted(root.iterdir()):
        step = checkpoint_step(path)
        if step is None or step in keep or not path.is_dir():
            continue
        if path.resolve().parent != root:
            raise RuntimeError(f"Refusing to prune path outside checkpoint root: {path}")
        shutil.rmtree(path)
        removed.append(path)
    return removed
