#!/usr/bin/env python3
"""Rescore a saved TeamGym RL artifact with the ordinary evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from teamgym.interaction.evaluation import evaluate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = Path(args.artifact).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = payload.get("paper_episode")
    if not isinstance(episode, dict):
        raise ValueError("Artifact has no completed paper_episode")
    trajectory = payload.get("rllm_trajectories") or []
    if not trajectory or not isinstance(trajectory[0], dict):
        raise ValueError("Artifact has no RLLM task view")
    task = trajectory[0].get("task") or {}
    datapoint_json = task.get("teamgym_datapoint_json")
    if not isinstance(datapoint_json, str):
        # New artifacts keep the normalized row outside the compact RLLM task.
        datapoint = payload.get("normalized_datapoint")
    else:
        datapoint = json.loads(datapoint_json)
    if not isinstance(datapoint, dict):
        raise ValueError("Artifact lacks the normalized datapoint needed for rescoring")
    result = evaluate_episode(episode, datapoint)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
