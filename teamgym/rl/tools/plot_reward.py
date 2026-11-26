#!/usr/bin/env python3
"""Render TeamGym task-reward curves from a console log or W&B history."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt


METRICS = (
    "batch/teamgym_task_success",
    "batch/teamgym_task_success_cuad",
    "batch/teamgym_task_success_maud",
    "batch/teamgym_reward",
    "batch/teamgym_reward/std",
    "batch/teamgym_base_reward",
    "batch/teamgym_local_penalty",
    "batch/teamgym_holistic_score",
    "batch/teamgym_strict_success",
    "batch/teamgym_required_step_coverage",
    "batch/teamgym_format_score",
    "batch/teamgym_first_pass_format_score",
    "batch/teamgym_written_judge_error",
)
NUMBER = r"(?:np\.float64\()?([-+0-9.eE]+)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", help="Console log written by the trainer")
    source.add_argument(
        "--wandb-history",
        help="JSONL exported with ``wandb.Api().run(...).scan_history()``",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--title", default=None, help="Optional figure-title override")
    args = parser.parse_args()

    rows = (
        parse_log(Path(args.log))
        if args.log
        else parse_wandb_history(Path(args.wandb_history))
    )
    if not rows:
        source_path = args.log or args.wandb_history
        raise ValueError(f"No TeamGym step metrics found in {source_path}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    written_run = any(
        row.get("batch/teamgym_holistic_score") is not None for row in rows
    )
    panel_count = 3 if written_run else 2
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(9, 9 if written_run else 7),
        sharex=True,
    )
    reward_ax = axes[0]
    format_ax = axes[1] if written_run else None
    spread_ax = axes[-1]
    steps = [int(row["step"]) for row in rows]
    dataset_series = [
        (metric, label)
        for metric, label in (
            ("batch/teamgym_task_success_cuad", "CUAD task success"),
            ("batch/teamgym_task_success_maud", "MAUD task success"),
        )
        if any(row.get(metric) is not None for row in rows)
    ]
    success_series = (
        [
            ("batch/teamgym_holistic_score", "holistic task score"),
            ("batch/teamgym_required_step_coverage", "required-step coverage"),
            ("batch/teamgym_strict_success", "strict success"),
        ]
        if written_run
        else (
            [
                ("batch/teamgym_task_success", "task success (mixed)"),
                *dataset_series,
            ]
            if len(dataset_series) > 1
            else dataset_series
            or [("batch/teamgym_task_success", "task success")]
        )
    )
    for metric, label in success_series:
        values = [row.get(metric) for row in rows]
        points = [(step, value) for step, value in zip(steps, values) if value is not None]
        if points:
            xs, ys = zip(*points)
            reward_ax.plot(xs, rolling_mean(list(ys), args.rolling_window), label=label)
    reward_ax.set_ylabel(
        f"{'semantic score' if written_run else 'success'} "
        f"(rolling {args.rolling_window})"
    )
    reward_ax.set_ylim(-0.02, 1.02)
    reward_ax.grid(alpha=0.25)
    reward_ax.legend(loc="best")

    if format_ax is not None:
        for metric, label in (
            ("batch/teamgym_format_score", "valid after retry"),
            ("batch/teamgym_first_pass_format_score", "valid first pass"),
        ):
            points = [
                (step, row.get(metric))
                for step, row in zip(steps, rows)
                if row.get(metric) is not None
            ]
            if points:
                xs, ys = zip(*points)
                format_ax.plot(
                    xs,
                    rolling_mean(list(ys), args.rolling_window),
                    label=label,
                )
        judge_points = [
            (step, 1.0 - float(row["batch/teamgym_written_judge_error"]))
            for step, row in zip(steps, rows)
            if row.get("batch/teamgym_written_judge_error") is not None
        ]
        if judge_points:
            xs, ys = zip(*judge_points)
            format_ax.plot(
                xs,
                rolling_mean(list(ys), args.rolling_window),
                label="judge reached",
            )
        format_ax.set_ylabel(
            f"protocol rate (rolling {args.rolling_window})"
        )
        format_ax.set_ylim(-0.02, 1.02)
        format_ax.grid(alpha=0.25)
        format_ax.legend(loc="best")

    reward_values = [row.get("batch/teamgym_reward") for row in rows]
    reward_points = [
        (step, value)
        for step, value in zip(steps, reward_values)
        if value is not None
    ]
    if reward_points:
        xs, ys = zip(*reward_points)
        spread_ax.plot(xs, ys, alpha=0.45, label="batch reward")
        spread_ax.plot(
            xs,
            rolling_mean(list(ys), args.rolling_window),
            linewidth=2,
            label="rolling reward",
        )
    if written_run:
        penalty_points = [
            (step, row.get("batch/teamgym_local_penalty"))
            for step, row in zip(steps, rows)
            if row.get("batch/teamgym_local_penalty") is not None
        ]
        if penalty_points:
            xs, ys = zip(*penalty_points)
            spread_ax.plot(
                xs,
                rolling_mean(list(ys), args.rolling_window),
                linestyle=":",
                label="rolling format penalty",
            )
    std_values = [row.get("batch/teamgym_reward/std") for row in rows]
    std_points = [(step, value) for step, value in zip(steps, std_values) if value is not None]
    if std_points:
        xs, ys = zip(*std_points)
        spread_ax.plot(xs, ys, linestyle="--", label="within-batch reward std")
    spread_ax.set_xlabel("optimizer step")
    spread_ax.set_ylabel("reward")
    spread_ax.grid(alpha=0.25)
    spread_ax.legend(loc="best")
    inferred_scope = (
        " + ".join(label.split()[0] for _, label in dataset_series)
        if dataset_series
        else "task"
    )
    fig.suptitle(
        args.title
        or (
            "TeamGym written-answer holistic RL"
            if written_run
            else f"TeamGym {inferred_scope} task reward"
        )
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")
    print(f"wrote {csv_path}")


def parse_log(path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = re.search(r"training/global_step:(\d+)", line)
        if not step_match or not any(
            f"{metric}:" in line for metric in METRICS
        ):
            continue
        row: dict[str, float | int | None] = {"step": int(step_match.group(1))}
        for metric in METRICS:
            match = re.search(re.escape(metric) + ":" + NUMBER, line)
            row[metric] = float(match.group(1)) if match else None
        rows.append(row)
    return rows


def parse_wandb_history(path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        payload = json.loads(line)
        step = payload.get("training/global_step")
        if step is None or not any(
            payload.get(metric) is not None for metric in METRICS
        ):
            continue
        row: dict[str, float | int | None] = {"step": int(step)}
        for metric in METRICS:
            value = payload.get(metric)
            row[metric] = float(value) if value is not None else None
        rows.append(row)
    return rows


def rolling_mean(values: list[float], window: int) -> list[float]:
    width = max(1, int(window))
    return [
        sum(values[max(0, index - width + 1) : index + 1])
        / len(values[max(0, index - width + 1) : index + 1])
        for index in range(len(values))
    ]


if __name__ == "__main__":
    main()
