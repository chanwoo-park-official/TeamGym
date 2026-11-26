#!/usr/bin/env python3
"""Assemble datasets, trajectories, and code into one verified paper release."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ARCHIVE_ROOT = Path(os.environ.get("TEAMGYM_ARCHIVE_ROOT", "artifacts/archives")).expanduser()


REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ARCHIVE_ROOT / 'legal_paper_release_v1'
DATASET_ZIP = ARCHIVE_ROOT / 'legal_allocated_v1/zips/legal_allocated_v1_all.zip'
TRAJECTORY_NAMES = (
    "model_evidence",
    "communication_structure",
    "allocation_communication",
    "human_constraints",
    "diagnostics",
)
CODE_ROOTS = (
    "teamgym",
    "evaluation",
    "scripts",
    "tests",
    "requirements",
    "examples",
    "rllm",
    "verl",
)
ROOT_FILES = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    ".gitignore",
    ".python-version",
    "pyproject.toml",
    "setup.py",
    "uv.lock",
)
TEAMGYM_RUNTIME_SCRIPTS = {
    "train_legal_roles.sh",
    "train_legal.sh",
    "train_backend.sh",
    "merge_full_checkpoint_hf.sh",
    "merge_joint_checkpoint_hf.sh",
    "require_four_gpu_allocation.sh",
}
TEAMGYM_RESEARCH_TOOLS = {
    "build_participant_control_replay_bank.py",
    "build_participant_replay_bank.py",
    "compact_maud_written_counterfactual.py",
    "export_wandb_history.py",
    "materialize_maud_counterfactual_rollouts.py",
    "summarize_participant_control_runs.py",
}
EXCLUDED_ROOT_PATHS = {
    "docs",
    "rebuttal",
    "figures",
    ".codex",
    ".local_data",
    ".cursorignore",
    ".latexmkrc",
    ".vscode",
    "distributed_reasoning_dataset_plan.md",
    "trajectories_merged_20260919.csv",
    "decision_option_permutation_map.jsonl",
    "decision_option_permutation_map.pretty.json",
    "decision_option_permutation_report.json",
}
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "slurm_logs",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".partial", ".synctex.gz", ".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".blg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-zip", type=Path, default=DATASET_ZIP)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trajectory_root = output_root / "trajectory_zips"
    code_zip = output_root / "code_snapshot.zip"
    release_zip = output_root / "legal_paper_release_v1.zip"
    if not args.overwrite:
        for path in (code_zip, release_zip):
            if path.exists():
                raise FileExistsError(path)
    code_zip.unlink(missing_ok=True)
    release_zip.unlink(missing_ok=True)

    required = [args.dataset_zip] + [
        trajectory_root / f"{name}.zip" for name in TRAJECTORY_NAMES
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        verify_zip(path)

    build_code_snapshot(repo, code_zip)
    verify_zip(code_zip)

    members = {
        "datasets/legal_allocated_v1_all.zip": args.dataset_zip,
        **{
            f"trajectories/{name}.zip": trajectory_root / f"{name}.zip"
            for name in TRAJECTORY_NAMES
        },
        "code/code_snapshot.zip": code_zip,
    }
    checksums = {archive_path: sha256(path) for archive_path, path in members.items()}
    sizes = {archive_path: path.stat().st_size for archive_path, path in members.items()}
    release_summary = {
        "schema": "legal_paper_release_v1",
        "dataset_instance_count": 1607,
        "paper_standard_task_count": 163,
        "paper_allocation_instance_count": 618,
        "paper_result_cell_count": 69577,
        "members": {
            name: {"bytes": sizes[name], "sha256": checksums[name]}
            for name in members
        },
    }
    readme = release_readme(release_summary)
    checksum_text = "".join(f"{digest}  {name}\n" for name, digest in checksums.items())

    temporary = release_zip.with_suffix(".zip.partial")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("README.md", readme.encode("utf-8"))
        archive.writestr(
            "release_summary.json",
            (json.dumps(release_summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        archive.writestr("checksums.sha256", checksum_text.encode("utf-8"))
        for archive_path, path in members.items():
            archive.write(path, archive_path)
    temporary.replace(release_zip)
    verify_zip(release_zip)

    release_summary["release_zip_bytes"] = release_zip.stat().st_size
    release_summary["release_zip_sha256"] = sha256(release_zip)
    (output_root / "release_summary.json").write_text(
        json.dumps(release_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "checksums.sha256").write_text(
        checksum_text
        + f"{release_summary['release_zip_sha256']}  legal_paper_release_v1.zip\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(release_summary, indent=2))


def build_code_snapshot(repo: Path, destination: Path) -> None:
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    metadata = {
        "schema": "legal_paper_code_snapshot_v1",
        "git_head": git(repo, "rev-parse", "HEAD", check=False).strip(),
        "git_status": git(repo, "status", "--short", check=False),
        "python_version": subprocess.run(
            [sys.executable, "--version"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "note": "This snapshot includes uncommitted files and modifications present at archive time.",
    }
    temporary = destination.with_suffix(".zip.partial")
    temporary.unlink(missing_ok=True)
    archived: set[str] = set()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(
            "snapshot_metadata.json",
            (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        archive.writestr("git_status.txt", metadata["git_status"].encode("utf-8"))
        archive.writestr("python_packages.txt", pip_freeze.encode("utf-8"))
        for relative in ROOT_FILES:
            path = repo / relative
            if include_code_file(path, repo):
                add_code_file(archive, path, relative, archived)
        for relative in CODE_ROOTS:
            root = repo / relative
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and include_code_file(path, repo):
                    add_code_file(archive, path, path.relative_to(repo).as_posix(), archived)
    temporary.replace(destination)


def include_code_file(path: Path, repo: Path) -> bool:
    relative = path.relative_to(repo)
    if relative.parts and relative.parts[0] in EXCLUDED_ROOT_PATHS:
        return False
    if relative.parts[:2] == ("scripts", "storage") and (
        path.suffix == ".sbatch" or path.name == "migrate_pool_artifacts_to_scratch_0714.sh"
    ):
        return False
    if relative.parts[:2] == ("evaluation", "submissions"):
        return False
    if relative.parts == ("tests", "test_legal_communication_process.py"):
        return False
    if relative.parts[:2] == ("evaluation", "reporting"):
        if len(relative.parts) != 3 or path.name not in {
            "rescore_numeric_short_answers.py", "render_long_context_rui_visualization.py",
        }:
            return False
    if relative.parts[:2] == ("teamgym", "rl"):
        if relative.parts[2:3] == ("analysis",):
            return False
        if len(relative.parts) == 3:
            if path.name in TEAMGYM_RESEARCH_TOOLS:
                return False
            if (
                path.suffix == ".py" and path.name != "plot_reward.py"
                and path.name.startswith(("analyze_", "compare_", "plot_", "run_"))
            ):
                return False
            if path.name in {"CUAD_WRITTEN_DATASET_LOG.md", "QUERY_PRIVATE_EXPERIMENT_LOG.md"}:
                return False
            if path.name.startswith("campaign_") and path.suffix == ".json":
                return False
            if path.suffix in {".sh", ".sbatch"} and path.name not in TEAMGYM_RUNTIME_SCRIPTS:
                return False
    if EXCLUDED_PARTS.intersection(relative.parts):
        return False
    if relative.parts[:3] == ("rllm", "data", "datasets"):
        return False
    if relative.parts == ("rllm", "registry", "dataset_registry.json"):
        return False
    name = path.name.lower()
    if name in {".env", "credentials", "credentials.json", "api_key", "api_key.txt"}:
        return False
    return not any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def add_code_file(
    archive: zipfile.ZipFile, source: Path, archive_path: str, archived: set[str]
) -> None:
    if not source.is_file() or archive_path in archived:
        return
    archive.write(source, archive_path)
    archived.add(archive_path)


def release_readme(summary: dict) -> str:
    return f"""# Legal Paper Reproducibility Release v1

This release preserves the six legal datasets, their information allocations,
the exact model transcripts and judgments used in the paper, and a snapshot of
the evaluation and analysis code.

## Contents

- `datasets/legal_allocated_v1_all.zip`: 1,607 allocated task instances across
  CUAD (538), MAUD (557), Casebook (77), MSLR (24), MultiLexSum (141), and
  SARA (270). It includes the fixed 163-task evaluation set and all 618
  task-allocation instances used by the allocation analysis.
- `trajectories/model_evidence.zip`: the model and evidence-condition matrix.
- `trajectories/communication_structure.zip`: the five communication structures
  and transcript-process analysis.
- `trajectories/allocation_communication.zip`: the allocation-by-communication
  matrix, including links for cells reused from earlier experiment families.
- `trajectories/human_constraints.zip`: bounded private reasoning, public
  messaging, and participant public-reading experiments.
- `trajectories/diagnostics.zip`: Single-Agent Full-Context runs and the
  repeated-judge consistency sample.
- `code/code_snapshot.zip`: the code and dependency profiles used to build and
  analyze these artifacts.

Each trajectory ZIP contains `metadata/cell_index.jsonl`. Join its
`archive_instance_id` to the dataset archive's `provenance/instance_index.jsonl`.
The archived actor and judge directories contain the complete transcripts,
outputs, run manifests, and judgments. Original absolute paths appear only in
`metadata/source_plans/` for provenance.

## Reproducibility

The archived outputs reproduce reported tables and analyses without new model
calls. The code snapshot and run metadata preserve the prompts, model names,
budgets, communication structures, and evaluation settings needed to rerun the
experiments after replacing site-specific paths. New calls to hosted models may
produce different wording or decisions because remote inference is not
guaranteed to be deterministic.

The final SARA evaluation uses the validated 45-task collection represented by
270 allocation instances. The earlier 24-task SARA collection is not part of
the paper evaluation; only the 21 exact examples sampled for the judge-consistency
diagnostic are retained in `diagnostics.zip`.

Total indexed paper-result cells: {summary['paper_result_cell_count']:,}.
"""


def verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC verification failed for {path}: {bad}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=check
    )
    return result.stdout


if __name__ == "__main__":
    main()
