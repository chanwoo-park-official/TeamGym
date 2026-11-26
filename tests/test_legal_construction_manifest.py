from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import pytest


DATASETS = ("mslr", "casebook", "multilexsum")
MODES = ("one_each", "paired", "overlap", "redundant_player", "common_anchor", "bottleneck")


def test_manifest_uses_rows_that_exist(tmp_path: Path) -> None:
    expected: list[tuple[str, str, int, str]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        for mode_index, mode in enumerate(MODES):
            if dataset == "multilexsum":
                source = tmp_path / "build" / dataset / mode / "common_datapoints.jsonl"
            else:
                source = (
                    tmp_path
                    / "build"
                    / dataset
                    / mode
                    / "03_option_exclusion"
                    / "solo_oracle_kept_common_datapoints.jsonl"
                )
            source.parent.mkdir(parents=True, exist_ok=True)
            count = (dataset_index + mode_index) % 3
            source.write_text("".join(f'{{"row": {index}}}\n' for index in range(count)), encoding="utf-8")
            expected.extend((dataset, mode, index, str(source.resolve())) for index in range(count))

    output = tmp_path / "manifest.tsv"
    repository_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(
                repository_root
                / "scripts"
                / "data_generation"
                / "slurm"
                / "legal_written_construction"
                / "build_manifest.py"
            ),
            "--run-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    actual = []
    for line in output.read_text(encoding="utf-8").splitlines():
        dataset, mode, index, source = line.split("\t")
        actual.append((dataset, mode, int(index), source))
    assert actual == expected


@pytest.mark.parametrize("legacy", [False, True], ids=["default", "legacy-override"])
def test_leakage_merge_requires_every_manifest_row_to_finish(tmp_path: Path, legacy: bool) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    merge_script = (
        repository_root
        / "scripts"
        / "data_generation"
        / "slurm"
        / "legal_written_construction"
        / "merge_leakage.py"
    )
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n{}\n", encoding="utf-8")
    manifest = tmp_path / "validation_manifest.tsv"
    manifest.write_text(
        f"mslr\tone_each\t0\t{source}\nmslr\tone_each\t1\t{source}\n",
        encoding="utf-8",
    )

    directory = "validation_v2" if legacy else "leakage_validation"
    merge_args = [sys.executable, str(merge_script), "--run-root", str(tmp_path)]
    if legacy:
        merge_args.extend(["--validation-dir", directory])
    rejected = tmp_path / directory / "mslr" / "one_each" / "row_0000" / "candidate_status.json"
    rejected.parent.mkdir(parents=True)
    rejected.write_text('{"status":"rejected_after_leakage"}\n', encoding="utf-8")

    incomplete = subprocess.run(
        merge_args,
        capture_output=True,
        text=True,
    )
    assert incomplete.returncode != 0
    summary = json.loads((tmp_path / "leakage_survivors" / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing_row_count"] == 1
    assert not (tmp_path / "leakage_survivors" / "common_datapoints.jsonl").exists()

    completed = (
        tmp_path
        / directory
        / "mslr"
        / "one_each"
        / "row_0001"
        / "01_leakage_deletion"
        / "leakage_deleted_common_datapoints.jsonl"
    )
    completed.parent.mkdir(parents=True)
    completed.write_text('{"dataset":"mslr","instance_id":"one"}\n', encoding="utf-8")

    subprocess.run(
        merge_args,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "leakage_survivors" / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest_row_count"] == 2
    assert summary["processed_row_count"] == 2
    assert summary["missing_row_count"] == 0
    assert summary["leakage_survivor_count"] == 1
    assert (tmp_path / "leakage_survivors" / "qualification_manifest.tsv").read_text(encoding="utf-8").startswith(
        "0\tmslr\tone_each\t1\tone"
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["default", "legacy-override"])
def test_qualification_merge_requires_strict_all_step_pass(tmp_path: Path, legacy: bool) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    merge_script = (
        repository_root
        / "scripts"
        / "data_generation"
        / "slurm"
        / "legal_written_construction"
        / "merge_qualification.py"
    )
    survivor_root = tmp_path / "leakage_survivors"
    survivor_root.mkdir()
    survivor_root.joinpath("common_datapoints.jsonl").write_text(
        '{"dataset":"mslr","instance_id":"pass"}\n'
        '{"dataset":"mslr","instance_id":"fail"}\n'
        '{"dataset":"mslr","instance_id":"judge_fail"}\n',
        encoding="utf-8",
    )
    survivor_root.joinpath("qualification_manifest.tsv").write_text(
        "0\tmslr\tone_each\t3\tpass\n1\tmslr\tone_each\t4\tfail\n"
        "2\tmslr\tone_each\t5\tjudge_fail\n",
        encoding="utf-8",
    )

    directory = "qualification_v1" if legacy else "all_piu_qualification"
    cases = ((3, "pass", 2), (4, "fail", 1), (5, "judge_fail", 2))
    for source_index, instance_id, covered in cases:
        judgment = (
            tmp_path
            / directory
            / "mslr"
            / "one_each"
            / f"row_{source_index:04d}"
            / "judge"
            / "judgments.jsonl"
        )
        judgment.parent.mkdir(parents=True)
        outcome = {
            "conclusion_correct": True,
            "covered_step_count": covered,
            "total_step_count": 2,
            "disqualifying_errors_found": [],
        }
        if instance_id == "judge_fail":
            outcome["strict_all_step_passed"] = False
        judgment.write_text(
            json.dumps({"instance_id": instance_id, "judgment": outcome}) + "\n",
            encoding="utf-8",
        )

    merge_args = [sys.executable, str(merge_script), "--run-root", str(tmp_path)]
    if legacy:
        merge_args.extend(["--qualification-dir", directory])
    subprocess.run(
        merge_args,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "all_riu_qualified" / "summary.json").read_text(encoding="utf-8"))
    assert summary["strict_passed_count"] == 1
    assert summary["strict_failed_count"] == 2
    kept = (tmp_path / "all_riu_qualified" / "common_datapoints.jsonl").read_text(encoding="utf-8")
    assert '"instance_id": "pass"' in kept
    assert '"instance_id": "fail"' not in kept
    assert '"instance_id": "judge_fail"' not in kept


@pytest.mark.parametrize("legacy", [False, True], ids=["default", "legacy-override"])
def test_leave_one_out_merge_keeps_only_tasks_requiring_every_rui(tmp_path: Path, legacy: bool) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    submission = repository_root / "scripts" / "data_generation" / "slurm" / "legal_written_construction"
    all_riu = tmp_path / "all_riu_qualified"
    all_riu.mkdir()
    source_rows = [
        {
            "dataset": "sara",
            "instance_id": instance_id,
            "generation_metadata": {"allocation_mode": "one_each"},
            "reference_information_units": [{"rui_id": "a"}, {"rui_id": "b"}],
            "participants": [],
        }
        for instance_id in ("all_required", "one_recoverable")
    ]
    all_riu.joinpath("common_datapoints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )

    variants = tmp_path / "necessity_variants"
    subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "data_generation" / "common" / "leave_one_rui_out.py"),
            "--input-jsonl",
            str(all_riu / "common_datapoints.jsonl"),
            "--output-root",
            str(variants),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, str(submission / "build_necessity_manifest.py"), "--variants-root", str(variants)],
        check=True,
        capture_output=True,
        text=True,
    )

    directory = "necessity_v1" if legacy else "leave_one_piu_out"
    manifest_lines = variants.joinpath("necessity_manifest.tsv").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 4
    for line in manifest_lines:
        variant_index, source_index, dataset, mode, *_rest = line.split("\t")
        strict_passed = int(source_index) == 1 and int(variant_index) == 2
        judgment = (
            tmp_path
            / directory
            / dataset
            / mode
            / f"variant_{int(variant_index):05d}"
            / "judge"
            / "judgments.jsonl"
        )
        judgment.parent.mkdir(parents=True)
        judgment.write_text(
            json.dumps(
                {
                    "judgment": {
                        "conclusion_correct": strict_passed,
                        "covered_step_count": 2 if strict_passed else 1,
                        "total_step_count": 2,
                        "disqualifying_errors_found": [],
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )

    merge_args = [sys.executable, str(submission / "merge_necessity.py"), "--run-root", str(tmp_path)]
    if legacy:
        merge_args.extend(["--necessity-dir", directory])
    subprocess.run(
        merge_args,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((tmp_path / "final_qualified" / "summary.json").read_text(encoding="utf-8"))
    assert summary["necessity_passed_source_count"] == 1
    assert summary["necessity_failed_source_count"] == 1
    kept = (tmp_path / "final_qualified" / "common_datapoints.jsonl").read_text(encoding="utf-8")
    assert '"instance_id": "all_required"' in kept
    assert '"instance_id": "one_recoverable"' not in kept
