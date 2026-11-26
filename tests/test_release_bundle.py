from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.storage import export_release_bundle as release


def _sources(tmp_path: Path) -> tuple[list[str], dict[str, bytes]]:
    metadata = tmp_path / "legal"
    legal = metadata / "prepared"
    finance = tmp_path / "finance"
    original = {}
    for kind, data, source_metadata in (("legal", legal, metadata), ("finance", finance, finance)):
        data.mkdir(parents=True)
        for split, count in release.SPLITS[kind].items():
            for suffix in release.SUFFIXES:
                name = split + suffix
                value = (b"".join(json.dumps({"dataset": "fixture", "instance_id": f"{split}_{i}",
                                              "text": "source\u2028text"}, ensure_ascii=False).encode() + b"\n"
                                  for i in range(count)) if suffix == ".jsonl" else b"opaque frozen parquet bytes")
                (data / name).write_bytes(value)
                original[f"{kind}/prepared/{name}"] = value
        for name in release.METADATA[kind]:
            (source_metadata / name).write_bytes(b"{}\n")
        (data / "private_notes.txt").write_text("must not be exported")
    benchmark = tmp_path / "benchmark.jsonl"
    tasks = []
    cells = []
    archive = tmp_path / "model_evidence.zip"
    with ZipFile(archive, "w") as output:
        for dataset in release.LEGAL_DATASETS:
            instance = dataset + "_task"
            written = dataset not in {"cuad", "maud"}
            tasks.append({"instance_id": instance, "dataset": dataset,
                          "task_description": {"answer_type": "short_answer_judged" if written else "finite_choice"}})
            actor, judge = f"episodes/{dataset}/actor", f"episodes/{dataset}/judge"
            evaluation = {"instance_id": instance}
            payload = {"episode": {"messages": []}, "evaluation": evaluation}
            output.writestr(actor + "/trajectories/saved.json", json.dumps(payload))
            output.writestr(actor + "/eval_results.jsonl", json.dumps(evaluation) + "\n")
            output.writestr(actor + "/eval_summary.json", "{}")
            output.writestr(actor + "/run_manifest.json", "{}")
            output.writestr(actor + "/llm_requests.jsonl", "private request log")
            if written:
                output.writestr(judge + "/judgments.jsonl", json.dumps({"instance_id": instance, "score": 0}) + "\n")
            cells.append({"dataset": dataset, "model_slug": "gpt54", "condition": "multi_full",
                          "status": "evaluated", "success": False,
                          "paper_cell_id": dataset + "_cell", "archive_instance_id": dataset + "_archive",
                          "actor_archive_dir": actor, "judge_archive_dir": judge})
        output.writestr("metadata/cell_index.jsonl", "".join(json.dumps(row) + "\n" for row in cells))
        output.writestr("code/private_snapshot.py", "not part of this export")
    benchmark.write_text("".join(json.dumps(row) + "\n" for row in tasks))
    args = ["--legal-data-dir", str(legal), "--finance-data-dir", str(finance),
            "--trajectory-archive", str(archive), "--benchmark-file", str(benchmark),
            "--code-revision", "fixture-revision"]
    return args, original


def _verify_checksums(files: dict[str, bytes]) -> None:
    expected = {}
    for line in files["FILES.sha256"].decode().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
        assert release.sha256(files[name]) == digest
    assert set(expected) == set(files) - {"FILES.sha256"}


def test_export_preserves_frozen_bytes_selects_failures_and_is_deterministic(tmp_path, capsys):
    args, original = _sources(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    release.main([*args, "--output-dir", str(first)])
    release.main([*args, "--output-dir", str(second)])
    capsys.readouterr()
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    _verify_checksums({path.name: path.read_bytes() for path in first.iterdir()})
    for kind in release.SPLITS:
        with ZipFile(first / f"teamgym_{kind}_rl_v1.zip") as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        _verify_checksums(files)
        assert not any("private_notes" in name for name in files)
        for name, value in original.items():
            if name.startswith(kind + "/"):
                assert files[name.removeprefix(kind + "/")] == value
    with ZipFile(first / "teamgym_trajectory_examples_v1.zip") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    _verify_checksums(files)
    assert not any("llm_requests" in name or name.endswith(".py") for name in files)
    selected = release.jsonl_rows(files["selection.jsonl"])
    assert len(selected) == 6
    assert all(row["recorded_success"] is False for row in selected)
    assert len(release.jsonl_rows(files["inputs.jsonl"])) == 6
    assert "examples/sara/trajectories/trajectory.json" in files
    assert "examples/sara/judgments.jsonl" in files
    assert "examples/cuad/judgments.jsonl" not in files


def test_bad_split_counts_leave_no_partial_release(tmp_path):
    args, _ = _sources(tmp_path)
    (tmp_path / "legal/prepared/train.jsonl").write_text("{}\n")
    output = tmp_path / "release"
    with pytest.raises(SystemExit) as error:
        release.main([*args, "--output-dir", str(output)])
    assert error.value.code == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".release-*"))


def test_existing_release_is_preserved(tmp_path):
    args, _ = _sources(tmp_path)
    output = tmp_path / "release"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original")
    with pytest.raises(SystemExit):
        release.main([*args, "--output-dir", str(output)])
    assert marker.read_text() == "original"


def test_mismatched_saved_judgment_cannot_be_paired_with_an_example(tmp_path):
    args, _ = _sources(tmp_path)
    archive = tmp_path / "model_evidence.zip"
    with ZipFile(archive) as source:
        files = {name: source.read(name) for name in source.namelist()}
    files["episodes/sara/judge/judgments.jsonl"] = b'{"instance_id":"another_task"}\n'
    with ZipFile(archive, "w") as output:
        for name, value in files.items():
            output.writestr(name, value)
    destination = tmp_path / "examples.zip"
    with pytest.raises(ValueError, match="matching input/judgment for sara"):
        release.export_examples(archive, tmp_path / "benchmark.jsonl", destination, "gpt54", "multi_full", None)
    assert not destination.exists()
