from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.core import dataset
from teamgym.core.schema import SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS
from teamgym.interaction import dataset as legacy_dataset


def _row(identity: str, **metadata) -> dict:
    return {
        "schema_version": dataset.COMMON_DATAPOINT_SCHEMA_VERSION,
        "instance_id": identity,
        **metadata,
    }


def _mixed_files(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    first = root / "a.jsonl"
    second = root / "nested" / "b.jsonl"
    second.parent.mkdir()
    records = [
        "",
        json.dumps({"schema_version": "unsupported", "instance_id": "skip"}),
        json.dumps(_row("first", text="source\u2028text"), ensure_ascii=False),
        json.dumps({"common_datapoint": _row("wrapped", _input_path="original.jsonl", _input_line_number=42)}),
        "[]",
    ]
    first.write_text("\n".join(records) + "\n", encoding="utf-8")
    second.write_text(json.dumps(_row("last")) + "\n", encoding="utf-8")
    (root / "notes.txt").write_text(json.dumps(_row("ignore-non-jsonl")), encoding="utf-8")
    return first, second


def test_legacy_dataset_exports_the_shared_loader() -> None:
    for name in (
        "COMMON_DATAPOINT_SCHEMA_VERSION", "DEFAULT_LEGAL_DATASET_PATH",
        "discover_common_datapoint_files", "extract_common_datapoint",
        "iter_common_datapoints", "load_common_datapoints",
    ):
        assert getattr(legacy_dataset, name) is getattr(dataset, name)
    from teamgym import interaction

    assert interaction.iter_common_datapoints is dataset.iter_common_datapoints
    assert interaction.load_common_datapoints is dataset.load_common_datapoints


def test_core_loader_imports_without_interaction_or_training_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import teamgym.core.dataset; "
            "blocked = ('teamgym.data', "
            "'teamgym.interaction', 'rllm', 'verl', 'torch', 'pandas'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_directory_loading_preserves_order_physical_lines_and_source_metadata(tmp_path) -> None:
    first, second = _mixed_files(tmp_path)
    assert dataset.discover_common_datapoint_files(tmp_path) == [first, second]
    rows = dataset.load_common_datapoints(tmp_path)
    assert [row["instance_id"] for row in rows] == ["first", "wrapped", "last"]
    assert rows[0]["text"] == "source\u2028text"
    assert rows[0]["_input_path"] == str(first)
    assert rows[0]["_input_line_number"] == 3
    assert rows[1]["_input_path"] == "original.jsonl"
    assert rows[1]["_input_line_number"] == 42
    assert rows[2]["_input_path"] == str(second)
    assert rows[2]["_input_line_number"] == 1


def test_explicit_file_order_and_limits(tmp_path) -> None:
    first, second = _mixed_files(tmp_path)
    paths = [second, first]
    expected = ["last", "first", "wrapped"]
    assert [row["instance_id"] for row in dataset.iter_common_datapoints(paths)] == expected
    assert [row["instance_id"] for row in dataset.iter_common_datapoints(iter(paths))] == expected
    for limit, count in ((None, 3), (-1, 0), (0, 0), (1, 1), (2, 2), (20, 3)):
        assert [row["instance_id"] for row in dataset.load_common_datapoints(paths, limit=limit)] == expected[:count]


def test_extract_accepts_native_or_wrapped_records_without_copying() -> None:
    native = _row("native")
    assert dataset.extract_common_datapoint(native) is native
    assert dataset.extract_common_datapoint({"common_datapoint": native}) is native
    for unsupported in (None, [], {}, {"schema_version": "unsupported"},
                        {"schema_version": []}, {"common_datapoint": []},
                        {"common_datapoint": {"schema_version": {}}}):
        assert dataset.extract_common_datapoint(unsupported) is None


@pytest.mark.parametrize("schema_version", SUPPORTED_COMMON_DATAPOINT_SCHEMA_VERSIONS)
def test_current_and_historical_schema_labels_load_without_rewriting(tmp_path, schema_version) -> None:
    row = _row("saved", schema_version=schema_version, text="source\u2028text")
    serialized = json.dumps(row, ensure_ascii=False)
    assert dataset.extract_common_datapoint(row) is row
    assert dataset.extract_common_datapoint({"common_datapoint": row}) is row
    path = tmp_path / "saved.jsonl"
    path.write_text(serialized + "\n" + json.dumps({"common_datapoint": row}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    before = path.read_bytes()
    loaded = dataset.load_common_datapoints(path)
    assert len(loaded) == 2
    for value in loaded:
        value = {key: item for key, item in value.items() if key not in {"_input_path", "_input_line_number"}}
        assert value["schema_version"] == schema_version
        assert json.dumps(value, ensure_ascii=False) == serialized
    assert path.read_bytes() == before


def test_missing_paths_and_lazy_json_errors_keep_their_existing_behavior(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Input path not found"):
        dataset.discover_common_datapoint_files(tmp_path / "missing")
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(_row("valid-first")) + "\n{invalid JSON}\n", encoding="utf-8")
    assert dataset.discover_common_datapoint_files(path) == [path]
    rows = dataset.iter_common_datapoints(path)
    assert next(rows)["instance_id"] == "valid-first"
    with pytest.raises(json.JSONDecodeError):
        next(rows)
