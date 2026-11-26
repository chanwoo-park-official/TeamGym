from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from teamgym.core import io
from teamgym.data.legal import io as legacy_io


def test_legacy_io_reexports_shared_helpers() -> None:
    for name in (
        "read_jsonl", "write_json", "write_jsonl", "prompt_text",
        "render_template", "estimate_tokens", "compact_text",
        "json", "os", "Path", "Any", "Iterable",
    ):
        assert getattr(legacy_io, name) is getattr(io, name)


def test_core_io_imports_without_domain_or_optional_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import teamgym.core.io; "
            "blocked = ('teamgym.data', "
            "'teamgym.interaction', 'rllm', 'verl', "
            "'torch', 'pandas', 'openai', 'tiktoken'); "
            "assert not [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_json_output_and_reading_preserve_utf8_and_physical_lines(tmp_path) -> None:
    rows = [{"text": "München\u2028detail\nnext", "value": 2}, {"text": "東京", "value": 3}]
    output = tmp_path / "nested"
    io.write_json(output / "record.json", rows[0])
    assert (output / "record.json").read_text(encoding="utf-8") == (
        '{\n  "text": "München\u2028detail\\nnext",\n  "value": 2\n}\n'
    )
    io.write_jsonl(output / "records.jsonl", (row for row in rows))
    expected = (
        '{"text": "München\u2028detail\\nnext", "value": 2}\n'
        '{"text": "東京", "value": 3}\n'
    )
    assert (output / "records.jsonl").read_text(encoding="utf-8") == expected
    assert io.read_jsonl(output / "records.jsonl") == rows
    (output / "with-blanks.jsonl").write_text("\n \t\n" + expected + "\n", encoding="utf-8")
    assert legacy_io.read_jsonl(output / "with-blanks.jsonl") == rows


def test_json_reader_preserves_file_and_parse_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        io.read_jsonl(tmp_path / "missing.jsonl")
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"valid": true}\n{invalid JSON}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        io.read_jsonl(malformed)
