from __future__ import annotations

import json
from pathlib import Path

import pytest

from teamgym.data.medical.mlcr.source import (
    build_case_indexes,
    load_inspectable,
    validate_case_index,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_external_mlcr_records_and_build_exact_index(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "cases_summaries.jsonl",
        [
            {
                "case_uuid": "case-1",
                "summary_file": "001.txt",
                "summary_text": "First exact sentence.\n\nSecond exact paragraph.",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {
                "case_uuid": "case-1",
                "prompt_id": "q01",
                "question": "What happened?",
                "answer": "The recorded event happened.",
            }
        ],
    )

    summaries, questions = load_inspectable(tmp_path)
    indexes = build_case_indexes(summaries, questions)

    assert len(indexes) == 1
    assert indexes[0]["case_uuid"] == "case-1"
    assert indexes[0]["documents"][0]["text"] == summaries[0]["summary_text"]
    assert validate_case_index(indexes[0]) == []


def test_load_external_mlcr_records_rejects_missing_fields(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "cases_summaries.jsonl",
        [{"case_uuid": "case-1", "summary_file": "001.txt"}],
    )
    _write_jsonl(tmp_path / "questions.jsonl", [])

    with pytest.raises(ValueError, match="missing fields: .*summary_text"):
        load_inspectable(tmp_path)
