from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from teamgym.rl.preparation.filter_prepared_split import main


@pytest.mark.parametrize("with_verl_source", [True, False])
def test_filter_preserves_source_reward_metadata(tmp_path, monkeypatch, with_verl_source):
    rows = [
        {"dataset": "maud", "instance_id": "maud_1"},
        {"dataset": "cuad", "instance_id": "cuad_1"},
    ]
    (tmp_path / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    if with_verl_source:
        pd.DataFrame([
            {
                "prompt": [{"role": "user", "content": row["instance_id"]}],
                "reward_model": {"style": "llm_judge", "ground_truth": None},
                "extra_info": row,
            }
            for row in rows
        ]).to_parquet(tmp_path / "validation_verl.parquet", index=False)
    monkeypatch.setattr(sys, "argv", [
        "filter_prepared_split", "--prepared-dir", str(tmp_path),
        "--dataset", "cuad", "--output-split", "validation_cuad",
    ])

    main()

    filtered = pd.read_parquet(tmp_path / "validation_cuad_verl.parquet").iloc[0]
    assert filtered["extra_info"] == rows[1]
    expected_style = "llm_judge" if with_verl_source else "rule"
    expected_prompt = "cuad_1" if with_verl_source else "TeamGym workflow task"
    assert filtered["reward_model"]["style"] == expected_style
    assert filtered["prompt"][0]["content"] == expected_prompt
