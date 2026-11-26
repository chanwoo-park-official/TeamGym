"""Offline integrity and split checks for prepared paper RL datasets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


def check_prepared_data(root: str | Path, *, finance: bool) -> dict:
    import pandas as pd

    from teamgym.rl.data import (
        answer_type, canonical_dataset_name, canonical_task_id, file_sha256,
        require_source_group_id, stable_json_hash,
    )

    root = Path(root).expanduser().resolve()
    splits = ("train", "validation", "test") if finance else ("train", "validation_cuad", "validation_maud")
    expected = {"train": 538, "validation": 60, "test": 40} if finance else {"train": 334, "validation_cuad": 36, "validation_maud": 26}
    seen = {"canonical_task_id": {}, "source_group_id": {}}
    report = {"prepared_data_dir": str(root), "splits": {}, "file_sha256": {}}
    for split in splits:
        paths = [root / f"{split}{suffix}" for suffix in (".jsonl", ".parquet", "_verl.parquet")]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            report["file_sha256"][path.name] = file_sha256(path)
        # Physical lines matter: source strings can contain Unicode line separators.
        with paths[0].open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        parquet = pd.read_parquet(paths[1]).to_dict("records")
        verl = pd.read_parquet(paths[2])["extra_info"].tolist()
        if not rows or len(rows) != len(parquet) or len(rows) != len(verl):
            raise ValueError(f"{split}: empty split or inconsistent JSONL/parquet row counts")
        ids = set()
        datasets = Counter()
        participants = Counter()
        for index, (row, pq_row, verl_row) in enumerate(zip(rows, parquet, verl), 1):
            label = f"{split}:{index}"
            for field in ("instance_id", "dataset", "canonical_task_id", "source_group_id", "context_variant", "participant_count", "split", "datapoint_sha256", "teamgym_datapoint_json"):
                if field not in row or any(peer.get(field) != row[field] for peer in (pq_row, verl_row)):
                    raise ValueError(f"{label}: inconsistent {field} across prepared formats")
            payload = json.loads(row["teamgym_datapoint_json"])
            if stable_json_hash(payload) != row["datapoint_sha256"]:
                raise ValueError(f"{label}: datapoint checksum mismatch")
            dataset = canonical_dataset_name(payload)
            if row["dataset"] != dataset or row["instance_id"] != payload["instance_id"]:
                raise ValueError(f"{label}: embedded datapoint identity mismatch")
            # Legal dataset-specific validation files preserve the original
            # generic 'validation' row label from before dataset filtering.
            split_labels = {split, "validation"} if split.startswith("validation_") else {split}
            if row["split"] not in split_labels or row["context_variant"] != ("riu_only" if finance else "full_context"):
                raise ValueError(f"{label}: wrong split or context variant")
            if row["instance_id"] in ids:
                raise ValueError(f"{label}: duplicate instance ID")
            ids.add(row["instance_id"])
            for field, resolver in (("canonical_task_id", canonical_task_id), ("source_group_id", require_source_group_id)):
                if row[field] != resolver(payload):
                    raise ValueError(f"{label}: embedded {field} mismatch")
                key = (dataset, row[field])
                previous = seen[field].setdefault(key, split)
                if previous != split:
                    raise ValueError(f"{label}: {field} overlaps {previous}")
            actors = payload["participants"]
            count = len(actors)
            if row["participant_count"] != count or row.get("answer_type") != "short_answer_judged" or answer_type(payload) != "short_answer_judged":
                raise ValueError(f"{label}: wrong participant count or answer type")
            if finance:
                if dataset not in {"findver", "finlongdocqa"}:
                    raise ValueError(f"{label}: unexpected finance dataset")
                if split == "test" and (dataset != "finlongdocqa" or count != 4):
                    raise ValueError(f"{label}: test must contain four-participant FinLong only")
                if split != "test" and dataset == "finlongdocqa" and count not in {2, 3}:
                    raise ValueError(f"{label}: FinLong training/validation must use two or three participants")
                if any(len(actor.get("owned_rui_ids", [])) != 1 or len(actor.get("private_context_unit_ids", [])) != 1 for actor in actors):
                    raise ValueError(f"{label}: finance requires one PIU/context unit per participant")
                contexts = {unit["context_unit_id"]: unit for unit in payload["private_context_units"]}
                if len(contexts) != count or any(
                    set(contexts.get(actor["private_context_unit_ids"][0], {}).get("hidden_contains_rui_ids", [])) != set(actor["owned_rui_ids"])
                    for actor in actors
                ):
                    raise ValueError(f"{label}: finance PIUs do not match assigned context units")
            elif dataset != ("maud" if split == "validation_maud" else "cuad"):
                raise ValueError(f"{label}: legal training/selection must use CUAD; held-out evaluation uses MAUD")
            datasets[dataset] += 1
            participants[str(count)] += 1
        report["splits"][split] = {"rows": len(rows), "datasets": dict(sorted(datasets.items())), "participant_counts": dict(sorted(participants.items()))}
    report["canonical_tasks_disjoint"] = True
    report["source_groups_disjoint"] = True
    report["matches_paper_split_counts"] = all(report["splits"][name]["rows"] == count for name, count in expected.items())
    return report
