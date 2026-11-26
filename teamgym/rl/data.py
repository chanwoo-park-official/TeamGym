"""Leak-closed preparation of normalized TeamGym data for RLLM/VERL."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

import pandas as pd

from teamgym.core.dataset import iter_common_datapoints, load_common_datapoints
from rllm.data.dataset import Dataset


PREPARATION_SCHEMA_VERSION = "teamgym_split_manifest_v1"


def canonical_dataset_name(row: dict[str, Any]) -> str:
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    explicit = (
        str(specific.get("benchmark_dataset") or row.get("dataset") or "")
        .strip()
        .casefold()
    )
    family = str(
        source.get("source_family") or specific.get("source_family") or ""
    ).casefold()
    if explicit and explicit != "long_context_reasoning":
        return explicit
    if family.startswith("ecfr_casebook"):
        return "casebook"
    if family.startswith("mslr_"):
        return "mslr"
    if family.startswith("sara_"):
        return "sara"
    if "multilexsum" in family or "multi-lexsum" in family:
        return "multilexsum"
    if explicit:
        return explicit
    raise ValueError(f"Cannot resolve dataset for instance {row.get('instance_id')!r}")


def canonical_task_id(row: dict[str, Any]) -> str:
    """Resolve identity before allocation/formulation suffixes are considered."""

    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    generation = (
        row.get("generation_metadata")
        if isinstance(row.get("generation_metadata"), dict)
        else {}
    )
    cross = row.get("_cross") if isinstance(row.get("_cross"), dict) else {}
    dataset = canonical_dataset_name(row)

    candidates = [
        cross.get("canonical_task_id"),
        specific.get("cross_canonical_task_id"),
        specific.get("canonical_task_id"),
        generation.get("canonical_task_id"),
    ]
    if dataset in {"cuad", "maud"}:
        candidates.extend(
            [specific.get("pilot_case_id"), specific.get("pilot_source_id")]
        )
    candidates.extend(
        [
            specific.get("source_instance_id"),
            source.get("source_instance_id"),
            generation.get("instance_id"),
            specific.get("cross_original_instance_id"),
        ]
    )
    value = next(
        (str(item).strip() for item in candidates if str(item or "").strip()), ""
    )
    if not value:
        stored = str(row.get("instance_id") or "").strip()
        value = re.sub(r"__partition_[A-Za-z0-9_-]+$", "", stored)
    # Archive manifests already provide the allocation-independent task ID,
    # including the legal issue suffix after ``__``. Older pilot IDs used that
    # separator for allocation variants, so retain the legacy cleanup only
    # when no archive-level canonical ID is present.
    if dataset in {"cuad", "maud"} and not str(
        cross.get("canonical_task_id") or ""
    ).strip():
        value = value.split("__", 1)[0]
    if not value:
        raise ValueError(
            f"Cannot resolve canonical task identity for {row.get('instance_id')!r}"
        )
    return value


def explicit_source_group_id(row: dict[str, Any]) -> str | None:
    cross = row.get("_cross") if isinstance(row.get("_cross"), dict) else {}
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    candidates = [
        cross.get("source_group_id"),
        row.get("source_group_id"),
        specific.get("source_group_id"),
        source.get("source_group_id"),
    ]
    return next(
        (item.strip() for item in candidates if isinstance(item, str) and item.strip()),
        None,
    )


def source_group_id(row: dict[str, Any]) -> str:
    """Read source groups from prepared data, with a fallback for task-only splits."""
    return explicit_source_group_id(row) or canonical_task_id(row)


def require_source_group_id(row: dict[str, Any]) -> str:
    value = explicit_source_group_id(row)
    if value is None:
        raise ValueError(
            f"{row.get('instance_id')}: source-group splitting requires an explicit "
            "source_group_id, for example _cross.source_group_id"
        )
    return value


def allocation_mode(row: dict[str, Any]) -> str:
    generation = (
        row.get("generation_metadata")
        if isinstance(row.get("generation_metadata"), dict)
        else {}
    )
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    private = generation.get("private_information_allocation")
    private = private if isinstance(private, dict) else {}
    return str(
        (row.get("_cross") or {}).get("partition_mode")
        or generation.get("allocation_mode")
        or private.get("mode")
        or specific.get("partition_mode")
        or specific.get("pilot_partition_mode")
        or "unknown"
    )


def answer_type(row: dict[str, Any]) -> str:
    task = (
        row.get("task_description")
        if isinstance(row.get("task_description"), dict)
        else {}
    )
    specific = (
        row.get("dataset_specific")
        if isinstance(row.get("dataset_specific"), dict)
        else {}
    )
    answer_format = (
        task.get("answer_format")
        if isinstance(task.get("answer_format"), dict)
        else {}
    )
    if (
        task.get("answer_type") == "short_answer_judged"
        or specific.get("answer_type") == "short_answer_judged"
        or answer_format.get("type") == "free_response"
        or row.get("decision_set") == ["free_response"]
    ):
        return "short_answer_judged"
    return "finite_choice"


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedPaths:
    root: Path
    split_jsonl: dict[str, Path]
    split_parquet: dict[str, Path]
    split_verl_parquet: dict[str, Path]
    manifest_path: Path
    leakage_audit_path: Path


class PreparedTeamGymDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        *,
        data_path: Path,
        verl_path: Path,
        split: str,
    ):
        super().__init__(data=data, name="teamgym_legal", split=split)
        self._data_path = str(data_path.resolve())
        self._verl_path = str(verl_path.resolve())

    def get_data_path(self) -> str:
        return self._data_path

    def get_verl_data_path(self) -> str:
        return self._verl_path


def prepare_data(
    *,
    input_paths: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    seed: int = 17,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    paper_heldout_input: str | Path | None = None,
    limit: int | None = None,
    split_group: str = "canonical_task",
    interleave_datasets: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    rows = load_common_datapoints(input_paths, limit=limit)
    if not rows:
        raise ValueError("No normalized TeamGym datapoints found")

    normalized_split_group = str(split_group).strip().casefold().replace("-", "_")
    if normalized_split_group not in {"canonical_task", "source_group"}:
        raise ValueError("split_group must be 'canonical_task' or 'source_group'")
    identity_fn = (
        require_source_group_id
        if normalized_split_group == "source_group"
        else canonical_task_id
    )
    heldout = (
        _heldout_keys(paper_heldout_input, identity_fn=identity_fn)
        if paper_heldout_input
        else set()
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(canonical_dataset_name(row), identity_fn(row))].append(row)

    available_keys = sorted(grouped)
    missing_heldout = sorted(heldout - set(available_keys))
    if missing_heldout:
        raise ValueError(
            f"Held-out canonical tasks missing from input: {missing_heldout[:10]}"
        )
    remaining = [key for key in available_keys if key not in heldout]
    random.Random(seed).shuffle(remaining)
    if heldout:
        test_keys = set(heldout)
    else:
        test_count = _fraction_count(len(remaining), test_fraction)
        test_keys = set(remaining[:test_count])
        remaining = remaining[test_count:]
    validation_count = _fraction_count(len(remaining), validation_fraction)
    validation_keys = set(remaining[:validation_count])
    train_keys = set(remaining[validation_count:])
    if not train_keys:
        raise ValueError("Split configuration produced no training canonical tasks")

    assignment = {
        **{key: "train" for key in train_keys},
        **{key: "validation" for key in validation_keys},
        **{key: "test" for key in test_keys},
    }
    if set(assignment) != set(available_keys):
        raise RuntimeError("Internal split assignment error")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    manifest_rows: list[dict[str, Any]] = []
    for key in available_keys:
        split = assignment[key]
        for row in grouped[key]:
            prepared = {
                "instance_id": str(row.get("instance_id") or ""),
                "dataset": key[0],
                "canonical_task_id": canonical_task_id(row),
                "source_group_id": source_group_id(row),
                "allocation": allocation_mode(row),
                "answer_type": answer_type(row),
                "participant_count": len(row.get("participant_directory", [])),
                "context_variant": str(
                    ((row.get("_cross") or {}).get("context_variant"))
                    or ((row.get("dataset_specific") or {}).get("context_variant"))
                    or "unknown"
                ),
                "paper_standard_163": bool(
                    (row.get("_cross") or {}).get("paper_standard_163", False)
                ),
                "source_path": str(row.get("_input_path") or ""),
                "source_line": int(row.get("_input_line_number") or 0),
                "split": split,
                "split_seed": int(seed),
                "datapoint_sha256": stable_json_hash(_without_loader_metadata(row)),
                "teamgym_datapoint_json": json.dumps(
                    _without_loader_metadata(row), sort_keys=True, ensure_ascii=False
                ),
            }
            split_rows[split].append(prepared)
            manifest_rows.append(
                {
                    key: value
                    for key, value in prepared.items()
                    if key != "teamgym_datapoint_json"
                }
            )

    if interleave_datasets:
        split_rows = {
            split: _interleave_by_dataset(values, seed=seed + index)
            for index, (split, values) in enumerate(split_rows.items())
        }

    paths = _write_prepared(output_root, split_rows, manifest_rows)
    input_file_list = _input_file_list(input_paths)
    input_hashes = {str(path): file_sha256(path) for path in input_file_list}
    split_group_sets = {
        split: sorted(
            [list(key) for key, assigned in assignment.items() if assigned == split]
        )
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": sorted(
            set(map(tuple, split_group_sets["train"]))
            & set(map(tuple, split_group_sets["validation"]))
        ),
        "train_test": sorted(
            set(map(tuple, split_group_sets["train"]))
            & set(map(tuple, split_group_sets["test"]))
        ),
        "validation_test": sorted(
            set(map(tuple, split_group_sets["validation"]))
            & set(map(tuple, split_group_sets["test"]))
        ),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Canonical split leakage detected: {overlaps}")
    canonical_sets = {
        split: {
            (str(row["dataset"]), str(row["canonical_task_id"]))
            for row in values
        }
        for split, values in split_rows.items()
    }
    audit = {
        "schema_version": "teamgym_leakage_audit_v1",
        "passed": True,
        "split_group": normalized_split_group,
        "interleave_datasets": bool(interleave_datasets),
        "input_hashes": input_hashes,
        "paper_heldout_input": str(paper_heldout_input)
        if paper_heldout_input
        else None,
        "split_group_counts": {
            split: len(split_group_sets[split]) for split in split_group_sets
        },
        "canonical_task_counts": {
            split: len(canonical_sets[split]) for split in canonical_sets
        },
        "row_counts": {split: len(split_rows[split]) for split in split_rows},
        "overlaps": overlaps,
        "manifest_sha256": file_sha256(paths.manifest_path),
    }
    paths.leakage_audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_root),
        "manifest_path": str(paths.manifest_path),
        "leakage_audit_path": str(paths.leakage_audit_path),
        "input_hashes": input_hashes,
        "canonical_task_counts": audit["canonical_task_counts"],
        "split_group_counts": audit["split_group_counts"],
        "row_counts": audit["row_counts"],
        "split_paths": {
            split: {
                "jsonl": str(paths.split_jsonl[split]),
                "parquet": str(paths.split_parquet[split]),
                "verl_parquet": str(paths.split_verl_parquet[split]),
            }
            for split in split_rows
        },
    }


def load_prepared_dataset(root: str | Path, split: str) -> PreparedTeamGymDataset:
    path = Path(root).expanduser().resolve()
    jsonl_path = path / f"{split}.jsonl"
    data_path = path / f"{split}.parquet"
    verl_path = path / f"{split}_verl.parquet"
    if not jsonl_path.is_file() or not data_path.is_file() or not verl_path.is_file():
        raise FileNotFoundError(f"Prepared split is incomplete: {path}/{split}")
    # File iteration respects physical JSONL newlines. ``str.splitlines()``
    # also splits on Unicode U+2028/U+2029 characters that legitimately occur
    # inside legal source strings and would corrupt otherwise valid records.
    with jsonl_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return PreparedTeamGymDataset(
        rows, data_path=data_path, verl_path=verl_path, split=split
    )


def _write_prepared(
    root: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    manifest_rows: list[dict[str, Any]],
) -> PreparedPaths:
    jsonl_paths: dict[str, Path] = {}
    parquet_paths: dict[str, Path] = {}
    verl_paths: dict[str, Path] = {}
    for split, rows in split_rows.items():
        jsonl_path = root / f"{split}.jsonl"
        parquet_path = root / f"{split}.parquet"
        verl_path = root / f"{split}_verl.parquet"
        jsonl_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        verl_rows = [
            {
                "prompt": [{"role": "user", "content": "TeamGym workflow task"}],
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": row,
            }
            for row in rows
        ]
        pd.DataFrame(verl_rows).to_parquet(verl_path, index=False)
        jsonl_paths[split] = jsonl_path
        parquet_paths[split] = parquet_path
        verl_paths[split] = verl_path
    manifest_path = root / "split_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return PreparedPaths(
        root=root,
        split_jsonl=jsonl_paths,
        split_parquet=parquet_paths,
        split_verl_parquet=verl_paths,
        manifest_path=manifest_path,
        leakage_audit_path=root / "leakage_audit.json",
    )


def _heldout_keys(
    path: str | Path, *, identity_fn=canonical_task_id
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    raw_path = Path(path).expanduser()
    try:
        for row in iter_common_datapoints(raw_path):
            keys.add((canonical_dataset_name(row), identity_fn(row)))
    except (ValueError, json.JSONDecodeError):
        keys.clear()
    if not keys:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row.get("dataset") or "").strip()
            identity_field = (
                "source_group_id" if identity_fn is require_source_group_id else "canonical_task_id"
            )
            task_id = str(row.get(identity_field) or "").strip()
            if not dataset or not task_id:
                raise ValueError(
                    f"Held-out manifest rows require dataset and {identity_field}"
                )
            keys.add((dataset, task_id))
    if not keys:
        raise ValueError(f"No canonical tasks found in held-out input {path}")
    return keys


def _without_loader_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"_input_path", "_input_line_number"}
    }


def _fraction_count(total: int, fraction: float) -> int:
    if total <= 1 or fraction <= 0:
        return 0
    return min(total - 1, max(1, int(round(total * fraction))))


def _interleave_by_dataset(
    rows: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    """Deterministically round-robin datasets after shuffling within each one."""

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    rng = random.Random(seed)
    for values in by_dataset.values():
        rng.shuffle(values)
    names = sorted(by_dataset)
    mixed: list[dict[str, Any]] = []
    while any(by_dataset.values()):
        for name in names:
            if by_dataset[name]:
                mixed.append(by_dataset[name].pop())
    return mixed


def _input_file_list(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path = Path(paths).expanduser()
        if path.is_file():
            return [path.resolve()]
        return sorted(
            item.resolve() for item in path.rglob("*.jsonl") if item.is_file()
        )
    return [Path(path).expanduser().resolve() for path in paths]
