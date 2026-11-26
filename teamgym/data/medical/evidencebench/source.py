"""Download and parse the original, full-paper EvidenceBench release."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any
import urllib.request


BASE_URL = "https://raw.githubusercontent.com/EvidenceBench/EvidenceBench/main/datasets"
SPLIT_FILES = {
    "train": "evidencebench_train_set.json",
    "dev": "evidencebench_dev_set.json",
    "test": "evidencebench_test_set.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    *,
    force: bool,
    attempts: int = 3,
) -> dict[str, Any]:
    target = destination.expanduser().resolve()
    downloaded = False
    if not target.is_file() or force:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "CollabGym-EvidenceBench-loader/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        out.write(chunk)
                os.replace(temporary, target)
                downloaded = True
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == attempts:
                    raise
                time.sleep(2 ** (attempt - 1))
    return {
        "path": str(target),
        "url": url,
        "downloaded": downloaded,
        "bytes": target.stat().st_size,
        "sha256": _sha256_file(target),
    }


def acquire(
    raw_root: str | Path = "data/medical_raw/evidencebench",
    *,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(raw_root).expanduser().resolve() / "original"
    downloads = []
    for filename in SPLIT_FILES.values():
        downloads.append(
            _download_file(f"{BASE_URL}/{filename}", root / filename, force=force)
        )
    return {
        "raw_root": str(root),
        "variant": "original",
        "downloads": downloads,
    }


def load_records(
    raw_root: str | Path,
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    if split not in SPLIT_FILES:
        raise ValueError(f"unsupported EvidenceBench split: {split}")
    root = Path(raw_root).expanduser().resolve()
    path = root / SPLIT_FILES[split]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("EvidenceBench split root must be a JSON object")
    records: list[dict[str, Any]] = []
    for instance_id, row in value.items():
        if not isinstance(row, dict):
            raise ValueError(f"EvidenceBench {instance_id} is not an object")
        sentences = row.get("paper_as_candidate_pool")
        sentence_types = row.get("sentence_types_in_candidate_pool")
        if not row.get("hypothesis") or not isinstance(sentences, list):
            raise ValueError(f"EvidenceBench {instance_id} lacks hypothesis/candidate pool")
        if not isinstance(sentence_types, list) or len(sentence_types) != len(sentences):
            raise ValueError(f"EvidenceBench {instance_id} has invalid sentence types")
        records.append({"instance_id": str(instance_id), **row})
    return records


def sentence_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    paper_id = str(record.get("paper_id") or record["instance_id"])
    return [
        {
            "block_id": f"sentence_{index:06d}",
            "document_id": paper_id,
            "block_type": str(sentence_type),
            "text": str(sentence),
            "sentence_index": index,
            "source_ref": {
                "evidencebench_instance_id": record["instance_id"],
                "paper_id": paper_id,
                "sentence_index": index,
            },
        }
        for index, (sentence, sentence_type) in enumerate(
            zip(
                record["paper_as_candidate_pool"],
                record["sentence_types_in_candidate_pool"],
                strict=True,
            )
        )
        if str(sentence).strip()
    ]


def optimal_evidence_indices(record: dict[str, Any]) -> list[int]:
    evaluation = record.get("evidence_retrieval_at_optimal_evaluation") or {}
    values = evaluation.get("one_selection_of_sentences") or []
    return sorted({int(value) for value in values})
