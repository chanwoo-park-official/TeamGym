"""Real legal source loading for fully collaborative dataset generation."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterator

from .constants import CUAD_LABEL_FAMILIES


_CONTRACT_SOURCE_REVISIONS = {
    "theatticusproject/cuad": "a3c393f5d103fd0c516374e4fdff676c8176dcb1",
    "theatticusproject/maud": "37d5c3b95d18dcd8404cc5ce3fd5069be062392f",
}


def cuad_candidates(cache_dir: str | Path) -> Iterator[dict[str, Any]]:
    json_path = _hf_download("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json", cache_dir)
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    for contract_index, contract in enumerate(payload.get("data", [])):
        title = str(contract.get("title") or f"contract_{contract_index}")
        paragraphs = contract.get("paragraphs") or []
        if not paragraphs:
            continue
        context = str(paragraphs[0].get("context") or "")
        qas = paragraphs[0].get("qas") or []
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for qa in qas:
            question = str(qa.get("question", ""))
            label = _cuad_label_from_question(question)
            if label not in CUAD_LABEL_FAMILIES:
                continue
            answers = qa.get("answers") or []
            if bool(qa.get("is_impossible")) or not answers:
                continue
            for answer_index, answer in enumerate(answers):
                text = str(answer.get("text") or "").strip()
                if not text:
                    continue
                by_family[label].append(
                    {
                        "qa_id": qa.get("id"),
                        "label": label,
                        "label_question": question,
                        "label_details": _cuad_details_from_question(question),
                        "answer_index": answer_index,
                        "text": text,
                        "answer_start": answer.get("answer_start"),
                    }
                )
        for label, spans in sorted(by_family.items()):
            cleaned_spans = _clean_cuad_spans(spans)
            if len(cleaned_spans) < 2 or len(cleaned_spans) > 5:
                continue
            indexed_spans = [
                {
                    **span,
                    "span_index": idx,
                }
                for idx, span in enumerate(cleaned_spans, start=1)
            ]
            yield {
                "dataset": "cuad",
                "source_id": f"{title}__{_slug(label)}",
                "contract_index": contract_index,
                "contract_title": title,
                "contract_text": context,
                "label_family": label,
                "label_details": indexed_spans[0].get("label_details", ""),
                "positive_spans": indexed_spans,
                "family_config": CUAD_LABEL_FAMILIES[label],
            }


def _hf_download(repo_id: str, filename: str, cache_dir: str | Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=_CONTRACT_SOURCE_REVISIONS.get(repo_id),
        cache_dir=str(Path(cache_dir).expanduser().resolve() / "hf"),
    )


def _maud_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized in {"dev", "validation", "valid"}:
        return "dev"
    if normalized in {"test"}:
        return "test"
    return "train"


def _cuad_label_from_question(question: str) -> str:
    if '"' in question:
        parts = question.split('"')
        if len(parts) >= 2:
            return parts[1]
    return question


def _cuad_details_from_question(question: str) -> str:
    marker = "Details:"
    if marker in question:
        return question.split(marker, 1)[1].strip()
    return ""


def _clean_cuad_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for span in spans:
        text = " ".join(str(span.get("text") or "").split())
        if len(text) < 30:
            continue
        key = text.lower()
        if key in seen:
            continue
        copy = dict(span)
        copy["text"] = text
        out.append(copy)
        seen.add(key)
    return out


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
