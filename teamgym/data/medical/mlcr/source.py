"""Load MLCR-AA inspectable records and build exact-span source indexes."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


REQUIRED_SUMMARY_FIELDS = {"case_uuid", "summary_file", "summary_text"}
REQUIRED_QUESTION_FIELDS = {"case_uuid", "prompt_id", "question", "answer"}
PARAGRAPH_RE = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
SENTENCE_END_RE = re.compile(r"(?:[.!?](?:[\"')\]]+)?|\Z)(?=\s+|\Z)")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path, required_fields: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted(required_fields - set(row))
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {missing}")
            rows.append(row)
    return rows


def exact_paragraph_spans(text: str) -> list[dict[str, Any]]:
    """Return natural paragraph spans without rewriting source text."""

    spans: list[dict[str, Any]] = []
    for position, match in enumerate(PARAGRAPH_RE.finditer(text), start=1):
        spans.append(
            {
                "span_index": position,
                "raw_start": match.start(),
                "raw_end": match.end(),
                "text": match.group(0),
                "sha256": hashlib.sha256(match.group(0).encode()).hexdigest(),
            }
        )
    return spans


def exact_sentence_atoms(text: str) -> list[dict[str, Any]]:
    """Split source into exact sentence-like atoms while retaining raw offsets.

    Line wrapping in MLCR summaries is presentation noise, so it is not treated as
    an atom boundary. A later RIU may join adjacent atoms by taking one exact raw
    slice from the first atom start through the last atom end.
    """

    atoms: list[dict[str, Any]] = []
    cursor = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        while cursor < end and text[cursor].isspace():
            cursor += 1
        while end > cursor and text[end - 1].isspace():
            end -= 1
        if end <= cursor:
            continue
        exact = text[cursor:end]
        atoms.append(
            {
                "atom_index": len(atoms) + 1,
                "raw_start": cursor,
                "raw_end": end,
                "text": exact,
                "sha256": hashlib.sha256(exact.encode()).hexdigest(),
            }
        )
        cursor = match.end()
    return atoms


def build_case_indexes(
    summaries: Iterable[dict[str, Any]],
    questions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    questions_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        summaries_by_case[str(row["case_uuid"])].append(row)
    for row in questions:
        questions_by_case[str(row["case_uuid"])].append(row)

    indexes: list[dict[str, Any]] = []
    for case_uuid in sorted(set(summaries_by_case) | set(questions_by_case)):
        case_summaries = sorted(
            summaries_by_case.get(case_uuid, []),
            key=lambda row: str(row["summary_file"]),
        )
        case_questions = sorted(
            questions_by_case.get(case_uuid, []),
            key=lambda row: str(row["prompt_id"]),
        )
        documents = []
        for summary in case_summaries:
            text = str(summary["summary_text"])
            document_id = f"{case_uuid}:{summary['summary_file']}"
            spans = exact_paragraph_spans(text)
            atoms = exact_sentence_atoms(text)
            for span in spans:
                span["span_id"] = f"{document_id}:p{span['span_index']:04d}"
            for atom in atoms:
                atom["atom_id"] = f"{document_id}:s{atom['atom_index']:04d}"
            documents.append(
                {
                    "document_id": document_id,
                    "summary_file": str(summary["summary_file"]),
                    "text": text,
                    "char_count": len(text),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "spans": spans,
                    "atoms": atoms,
                }
            )
        indexes.append(
            {
                "schema": "mlcr_case_source_index_v1",
                "case_uuid": case_uuid,
                "source_kind": "mlcr_release_visit_summary",
                "document_count": len(documents),
                "question_count": len(case_questions),
                "documents": documents,
                "source_questions": [
                    {
                        "prompt_id": str(row["prompt_id"]),
                        "question": str(row["question"]),
                        "answer": str(row["answer"]),
                        "difficulty": str(row.get("difficulty") or ""),
                    }
                    for row in case_questions
                ],
            }
        )
        errors = validate_case_index(indexes[-1])
        if errors:
            raise ValueError(f"{case_uuid}: invalid source index: {'; '.join(errors)}")
    return indexes


def load_inspectable(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root_path = Path(root).expanduser().resolve()
    summaries = read_jsonl(root_path / "cases_summaries.jsonl", REQUIRED_SUMMARY_FIELDS)
    questions = read_jsonl(root_path / "questions.jsonl", REQUIRED_QUESTION_FIELDS)
    return summaries, questions


def validate_case_index(index: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not index.get("case_uuid"):
        errors.append("case_uuid is required")
    documents = index.get("documents")
    if not isinstance(documents, list) or not documents:
        errors.append("at least one source document is required")
        return errors
    seen_ids: set[str] = set()
    for document in documents:
        text = str(document.get("text") or "")
        if hashlib.sha256(text.encode()).hexdigest() != document.get("sha256"):
            errors.append(f"{document.get('document_id')}: document checksum mismatch")
        previous_end = -1
        for span in document.get("spans") or []:
            span_id = str(span.get("span_id") or "")
            start, end = int(span.get("raw_start", -1)), int(span.get("raw_end", -1))
            if not span_id or span_id in seen_ids:
                errors.append("span IDs must be nonempty and unique")
            seen_ids.add(span_id)
            if start < 0 or end <= start or end > len(text) or start < previous_end:
                errors.append(f"{span_id}: invalid or overlapping offsets")
                continue
            exact = text[start:end]
            if exact != span.get("text"):
                errors.append(f"{span_id}: text is not the exact source slice")
            if hashlib.sha256(exact.encode()).hexdigest() != span.get("sha256"):
                errors.append(f"{span_id}: checksum mismatch")
            previous_end = end
        previous_end = -1
        for atom in document.get("atoms") or []:
            atom_id = str(atom.get("atom_id") or "")
            start, end = int(atom.get("raw_start", -1)), int(atom.get("raw_end", -1))
            if not atom_id or atom_id in seen_ids:
                errors.append("atom IDs must be nonempty and unique")
            seen_ids.add(atom_id)
            if start < 0 or end <= start or end > len(text) or start < previous_end:
                errors.append(f"{atom_id}: invalid or overlapping offsets")
                continue
            exact = text[start:end]
            if exact != atom.get("text"):
                errors.append(f"{atom_id}: text is not the exact source slice")
            if hashlib.sha256(exact.encode()).hexdigest() != atom.get("sha256"):
                errors.append(f"{atom_id}: checksum mismatch")
            previous_end = end
    return list(dict.fromkeys(errors))


def acquire_inspectable(raw_root: str | Path, *, revision: str) -> dict[str, Any]:
    """Download immutable upstream Parquet and derive inspectable JSONL losslessly.

    Requires the optional data dependencies; no generated construction artifact
    is involved. The caller chooses and records the full upstream revision.
    """
    import urllib.request
    import os
    import pyarrow.parquet as pq
    if not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('MLCR acquisition requires a full immutable source revision')
    root = Path(raw_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for subset in ('cases_summaries', 'questions'):
        url = f'https://huggingface.co/datasets/Wisedocs/mlcr-dataset/resolve/{revision}/data/{subset}/train-00000-of-00001.parquet'
        parquet = root / f'{subset}.parquet'
        temporary = parquet.with_suffix('.parquet.part')
        with urllib.request.urlopen(url, timeout=180) as source, temporary.open('wb') as stream:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                stream.write(chunk)
        os.replace(temporary, parquet)
        rows = pq.read_table(parquet).to_pylist()
        output = root / f'{subset}.jsonl'
        temporary = output.with_suffix('.jsonl.part')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
        os.replace(temporary, output)
        files[subset] = {'url': url, 'parquet_sha256': sha256_file(parquet),
                         'jsonl_sha256': sha256_file(output), 'rows': len(rows)}
    manifest = {'repository': 'Wisedocs/mlcr-dataset', 'revision': revision,
                'conversion': 'pyarrow.read_table.to_pylist; JSON UTF-8, no text rewriting', 'files': files}
    temporary = root / 'source_manifest.json.part'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, root / 'source_manifest.json')
    return manifest
