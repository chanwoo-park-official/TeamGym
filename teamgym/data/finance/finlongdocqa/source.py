"""FinLongDocQA source loading and deterministic report block indexing."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PAGE_RE = re.compile(r"(?m)^# Page (\d+)\s*$")
PARSER_VERSION = "finlongdocqa_markdown_blocks_v4_unit_context"


def extract_unit_conventions(text: str) -> list[dict[str, Any]]:
    """Retain explicit report-wide tabular currency declarations verbatim.

    Capture the declaration sentence only, never nearby financial results.
    Table-specific units still override a declaration's stated default.
    """
    pattern = re.compile(r"\bTabular\s+amounts\s+are\s+in\s+(?:U\.S\.\s+)?Dollars\s+in\s+"
                         r"(?:thousands|millions|billions)\b[^.]{0,200}\.", re.I)
    rows, seen = [], set()
    pages = list(PAGE_RE.finditer(text))
    for match in pattern.finditer(text):
        exact = match.group()
        key = " ".join(exact.lower().split())
        if key in seen: continue
        seen.add(key)
        preceding = [p for p in pages if p.start() <= match.start()]
        rows.append({"text": exact, "raw_start": match.start(), "raw_end": match.end(),
            "page_number": int(preceding[-1].group(1)) if preceding else None,
            "sha256": hashlib.sha256(exact.encode()).hexdigest()})
    return rows


def load_qa_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_number"] = line_number
            rows.append(row)
    return rows


def group_qas_by_report(
    rows: Iterable[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        company = str(row.get("company") or "")
        year = str(row.get("year") or "")
        groups.setdefault(f"{company}_{year}", []).append(row)
    return list(groups.items())


def qa_for_generation(row: dict[str, Any]) -> dict[str, Any]:
    """Exclude hidden thoughts and code from every LLM input."""

    return {
        "qa_id": str(row.get("id") or row.get("_source_line_number") or ""),
        "question": str(row.get("question") or ""),
        "answer": row.get("answer"),
        "answer_type": str(row.get("type") or ""),
        "page_numbers": [int(value) for value in row.get("page_numbers") or []],
    }


class ReportStore:
    def __init__(self, reports_zip: str | Path):
        self.path = Path(reports_zip).expanduser().resolve()
        self.archive = zipfile.ZipFile(self.path)

    def load(self, company: str, year: str) -> tuple[str, dict[str, Any]]:
        member = f"reports/{company}/{year}.md"
        text = self.archive.read(member).decode("utf-8")
        return text, {
            "kind": "zip_member",
            "path": str(self.path),
            "member": member,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    def close(self) -> None:
        self.archive.close()


def index_report(report_id: str, report_text: str) -> dict[str, Any]:
    pages = split_pages(report_text)
    parsed_blocks = [
        block
        for page in pages
        for block in segment_page(
            report_id=report_id,
            page_number=page["page_number"],
            page_text=page["text"],
            page_raw_start=page["raw_start"],
        )
    ]
    blocks = normalize_adjacent_fragments(parsed_blocks, report_text)
    return {
        "schema": "finlongdocqa_report_block_index_v1",
        "parser_version": PARSER_VERSION,
        "report_id": report_id,
        "report_char_count": len(report_text),
        "page_count": len(pages),
        "block_count": len(blocks),
        "pages": [
            {
                "page_number": page["page_number"],
                "raw_start": page["raw_start"],
                "raw_end": page["raw_end"],
            }
            for page in pages
        ],
        "blocks": blocks,
        "source_unit_conventions": extract_unit_conventions(report_text),
    }


def normalize_adjacent_fragments(
    blocks: list[dict[str, Any]],
    report_text: str,
) -> list[dict[str, Any]]:
    """Merge only adjacent short label/value fragments from the same page.

    The merged text remains an exact raw span. Separate complete paragraphs,
    tables, non-adjacent disclosures, and semantic duplicates are not merged.
    """

    normalized: list[dict[str, Any]] = []
    index = 0
    while index < len(blocks):
        current = blocks[index]
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if following is not None and _should_merge_fragments(current, following):
            raw_start = int(current["raw_start"])
            raw_end = int(following["raw_end"])
            child_ids = [
                str(current.get("block_id") or ""),
                str(following.get("block_id") or ""),
            ]
            normalized.append(
                {
                    "block_id": (
                        f"p{int(current['page_number']):03d}_composite_"
                        + "_".join(
                            child_id.rsplit("_", 1)[-1]
                            for child_id in child_ids
                        )
                    ),
                    "report_id": current.get("report_id"),
                    "page_number": current.get("page_number"),
                    "block_type": "composite",
                    "heading": current.get("heading") or following.get("heading"),
                    "text": report_text[raw_start:raw_end],
                    "raw_start": raw_start,
                    "raw_end": raw_end,
                    "char_count": raw_end - raw_start,
                    "needs_segmentation_review": False,
                    "normalization": {
                        "kind": "adjacent_label_value_merge",
                        "child_block_ids": child_ids,
                    },
                }
            )
            index += 2
            continue
        normalized.append(current)
        index += 1
    return normalized


def split_pages(report_text: str) -> list[dict[str, Any]]:
    matches = list(PAGE_RE.finditer(report_text))
    return [
        {
            "page_number": int(match.group(1)),
            "raw_start": match.start(),
            "raw_end": (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(report_text)
            ),
            "text": report_text[
                match.start() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(report_text)
                )
            ],
        }
        for index, match in enumerate(matches)
    ]


def segment_page(
    *,
    report_id: str,
    page_number: int,
    page_text: str,
    page_raw_start: int,
) -> list[dict[str, Any]]:
    """Split one page into exact, contiguous paragraph and table spans."""

    lines = page_text.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    blocks: list[dict[str, Any]] = []
    heading = ""
    block_index = 0
    index = 1 if lines and lines[0].lstrip().startswith("# Page ") else 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped == "---":
            index += 1
            continue
        if stripped.startswith("##"):
            heading = stripped.lstrip("#").strip()
            index += 1
            continue
        start_line = index
        if stripped.startswith("|"):
            kind = "table"
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|"):
                index += 1
        else:
            kind = "paragraph"
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if (
                    not candidate
                    or candidate == "---"
                    or candidate.startswith(("##", "|"))
                ):
                    break
                index += 1
        local_start = offsets[start_line]
        local_end = offsets[index] if index < len(offsets) else len(page_text)
        while local_start < local_end and page_text[local_start].isspace():
            local_start += 1
        while local_end > local_start and page_text[local_end - 1].isspace():
            local_end -= 1
        text = page_text[local_start:local_end]
        if not text or text == "<!-- image -->":
            continue
        block_index += 1
        raw_start = page_raw_start + local_start
        raw_end = page_raw_start + local_end
        blocks.append(
            {
                "block_id": f"p{page_number:03d}_{kind}_{block_index:02d}",
                "report_id": report_id,
                "page_number": page_number,
                "block_type": kind,
                "heading": heading,
                "text": text,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "char_count": len(text),
                "needs_segmentation_review": _needs_review(kind, text),
            }
        )
    return blocks


def blocks_for_pages(
    report_index: dict[str, Any],
    page_numbers: Iterable[int],
) -> list[dict[str, Any]]:
    selected = {int(value) for value in page_numbers}
    return [
        block
        for block in report_index.get("blocks") or []
        if int(block.get("page_number") or -1) in selected
    ]


def validate_report_index(
    report_index: dict[str, Any],
    report_text: str,
) -> list[str]:
    errors: list[str] = []
    if report_index.get("parser_version") != PARSER_VERSION:
        errors.append("unrecognized parser version")
    if not report_index.get("pages") or not report_index.get("blocks"):
        errors.append("report index requires recognized pages and nonempty blocks")
    if report_index.get("page_count") != len(report_index.get("pages") or []) or report_index.get("block_count") != len(report_index.get("blocks") or []):
        errors.append("index counts disagree with contents")
    if report_index.get("report_char_count") != len(report_text):
        errors.append("report character count mismatch")
    pages = [row.get("page_number") for row in report_index.get("pages") or []]
    if any(type(value) is not int or value < 1 for value in pages) or len(pages) != len(set(pages)):
        errors.append("invalid or duplicate page numbers")
    block_ids: set[str] = set()
    for block in report_index.get("blocks") or []:
        block_id = str(block.get("block_id") or "")
        if not block_id or block_id in block_ids:
            errors.append(f"missing or duplicate block_id {block_id!r}")
        block_ids.add(block_id)
        start = int(block.get("raw_start") or 0)
        end = int(block.get("raw_end") or 0)
        if not (0 <= start < end <= len(report_text)):
            errors.append(f"{block_id}: invalid raw range {start}:{end}")
        elif report_text[start:end] != str(block.get("text") or ""):
            errors.append(f"{block_id}: text does not match its raw report span")
    return list(dict.fromkeys(errors))


def _needs_review(kind: str, text: str) -> bool:
    if len(text) > 20_000:
        return True
    if kind == "paragraph" and len(text) > 10_000:
        return True
    if kind == "table":
        lines = [line for line in text.splitlines() if line.strip()]
        return bool(lines) and not all(line.lstrip().startswith("|") for line in lines)
    return False


def _should_merge_fragments(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    if first.get("page_number") != second.get("page_number"):
        return False
    if first.get("block_type") != "paragraph" or second.get(
        "block_type"
    ) != "paragraph":
        return False
    if first.get("heading") != second.get("heading"):
        return False
    first_text = str(first.get("text") or "").strip()
    second_text = str(second.get("text") or "").strip()
    if not first_text or not second_text:
        return False
    return (
        _is_short_value_fragment(first_text)
        and _is_short_label_fragment(second_text)
    ) or (
        _is_short_label_fragment(first_text)
        and _is_short_value_fragment(second_text)
    )


def _is_short_value_fragment(text: str) -> bool:
    if len(text) > 80 or len(text.split()) > 8 or not any(
        character.isdigit() for character in text
    ):
        return False
    return bool(
        re.fullmatch(
            r"[\s$€£¥₩()\-+.,:%/0-9A-Za-z]+",
            text,
        )
    )


def _is_short_label_fragment(text: str) -> bool:
    if len(text) > 100 or len(text.split()) > 12:
        return False
    if any(character.isdigit() for character in text):
        return False
    if text.endswith((".", "?", "!", ";")):
        return False
    return bool(re.search(r"[A-Za-z]", text))


def restrict_table_rows(block: dict[str, Any], row_numbers: list[int]) -> dict[str, Any]:
    """Select exact original table rows, retaining headers and absolute provenance.

    This is the explicit table exception to complete-block copying. One source
    block still maps to one PIU; rows are never split into separate PIUs.
    """
    if block.get("block_type") != "table":
        raise ValueError("row restriction requires a table block")
    lines = str(block["text"]).splitlines(keepends=True)
    if not isinstance(row_numbers, list) or not row_numbers or any(type(n) is not int for n in row_numbers):
        raise ValueError("source_row_numbers must be a nonempty integer array")
    if row_numbers != sorted(set(row_numbers)) or not set(row_numbers) <= set(range(1, len(lines) + 1)):
        raise ValueError("source_row_numbers must be unique ordered original rows")
    # Markdown table header and separator are always retained. Unit/period
    # labels within further rows remain the constructor's grounding obligation.
    if not {1, 2} <= set(row_numbers) or len(row_numbers) < 3:
        raise ValueError("table restriction must retain header, separator and data")
    starts = []
    cursor = int(block["raw_start"])
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    spans = [{"source_row_number": n, "raw_start": starts[n - 1],
              "raw_end": starts[n - 1] + len(lines[n - 1]), "text": lines[n - 1]}
             for n in row_numbers]
    return {"text": "".join(lines[n - 1] for n in row_numbers),
            "source_row_numbers": list(row_numbers), "source_row_spans": spans,
            "original_block_sha256": hashlib.sha256(str(block["text"]).encode()).hexdigest()}
