"""Prepare original-document packets and attach them to MultiLexSum tasks.

This is a documented reconstruction of packet preparation. Frozen paper inputs
remain the reference for the published experiments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.data.legal.multilexsum.source import load_eligible_metadata
from teamgym.core.schema import validate_common_datapoint


PACKET_SCHEMA = "multilexsum_original_source_packet_v1"


def iter_source_documents(path: Path, *, read_size: int = 1 << 20) -> Iterator[tuple[str, Any]]:
    """Read a sources.json object without loading the entire corpus into memory."""
    if read_size < 1:
        raise ValueError("read_size must be positive")
    decoder = json.JSONDecoder(parse_constant=lambda _value: None)
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            chunk = handle.read(read_size)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def character() -> str:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    raise ValueError(f"Unexpected end of source JSON: {path}")
                refill()

        def value() -> Any:
            nonlocal position
            character()
            while True:
                try:
                    item, position_after = decoder.raw_decode(buffer, position)
                    position = position_after
                    return item
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Invalid source JSON: {path}: {exc.msg}") from exc
                    refill()

        if character() != "{":
            raise ValueError("sources.json must contain an object keyed by document ID")
        position += 1
        if character() == "}":
            return
        while True:
            key = value()
            if not isinstance(key, str) or character() != ":":
                raise ValueError("Invalid document ID in sources.json")
            position += 1
            document = value()
            yield key, document
            delimiter = character()
            position += 1
            if delimiter == "}":
                return
            if delimiter != ",":
                raise ValueError("Expected a comma between source documents")


def load_documents(path: Path, needed: set[str]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    if not needed:
        return documents
    for document_id, document in iter_source_documents(path):
        if document_id in needed:
            if document_id in documents or not isinstance(document, dict):
                raise ValueError(f"Duplicate or invalid source document: {document_id}")
            if document.get("doc_id", document_id) != document_id:
                raise ValueError(f"Source document ID does not match its key: {document_id}")
            documents[document_id] = document
            if documents.keys() >= needed:
                return documents
    raise ValueError(f"Missing source documents: {sorted(needed - documents.keys())}")


def spaced(items: list[Any], limit: int) -> list[Any]:
    if limit < 1:
        raise ValueError("Selection limits must be positive")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    return [items[index * (len(items) - 1) // (limit - 1)] for index in range(limit)]


def source_chunks(text: str, words: int) -> list[tuple[int, int, str]]:
    if words < 1:
        raise ValueError("chunk_words must be positive")
    spans = list(re.finditer(r"\S+", text))
    return [
        (spans[index].start(), spans[min(index + words, len(spans)) - 1].end(),
         text[spans[index].start():spans[min(index + words, len(spans)) - 1].end()])
        for index in range(0, len(spans), words)
    ]


def build_packet(
    case: dict[str, Any], documents: dict[str, dict[str, Any]], *,
    chunk_words: int = 250, max_chunks_per_document: int = 8, excerpts_per_document: int = 2,
) -> dict[str, Any]:
    """Select bounded original passages; summary text never becomes task evidence."""
    if min(chunk_words, max_chunks_per_document, excerpts_per_document) < 1:
        raise ValueError("Packet preparation settings must be positive")
    case_id = str(case["case_id"])
    if not case_id or len(set(case["case_documents"])) != len(case["case_documents"]):
        raise ValueError("A collection needs a case ID and unique document IDs")
    units, excerpts = [], []
    for document_id in case["case_documents"]:
        document_id = str(document_id)
        document = documents[document_id]
        text = document.get("doc_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing source text for {document_id}")
        selected = spaced(list(enumerate(source_chunks(text, chunk_words))), max_chunks_per_document)
        document_units = []
        for chunk_index, (start, end, passage) in selected:
            context_id = f"{document_id}_chunk_{chunk_index:05d}"
            metadata = {
                "context_kind": "multilexsum_selected_original_case_chunk",
                "source_doc_id": document_id,
                "doc_title": document.get("doc_title"),
                "doc_type": document.get("doc_type"),
                "source_char_start": start,
                "source_char_end": end,
                "source_document_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            document_units.append({
                "context_unit_id": context_id,
                "text": passage,
                "hidden_contains_rui_ids": [],
                "metadata": metadata,
            })
        units.extend(document_units)
        for unit in spaced(document_units, excerpts_per_document):
            excerpts.append({
                "rui_id": unit["context_unit_id"],
                "text": unit["text"],
                "source_ref": {**unit["metadata"], "source_context_unit_id": unit["context_unit_id"]},
            })
    if len({unit["source_ref"]["source_doc_id"] for unit in excerpts}) < 3:
        raise ValueError("A packet must provide excerpts from at least three documents")
    return {
        "packet_schema": PACKET_SCHEMA,
        "instance_id": f"multilexsum_source_{case_id}",
        "source": {"source_family": "Multi-LexSum", "collection_id": case_id, "split": case.get("_split")},
        "reference_information_units": excerpts,
        "private_context_units": units,
        "packet_policy": {
            "schema": PACKET_SCHEMA,
            "chunk_words": chunk_words,
            "max_chunks_per_document": max_chunks_per_document,
            "excerpts_per_document": excerpts_per_document,
            "selection": "evenly spaced chunks and excerpts in original document order",
            "summary_usage": "eligibility_only",
        },
    }


def source_identity(row: dict[str, Any]) -> str:
    return str((row.get("dataset_specific") or {}).get("source_instance_id")
               or (row.get("source") or {}).get("source_instance_id") or row["instance_id"])


def expand_packet(task: dict[str, Any], packet: dict[str, Any], *, seed: int = 20260712) -> dict[str, Any]:
    """Preserve the decision and PIU ownership while rebuilding original contexts."""
    if source_identity(task) not in {str(packet["instance_id"]), source_identity(packet)}:
        raise ValueError(f"Source packet does not match task: {task['instance_id']}")
    out = copy.deepcopy(task)
    participants = out["participants"]
    participant_ids = [str(participant["participant_id"]) for participant in participants]
    if len(participant_ids) != len(set(participant_ids)) or len(participant_ids) < 2:
        raise ValueError("Task needs at least two unique participants")
    rui_ids = {unit["rui_id"] for unit in out["reference_information_units"]}
    if len(rui_ids) != len(out["reference_information_units"]):
        raise ValueError("Duplicate PIU IDs")
    owners = {
        rui_id: [p["participant_id"] for p in participants if rui_id in p.get("owned_rui_ids", [])]
        for rui_id in rui_ids
    }
    public_ids = {unit["rui_id"] for unit in out.get("initial_common_information", {}).get("common_reference_units", [])}
    source_excerpts = {unit["rui_id"]: unit["text"] for unit in packet["reference_information_units"]}
    existing_windows = {unit["context_unit_id"]: unit["text"] for unit in packet["private_context_units"]
                        if unit.get("hidden_contains_rui_ids")}
    windows, covered, used_sources = [], set(), set()
    for unit in out["private_context_units"]:
        hidden = unit.get("hidden_contains_rui_ids") or []
        if not hidden:
            continue
        if len(hidden) != 1 or hidden[0] not in rui_ids:
            raise ValueError("Each source window must support exactly one known PIU")
        source_id = (unit.get("metadata") or {}).get("source_rui_id")
        passage = " ".join(unit["text"].split())
        if not passage:
            raise ValueError("Empty PIU source window")
        if source_id in source_excerpts:
            if passage not in " ".join(source_excerpts[source_id].split()):
                raise ValueError(f"Source window is not present in its original excerpt: {source_id}")
            used_sources.add(source_id)
        elif existing_windows.get(unit["context_unit_id"]) != unit["text"]:
            raise ValueError(f"No original source evidence for {unit['context_unit_id']}")
        unit["assigned_to"] = list(participant_ids if hidden[0] in public_ids else owners[hidden[0]])
        if not unit["assigned_to"]:
            raise ValueError(f"PIU has no owner: {hidden[0]}")
        windows.append(unit)
        covered.update(hidden)
    if covered != rui_ids:
        raise ValueError(f"Missing original source windows for PIUs: {sorted(rui_ids - covered)}")
    background = [copy.deepcopy(unit) for unit in packet["private_context_units"]
                  if not unit.get("hidden_contains_rui_ids") and unit["context_unit_id"] not in used_sources]
    rng = random.Random(f"{seed}:{source_identity(task)}")
    rng.shuffle(background)
    for index, unit in enumerate(background):
        unit["assigned_to"] = [participant_ids[index % len(participant_ids)]]
    contexts = windows + background
    if len({unit["context_unit_id"] for unit in contexts}) != len(contexts):
        raise ValueError("Duplicate context IDs in expanded source packet")
    for participant in participants:
        participant["private_context_unit_ids"] = [unit["context_unit_id"] for unit in contexts
                                                    if participant["participant_id"] in unit["assigned_to"]]
    for unit in out["reference_information_units"]:
        unit["owned_by"] = owners[unit["rui_id"]]
    out["private_context_units"] = contexts
    out["participant_directory"] = [{"participant_id": p["participant_id"], "role_name": p["role_name"]} for p in participants]
    out["generation_metadata"] = {
        **out.get("generation_metadata", {}),
        "packet_builder": "teamgym.data.legal.multilexsum.packets",
        "packet_expansion_schema": "multilexsum_packet_expansion_v1",
        "packet_policy": packet.get("packet_policy", {"source": "preserved original task packets"}),
        "packet_contexts_require_qualification": True,
        "packet_expansion_seed": seed,
        "excluded_background_source_chunks": sorted(used_sources),
    }
    # Cached checks concern earlier contexts; the expanded contexts need fresh qualification.
    for key in ["participant_leakage_deletion_funnel", "participant_option_exclusion_funnel"]:
        out["generation_metadata"].pop(key, None)
    prior_qualification = out["generation_metadata"].pop("qualification_source", None)
    if prior_qualification:
        out["generation_metadata"]["prior_qualification_source"] = prior_qualification
    out.get("dataset_specific", {}).pop("qualification_survivor", None)
    out["format_validation"] = validate_common_datapoint(out)
    if not out["format_validation"]["valid"]:
        raise ValueError(f"Expanded task failed schema validation: {task['instance_id']}")
    return out


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bounded original-document packets for task generation.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--chunk-words", type=int, default=250)
    parser.add_argument("--max-chunks-per-document", type=int, default=8)
    parser.add_argument("--excerpts-per-document", type=int, default=2)
    args = parser.parse_args()
    if min(args.limit, args.chunk_words, args.max_chunks_per_document, args.excerpts_per_document) < 1:
        parser.error("limits and chunk settings must be positive")
    input_root = args.input_root.expanduser().resolve()
    cases = load_eligible_metadata(input_root=input_root)
    random.Random(args.seed).shuffle(cases)
    cases = cases[:args.limit]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        parser.error("No eligible collections or duplicate collection IDs")
    documents = load_documents(input_root / "sources.json", {str(doc) for case in cases for doc in case["case_documents"]})
    packets = [build_packet(case, documents, chunk_words=args.chunk_words,
                           max_chunks_per_document=args.max_chunks_per_document,
                           excerpts_per_document=args.excerpts_per_document) for case in cases]
    write_jsonl(args.output_root / "source_packets.jsonl", packets)
    write_json(args.output_root / "packet_report.json", {
        "schema": PACKET_SCHEMA, "input_root": str(input_root), "seed": args.seed,
        "collection_count": len(packets), "source_document_count": len(documents),
        "policy": packets[0]["packet_policy"],
        "note": "Prepared source passages require task generation and model-based qualification before release.",
    })


def expand_main() -> None:
    parser = argparse.ArgumentParser(description="Attach original source packets to generated tasks.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--source-packets", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    tasks, packets = read_jsonl(args.input_root), read_jsonl(args.source_packets)
    by_id: dict[str, dict[str, Any]] = {}
    for packet in packets:
        for key in {str(packet["instance_id"]), source_identity(packet)}:
            if key in by_id and by_id[key] is not packet:
                raise ValueError(f"Ambiguous source packet ID: {key}")
            by_id[key] = packet
    if not tasks or len({task["instance_id"] for task in tasks}) != len(tasks):
        raise ValueError("Expected nonempty tasks with unique IDs")
    expanded = []
    for task in tasks:
        key = source_identity(task)
        if key not in by_id:
            raise ValueError(f"Missing source packet: {key}")
        expanded.append(expand_packet(task, by_id[key], seed=args.seed))
    write_jsonl(args.output_root / "common_datapoints.jsonl", expanded)
    write_json(args.output_root / "packet_expansion_report.json", {
        "schema": "multilexsum_packet_expansion_report_v1", "task_count": len(expanded),
        "seed": args.seed, "input_root": str(args.input_root), "source_packets": str(args.source_packets),
        "context_unit_count": sum(len(row["private_context_units"]) for row in expanded),
        "requires_fresh_qualification": True,
    })
