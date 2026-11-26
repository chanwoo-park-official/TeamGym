#!/usr/bin/env python
"""Build distributed casebook tasks from original verified Multi-LexSum documents."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.core.schema import COMMON_DATAPOINT_SCHEMA_VERSION, validate_common_datapoint
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


SYSTEM_PROMPT = """You construct difficult but determinate legal casebook questions from original litigation records.

The input contains sentence-indexed excerpts from original case documents and may include a previously verified task.
Construct the question and answer from the document excerpts. Reject the case unless the excerpts support a meaningful
multi-step procedural or legal conclusion. Do not invent any fact,
rule, date, disposition, or remedy.

For an accepted case:
- Write an anonymized question that does not reveal party names, case names, docket numbers, judges, courts, cities,
  or uniquely identifying organizations. It must ask for one legal/procedural characterization and its reasoning.
- Construct four mutually exclusive, legally plausible outcomes. They must use the same syntactic frame, address the
  same legal facets, and have similar lengths. Distractors must be counterfactual combinations that would be correct
  under different source facts, not facially invalid legal rules.
- The correct outcome must follow straightforwardly when all selected evidence atoms are available.
- Select 8 to 12 atomic propositions grounded in the indexed source sentences. Each atom must state one concise
  rule, fact, order, chronology relation, exception, or legal consequence. Use only supplied sentence IDs.
- The atoms must form an inference graph with at least two branches and depth of at least three: for example,
  requested relief versus ordered relief, an interim order plus a later event, or rule plus exception plus application.
- Every atom must distinguish the correct outcome from at least one distractor. Do not select incidental names,
  background, amounts, dates, or narrative details unless changing that proposition changes the correct outcome.
- Use at least three source RIUs and at least two atoms from each used source RIU.
- Do not request a checklist, exhaustive summary, table, JSON response, or reproduction of all details.

Return one JSON object only."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    rows = read_jsonl(args.input_root)
    selected = [(i, row) for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAIJsonClient(OpenAIJsonClientConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        credentials_file=args.credentials_file,
        temperature=0.0,
        max_output_tokens=7000,
        timeout_sec=300,
        request_retry_attempts=4,
        request_log_path=str(output / "requests.jsonl"),
        log_prompts=False,
        log_responses=False,
    ))
    accepted, rejected, errors = [], [], []
    for global_index, row in selected:
        try:
            source = prepare_source(row)
            generated = generate(row, source, client, global_index)
            if not generated["suitable"]:
                rejected.append({"instance_id": row.get("instance_id"), **generated})
                continue
            task = convert(row, source, generated, global_index, args.seed)
            task["generation_metadata"].update(model=args.model, provider=args.provider)
            accepted.append(task)
            print(f"accepted {row.get('instance_id')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"instance_id": row.get("instance_id"), "error": f"{type(exc).__name__}: {exc}"})
            print(f"error {row.get('instance_id')}: {exc}", flush=True)
    write_jsonl(output / "common_datapoints.jsonl", accepted)
    write_jsonl(output / "rejected.jsonl", rejected)
    write_jsonl(output / "errors.jsonl", errors)
    write_json(output / "report.json", {
        "schema": "multilexsum_original_distributed_casebook_build_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_count": len(selected),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "error_count": len(errors),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "model": args.model,
    })


def prepare_source(row: dict[str, Any]) -> dict[str, Any]:
    sources = []
    sentence_lookup = {}
    for rui in row.get("reference_information_units", []):
        rui_id = str(rui.get("rui_id"))
        source_text = normalize(rui.get("text"))
        sentences = split_sentences(source_text)
        sentence_spans = []
        cursor = 0
        indexed = []
        for index, text in enumerate(sentences, start=1):
            start = source_text.index(text, cursor)
            cursor = start + len(text)
            sentence_spans.append((start, cursor))
            sentence_id = f"{rui_id}_s{index:02d}"
            indexed.append({"sentence_id": sentence_id, "text": text})
            sentence_lookup[sentence_id] = {
                "rui_id": rui_id,
                "index": index - 1,
                "text": text,
                "source_text": source_text,
                "sentence_spans": sentence_spans,
            }
        sources.append({
            "source_rui_id": rui_id,
            "document_type": (rui.get("source_ref") or {}).get("doc_type"),
            "document_title": (rui.get("source_ref") or {}).get("doc_title"),
            "sentences": indexed,
        })
    return {"sources": sources, "sentence_lookup": sentence_lookup}


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[\d])", normalized)
    pieces = [piece.strip() for piece in pieces if len(piece.split()) >= 4]
    if len(pieces) < 4:
        words = normalized.split()
        pieces = [" ".join(words[i:i + 45]) for i in range(0, len(words), 45) if len(words[i:i + 45]) >= 8]
    return pieces


def generate(
    row: dict[str, Any],
    source: dict[str, Any],
    client: OpenAIJsonClient,
    index: int,
) -> dict[str, Any]:
    oracle = row.get("oracle") or {}
    task = row.get("task_description") or {}
    payload = {
        "original_question": task.get("question") or task.get("objective"),
        "prior_verified_answer": oracle.get("canonical_response") or oracle.get("short_answer"),
        "original_sources": source["sources"],
        "response_format": {
            "suitable": True,
            "suitability_reason": "substantive reason",
            "question": "anonymized natural legal question",
            "correct_outcome": "complete outcome and controlling rationale",
            "distractors": [
                {
                    "outcome": "counterfactual complete outcome",
                    "distinguished_by_atom_ids": ["atom IDs needed to reject it"],
                }
            ],
            "atoms": [
                {
                    "atom_id": "atom_01",
                    "source_rui_id": "source RIU ID",
                    "sentence_ids": ["one or more supplied sentence IDs"],
                    "proposition": "concise decision-changing proposition",
                    "inferential_role": "how this proposition changes the outcome",
                }
            ],
            "reasoning_steps": [
                {"uses": ["atom IDs"], "claim": "intermediate or final legal inference"}
            ],
            "canonical_rationale": "concise multi-step explanation",
        },
    }
    previous = None
    error = None
    for attempt in range(1, 4):
        request = payload if previous is None else {
            "original_task": payload,
            "previous_response": previous,
            "validation_error": error,
            "instruction": "Correct the validation error without inventing information. Reject the case if compliance is impossible.",
        }
        previous = client.generate_json(
            prompt_id=f"{safe(row.get('instance_id'))}.original_distributed.{index}.attempt_{attempt}",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=json.dumps(request, indent=2, ensure_ascii=False),
        )
        try:
            return normalize_generated(previous, source)
        except ValueError as exc:
            error = str(exc)
    raise ValueError(error or "generation failed")


def normalize_generated(generated: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    if not bool(generated.get("suitable")):
        return {"suitable": False, "suitability_reason": normalize(generated.get("suitability_reason"))}
    question = normalize(generated.get("question"))
    correct = normalize(generated.get("correct_outcome"))
    rationale = normalize(generated.get("canonical_rationale"))
    if min(len(question.split()), len(correct.split()), len(rationale.split())) < 15:
        raise ValueError("question, correct outcome, or rationale too short")
    if any(term in question.casefold() for term in ["rui", "reference information", "checklist", "json", "table"]):
        raise ValueError("question exposes benchmark structure or requests extraction")

    valid_sentence_ids = set(source["sentence_lookup"])
    atoms = []
    atom_ids = set()
    source_counts: dict[str, int] = {}
    for item in generated.get("atoms") or []:
        if not isinstance(item, dict):
            continue
        atom_id = normalize(item.get("atom_id"))
        source_rui_id = normalize(item.get("source_rui_id"))
        sentence_ids = dedupe(str(value) for value in item.get("sentence_ids") or [] if str(value) in valid_sentence_ids)
        proposition = normalize(item.get("proposition"))
        role = normalize(item.get("inferential_role"))
        if not atom_id or atom_id in atom_ids or not sentence_ids or len(proposition.split()) < 8 or len(role.split()) < 6:
            continue
        if any(source["sentence_lookup"][sid]["rui_id"] != source_rui_id for sid in sentence_ids):
            continue
        atom_ids.add(atom_id)
        source_counts[source_rui_id] = source_counts.get(source_rui_id, 0) + 1
        atoms.append({
            "atom_id": atom_id,
            "source_rui_id": source_rui_id,
            "sentence_ids": sentence_ids,
            "proposition": proposition,
            "inferential_role": role,
        })
    if not 8 <= len(atoms) <= 12:
        raise ValueError(f"expected 8-12 valid atoms, got {len(atoms)}")
    if len(source_counts) < 3 or min(source_counts.values()) < 2:
        raise ValueError(f"atoms must use at least 3 sources with at least 2 atoms each: {source_counts}")

    distractors = []
    covered = set()
    for item in generated.get("distractors") or []:
        if not isinstance(item, dict):
            continue
        outcome = normalize(item.get("outcome"))
        ids = dedupe(str(value) for value in item.get("distinguished_by_atom_ids") or [] if str(value) in atom_ids)
        if len(outcome.split()) >= 15 and ids:
            distractors.append({"outcome": outcome, "distinguished_by_atom_ids": ids})
            covered.update(ids)
    if len(distractors) != 3:
        raise ValueError("expected exactly three valid distractors")
    if covered != atom_ids:
        raise ValueError("every atom must distinguish at least one distractor")
    lengths = [len(correct.split())] + [len(item["outcome"].split()) for item in distractors]
    if min(lengths) == 0 or max(lengths) / min(lengths) > 1.30:
        raise ValueError(f"outcomes not length-balanced: {lengths}")

    steps = []
    step_coverage = set()
    for item in generated.get("reasoning_steps") or []:
        if not isinstance(item, dict):
            continue
        uses = dedupe(str(value) for value in item.get("uses") or [] if str(value) in atom_ids)
        claim = normalize(item.get("claim"))
        if uses and len(claim.split()) >= 8:
            steps.append({"uses": uses, "claim": claim})
            step_coverage.update(uses)
    if step_coverage != atom_ids or len(steps) < 4:
        raise ValueError("reasoning graph does not cover every atom with at least four steps")
    return {
        "suitable": True,
        "suitability_reason": normalize(generated.get("suitability_reason")),
        "question": question,
        "correct_outcome": correct,
        "distractors": distractors,
        "atoms": atoms,
        "reasoning_steps": steps,
        "canonical_rationale": rationale,
    }


def convert(
    source_row: dict[str, Any],
    source: dict[str, Any],
    generated: dict[str, Any],
    index: int,
    seed: int,
) -> dict[str, Any]:
    row = copy.deepcopy(source_row)
    row["schema_version"] = COMMON_DATAPOINT_SCHEMA_VERSION
    rng = random.Random(seed * 100_000 + index)
    participants = [
        {
            "participant_id": f"participant_{chr(97 + i)}",
            "role_name": "Case-file analyst",
            "private_context_unit_ids": [],
            "owned_rui_ids": [],
            "metadata": {"full_context_partition_role": "original Multi-LexSum source windows plus unrelated case-file chunks"},
        }
        for i in range(6)
    ]
    atoms = list(generated["atoms"])
    rng.shuffle(atoms)
    atom_owner = {}
    for atom_index, atom in enumerate(atoms):
        atom_owner[atom["atom_id"]] = participants[atom_index % len(participants)]["participant_id"]

    reference_units = []
    private_units = []
    source_refs = {unit["rui_id"]: unit.get("source_ref", {}) for unit in source_row.get("reference_information_units", [])}
    for atom in atoms:
        atom_id = atom["atom_id"]
        owner = atom_owner[atom_id]
        participant = next(item for item in participants if item["participant_id"] == owner)
        rui_id = f"derived_{atom_id}"
        participant["owned_rui_ids"].append(rui_id)
        quote = source_window(atom["sentence_ids"], source["sentence_lookup"])
        context_id = f"source_window_{safe(row.get('instance_id'))}_{safe(atom_id)}"
        participant["private_context_unit_ids"].append(context_id)
        private_units.append({
            "context_unit_id": context_id,
            "assigned_to": [owner],
            "text": quote,
            "hidden_contains_rui_ids": [rui_id],
            "metadata": {
                **source_refs.get(atom["source_rui_id"], {}),
                "context_kind": "original_multilexsum_source_window",
                "source_rui_id": atom["source_rui_id"],
                "sentence_ids": atom["sentence_ids"],
            },
        })
        reference_units.append({
            "rui_id": rui_id,
            "text": atom["proposition"],
            "summary": atom["proposition"],
            "information_feature": atom["inferential_role"],
            "rui_type": "verified_atomic_proposition_from_original_multilexsum",
            "source_ref": {
                **source_refs.get(atom["source_rui_id"], {}),
                "source_dataset": "Multi-LexSum",
                "source_rui_id": atom["source_rui_id"],
                "sentence_ids": atom["sentence_ids"],
            },
            "owned_by": [owner],
            "metadata": {"decision_critical": True, "original_source_grounded": False},
        })

    decoys = [unit for unit in source_row.get("private_context_units", []) if str(unit.get("text") or "").strip()]
    rng.shuffle(decoys)
    for decoy_index, decoy in enumerate(decoys[:24]):
        participant = participants[decoy_index % len(participants)]
        context_id = f"decoy_{index:03d}_{decoy_index:02d}"
        participant["private_context_unit_ids"].append(context_id)
        private_units.append({
            "context_unit_id": context_id,
            "assigned_to": [participant["participant_id"]],
            "text": str(decoy.get("text") or ""),
            "hidden_contains_rui_ids": [],
            "metadata": {"context_kind": "original_multilexsum_case_chunk", **(decoy.get("metadata") or {})},
        })
    rng.shuffle(private_units)

    option_rows = [{"correct": True, "outcome": generated["correct_outcome"]}] + [
        {"correct": False, "outcome": item["outcome"]} for item in generated["distractors"]
    ]
    option_rows.sort(key=lambda item: hashlib.sha256(f"{row.get('instance_id')}:{item['outcome']}".encode()).hexdigest())
    labels = ["decision_a", "decision_b", "decision_c", "decision_d"]
    definitions = {label: option["outcome"] for label, option in zip(labels, option_rows, strict=True)}
    answer = next(label for label, option in zip(labels, option_rows, strict=True) if option["correct"])
    atom_to_rui = {atom["atom_id"]: f"derived_{atom['atom_id']}" for atom in atoms}
    steps = [
        {
            "step": step_index,
            "uses": [atom_to_rui[atom_id] for atom_id in step["uses"]],
            "claim": step["claim"],
            "judge_focus": "Recover and combine the source-grounded premises used by this inference.",
        }
        for step_index, step in enumerate(generated["reasoning_steps"], start=1)
    ]
    required_ids = [unit["rui_id"] for unit in reference_units]

    old_id = str(row.get("instance_id"))
    row["instance_id"] = f"{old_id}_original_distributed_casebook"
    row["dataset"] = "multilexsum_original_distributed_casebook"
    row["scenario_family"] = "multilexsum_original_distributed_reasoning"
    row["task_description"] = {
        "title": "Anonymized Multi-LexSum procedural casebook decision",
        "objective": generated["question"],
        "question": generated["question"],
        "answer_format": {
            "type": "single_choice",
            "choices": labels,
            "instruction": "Select the correct legal or procedural characterization and explain the controlling reasoning.",
        },
    }
    row["decision_set"] = labels
    row["decision_option_definitions"] = definitions
    row["initial_common_information"] = {
        "task": generated["question"],
        "decision_options": definitions,
        "common_reference_units": [],
        "collaboration_note": "Participants hold different portions of the anonymized case file.",
    }
    row["participants"] = participants
    row["participant_directory"] = [{"participant_id": p["participant_id"], "role_name": p["role_name"]} for p in participants]
    row["private_context_units"] = private_units
    row["reference_information_units"] = reference_units
    row["oracle"] = {
        "answer": answer,
        "short_answer": generated["canonical_rationale"],
        "canonical_response": generated["canonical_rationale"],
        "required_rui_ids": required_ids,
        "reasoning_steps": steps,
        "required_decision_steps": steps,
        "smallest_sufficient_rui_sets": [{"rui_ids": required_ids, "answer": answer}],
        "rationale": generated["canonical_rationale"],
        "judge_rubric": {
            "correct": "The selected decision equals the source-grounded oracle decision.",
            "incorrect": "The selected decision differs from the source-grounded oracle decision.",
        },
    }
    row["dataset_specific"] = {
        "task_variant": "original_multilexsum_distributed_casebook",
        "source_instance_id": old_id,
        "original_source_grounded": False,
        "atom_count": len(atoms),
        "participant_count": len(participants),
        "suitability_reason": generated["suitability_reason"],
    }
    row["generation_metadata"] = {
        "builder": "teamgym.data.legal.multilexsum.generation",
        "schema": "multilexsum_original_distributed_casebook_v1",
        "source_policy": "original Multi-LexSum excerpts and case-file chunks; grounding audit and qualification required",
        "requires_source_grounding_audit": True,
        "source_packet_schema": source_row.get("packet_schema"),
        "source_packet_policy": source_row.get("packet_policy"),
    }
    row["source"] = {**source_row.get("source", {}), "source_family": "Multi-LexSum", "source_instance_id": old_id}
    row.pop("packet_schema", None)
    row.pop("packet_policy", None)
    validation = validate_common_datapoint(row)
    row["format_validation"] = validation
    if not validation["valid"]:
        raise ValueError(f"schema validation failed: {validation}")
    return row


def source_window(sentence_ids: list[str], lookup: dict[str, dict[str, Any]]) -> str:
    selected = [lookup[sid] for sid in sentence_ids]
    grouped: dict[str, set[int]] = {}
    for item in selected:
        grouped.setdefault(item["rui_id"], set()).add(item["index"])
    windows = []
    for rui_id, indices in grouped.items():
        sample = next(item for item in selected if item["rui_id"] == rui_id)
        spans = sample["sentence_spans"]
        start = spans[max(0, min(indices) - 1)][0]
        end = spans[min(len(spans) - 1, max(indices) + 1)][1]
        windows.append(sample["source_text"][start:end])
    return "\n\n".join(windows)


def dedupe(values: Any) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def safe(value: Any) -> str:
    text = "".join(character if character.isalnum() else "_" for character in str(value or "item"))
    return "_".join(part for part in text.split("_") if part)[:160]


if __name__ == "__main__":
    main()
