"""Disjoint paper allocation, foreign-RIU leakage audit, and public modes."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from teamgym.tokenization import count_tokens

from teamgym.data.medical.evidencebench.boundary import display_materials
from teamgym.data.medical.evidencebench.llm import (
    ResponseCache,
    structured_call,
)
from teamgym.data.medical.evidencebench.source import sentence_blocks
from teamgym.data.medical.evidencebench.validation import (
    string_list,
    validate_leakage_audit,
)


def _token_estimate(text: str) -> int:
    return count_tokens(str(text))


def build_disjoint_packets(
    *,
    record: dict[str, Any],
    final_rius: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Reserve canonical sentences to owners and assign every other sentence once."""

    blocks = sentence_blocks(record)
    contexts = {
        int(row["sentence_index"]): {
            "context_id": f"context_{int(row['sentence_index']):06d}",
            "text": row["text"],
            "sentence_index": int(row["sentence_index"]),
            "sentence_type": row["block_type"],
            "source_ref": row["source_ref"],
            "hidden_contains_riu_ids": [],
            "hidden_canonical": False,
        }
        for row in blocks
    }
    participants: list[dict[str, Any]] = []
    canonical_owner: dict[int, str] = {}
    for index, riu in enumerate(final_rius, start=1):
        participant_id = f"participant_{index:02d}"
        canonical_indices = sorted(
            {
                int(sentence_index)
                for atom in riu["evidence_atoms"]
                for sentence_index in atom["sentence_indices"]
            }
        )
        for sentence_index in canonical_indices:
            if sentence_index in canonical_owner:
                raise ValueError(f"canonical sentence {sentence_index} belongs to multiple RIUs")
            canonical_owner[sentence_index] = participant_id
            contexts[sentence_index]["hidden_contains_riu_ids"].append(riu["riu_id"])
            contexts[sentence_index]["hidden_canonical"] = True
        participants.append(
            {
                "participant_id": participant_id,
                "hidden_owned_riu_ids": [riu["riu_id"]],
                "hidden_canonical_context_ids": [contexts[value]["context_id"] for value in canonical_indices],
                "context_ids": [contexts[value]["context_id"] for value in canonical_indices],
            }
        )
    by_participant = {row["participant_id"]: row for row in participants}
    loads = {
        row["participant_id"]: sum(
            _token_estimate(contexts[index]["text"])
            for index, owner in canonical_owner.items()
            if owner == row["participant_id"]
        )
        for row in participants
    }
    remaining = [index for index in contexts if index not in canonical_owner]
    random.Random(seed).shuffle(remaining)
    for sentence_index in remaining:
        owner = min(loads, key=lambda item: (loads[item], item))
        by_participant[owner]["context_ids"].append(contexts[sentence_index]["context_id"])
        loads[owner] += _token_estimate(contexts[sentence_index]["text"])
    context_by_id = {row["context_id"]: row for row in contexts.values()}
    for participant in participants:
        participant["context_ids"].sort(key=lambda value: context_by_id[value]["sentence_index"])
        participant["context_tokens"] = _participant_token_estimate(participant, context_by_id)
    assigned = [value for participant in participants for value in participant["context_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(context_by_id):
        raise ValueError("paper context partition is not exhaustive and disjoint")
    return {
        "allocation_policy": "one_each_disjoint_full_paper",
        "participants": participants,
        "contexts": list(context_by_id.values()),
    }


def _participant_token_estimate(
    participant: dict[str, Any],
    context_by_id: dict[str, dict[str, Any]],
) -> int:
    return sum(
        _token_estimate(context_by_id[value]["text"])
        for value in participant["context_ids"]
        if value in context_by_id
    )


def _refresh_context_tokens(
    participants: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
) -> None:
    for participant in participants:
        participant["context_tokens"] = _participant_token_estimate(participant, contexts)


def _public_packet(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "context_id": row["context_id"],
            "material_type": "paper_sentence",
            "text": row["text"],
            "source_order": row["sentence_index"],
        }
        for row in packet
    ]


def audit_and_sanitize_packets(
    *,
    instance_id: str,
    public_task: dict[str, Any],
    final_rius: list[dict[str, Any]],
    partition: dict[str, Any],
    client: Any,
    cache: ResponseCache,
    trace_root: Path,
    attempts: int,
    max_rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contexts = {str(row["context_id"]): dict(row) for row in partition["contexts"]}
    participants = [dict(row) for row in partition["participants"]]
    riu_by_id = {str(row["riu_id"]): row for row in final_rius}
    rounds: list[dict[str, Any]] = []
    removed_total: set[str] = set()
    # max_rounds is the repair budget; the last repair always gets a recheck.
    for round_index in range(1, max_rounds + 2):
        audits: list[dict[str, Any]] = []
        removable_total: set[str] = set()
        round_errors: list[str] = []
        dirty = False
        for participant in participants:
            participant_id = str(participant["participant_id"])
            owned = set(string_list(participant.get("hidden_owned_riu_ids")))
            foreign_ids = set(riu_by_id) - owned
            packet = [contexts[value] for value in participant["context_ids"] if value in contexts]
            packet_ids = {str(row["context_id"]) for row in packet}
            canonical_ids = set(string_list(participant.get("hidden_canonical_context_ids")))
            response, errors = structured_call(
                client=client,
                cache=cache,
                trace_root=trace_root,
                instance_id=instance_id,
                stage=f"packet_leakage.round_{round_index}.{participant_id}",
                user_prompt_name="06_packet_leakage_audit_user.txt",
                variables={
                    "PUBLIC_TASK_JSON": public_task,
                    "PARTICIPANT_JSON": {
                        "participant_id": participant_id,
                        "canonical_context_ids": sorted(canonical_ids),
                    },
                    "FOREIGN_RIUS_JSON": [
                        {
                            "riu_id": value,
                            "hidden_claim": riu_by_id[value]["hidden_claim"],
                            "required_claim_ids": riu_by_id[value]["required_claim_ids"],
                            "exact_evidence": riu_by_id[value]["evidence_atoms"],
                        }
                        for value in sorted(foreign_ids)
                    ],
                    "PACKET_JSON": _public_packet(packet),
                },
                validator=lambda row, pid=participant_id, foreign=foreign_ids, ids=packet_ids, canon=canonical_ids: validate_leakage_audit(
                    row,
                    participant_id=pid,
                    foreign_riu_ids=foreign,
                    packet_ids=ids,
                    canonical_ids=canon,
                ),
                attempts=attempts,
            )
            for result in response.get("foreign_riu_results") or []:
                if result.get("reconstructable_from_packet") is not True:
                    continue
                dirty = True
                if result.get("canonical_overlap") is True:
                    errors.append(f"{participant_id}: canonical overlap with {result.get('riu_id')}")
                removable = set(string_list(result.get("removable_noncanonical_context_ids")))
                if not removable:
                    errors.append(f"{participant_id}: leakage has no removable noncanonical context")
                removable_total.update(removable)
            round_errors.extend(errors)
            audits.append({"participant_id": participant_id, "response": response, "errors": errors})
        rounds.append(
            {
                "round": round_index,
                "audits": audits,
                "removed_context_ids": sorted(removable_total),
                "errors": round_errors,
            }
        )
        if round_errors:
            _refresh_context_tokens(participants, contexts)
            return {**partition, "participants": participants, "contexts": list(contexts.values())}, {
                "passed": False,
                "rounds": rounds,
                "removed_context_ids": sorted(removed_total),
                "errors": round_errors,
            }
        if not dirty:
            _refresh_context_tokens(participants, contexts)
            return {**partition, "participants": participants, "contexts": list(contexts.values())}, {
                "passed": True,
                "rounds": rounds,
                "removed_context_ids": sorted(removed_total),
                "errors": [],
            }
        if round_index > max_rounds:
            break
        for context_id in removable_total:
            contexts.pop(context_id, None)
        removed_total.update(removable_total)
        for participant in participants:
            participant["context_ids"] = [
                value for value in participant["context_ids"] if value in contexts
            ]
    _refresh_context_tokens(participants, contexts)
    return {**partition, "participants": participants, "contexts": list(contexts.values())}, {
        "passed": False,
        "rounds": rounds,
        "removed_context_ids": sorted(removed_total),
        "errors": ["packet leakage audit did not converge"],
    }


def materialize_modes(
    *,
    final_rius: list[dict[str, Any]],
    partition: dict[str, Any],
    public_task: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    canonical = []
    for riu in final_rius:
        for atom in riu["evidence_atoms"]:
            canonical.append(
                {
                    "candidate_id": riu["candidate_id"],
                    "atom_id": atom["atom_id"],
                    "sentence_indices": atom["sentence_indices"],
                    "text": atom["text"],
                }
            )
    single_display, _ = display_materials(canonical, seed=seed)
    context_by_id = {str(row["context_id"]): row for row in partition["contexts"]}
    riu_by_id = {str(row["riu_id"]): row for row in final_rius}
    riu_only_participants = []
    full_context_participants = []
    for participant in partition["participants"]:
        owned_riu = riu_by_id[string_list(participant["hidden_owned_riu_ids"])[0]]
        own_rows = [
            {
                "candidate_id": owned_riu["candidate_id"],
                "atom_id": atom["atom_id"],
                "sentence_indices": atom["sentence_indices"],
                "text": atom["text"],
            }
            for atom in owned_riu["evidence_atoms"]
        ]
        own_display, _ = display_materials(own_rows, seed=seed + len(riu_only_participants) + 101)
        riu_only_participants.append(
            {"participant_id": participant["participant_id"], "private_material": own_display}
        )
        packet = [context_by_id[value] for value in participant["context_ids"] if value in context_by_id]
        full_context_participants.append(
            {"participant_id": participant["participant_id"], "private_material": _public_packet(packet)}
        )
    return {
        "public_task": public_task,
        "single_all_rius": {"materials": single_display},
        "multi_riu_only": {"participants": riu_only_participants},
        "multi_full_context": {"participants": full_context_participants},
    }
