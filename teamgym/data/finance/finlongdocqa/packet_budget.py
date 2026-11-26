"""Canonical-preserving whole-source materialization under the finance 40k cap."""
from __future__ import annotations

from copy import deepcopy
import hashlib

from teamgym.interaction.actor_utils import count_tokens
from teamgym.interaction.llm_actors import _private_context_payload, _public_task_payload
from teamgym.interaction.types import BudgetSpec

from .provenance import digest

PAPER_CONTEXT_BUDGET = 40_000


class PacketBudgetError(ValueError):
    """The exact pivotal source cannot be represented under the fixed cap."""


def complete_private_text(datapoint, participant):
    allowed = set(participant["private_context_unit_ids"])
    return "\n\n".join(str(row["text"]).strip() for row in datapoint["private_context_units"]
                       if row["context_unit_id"] in allowed and str(row["text"]).strip())


def segment_tokens(rows):
    texts = [str(row["text"]).strip() for row in rows if str(row["text"]).strip()]
    return sum(count_tokens(text) for text in texts) + max(0, len(texts) - 1) * count_tokens("\n\n")


def materialize_context_cap(datapoint, *, token_budget=PAPER_CONTEXT_BUDGET):
    """Keep complete original blocks; canonical blocks are reserved first.

    The original balanced allocation is fixed before this cap. Remaining whole
    contexts are considered in original report order and are never redistributed
    or cut into token windows. Exclusions and exact token accounting are hidden
    provenance, not participant annotations.
    """
    if type(token_budget) is not int or not 0 < token_budget <= PAPER_CONTEXT_BUDGET:
        raise PacketBudgetError("finance private context cap must be between 1 and 40000")
    contexts = datapoint["private_context_units"]
    by_id = {row["context_unit_id"]: row for row in contexts}
    if len(by_id) != len(contexts):
        raise PacketBudgetError("duplicate private context IDs")
    chosen, exclusions, before = {}, [], {}
    for participant in datapoint["participants"]:
        pid = participant["participant_id"]
        references = participant["private_context_unit_ids"]
        assigned = [row for row in contexts if pid in row["assigned_to"]]
        if len(set(references)) != len(references) or set(references) != {row["context_unit_id"] for row in assigned}:
            raise PacketBudgetError("participant source references disagree with allocation")
        before[pid] = segment_tokens(assigned)
        canonical = [row for row in assigned if row.get("hidden_contains_rui_ids")]
        if segment_tokens(canonical) > token_budget:
            raise PacketBudgetError(f"{pid}: complete canonical evidence exceeds {token_budget} tokens; no PIU truncation allowed")
        retained = {row["context_unit_id"] for row in canonical}
        costs = {row["context_unit_id"]: count_tokens(str(row["text"]).strip()) for row in assigned}
        used = segment_tokens(canonical)
        nonempty_count = sum(bool(str(row["text"]).strip()) for row in canonical)
        separator_cost = count_tokens("\n\n")
        for row in assigned:
            cid = row["context_unit_id"]
            if cid in retained:
                continue
            nonempty = bool(str(row["text"]).strip())
            proposed_cost = used + costs[cid] + (separator_cost if nonempty and nonempty_count else 0)
            if proposed_cost <= token_budget:
                retained.add(cid)
                used = proposed_cost
                nonempty_count += int(nonempty)
            else:
                exclusions.append({"participant_id": pid, "context_unit_id": cid,
                    "reason": "complete_noncanonical_context_exceeds_fixed_initial_local_cap",
                    "context_tokens": count_tokens(str(row["text"]).strip()),
                    "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
                    "source_ref": deepcopy(row.get("source_ref", {}))})
        chosen[pid] = retained
    kept = []
    for row in contexts:
        owners = [pid for pid in row["assigned_to"] if row["context_unit_id"] in chosen[pid]]
        if owners:
            row["assigned_to"] = owners
            kept.append(row)
    datapoint["private_context_units"] = kept
    for participant in datapoint["participants"]:
        pid = participant["participant_id"]
        participant["private_context_unit_ids"] = [row["context_unit_id"] for row in kept if pid in row["assigned_to"]]
        participant["private_word_count"] = sum(len(str(row["text"]).split()) for row in kept if pid in row["assigned_to"])
    datapoint.setdefault("generation_metadata", {})["source_packet_materialization"] = {
        "schema": "finlongdocqa_complete_source_cap_v1", "context_token_budget": token_budget,
        "tokenizer": "cl100k_base", "policy": "reserve_complete_canonical_blocks_then_original_order_complete_contexts",
        "original_balanced_allocation_preserved": True, "source_contexts_redistributed": False,
        "original_segment_tokens": before, "excluded_contexts": exclusions,
        "original_context_count": len(contexts), "retained_context_count": len(kept)}
    certify_materialized_views(datapoint, token_budget=token_budget)
    return datapoint


def certify_materialized_views(datapoint, *, token_budget=PAPER_CONTEXT_BUDGET):
    """Verify exact actor-visible source at this budget, without truncation."""
    from teamgym.interaction.run_single_agent_rui_evaluation import _budgeted_all_rui_context
    canonical_all = [{"text": row["text"]} for row in datapoint["reference_information_units"]]
    expected_all = "\n\n".join(str(row["text"]).strip() for row in canonical_all)
    if segment_tokens(canonical_all) > token_budget or _budgeted_all_rui_context(datapoint, token_budget) != expected_all:
        raise PacketBudgetError("Single-Agent All-PIU complete evidence exceeds the common context budget; no PIU truncation allowed")
    receipts = {}
    for participant in datapoint["participants"]:
        pid = participant["participant_id"]
        expected = complete_private_text(datapoint, participant)
        selected = [row for row in datapoint["private_context_units"] if row["context_unit_id"] in participant["private_context_unit_ids"]]
        actual = _private_context_payload(datapoint, participant, BudgetSpec(participant_context_token_budget=token_budget))
        if segment_tokens(selected) > token_budget or count_tokens(actual) > token_budget or actual != expected:
            raise PacketBudgetError(f"{pid}: exact complete source differs from budgeted actual consumer input")
        for unit in datapoint["reference_information_units"]:
            if unit["rui_id"] in participant["owned_rui_ids"] and not any(str(row["text"]).strip() == str(unit["text"]).strip()
                    and unit["rui_id"] in row.get("hidden_contains_rui_ids", []) for row in selected):
                raise PacketBudgetError(f"{pid}: canonical PIU source was lost")
        receipts[pid] = {"context_token_budget": token_budget, "private_tokens": count_tokens(actual),
            "segmented_tokens": segment_tokens(selected), "private_source_sha256": digest(actual),
            "public_input_sha256": digest(_public_task_payload(datapoint)),
            "context_unit_ids": list(participant["private_context_unit_ids"]), "exact_consumer_source_preserved": True}
    metadata = datapoint.setdefault("generation_metadata", {}).setdefault("source_packet_materialization", {})
    if metadata.get("context_token_budget", token_budget) != token_budget:
        raise PacketBudgetError("consumer budget differs from materialization budget")
    metadata.update({"context_token_budget": token_budget, "final_retained_context_count": len(datapoint["private_context_units"]),
                     "participants": receipts, "all_piu_tokens": count_tokens(expected_all), "all_piu_source_sha256": digest(expected_all)})
    return receipts
