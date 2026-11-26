"""Render Medical allocation packets and the exact common consumer export."""
from __future__ import annotations

from copy import deepcopy
import random

from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES, assign_reference_units, validate_reference_allocation
from teamgym.tokenization import count_tokens
from .allocation_source import content_hash
from .certification import positive_token_budget
from teamgym.core.schema import build_common_datapoint


class AllocationNotApplicable(ValueError):
    """The requested distribution is not admissible for this fixed PIU set."""


def render_contexts(source, context_ids):
    allowed = set(context_ids)
    rows = sorted((r for r in source["contexts"] if r["context_id"] in allowed),
                  key=lambda r: (r["document_order"], r["source_order"]))
    parts, previous = [], None
    for row in rows:
        if source["dataset"] == "mlcr" and row["source_file"] != previous:
            parts.append(f"Source file: {row['source_file']}")
            previous = row["source_file"]
        parts.append(row["text"])
    return "\n\n".join(parts).strip()


def common_information(source, partition):
    units = {u["unit_id"]: u for u in source["units"]}
    if not partition["allocation"]["common_rui_ids"]:
        return {}
    return {"common_reference_units": [
        {"rui_id": uid, "text": render_contexts(source, units[uid]["context_ids"])}
        for uid in partition["allocation"]["common_rui_ids"]]}


def visible_context_ids(partition, participant):
    return list(dict.fromkeys([*partition["common_context_ids"], *participant["context_ids"]]))


def packet_tokens(source, partition, participant):
    common = "\n\n".join(r["text"] for r in common_information(source, partition).get("common_reference_units", []))
    private = render_contexts(source, participant["context_ids"])
    return count_tokens("\n\n".join(p for p in (common, private) if p))


def build_partition(source, *, mode, seed, token_budget):
    positive_token_budget(token_budget)
    units = {u["unit_id"]: u for u in source["units"]}
    contexts = {r["context_id"]: r for r in source["contexts"]}
    if len(units) != len(source["units"]) or len(contexts) != len(source["contexts"]):
        raise ValueError("PIU and source context IDs must be unique")
    canonical = set()
    for unit in units.values():
        ids = set(unit["context_ids"])
        if not ids or not ids <= contexts.keys() or canonical & ids:
            raise ValueError("PIUs must reference distinct nonempty exact source contexts")
        canonical.update(ids)
    try:
        allocation = assign_reference_units(units, mode=mode, rng=random.Random(seed),
                                            smallest_sufficient_sets=[list(units)], allowed_modes=PAPER_ALLOCATION_MODES)
    except ValueError as exc:
        if "No eligible allocation mode" in str(exc):
            raise AllocationNotApplicable(str(exc)) from exc
        raise
    if not allocation.validation["valid"]:
        raise AllocationNotApplicable(f"{mode}: no admissible collaborative allocation for {len(units)} PIUs")
    common_ids = list(dict.fromkeys(c for uid in allocation.common_rui_ids for c in units[uid]["context_ids"]))
    result = {"allocation": allocation.to_metadata(), "seed": seed, "common_context_ids": common_ids,
              "participant_token_budget": token_budget, "excluded_context_ids": [],
              "participants": [{"participant_id": pid, "owned_unit_ids": list(owned),
                  "context_ids": list(dict.fromkeys(c for uid in owned for c in units[uid]["context_ids"]))}
                  for pid, owned in allocation.private_assignment.items()]}
    for participant in result["participants"]:
        if packet_tokens(source, result, participant) > token_budget:
            raise ValueError("Required private/common PIU evidence exceeds participant token budget")
    filler = [r["context_id"] for r in source["contexts"] if r["context_id"] not in canonical]
    random.Random(seed).shuffle(filler)
    for context_id in filler:
        choices = []
        for p in result["participants"]:
            candidate = {**p, "context_ids": [*p["context_ids"], context_id]}
            tokens = packet_tokens(source, result, candidate)
            if tokens <= token_budget:
                choices.append((tokens, p["participant_id"], p))
        if choices:
            min(choices, key=lambda r: (r[0], r[1]))[2]["context_ids"].append(context_id)
        else:
            result["excluded_context_ids"].append(context_id)
    order = {r["context_id"]: (r["document_order"], r["source_order"]) for r in source["contexts"]}
    for p in result["participants"]:
        p["context_ids"].sort(key=order.get)
    validate_partition(source, result)
    return result


def validate_partition(source, partition):
    units = {u["unit_id"]: u for u in source["units"]}
    contexts = {r["context_id"] for r in source["contexts"]}
    allocation = partition["allocation"]
    expected = assign_reference_units(units, mode=allocation["requested_mode"],
        rng=random.Random(partition["seed"]), smallest_sufficient_sets=[list(units)], allowed_modes=PAPER_ALLOCATION_MODES)
    if expected.to_metadata() != allocation:
        raise ValueError("Allocation metadata does not match the requested distribution and seed")
    participants = partition["participants"]
    assignment = {p["participant_id"]: p["owned_unit_ids"] for p in participants}
    common = set(allocation["common_rui_ids"])
    if len(assignment) != len(participants) or assignment != allocation["private_assignment"]:
        raise ValueError("Participant ownership differs from the requested allocation")
    check = validate_reference_allocation(all_rui_ids=units, private_assignment=assignment,
        common_rui_ids=common, smallest_sufficient_sets=[list(units)])
    if not check["valid"]:
        raise ValueError("Invalid collaborative private/common allocation")
    expected_common = {c for uid in common for c in units[uid]["context_ids"]}
    if set(partition["common_context_ids"]) != expected_common:
        raise ValueError("Common source packet differs from its intended PIUs")
    canonical = {c for u in units.values() for c in u["context_ids"]}
    filler_seen = set()
    included = set(expected_common)
    for p in participants:
        ids = p["context_ids"]
        if len(ids) != len(set(ids)) or not set(ids) <= contexts or set(ids) & expected_common:
            raise ValueError("Invalid, repeated or common context in private packet")
        visible = set(visible_context_ids(partition, p))
        actual = {uid for uid, u in units.items() if set(u["context_ids"]) <= visible}
        if actual != set(p["owned_unit_ids"]) | common:
            raise ValueError("Canonical source visibility disagrees with intended PIU allocation")
        filler = set(ids) - canonical
        if filler & filler_seen:
            raise ValueError("Non-PIU filler cannot be silently duplicated")
        filler_seen.update(filler); included.update(ids)
        if packet_tokens(source, partition, p) > positive_token_budget(partition["participant_token_budget"]):
            raise ValueError("Participant private/common material exceeds audited token budget")
    excluded = set(partition["excluded_context_ids"])
    if excluded & included or excluded | included != contexts:
        raise ValueError("Allocation does not account for the complete source corpus")


def material_snapshot(datapoint, *, token_budget):
    """Read the real consumer's common and private model-visible evidence."""
    from teamgym.interaction.llm_actors import _private_context_payload, _public_task_payload
    from teamgym.interaction.types import BudgetSpec
    common = _public_task_payload(datapoint)["initial_common_information"]
    return {p["participant_id"]: {"initial_common_information": deepcopy(common),
        "private_source_material": _private_context_payload(datapoint, p,
            BudgetSpec(participant_context_token_budget=token_budget))}
        for p in datapoint["participants"]}


def build_datapoint(source, partition, *, execution_identity=None):
    validate_partition(source, partition)
    task = source["task"]; dataset = source["dataset"]
    units = {u["unit_id"]: u for u in source["units"]}
    candidate_to_unit = {u["candidate_id"]: uid for uid, u in units.items()}
    claims = task["rubric"]["required_claims"]
    steps = [{"step": i, "claim_id": c["claim_id"], "claim": c.get("criterion", c.get("description", "")),
              "expected_answer_element": c.get("expected_answer_element", c.get("criterion", "")),
              "grading_rule": c.get("grading_rule", c.get("support_logic", "")), "required": True,
              "uses": [candidate_to_unit[cid] for cid in c["supporting_candidate_ids"]]}
             for i, c in enumerate(claims, 1)]
    participants, private = [], []
    for p in partition["participants"]:
        text = render_contexts(source, p["context_ids"])
        cid = f"{p['participant_id']}:source_packet"
        ids = [cid] if text else []
        if text:
            private.append({"context_unit_id": cid, "text": text, "assigned_to": [p["participant_id"]],
                "word_count": len(text.split()), "hidden_contains_rui_ids": p["owned_unit_ids"],
                "source_ref": {"context_ids": p["context_ids"]}})
        participants.append({"participant_id": p["participant_id"], "role_name": "medical evidence reviewer",
            "public_role": "medical evidence reviewer", "owned_rui_ids": p["owned_unit_ids"],
            "private_context_unit_ids": ids, "private_token_count": count_tokens(text),
            "private_word_count": len(text.split())})
    public = task["public_task"]
    note = public.get("response_format_note") or " ".join(public.get("response_requirements") or [])
    answer = task.get("canonical_answer") if dataset == "evidencebench" else task["oracle"]["reference_answer"]
    rubric = deepcopy(task["rubric"])
    errors = list(rubric.get("prohibited_errors") or []) + list(rubric.get("disqualifying_errors") or [])
    mode = partition["allocation"]["mode"]
    row = build_common_datapoint(
        instance_id=f"{source['instance_id']}__{mode}__{content_hash({'task': task, 'partition': partition})[:12]}",
        dataset="evidencebench" if dataset == "evidencebench" else "mlcr_aa",
        source={"lineage": source["source_lineage"], "allocation_source_sha256": content_hash(source)},
        scenario_family=f"{dataset}_medical_free_response",
        task_description={"answer_type": "short_answer_judged", **public, "objective": public["question"],
            "response_format_note": note, "answer_format": {"type": "free_response", "instruction": note}},
        decision_set=["free_response"], decision_option_definitions={"free_response": note or "Answer the medical question."},
        initial_common_information=common_information(source, partition), participants=participants,
        private_context_units=private,
        reference_information_units=[{"rui_id": uid, "text": u["text"], "summary": "",
            "supports_claim_ids": u["supports_claim_ids"], "source_ref": u["source_ref"]} for uid, u in units.items()],
        oracle={"answer": "free_response", "canonical_response": answer, "short_answer": answer,
            "judge_rubric": {**rubric, "require_all_steps_for_pass": True},
            "required_decision_steps": steps, "reasoning_steps": steps, "disqualifying_errors": errors,
            "pass_policy": "all_required_decision_steps", "semantic_judge": {"enabled": True, "primary_answer_field": "written_answer"},
            "smallest_sufficient_rui_sets": [{"rui_ids": list(units)}]},
        generation_metadata={"method": "medical_frozen_task_allocation_v1", "allocation_mode": mode,
            "private_information_allocation": partition["allocation"], "producer_identity": source["producer_identity"],
            "execution_identity": execution_identity},
        dataset_specific={"allocation_mode": mode, "full_context_materialized_at_construction": True,
            "full_context_redistribution_allowed": False,
            "audited_participant_context_token_budget": partition["participant_token_budget"]})
    if not row["format_validation"]["valid"]:
        raise ValueError(f"Invalid Medical allocation export: {row['format_validation']}")
    actual = material_snapshot(row, token_budget=partition["participant_token_budget"])
    for p in partition["participants"]:
        if actual[p["participant_id"]]["private_source_material"] != render_contexts(source, p["context_ids"]):
            raise ValueError("Actual consumer changed or truncated the allocated source packet")
    return row
