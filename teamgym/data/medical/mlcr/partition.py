"""Token-budgeted RIU-only/full-context allocation and leakage repair for MLCR."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from teamgym.tokenization import TOKENIZER, count_tokens

from .construction import string_list
from .llm import ResponseCache, structured_call


def _contexts(case_index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_order, document in enumerate(case_index.get("documents") or []):
        for atom in document.get("atoms") or []:
            rows.append(
                {
                    "context_id": str(atom["atom_id"]),
                    "source_document_id": str(document["document_id"]),
                    "source_file": str(document["summary_file"]),
                    "document_order": document_order,
                    "source_order": int(atom["atom_index"]),
                    "text": str(atom["text"]),
                    "hidden_canonical_unit_ids": [],
                }
            )
    return rows


def render_full_context_packet(contexts: list[dict[str, Any]]) -> str:
    """Render participant-visible source without construction IDs or gold marks."""

    ordered = sorted(contexts, key=lambda row: (row["document_order"], row["source_order"]))
    parts: list[str] = []
    current_file: str | None = None
    for row in ordered:
        source_file = str(row["source_file"])
        if source_file != current_file:
            parts.append(f"Source file: {source_file}")
            current_file = source_file
        parts.append(str(row["text"]))
    return "\n\n".join(parts)


def refresh_partition_coverage(partition: dict[str, Any]) -> None:
    """Recount the actual post-sanitization union, including excluded spans."""

    contexts = {str(row["context_id"]): row for row in partition["contexts"]}
    included = {
        str(context_id)
        for participant in partition["participants"]
        for context_id in participant["context_ids"]
    }
    if not included <= set(contexts):
        raise ValueError("participant references an unknown source context")
    total = sum(count_tokens(row["text"]) for row in contexts.values())
    retained = sum(count_tokens(contexts[value]["text"]) for value in included)
    partition["included_context_count"] = len(included)
    partition["included_tokens_sum_of_atoms"] = retained
    partition["excluded_context_ids"] = sorted(set(contexts) - included)
    partition["token_coverage"] = retained / total if total else 1.0


def build_token_budgeted_partition(
    *,
    case_index: dict[str, Any],
    evidence_units: list[dict[str, Any]],
    participant_token_budget: int,
    ownership_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    if participant_token_budget <= 0:
        raise ValueError("participant_token_budget must be positive")
    contexts = _contexts(case_index)
    by_id = {row["context_id"]: row for row in contexts}
    participants: list[dict[str, Any]] = []
    canonical_owner: dict[str, str] = {}
    units_by_id = {str(row["unit_id"]): row for row in evidence_units}
    groups = ownership_groups or [[unit_id] for unit_id in units_by_id]
    flattened = [unit_id for group in groups for unit_id in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(units_by_id):
        raise ValueError("ownership groups must partition every RIU exactly once")
    if len(groups) < 2:
        raise ValueError("collaborative allocation requires at least two participants")
    for position, group in enumerate(groups, start=1):
        participant_id = f"participant_{position:02d}"
        atom_ids: list[str] = []
        for unit_id in group:
            unit_atoms = string_list(units_by_id[unit_id].get("atom_ids"))
            if not unit_atoms or not set(unit_atoms) <= set(by_id):
                raise ValueError(f"{unit_id}: missing or unknown canonical atom IDs")
            for atom_id in unit_atoms:
                if atom_id in canonical_owner:
                    raise ValueError(f"canonical atom belongs to multiple RIUs: {atom_id}")
                canonical_owner[atom_id] = participant_id
                by_id[atom_id]["hidden_canonical_unit_ids"].append(unit_id)
            atom_ids.extend(unit_atoms)
        participants.append(
            {
                "participant_id": participant_id,
                "hidden_owned_unit_ids": list(group),
                "hidden_canonical_context_ids": atom_ids,
                "context_ids": list(atom_ids),
            }
        )

    def packet_rows(participant: dict[str, Any]) -> list[dict[str, Any]]:
        return [by_id[value] for value in participant["context_ids"]]

    for participant in participants:
        tokens = count_tokens(render_full_context_packet(packet_rows(participant)))
        if tokens > participant_token_budget:
            raise ValueError(
                f"{participant['participant_id']}: canonical RIU context alone exceeds "
                f"token budget ({tokens} > {participant_token_budget})"
            )

    remaining = [row for row in contexts if row["context_id"] not in canonical_owner]
    excluded: list[str] = []
    for context in remaining:
        viable: list[tuple[int, str, dict[str, Any]]] = []
        for participant in participants:
            candidate_rows = packet_rows(participant) + [context]
            tokens = count_tokens(render_full_context_packet(candidate_rows))
            if tokens <= participant_token_budget:
                viable.append((tokens, str(participant["participant_id"]), participant))
        if not viable:
            excluded.append(str(context["context_id"]))
            continue
        _, _, owner = min(viable, key=lambda item: (item[0], item[1]))
        owner["context_ids"].append(str(context["context_id"]))

    assigned = [value for participant in participants for value in participant["context_ids"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("full-context allocation is not disjoint")
    included = set(assigned)
    if included | set(excluded) != set(by_id):
        raise ValueError("full-context allocation did not account for the entire corpus")
    for participant in participants:
        rows = packet_rows(participant)
        rendered = render_full_context_packet(rows)
        participant["context_ids"].sort(
            key=lambda value: (by_id[value]["document_order"], by_id[value]["source_order"])
        )
        participant["context_tokens"] = count_tokens(rendered)
        participant["private_context"] = rendered
    total_source_tokens = sum(count_tokens(row["text"]) for row in contexts)
    result = {
        "schema": "mlcr_full_context_partition_v1",
        "tokenizer": TOKENIZER,
        "participant_token_budget": participant_token_budget,
        "participants": participants,
        "contexts": contexts,
        "excluded_context_ids": excluded,
        "source_context_count": len(contexts),
        "included_context_count": len(included),
        "source_tokens_sum_of_atoms": total_source_tokens,
        "included_tokens_sum_of_atoms": 0,
        "token_coverage": 0.0,
    }
    refresh_partition_coverage(result)
    return result


def validate_leakage_audit(
    value: Any,
    *,
    foreign_unit_ids: set[str],
    packet_context_ids: set[str],
    canonical_context_ids: set[str],
) -> list[str]:
    if not isinstance(value, dict):
        return ["leakage audit must be an object"]
    errors: list[str] = []
    audits = value.get("foreign_riu_audits")
    if not isinstance(audits, list):
        return ["foreign_riu_audits must be a list"]
    by_id = {str(row.get("foreign_unit_id") or ""): row for row in audits if isinstance(row, dict)}
    if len(audits) != len(foreign_unit_ids) or set(by_id) != foreign_unit_ids:
        errors.append("foreign RIU audits must cover every foreign unit exactly once")
    for unit_id, audit in by_id.items():
        if not isinstance(audit.get("recoverable"), bool):
            errors.append(f"{unit_id}: recoverable must be boolean")
        if not isinstance(audit.get("canonical_overlap"), bool):
            errors.append(f"{unit_id}: canonical_overlap must be boolean")
        removable = set(string_list(audit.get("removable_context_ids")))
        if not removable <= packet_context_ids:
            errors.append(f"{unit_id}: removable IDs are outside the packet")
        if removable & canonical_context_ids:
            errors.append(f"{unit_id}: canonical contexts cannot be removed")
        if audit.get("canonical_overlap") is True and audit.get("recoverable") is not True:
            errors.append(f"{unit_id}: canonical overlap implies recoverability")
        if audit.get("recoverable") is False and removable:
            errors.append(f"{unit_id}: non-recoverable unit cannot request removals")
        if audit.get("recoverable") is True and not removable and audit.get("canonical_overlap") is not True:
            errors.append(f"{unit_id}: recoverable leakage needs removable context or canonical_overlap")
    if not isinstance(value.get("passed"), bool):
        errors.append("passed must be boolean")
    if value.get("passed") is not (not any(row.get("recoverable") is True for row in by_id.values())):
        errors.append("passed contradicts per-unit recoverability")
    if not isinstance(value.get("issues"), list):
        errors.append("issues must be a list")
    elif value["issues"] and not any(row.get("recoverable") is True for row in by_id.values()):
        errors.append("reported issues have no corresponding recoverable unit")
    return list(dict.fromkeys(errors))


def audit_and_sanitize_partition(
    *,
    case_id: str,
    public_task: dict[str, Any],
    evidence_units: list[dict[str, Any]],
    partition: dict[str, Any],
    client: Any,
    cache: ResponseCache,
    trace_root: Path,
    attempts: int,
    max_rounds: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = deepcopy(partition)
    contexts = {str(row["context_id"]): row for row in current["contexts"]}
    units = {str(row["unit_id"]): row for row in evidence_units}
    rounds: list[dict[str, Any]] = []
    removed_total: set[str] = set()
    for round_number in range(1, max_rounds + 2):
        round_rows: list[dict[str, Any]] = []
        removals: set[str] = set()
        errors: list[str] = []
        any_leakage = False
        ownership_merge_pairs: set[tuple[str, str]] = set()
        for participant in current["participants"]:
            owned = set(string_list(participant["hidden_owned_unit_ids"]))
            foreign = set(units) - owned
            packet_ids = set(string_list(participant["context_ids"]))
            canonical_ids = set(string_list(participant["hidden_canonical_context_ids"]))
            packet = [contexts[value] for value in participant["context_ids"] if value in contexts]
            response, response_errors = structured_call(
                client=client,
                cache=cache,
                trace_root=trace_root,
                case_id=case_id,
                stage=f"packet_leakage.round_{round_number}.{participant['participant_id']}",
                user_prompt_name="06_packet_leakage_audit_user.txt",
                variables={
                    "PACKET_JSON": [
                        {"context_id": row["context_id"], "source_file": row["source_file"], "text": row["text"]}
                        for row in packet
                    ],
                    "OWN_RIU_JSON": [units[value] for value in sorted(owned)],
                    "FOREIGN_RIUS_JSON": [units[value] for value in sorted(foreign)],
                },
                validator=lambda row, f=foreign, p=packet_ids, c=canonical_ids: validate_leakage_audit(
                    row, foreign_unit_ids=f, packet_context_ids=p, canonical_context_ids=c
                ),
                attempts=attempts,
            )
            response_removals: set[str] = set()
            for audit in response.get("foreign_riu_audits") or []:
                if audit.get("recoverable") is not True:
                    continue
                any_leakage = True
                if audit.get("canonical_overlap") is True:
                    foreign_unit = str(audit.get("foreign_unit_id") or "")
                    ownership_merge_pairs.update(
                        tuple(sorted((owned_unit, foreign_unit)))
                        for owned_unit in owned
                        if foreign_unit
                    )
                response_removals.update(string_list(audit.get("removable_context_ids")))
            removals.update(response_removals)
            errors.extend(response_errors)
            round_rows.append({
                "participant_id": participant["participant_id"],
                "response": response,
                "errors": response_errors,
            })
        rounds.append({
            "round": round_number,
            "participant_audits": round_rows,
            "removed_context_ids": sorted(removals),
            "ownership_merge_pairs": [list(pair) for pair in sorted(ownership_merge_pairs)],
            "errors": errors,
        })
        if errors:
            return current, {"passed": False, "rounds": rounds, "removed_context_ids": sorted(removed_total), "errors": errors}
        if ownership_merge_pairs:
            return current, {
                "passed": False,
                "rounds": rounds,
                "removed_context_ids": sorted(removed_total),
                "ownership_merge_pairs": [list(pair) for pair in sorted(ownership_merge_pairs)],
                "errors": [],
            }
        if not any_leakage:
            return current, {"passed": True, "rounds": rounds, "removed_context_ids": sorted(removed_total), "errors": []}
        if not removals:
            return current, {"passed": False, "rounds": rounds, "removed_context_ids": sorted(removed_total), "errors": ["leakage found without removable noncanonical context"]}
        if round_number > max_rounds:
            break
        for participant in current["participants"]:
            participant["context_ids"] = [value for value in participant["context_ids"] if value not in removals]
            packet = [contexts[value] for value in participant["context_ids"]]
            participant["private_context"] = render_full_context_packet(packet)
            participant["context_tokens"] = count_tokens(participant["private_context"])
        removed_total.update(removals)
        refresh_partition_coverage(current)
    return current, {"passed": False, "rounds": rounds, "removed_context_ids": sorted(removed_total), "errors": ["leakage audit did not converge"]}


def materialize_modes(*, task: dict[str, Any], partition: dict[str, Any]) -> dict[str, Any]:
    units = list(task["evidence_units"])
    unit_by_id = {str(row["unit_id"]): row for row in units}
    return {
        "public_task": task["public_task"],
        "single_agent_all_rius": {
            "private_material": [{"material_type": "clinical_excerpt", "text": row["text"]} for row in units]
        },
        "collaborative_riu_only": {
            "participants": [
                {
                    "participant_id": participant["participant_id"],
                    "private_material": [
                        {"material_type": "clinical_excerpt", "text": unit_by_id[unit_id]["text"]}
                        for unit_id in participant["hidden_owned_unit_ids"]
                    ],
                }
                for participant in partition["participants"]
            ]
        },
        "collaborative_full_context": {
            "participants": [
                {"participant_id": row["participant_id"], "private_context": row["private_context"]}
                for row in partition["participants"]
            ]
        },
    }


def merge_ownership_groups(
    unit_ids: list[str],
    pairs: list[list[str]],
) -> list[list[str]]:
    """Return deterministic connected components induced by canonical leakage."""

    parent = {unit_id: unit_id for unit_id in unit_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for pair in pairs:
        if len(pair) != 2 or not set(pair) <= set(parent):
            raise ValueError(f"invalid ownership merge pair: {pair}")
        left, right = find(pair[0]), find(pair[1])
        if left != right:
            parent[max(left, right)] = min(left, right)
    components: dict[str, list[str]] = {}
    for unit_id in unit_ids:
        components.setdefault(find(unit_id), []).append(unit_id)
    return sorted((sorted(group) for group in components.values()), key=lambda group: group[0])


def validate_collaborative_ownership(task: dict[str, Any], partition: dict[str, Any]) -> list[str]:
    """Reject any owner whose RIUs structurally cover the entire answer."""

    errors: list[str] = []
    by_candidate = {
        str(unit["candidate_id"]): str(unit["unit_id"])
        for unit in task["evidence_units"]
    }
    required = [
        {by_candidate[str(candidate_id)] for candidate_id in claim["supporting_candidate_ids"]}
        for claim in task["rubric"]["required_claims"]
    ]
    for participant in partition["participants"]:
        owned = set(participant["hidden_owned_unit_ids"])
        if required and all(supporters <= owned for supporters in required):
            errors.append(f"{participant['participant_id']}: owns all required claim supports")
    return errors
