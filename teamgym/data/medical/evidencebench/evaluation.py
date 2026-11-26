"""Materialize accepted EvidenceBench artifacts for three-mode evaluation."""
from __future__ import annotations
from teamgym.data.medical.certification import (
    validate_export_certification, require_passed_audit, require_participant_audits,
    require_boundary_cases, positive_token_budget,
)
from .execution import export_artifacts

import json
import hashlib
from pathlib import Path
from typing import Any

from teamgym.core.release import write_release_datapoints
from teamgym.core.schema import build_common_datapoint


def consumer_full_context_materials(partition: dict, *, token_budget: int = 40000) -> dict[str, str]:
    from teamgym.interaction.llm_actors import _private_context_payload
    from teamgym.interaction.types import BudgetSpec
    from teamgym.tokenization import count_tokens
    contexts = {r['context_id']: r for r in partition['contexts']}
    owners = {
        context_id: participant['participant_id']
        for participant in partition['participants']
        for context_id in participant['context_ids']
    }
    data = {'private_context_units': [{'context_unit_id': c, 'text': r['text'],
             'assigned_to': [owners[c]],
             'hidden_contains_rui_ids': r.get('hidden_contains_riu_ids', [])} for c, r in contexts.items()]}
    materials = {}
    for participant in partition['participants']:
        expected = '\n\n'.join(str(contexts[c]['text']).strip() for c in participant['context_ids'])
        actual = _private_context_payload(data, {'participant_id': participant['participant_id'],
                                                  'private_context_unit_ids': participant['context_ids']},
                                         BudgetSpec(participant_context_token_budget=token_budget))
        if count_tokens(actual) > token_budget or actual != expected:
            raise ValueError('EvidenceBench actual consumer text differs from audited packet or exceeds its token budget')
        materials[participant['participant_id']] = actual
    return materials


def build_evaluation_datapoint(instance_dir: str | Path) -> dict[str, Any]:
    """Build one common datapoint without changing the audited packet allocation."""

    root = Path(instance_dir).expanduser().resolve()
    summary = _read_json(root / "summary.json")
    if summary.get("status") != "accepted":
        raise ValueError(f"EvidenceBench instance is not accepted: {root}")
    validate_export_certification(summary, export_artifacts(root))
    task = _read_json(root / "02_task_rubric.json")
    rius = _read_json(root / "05_final_rius.hidden.json")
    partition = _read_json(root / "06_participant_packets.hidden.json")
    from .validation import validate_task_audit
    audit_errors = validate_task_audit(_read_json(root / "03_task_audit.json"), task_rubric=task)
    if audit_errors:
        raise ValueError(f"EvidenceBench export task audit failed: {audit_errors}")
    require_boundary_cases(_read_json(root / "04_boundaries.json"), [r["candidate_id"] for r in rius])
    require_passed_audit(_read_json(root / "07_leakage_audit.json"), "packet leakage audit")
    token_budget = positive_token_budget(partition.get("participant_token_budget", 40_000))
    if summary.get("execution_identity", {}).get("participant_context_token_budget") != token_budget:
        raise ValueError("EvidenceBench export budget differs from its execution identity")
    actual_materials = consumer_full_context_materials(partition, token_budget=token_budget)
    audit_path = root / '09_participant_final_answer_audit.json'
    participant_audit = _read_json(audit_path)
    require_participant_audits(participant_audit, partition["participants"], token_budget=token_budget)
    for result in participant_audit['participants']:
        audited = '\n\n'.join(r['text'] for r in result['display_material'])
        if actual_materials[result['participant_id']] != audited:
            raise ValueError('EvidenceBench export differs from its actually audited consumer text')
    if partition.get("allocation_policy") != "one_each_disjoint_full_paper":
        raise ValueError(f"unsupported allocation policy: {partition.get('allocation_policy')}")

    context_by_id = {
        str(row["context_id"]): row for row in partition.get("contexts") or []
    }
    owner_by_context: dict[str, str] = {}
    participants: list[dict[str, Any]] = []
    for packet in partition.get("participants") or []:
        participant_id = str(packet["participant_id"])
        packet_ids = [str(value) for value in packet.get("context_ids") or []]
        for context_id in packet_ids:
            if context_id not in context_by_id:
                raise ValueError(f"{participant_id} references unknown context {context_id}")
            if context_id in owner_by_context:
                raise ValueError(f"context {context_id} has multiple packet owners")
            owner_by_context[context_id] = participant_id
        participants.append(
            {
                "participant_id": participant_id,
                "role_name": "medical evidence reviewer",
                "public_role": "medical evidence reviewer",
                "owned_rui_ids": [str(value) for value in packet.get("hidden_owned_riu_ids") or []],
                "private_context_unit_ids": packet_ids,
                "private_token_count": int(packet.get("context_tokens") or 0),
                "private_word_count": sum(
                    len(str(context_by_id[value].get("text") or "").split())
                    for value in packet_ids
                ),
            }
        )
    if set(owner_by_context) != set(context_by_id):
        missing = sorted(set(context_by_id) - set(owner_by_context))
        raise ValueError(f"disjoint packet partition is not exhaustive: {missing}")

    private_context_units = [
        {
            "context_unit_id": context_id,
            "text": str(row.get("text") or ""),
            "word_count": len(str(row.get("text") or "").split()),
            "assigned_to": [owner_by_context[context_id]],
            "hidden_contains_rui_ids": [
                str(value) for value in row.get("hidden_contains_riu_ids") or []
            ],
            "source_ref": {
                **dict(row.get("source_ref") or {}),
                "sentence_index": row.get("sentence_index"),
            },
        }
        for context_id, row in sorted(
            context_by_id.items(),
            key=lambda item: (int(item[1].get("sentence_index") or 0), item[0]),
        )
    ]
    reference_information_units = [
        {
            "rui_id": str(row["riu_id"]),
            "text": "\n".join(
                str(atom.get("text") or "") for atom in row.get("evidence_atoms") or []
            ),
            "summary": "",
            "supports_claim_ids": [
                str(value) for value in row.get("required_claim_ids") or []
            ],
            "source_ref": {
                "evidencebench_instance_id": summary["instance_id"],
                "sentence_indices": sorted(
                    {
                        int(value)
                        for atom in row.get("evidence_atoms") or []
                        for value in atom.get("sentence_indices") or []
                    }
                ),
            },
        }
        for row in rius
    ]
    candidate_to_riu = {
        str(row["candidate_id"]): str(row["riu_id"]) for row in rius
    }
    required_claims = (task.get("rubric") or {}).get("required_claims") or []
    reasoning_steps = [
        {
            "step": index,
            "claim_id": str(row["claim_id"]),
            "claim": str(row.get("criterion") or ""),
            "uses": list(
                dict.fromkeys(
                    candidate_to_riu[str(candidate_id)]
                    for candidate_id in row.get("supporting_candidate_ids") or []
                    if str(candidate_id) in candidate_to_riu
                )
            ),
            "grading_rule": str(row.get("support_logic") or "ALL_REQUIRED"),
        }
        for index, row in enumerate(required_claims, 1)
    ]
    gold_ids = [str(row["riu_id"]) for row in rius]
    public = task.get("public_task") or {}
    datapoint = build_common_datapoint(
        instance_id=str(summary["instance_id"]),
        dataset="evidencebench",
        source={"artifact_dir": str(root), "lineage": summary.get("source_lineage")},
        scenario_family="evidencebench_medical_evidence_free_response",
        task_description={
            "answer_type": "short_answer_judged",
            "title": public.get("title"),
            "question": public.get("question"),
            "objective": public.get("question"),
            "response_format_note": public.get("response_format_note"),
            "answer_format": {
                "type": "free_response",
                "instruction": public.get("response_format_note"),
            },
        },
        decision_set=["free_response"],
        decision_option_definitions={
            "free_response": str(
                public.get("response_format_note")
                or "Return a complete evidence-grounded synthesis."
            )
        },
        participants=participants,
        private_context_units=private_context_units,
        reference_information_units=reference_information_units,
        oracle={
            "answer": "free_response",
            "short_answer": task.get("canonical_answer"),
            "canonical_response": task.get("canonical_answer"),
            "answer_spec": {
                "type": "composite",
                "items": [
                    {
                        "item_id": "evidence_set",
                        "type": "exact_set",
                        "values": gold_ids,
                    },
                    {
                        "item_id": "synthesis",
                        "type": "text",
                        "claim_ids": [row["claim_id"] for row in reasoning_steps],
                    },
                ],
            },
            "reasoning_steps": reasoning_steps,
            "required_decision_steps": reasoning_steps,
            "judge_rubric": {**(task.get("rubric") or {}), "require_all_steps_for_pass": True},
            "pass_policy": "all_required_decision_steps",
            "disqualifying_errors": (task.get("rubric") or {}).get(
                "disqualifying_errors", []
            ),
            "smallest_sufficient_rui_sets": [{"rui_ids": gold_ids}],
            "semantic_judge": {
                "enabled": True,
                "primary_answer_field": "written_answer",
            },
        },
        generation_metadata={
            "execution_identity": summary.get('execution_identity'),
            "export_grading_contract": {"schema": "medical_all_required_claims_v3",
                "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "producer_dependency_sha256": summary.get('execution_identity', {}).get('dependencies', {}).get('sha256')},
            "method": "evidencebench_independent_riu_pipeline_v1",
            "allocation_mode": "one_each",
            "allocation_policy": partition["allocation_policy"],
            "source_summary": summary,
        },
        dataset_specific={
            "audited_participant_context_token_budget": token_budget,
            "allocation_mode": "one_each",
            "full_context_materialized_at_construction": True,
            "full_context_redistribution_allowed": False,
        },
    )
    if not datapoint["format_validation"]["valid"]:
        raise ValueError(
            f"invalid EvidenceBench evaluation datapoint: {datapoint['format_validation']}"
        )
    return datapoint


def write_evaluation_jsonl(
    instance_dirs: list[str | Path], output_path: str | Path
) -> Path:
    """Write accepted instances as common-datapoint JSONL."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [build_evaluation_datapoint(path) for path in instance_dirs]
    write_release_datapoints(destination, rows)
    return destination


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
