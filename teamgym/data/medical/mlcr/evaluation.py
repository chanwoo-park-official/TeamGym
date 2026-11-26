"""Materialize accepted MLCR reports for the common three-mode evaluator."""
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


def private_context_units(partition: dict) -> list[dict]:
    """Preserve the audited neutral file headings inside model-visible text."""
    contexts = {r['context_id']: r for r in partition['contexts']}
    rendered = []
    for participant in partition['participants']:
        previous_file = None
        for context_id in sorted(participant['context_ids'], key=lambda c:
                (contexts[c]['document_order'], contexts[c]['source_order'])):
            row = contexts[context_id]
            text = str(row['text'])
            if row['source_file'] != previous_file:
                text = f"Source file: {row['source_file']}\n\n" + text
                previous_file = row['source_file']
            rendered.append({'context_unit_id': context_id, 'text': text,
                'word_count': len(text.split()), 'assigned_to': [participant['participant_id']],
                'hidden_contains_rui_ids': list(row.get('hidden_canonical_unit_ids') or []),
                'source_ref': {'document_id': row['source_document_id'], 'source_file': row['source_file'],
                               'document_order': row['document_order'], 'source_order': row['source_order'],
                               'raw_text_sha256': hashlib.sha256(str(row['text']).encode()).hexdigest(),
                               'neutral_file_header_preserved': True}})
    return rendered


def consumer_full_context_materials(partition: dict, *, token_budget: int = 40000) -> dict[str, str]:
    """Use the real shared renderer and require exact untruncated audited text."""
    from teamgym.interaction.llm_actors import _private_context_payload
    from teamgym.interaction.types import BudgetSpec
    from .partition import render_full_context_packet
    from teamgym.tokenization import count_tokens
    contexts = {r['context_id']: r for r in partition['contexts']}
    datapoint = {'private_context_units': private_context_units(partition)}
    result = {}
    for participant in partition['participants']:
        expected = render_full_context_packet([contexts[c] for c in participant['context_ids']])
        proxy = {'participant_id': participant['participant_id'],
                 'private_context_unit_ids': participant['context_ids']}
        actual = _private_context_payload(datapoint, proxy, BudgetSpec(participant_context_token_budget=token_budget))
        if count_tokens(actual) > token_budget or actual != expected or (participant.get('private_context') is not None and participant['private_context'] != expected):
            raise ValueError('MLCR actual consumer material differs from the audited grouped packet or exceeds its budget')
        result[participant['participant_id']] = actual
    return result


def build_evaluation_datapoint(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "accepted":
        raise ValueError(f"MLCR report is not accepted: {path}")
    validate_export_certification(report, export_artifacts(path.parent))
    for key in ("grounding_audit", "boundary_audits", "packet_leakage_audit"):
        require_passed_audit(report.get(key), key)

    task = report["task"]
    partition = report["partition"]
    units = task["evidence_units"]
    from .construction import validate_grounding_audit
    grounding_errors = validate_grounding_audit(report["grounding_audit"], task=task)
    if grounding_errors:
        raise ValueError(f"MLCR export grounding audit failed: {grounding_errors}")
    require_boundary_cases(report["boundary_audits"], [unit["candidate_id"] for unit in units])
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    contexts = {str(row["context_id"]): row for row in partition["contexts"]}
    owners: dict[str, str] = {}
    participants = []
    for packet in partition["participants"]:
        participant_id = str(packet["participant_id"])
        context_ids = [str(value) for value in packet["context_ids"]]
        for context_id in context_ids:
            if context_id not in contexts or context_id in owners:
                raise ValueError(f"unknown or multiply assigned context: {context_id}")
            owners[context_id] = participant_id
        participants.append({
            "participant_id": participant_id,
            "role_name": "medical record reviewer",
            "public_role": "medical record reviewer",
            "owned_rui_ids": list(packet["hidden_owned_unit_ids"]),
            "private_context_unit_ids": context_ids,
            "private_token_count": int(packet["context_tokens"]),
            "private_word_count": sum(len(contexts[c]["text"].split()) for c in context_ids),
        })
    if set(owners) != set(contexts) - set(partition["excluded_context_ids"]):
        raise ValueError("packet allocation disagrees with the audited context exclusions")
    if {uid for p in participants for uid in p["owned_rui_ids"]} != set(unit_by_id):
        raise ValueError("packet allocation does not cover every RIU")

    exported_contexts = private_context_units(partition)
    token_budget = positive_token_budget(partition.get('participant_token_budget', 40000))
    if report.get("execution_identity", {}).get("participant_context_token_budget") != token_budget:
        raise ValueError("MLCR export budget differs from its execution identity")
    require_participant_audits(report.get('participant_final_answer_audit'), partition['participants'], token_budget=token_budget)
    actual_materials = consumer_full_context_materials(partition, token_budget=token_budget)
    for result in report['participant_final_answer_audit']['participants']:
        audited = result['display_materials'][0]['text']
        if actual_materials[result['participant_id']] != audited:
            raise ValueError('MLCR export differs from its actually audited consumer text')
    for participant in participants:
        participant['private_word_count'] = len(actual_materials[participant['participant_id']].split())
    reference_information_units = [{
        "rui_id": str(unit["unit_id"]),
        "text": str(unit["text"]),
        "summary": "",
        "supports_claim_ids": list(unit["supports_claim_ids"]),
        "source_ref": {
            "document_id": unit["source_document_id"],
            "raw_start": unit["raw_start"],
            "raw_end": unit["raw_end"],
            "sha256": unit["sha256"],
        },
    } for unit in units]
    claims = task["rubric"]["required_claims"]
    candidate_to_unit = {str(unit['candidate_id']): str(unit['unit_id']) for unit in units}
    judge_claims = [{
        **claim,
        "claim_id": str(claim["claim_id"]),
        "criterion": f"{claim['description']} Expected: {claim['expected_answer_element']}",
        "grading_rule": str(claim["grading_rule"]),
    } for claim in claims]
    required_steps = [{
        'step': index, 'claim_id': str(claim['claim_id']), 'claim': str(claim['description']),
        'expected_answer_element': claim['expected_answer_element'],
        'grading_rule': claim['grading_rule'],
        'uses': [candidate_to_unit[str(c)] for c in claim.get('supporting_candidate_ids', [])],
        'required': True,
    } for index, claim in enumerate(claims, 1)]
    prohibited_errors = list(task['rubric'].get('prohibited_errors') or []) + list(task['rubric'].get('disqualifying_errors') or [])
    public = task["public_task"]
    response_note = " ".join(str(x) for x in public.get("response_requirements") or [])
    datapoint = build_common_datapoint(
        instance_id=report.get("instance_id") or "mlcr_" + hashlib.sha256(json.dumps(
            {"case_id": report["case_id"], "task": task}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24],
        dataset="mlcr_aa",
        source={"report_path": str(path), "case_id": report["case_id"], "lineage": report.get("source_lineage")},
        scenario_family="mlcr_medical_record_free_response",
        task_description={
            "answer_type": "short_answer_judged",
            "title": public.get("title"),
            "context": public.get("context"),
            "question": public.get("question"),
            "objective": public.get("question"),
            "response_format_note": response_note,
            "answer_format": {"type": "free_response", "instruction": response_note},
        },
        decision_set=["free_response"],
        decision_option_definitions={"free_response": response_note or "Answer the clinical question."},
        participants=participants,
        private_context_units=exported_contexts,
        reference_information_units=reference_information_units,
        oracle={
            "answer": "free_response",
            "canonical_response": task["oracle"]["reference_answer"],
            "judge_rubric": {**task['rubric'], "required_claims": judge_claims,
                             "disqualifying_errors": prohibited_errors,
                             "require_all_steps_for_pass": True},
            "required_decision_steps": required_steps,
            "reasoning_steps": required_steps,
            "disqualifying_errors": prohibited_errors,
            "pass_policy": "all_required_decision_steps",
            "smallest_sufficient_rui_sets": [{"rui_ids": list(unit_by_id)}],
            "semantic_judge": {"enabled": True, "primary_answer_field": "written_answer"},
        },
        generation_metadata={"method": "mlcr_accepted_riu_pipeline", "report_path": str(path),
                             "execution_identity": report.get('execution_identity'),
                             "export_grading_contract": {"schema": "medical_all_required_claims_v3",
                                 "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 "producer_dependency_sha256": report.get('execution_identity', {}).get('dependencies', {}).get('sha256')},
                             "ownership_policy": report.get("ownership_policy"),
                             "ownership_vector": [len(p["owned_rui_ids"]) for p in participants]},
        dataset_specific={"full_context_materialized_at_construction": True,
                          "audited_participant_context_token_budget": partition.get('participant_token_budget', 40000),
                          "full_context_redistribution_allowed": False},
    )
    if not datapoint["format_validation"]["valid"]:
        raise ValueError(f"invalid MLCR datapoint: {datapoint['format_validation']}")
    return datapoint


def write_evaluation_jsonl(report_paths: list[str | Path], output_path: str | Path,
                           *, require_distinct_cases: bool = True) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [build_evaluation_datapoint(path) for path in report_paths]
    case_ids = [row["source"]["case_id"] for row in rows]
    if require_distinct_cases and len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation input must contain distinct MLCR cases")
    instance_ids = [row["instance_id"] for row in rows]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("evaluation input contains duplicate task IDs")
    write_release_datapoints(destination, rows)
    return destination
