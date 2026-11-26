"""Exact-source restriction of canonical tables, followed by full revalidation."""
from __future__ import annotations

from copy import deepcopy

from .runtime import call_json
from .source import restrict_table_rows


def repair_canonical_tables(*, client, datapoint, task, report_index, canonical_overlaps, trace_root):
    """Mutate canonical table text consistently; caller must rerun ALL/NONE/LOO.

    A table stays one PIU. No model-written table text is accepted. Paragraph
    overlaps and tables without a safe row restriction remain rejected.
    """
    affected = {participant_id for participant_id, _ in canonical_overlaps}
    blocks = {row["block_id"]: row for row in report_index["blocks"]}
    contexts = [row for row in datapoint["private_context_units"]
                if row.get("hidden_contains_rui_ids") and affected.intersection(row["assigned_to"])
                and (row.get("source_ref") or {}).get("block_type") == "table"]
    allowed = {(row["source_ref"] or {})["block_id"] for row in contexts}
    if not allowed:
        return False, []
    proposal = call_json(client, system_prompt_name="generation_system.txt",
        user_prompt_name="06_table_row_restriction_user.txt",
        variables={"PUBLIC_TASK_JSON": task["public_task"], "RUBRIC_JSON": task["rubric"],
            "EVIDENCE_UNITS_JSON": task["evidence_units"],
            "OVERLAPPING_TABLES_JSON": [blocks[block_id] for block_id in sorted(allowed)],
            "CANONICAL_OVERLAPS_JSON": sorted(canonical_overlaps)}, trace_dir=trace_root)
    proposals = proposal.get("restrictions")
    if not isinstance(proposals, list) or not proposals:
        return False, ["no safe canonical-table row restriction proposed"]
    planned, seen = [], set()
    for item in proposals:
        if not isinstance(item, dict) or set(item) != {"source_block_id", "source_row_numbers", "reason"}:
            return False, ["invalid canonical-table restriction shape"]
        block_id = item["source_block_id"]
        if block_id not in allowed or block_id in seen or not isinstance(item["reason"], str) or not item["reason"].strip():
            return False, ["canonical-table restriction has unknown/duplicate block or missing rationale"]
        seen.add(block_id)
        try:
            restriction = restrict_table_rows(blocks[block_id], item["source_row_numbers"])
        except (ValueError, TypeError, KeyError) as exc:
            return False, [f"invalid canonical-table restriction: {exc}"]
        unit = next(row for row in task["evidence_units"] if row["source_block_id"] == block_id)
        old_rows = set(unit.get("source_row_numbers") or range(1, len(blocks[block_id]["text"].splitlines()) + 1))
        if not set(restriction["source_row_numbers"]) < old_rows:
            return False, ["canonical-table repair must strictly remove rows, without adding evidence"]
        planned.append((block_id, unit, restriction, item["reason"]))
    history = datapoint.setdefault("generation_metadata", {}).setdefault("canonical_table_restrictions", [])
    for block_id, unit, restriction, reason in planned:
        unit["text"] = restriction["text"]
        unit["source_row_numbers"] = restriction["source_row_numbers"]
        source_fields = {k: restriction[k] for k in ("source_row_numbers", "source_row_spans", "original_block_sha256")}
        unit["source_ref"].update(deepcopy(source_fields))
        for riu in datapoint["reference_information_units"]:
            if riu["rui_id"] == unit["unit_id"]:
                riu["text"] = restriction["text"]
                riu["source_ref"].update(deepcopy(source_fields))
        for context in datapoint["private_context_units"]:
            if context["source_ref"]["block_id"] == block_id:
                context["text"] = restriction["text"]
                context["word_count"] = len(restriction["text"].split())
                context["source_ref"].update(deepcopy(source_fields))
        for common in datapoint["initial_common_information"].get("common_reference_units", []):
            if common["rui_id"] == unit["unit_id"]:
                common["text"] = restriction["text"]
        history.append({"source_block_id": block_id, "unit_id": unit["unit_id"],
                        **deepcopy(source_fields), "reason": reason})
    for participant in datapoint["participants"]:
        participant["private_word_count"] = sum(row["word_count"] for row in datapoint["private_context_units"]
            if participant["participant_id"] in row["assigned_to"])
    return True, []
