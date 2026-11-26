"""Render a static HTML browser for generated legal dataset examples."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from teamgym.core.io import estimate_tokens, read_jsonl


@dataclass(frozen=True)
class PromptText:
    label: str
    text: str
    token_count: int


def build_report(
    *,
    input_root: str | Path,
    output_path: str | Path,
    prompt_root: str | Path = "teamgym/prompts/legal",
) -> Path:
    """Build a self-contained HTML report from generated JSONL files."""

    root = Path(input_root).expanduser().resolve()
    examples = _load_examples(root)
    prompt_templates = _load_prompt_templates(prompt_root)
    html = _render_html(
        title="TeamGym Legal Dataset",
        input_root=root,
        examples=examples,
        prompt_templates=prompt_templates,
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _load_examples(input_root: Path) -> list[dict[str, Any]]:
    paths = sorted(path for path in input_root.glob("*/examples_*.jsonl") if path.is_file())
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["_report_source_path"] = str(path)
            rows.append(row)
    return sorted(rows, key=lambda row: (str(row.get("dataset")), str(row.get("instance_id"))))


def _load_prompt_templates(prompt_root: str | Path) -> list[PromptText]:
    root = Path(prompt_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.is_dir():
        return []
    prompts: list[PromptText] = []
    for path in sorted(root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        prompts.append(PromptText(str(path.relative_to(root)), text, estimate_tokens(text)))
    return prompts


def _render_html(
    *,
    title: str,
    input_root: Path,
    examples: list[dict[str, Any]],
    prompt_templates: list[PromptText],
) -> str:
    dataset_stats = _dataset_stats(examples)
    role_stats = _role_stats(examples)
    dataset_names = sorted(dataset_stats)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            "<header>",
            f"<h1>{escape(title)}</h1>",
            f'<p class="subtle">Input root: <code>{escape(str(input_root))}</code></p>',
            _controls(dataset_names),
            "</header>",
            "<main>",
            _overview(examples, dataset_stats),
            _role_overview(role_stats),
            _prompt_templates(prompt_templates),
            "<section>",
            "<h2>Generated Examples</h2>",
            "\n".join(_example_card(row) for row in examples) or '<p class="empty">No examples found.</p>',
            "</section>",
            "</main>",
            f"<script>{_js()}</script>",
            "</body>",
            "</html>",
        ]
    )


def _dataset_stats(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        grouped[str(row.get("dataset") or "unknown")].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for dataset, rows in grouped.items():
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        for row in rows:
            usage = (((row.get("llm_trace") or {}).get("request_meta") or {}).get("usage") or {})
            total_tokens += int(usage.get("total_tokens") or 0)
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
        stats[dataset] = {
            "count": len(rows),
            "valid": sum(1 for row in rows if (row.get("validation") or {}).get("valid")),
            "participants": sum(len(row.get("participants") or []) for row in rows),
            "evidence_units": sum(len(row.get("evidence_units") or []) for row in rows),
            "llm_total_tokens": total_tokens,
            "llm_input_tokens": input_tokens,
            "llm_output_tokens": output_tokens,
        }
    return stats


def _role_stats(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        units_by_id = {str(unit.get("unit_id")): unit for unit in row.get("evidence_units") or []}
        token_map = (row.get("token_accounting") or {}).get("participant_private_tokens") or {}
        required_ids = {str(item) for item in (row.get("hidden_world") or {}).get("required_evidence_unit_ids", [])}
        for participant in row.get("participants") or []:
            packet = participant.get("private_packet") or {}
            ids = [str(item) for item in packet.get("evidence_unit_ids", [])]
            role = str(participant.get("public_role") or participant.get("participant_id") or "unknown_role")
            participant_id = str(participant.get("participant_id") or "")
            grouped[(str(row.get("dataset") or "unknown"), role)].append(
                {
                    "token_count": int(token_map.get(participant_id) or _sum_unit_tokens(ids, units_by_id)),
                    "evidence_count": len(ids),
                    "required_count": len(set(ids) & required_ids),
                }
            )
    stats: list[dict[str, Any]] = []
    for (dataset, role), rows in grouped.items():
        stats.append(
            {
                "dataset": dataset,
                "role": role,
                "count": len(rows),
                "avg_tokens": round(mean(row["token_count"] for row in rows), 1),
                "avg_evidence_units": round(mean(row["evidence_count"] for row in rows), 1),
                "avg_required_units": round(mean(row["required_count"] for row in rows), 1),
            }
        )
    return sorted(stats, key=lambda row: (row["dataset"], -row["avg_tokens"], row["role"]))


def _overview(examples: list[dict[str, Any]], dataset_stats: dict[str, dict[str, Any]]) -> str:
    total_valid = sum(stats["valid"] for stats in dataset_stats.values())
    total_tokens = sum(stats["llm_total_tokens"] for stats in dataset_stats.values())
    cards = [
        _metric_card("Examples", str(len(examples))),
        _metric_card("Valid", f"{total_valid}/{len(examples)}"),
        _metric_card("Datasets", str(len(dataset_stats))),
        _metric_card("LLM Tokens", _fmt_int(total_tokens)),
    ]
    rows = []
    for dataset, stats in sorted(dataset_stats.items()):
        rows.append(
            "<tr>"
            f"<td><span class=\"chip\">{escape(dataset)}</span></td>"
            f"<td>{stats['count']}</td>"
            f"<td>{stats['valid']}/{stats['count']}</td>"
            f"<td>{stats['participants']}</td>"
            f"<td>{stats['evidence_units']}</td>"
            f"<td>{_fmt_int(stats['llm_input_tokens'])}</td>"
            f"<td>{_fmt_int(stats['llm_output_tokens'])}</td>"
            f"<td>{_fmt_int(stats['llm_total_tokens'])}</td>"
            "</tr>"
        )
    return (
        "<section>"
        "<h2>Overview</h2>"
        f"<div class=\"metrics\">{''.join(cards)}</div>"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>Dataset</th><th>Examples</th><th>Valid</th><th>Role Packets</th>"
        "<th>Evidence Units</th><th>Input Tokens</th><th>Output Tokens</th><th>Total Tokens</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</section>"
    )


def _role_overview(role_stats: list[dict[str, Any]]) -> str:
    if not role_stats:
        return ""
    max_tokens = max(float(row["avg_tokens"]) for row in role_stats) or 1.0
    rows = []
    for row in role_stats:
        rows.append(
            "<tr>"
            f"<td>{escape(row['dataset'])}</td>"
            f"<td><code>{escape(row['role'])}</code></td>"
            f"<td>{row['count']}</td>"
            f"<td>{row['avg_evidence_units']}</td>"
            f"<td>{row['avg_required_units']}</td>"
            f"<td>{_token_bar(float(row['avg_tokens']), max_tokens)}</td>"
            "</tr>"
        )
    return (
        "<section>"
        "<h2>Role Information Overview</h2>"
        '<p class="subtle">Average private evidence per public role across the loaded examples.</p>'
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>Dataset</th><th>Role</th><th>Packets</th><th>Avg Evidence Units</th>"
        "<th>Avg Required Units</th><th>Avg Private Tokens</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</section>"
    )


def _prompt_templates(prompt_templates: list[PromptText]) -> str:
    if not prompt_templates:
        return (
            "<section>"
            "<h2>Prompt Templates</h2>"
            '<p class="empty">No prompt templates found.</p>'
            "</section>"
        )
    blocks = []
    for prompt in prompt_templates:
        blocks.append(
            "<details class=\"prompt-template\">"
            f"<summary><code>{escape(prompt.label)}</code> "
            f"<span class=\"muted\">{_fmt_int(prompt.token_count)} tokens</span></summary>"
            f"<pre>{escape(prompt.text)}</pre>"
            "</details>"
        )
    return (
        "<section>"
        "<h2>Prompt Templates</h2>"
        '<p class="subtle">These are the editable prompt files used by the converters.</p>'
        f"{''.join(blocks)}"
        "</section>"
    )


def _example_card(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or "unknown")
    instance_id = str(row.get("instance_id") or "")
    valid = bool((row.get("validation") or {}).get("valid"))
    trace = row.get("llm_trace") or {}
    meta = trace.get("request_meta") or {}
    usage = meta.get("usage") or {}
    public_task = row.get("public_task") or {}
    answer = (row.get("hidden_world") or {}).get("answer")
    searchable = " ".join(
        [
            dataset,
            instance_id,
            str(public_task.get("title") or ""),
            str(public_task.get("objective") or ""),
            " ".join(str(p.get("public_role") or "") for p in row.get("participants") or []),
        ]
    ).lower()
    return (
        f'<article class="example-card" data-dataset="{escape(dataset)}" data-search="{escape(searchable)}">'
        "<div class=\"example-head\">"
        "<div>"
        f"<h3>{escape(instance_id)}</h3>"
        f"<p><span class=\"chip\">{escape(dataset)}</span> "
        f"<span class=\"status {'ok' if valid else 'bad'}\">{'valid' if valid else 'invalid'}</span></p>"
        "</div>"
        "<div class=\"trace-summary\">"
        f"<span>model <code>{escape(str(meta.get('model') or trace.get('model') or 'unknown'))}</code></span>"
        f"<span>tokens <strong>{_fmt_int(int(usage.get('total_tokens') or 0))}</strong></span>"
        f"<span>response <code>{escape(str(meta.get('response_id') or 'n/a'))}</code></span>"
        "</div>"
        "</div>"
        f"{_public_task_block(public_task)}"
        f"{_participant_table(row)}"
        f"{_span_necessity_block(row)}"
        f"{_evidence_block(row)}"
        f"{_hidden_block(answer, row.get('hidden_world') or {})}"
        f"{_trace_prompt_block(trace)}"
        "</article>"
    )


def _public_task_block(public_task: dict[str, Any]) -> str:
    title = str(public_task.get("title") or "Public task")
    objective = str(public_task.get("objective") or "")
    decision_set = public_task.get("decision_set") or []
    return (
        "<div class=\"public-task\">"
        f"<h4>{escape(title)}</h4>"
        f"<p>{escape(objective)}</p>"
        f"<p class=\"subtle\">Decision set: {_json_inline(decision_set)}</p>"
        "<details><summary>Full public task JSON</summary>"
        f"<pre>{escape(json.dumps(public_task, indent=2, ensure_ascii=False))}</pre>"
        "</details>"
        "</div>"
    )


def _participant_table(row: dict[str, Any]) -> str:
    units_by_id = {str(unit.get("unit_id")): unit for unit in row.get("evidence_units") or []}
    token_map = (row.get("token_accounting") or {}).get("participant_private_tokens") or {}
    required_ids = {str(item) for item in (row.get("hidden_world") or {}).get("required_evidence_unit_ids", [])}
    participants = row.get("participants") or []
    private_tokens = [
        int(token_map.get(str(participant.get("participant_id") or "")) or _sum_unit_tokens(
            (participant.get("private_packet") or {}).get("evidence_unit_ids", []),
            units_by_id,
        ))
        for participant in participants
    ]
    max_tokens = max(private_tokens or [1])
    rows = []
    for participant, token_count in zip(participants, private_tokens):
        packet = participant.get("private_packet") or {}
        ids = [str(item) for item in packet.get("evidence_unit_ids", [])]
        required_count = len(set(ids) & required_ids)
        critical_count = sum(
            1
            for unit_id in ids
            if str((units_by_id.get(unit_id) or {}).get("criticality") or "").lower() in {"critical", "silver_rationale"}
        )
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(participant.get('participant_id') or ''))}</code></td>"
            f"<td>{escape(str(participant.get('public_role') or ''))}</td>"
            f"<td>{escape(str(packet.get('title') or ''))}</td>"
            f"<td>{len(ids)}</td>"
            f"<td>{critical_count}</td>"
            f"<td>{required_count}</td>"
            f"<td>{_token_bar(token_count, max_tokens)}</td>"
            f"<td><code>{escape(', '.join(ids))}</code></td>"
            "</tr>"
        )
    return (
        "<div class=\"table-block\">"
        "<h4>Role And Private Information Amount</h4>"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>ID</th><th>Public Role</th><th>Private Packet</th><th>Evidence</th>"
        "<th>Critical</th><th>Required</th><th>Private Tokens</th><th>Evidence IDs</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</div>"
    )


def _evidence_block(row: dict[str, Any]) -> str:
    units = row.get("evidence_units") or []
    if not units:
        return ""
    rows = []
    for unit in units:
        tags = " ".join(f'<span class="tag">{escape(str(tag))}</span>' for tag in unit.get("tags", [])[:6])
        summary = str(unit.get("summary") or unit.get("text") or "")
        relevance = _decision_relevance_summary(unit.get("decision_relevance"))
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(unit.get('unit_id') or ''))}</code></td>"
            f"<td>{escape(str(unit.get('criticality') or ''))}</td>"
            f"<td>{_fmt_int(int(unit.get('token_count') or 0))}</td>"
            f"<td>{tags}</td>"
            f"<td>{escape(_truncate(summary, 260))}</td>"
            f"<td>{relevance}</td>"
            "</tr>"
        )
    return (
        "<details class=\"evidence-details\">"
        "<summary>Evidence Units</summary>"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>ID</th><th>Criticality</th><th>Tokens</th><th>Tags</th><th>Summary/Text</th><th>Decision Relevance</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</details>"
    )


def _span_necessity_block(row: dict[str, Any]) -> str:
    hidden = row.get("hidden_world") or {}
    metadata = hidden.get("native_metadata_hidden_from_participants") or {}
    assessment = metadata.get("candidate_assessment") or {}
    tests = hidden.get("oracle_necessity_tests") or []
    validation = hidden.get("necessity_validation") or {}
    smallest_sets = hidden.get("smallest_deciding_rui_sets") or []
    min_size = hidden.get("minimum_sufficient_rui_set_size")
    mode = hidden.get("rui_necessity_mode")
    if not assessment and not tests and not validation:
        return ""
    selected = assessment.get("selected_span_indices") or []
    excluded = assessment.get("excluded_spans") or []
    test_rows = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        test_rows.append(
            "<tr>"
            f"<td><code>{escape(str(test.get('omitted_unit_id') or test.get('unit_id') or ''))}</code></td>"
            f"<td>{escape(json.dumps(test.get('remaining_unit_ids') or [], ensure_ascii=False))}</td>"
            f"<td>{escape(str(test.get('can_determine_oracle_without_omitted')))}</td>"
            f"<td>{escape(str(test.get('effect') or ''))}</td>"
            f"<td>{escape(_truncate(str(test.get('without_this_span') or ''), 220))}</td>"
            f"<td>{escape(_truncate(json.dumps(test.get('ambiguous_or_plausible_decisions_without_it') or [], ensure_ascii=False), 220))}</td>"
            f"<td>{escape(_truncate(str(test.get('why_required') or ''), 260))}</td>"
            "</tr>"
        )
    excluded_rows = []
    for item in excluded:
        if not isinstance(item, dict):
            continue
        excluded_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('span_index') or ''))}</td>"
            f"<td>{escape(str(item.get('reason') or ''))}</td>"
            "</tr>"
        )
    validator_status = "valid" if validation.get("valid") is True else "not recorded"
    excluded_html = (
        "<details><summary>Excluded Supplied Spans</summary><div class=\"table-wrap\"><table>"
        "<thead><tr><th>Span</th><th>Reason</th></tr></thead>"
        f"<tbody>{''.join(excluded_rows)}</tbody></table></div></details>"
        if excluded_rows
        else ""
    )
    smallest_html = _smallest_deciding_sets_block(smallest_sets, min_size)
    validator_loo_html = _validator_leave_one_out_block(validation.get("leave_one_out_results") or [])
    return (
        "<details class=\"necessity-block\" open>"
        f"<summary>Span Necessity <span class=\"muted\">{escape(validator_status)}</span></summary>"
        f"<p class=\"subtle\">RUI necessity mode: <code>{escape(str(mode or 'not recorded'))}</code></p>"
        f"<p class=\"subtle\">Selected source spans: <code>{escape(json.dumps(selected, ensure_ascii=False))}</code></p>"
        f"<p>{escape(str(assessment.get('question_design_rationale') or ''))}</p>"
        f"<p class=\"subtle\">{escape(str(assessment.get('coverage_rationale') or ''))}</p>"
        f"{smallest_html}"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>Omitted RUI</th><th>Remaining RUIs</th><th>Can Still Decide?</th><th>Removal Effect</th><th>Without This RUI</th><th>Plausible Decisions</th><th>Why Required</th></tr></thead>"
        f"<tbody>{''.join(test_rows)}</tbody></table></div>"
        f"{validator_loo_html}"
        f"{excluded_html}"
        "</details>"
    )


def _decision_relevance_summary(relevance: Any) -> str:
    if not isinstance(relevance, list) or not relevance:
        return ""
    rows: list[str] = []
    for item in relevance[:4]:
        if not isinstance(item, dict):
            continue
        decision = escape(str(item.get("decision") or ""))
        relation = escape(str(item.get("relation") or ""))
        reason = escape(_truncate(str(item.get("reason") or ""), 160))
        rows.append(f"<div><code>{decision}</code> <span class=\"tag\">{relation}</span> {reason}</div>")
    return "".join(rows)


def _smallest_deciding_sets_block(rows: Any, min_size: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        table_rows.append(
            "<tr>"
            f"<td>{escape(json.dumps(row.get('unit_ids') or [], ensure_ascii=False))}</td>"
            f"<td>{escape(str(row.get('answer') or ''))}</td>"
            f"<td>{escape(_truncate(str(row.get('why_sufficient') or ''), 240))}</td>"
            f"<td>{escape(_truncate(str(row.get('why_minimal') or row.get('minimality_rationale') or ''), 240))}</td>"
            "</tr>"
        )
    return (
        "<details open><summary>Smallest Deciding RUI Set "
        f"<span class=\"muted\">min size: {escape(str(min_size))}</span></summary>"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>RUI Set</th><th>Answer</th><th>Why Sufficient</th><th>Why Minimal</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table></div>"
        "</details>"
    )


def _validator_leave_one_out_block(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return ""
    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        table_rows.append(
            "<tr>"
            f"<td><code>{escape(str(row.get('omitted_unit_id') or row.get('unit_id') or ''))}</code></td>"
            f"<td>{escape(json.dumps(row.get('remaining_unit_ids') or [], ensure_ascii=False))}</td>"
            f"<td>{escape(str(row.get('can_determine_oracle_without_omitted')))}</td>"
            f"<td>{escape(_truncate(json.dumps(row.get('plausible_decisions_without_omitted') or [], ensure_ascii=False), 220))}</td>"
            f"<td>{escape(_truncate(str(row.get('why_required') or ''), 260))}</td>"
            "</tr>"
        )
    return (
        "<details><summary>Validator Leave-One-Out Audit</summary>"
        "<div class=\"table-wrap\"><table>"
        "<thead><tr><th>Omitted RUI</th><th>Remaining RUIs</th><th>Can Still Decide?</th><th>Plausible Decisions</th><th>Why Required</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table></div>"
        "</details>"
    )


def _hidden_block(answer: Any, hidden_world: dict[str, Any]) -> str:
    answer_text = json.dumps(answer, ensure_ascii=False)
    return (
        "<details class=\"hidden-block\">"
        f"<summary>Hidden Oracle <span class=\"muted\">answer: {escape(_truncate(answer_text, 160))}</span></summary>"
        f"<pre>{escape(json.dumps(hidden_world, indent=2, ensure_ascii=False))}</pre>"
        "</details>"
    )


def _trace_prompt_block(trace: dict[str, Any]) -> str:
    trace_dir = trace.get("trace_dir")
    prompts = _load_trace_prompts(trace_dir) if trace_dir else []
    if not prompts:
        return (
            "<details class=\"trace-block\">"
            "<summary>Rendered Prompt Trace</summary>"
            '<p class="empty">No rendered prompt files found for this row.</p>'
            "</details>"
        )
    blocks = []
    for prompt in prompts:
        blocks.append(
            "<details class=\"prompt-rendered\">"
            f"<summary><code>{escape(prompt.label)}</code> "
            f"<span class=\"muted\">{_fmt_int(prompt.token_count)} tokens</span></summary>"
            f"<pre>{escape(prompt.text)}</pre>"
            "</details>"
        )
    return (
        "<details class=\"trace-block\">"
        f"<summary>Rendered Prompt Trace <span class=\"muted\">{escape(str(trace_dir))}</span></summary>"
        f"{''.join(blocks)}"
        "</details>"
    )


def _load_trace_prompts(trace_dir: str | Path) -> list[PromptText]:
    path = Path(trace_dir).expanduser()
    if not path.is_dir():
        return []
    prompts: list[PromptText] = []
    for file_path in sorted([*path.glob("*.system.txt"), *path.glob("*.user.txt")]):
        text = file_path.read_text(encoding="utf-8")
        prompts.append(PromptText(file_path.name, text, estimate_tokens(text)))
    return prompts


def _controls(dataset_names: list[str]) -> str:
    options = ['<option value="">all datasets</option>']
    options.extend(f'<option value="{escape(name)}">{escape(name)}</option>' for name in dataset_names)
    return (
        '<div class="controls">'
        '<label>Dataset <select id="datasetFilter">'
        f"{''.join(options)}"
        "</select></label>"
        '<label>Search <input id="searchBox" type="search" placeholder="instance, role, objective"></label>'
        "</div>"
    )


def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'


def _token_bar(value: float, max_value: float) -> str:
    width = 0 if max_value <= 0 else max(3, min(100, int(round((value / max_value) * 100))))
    return (
        '<div class="bar-cell">'
        f'<div class="bar"><span style="width:{width}%"></span></div>'
        f"<strong>{_fmt_int(round(value, 1) if value % 1 else int(value))}</strong>"
        "</div>"
    )


def _sum_unit_tokens(unit_ids: Iterable[Any], units_by_id: dict[str, dict[str, Any]]) -> int:
    return sum(int((units_by_id.get(str(unit_id)) or {}).get("token_count") or 0) for unit_id in unit_ids)


def _json_inline(value: Any) -> str:
    return f"<code>{escape(json.dumps(value, ensure_ascii=False))}</code>"


def _fmt_int(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(value):,}"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f8fafc;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #d9e2ec;
  --accent: #176b87;
  --accent-2: #2f7d59;
  --warn: #9b2c2c;
  --chip: #eef6f8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  background: rgba(248, 250, 252, 0.96);
  border-bottom: 1px solid var(--line);
  padding: 18px 28px;
  backdrop-filter: blur(8px);
}
main { max-width: 1440px; margin: 0 auto; padding: 24px 28px 56px; }
h1, h2, h3, h4 { margin: 0; line-height: 1.2; letter-spacing: 0; }
h1 { font-size: 24px; }
h2 { font-size: 19px; margin-bottom: 12px; }
h3 { font-size: 18px; }
h4 { font-size: 15px; margin-bottom: 8px; }
section, .example-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  margin-bottom: 18px;
}
code {
  background: #f0f3f5;
  border: 1px solid #e3e8ee;
  border-radius: 4px;
  padding: 1px 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
pre {
  white-space: pre-wrap;
  overflow: auto;
  max-height: 520px;
  background: #0f1720;
  color: #e6edf3;
  border-radius: 6px;
  padding: 12px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }
th { font-size: 12px; color: #475467; background: #f7f9fb; position: sticky; top: 0; }
tr:last-child td { border-bottom: 0; }
summary { cursor: pointer; font-weight: 650; }
details { margin-top: 12px; }
.subtle, .muted { color: var(--muted); }
.empty { color: var(--muted); font-style: italic; }
.controls { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; align-items: center; }
.controls label { display: flex; align-items: center; gap: 8px; color: #475467; font-weight: 600; }
select, input {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 9px;
  min-width: 220px;
  background: white;
  color: var(--ink);
}
.metrics { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }
.metric { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; }
.metric span { display: block; color: var(--muted); font-size: 12px; font-weight: 650; text-transform: uppercase; }
.metric strong { display: block; margin-top: 2px; font-size: 22px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 7px; }
.chip, .tag, .status {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 650;
}
.chip { background: var(--chip); color: #17566b; }
.tag { background: #f2f4f7; color: #475467; margin: 0 3px 3px 0; }
.status.ok { background: #ecfdf3; color: var(--accent-2); }
.status.bad { background: #fef3f2; color: var(--warn); }
.example-head { display: flex; justify-content: space-between; gap: 18px; margin-bottom: 12px; }
.trace-summary { display: flex; flex-direction: column; gap: 3px; align-items: flex-end; color: var(--muted); font-size: 12px; }
.public-task { border-left: 3px solid var(--accent); padding-left: 12px; margin: 12px 0; }
.table-block { margin: 14px 0; }
.bar-cell { display: grid; grid-template-columns: minmax(110px, 1fr) 70px; gap: 8px; align-items: center; min-width: 190px; }
.bar { height: 10px; border-radius: 999px; background: #e8eef3; overflow: hidden; }
.bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.prompt-template, .prompt-rendered { border-top: 1px solid var(--line); padding-top: 10px; }
.hidden-block summary { color: #7a4f01; }
.trace-block summary { color: #344054; }
.necessity-block summary { color: #17566b; }
@media (max-width: 760px) {
  header { position: static; padding: 16px; }
  main { padding: 16px; }
  .metrics { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
  .example-head { display: block; }
  .trace-summary { align-items: flex-start; margin-top: 8px; }
  select, input { min-width: 160px; width: 100%; }
}
"""


def _js() -> str:
    return """
const datasetFilter = document.getElementById('datasetFilter');
const searchBox = document.getElementById('searchBox');
function applyFilters() {
  const dataset = datasetFilter.value;
  const query = searchBox.value.trim().toLowerCase();
  document.querySelectorAll('.example-card').forEach(card => {
    const datasetOk = !dataset || card.dataset.dataset === dataset;
    const searchOk = !query || card.dataset.search.includes(query);
    card.style.display = datasetOk && searchOk ? '' : 'none';
  });
}
datasetFilter.addEventListener('change', applyFilters);
searchBox.addEventListener('input', applyFilters);
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", default="analysis_outputs/legal_dataset_report.html")
    parser.add_argument("--prompt-root", default="teamgym/prompts/legal")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = build_report(
        input_root=args.input_root,
        output_path=args.output,
        prompt_root=args.prompt_root,
    )
    print(output_path)


if __name__ == "__main__":
    main()
