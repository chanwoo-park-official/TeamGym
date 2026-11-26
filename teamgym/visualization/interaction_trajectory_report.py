"""Render a static HTML report for fully collaborative interaction runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any


def build_report(*, input_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(input_path).expanduser().resolve()
    trajectories, run_dir = _load_trajectories(source)
    manifest = _read_json(run_dir / "run_manifest.json") if run_dir else {}
    summary = _read_json(run_dir / "eval_summary.json") if run_dir else {}
    if output_path is None:
        out = (run_dir or source.parent) / "interaction_report.html"
    else:
        out = Path(output_path).expanduser().resolve()
    html = _render_html(
        title="Fully Collaborative Interaction Report",
        source=source,
        manifest=manifest,
        summary=summary,
        trajectories=trajectories,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _load_trajectories(source: Path) -> tuple[list[dict[str, Any]], Path | None]:
    if source.is_file():
        return [_trajectory_payload(source)], _infer_run_dir(source)
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    trajectory_dir = source / "trajectories"
    if trajectory_dir.is_dir():
        paths = sorted(path for path in trajectory_dir.glob("*.json") if path.is_file())
        return [_trajectory_payload(path) for path in paths], source
    paths = sorted(path for path in source.glob("*.json") if path.is_file())
    if not paths:
        raise ValueError(f"No trajectory JSON files found under {source}")
    return [_trajectory_payload(path) for path in paths], source


def _trajectory_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if "episode" not in payload and "evaluation" not in payload:
        raise ValueError(f"Not an interaction trajectory JSON: {path}")
    payload = dict(payload)
    payload["_path"] = str(path)
    return payload


def _infer_run_dir(path: Path) -> Path | None:
    if path.parent.name == "trajectories":
        return path.parent.parent
    return path.parent


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _render_html(
    *,
    title: str,
    source: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    trajectories: list[dict[str, Any]],
) -> str:
    datasets = sorted(
        {
            str(((row.get("episode") or {}).get("datapoint") or {}).get("dataset") or "unknown")
            for row in trajectories
        }
    )
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
            f'<p class="subtle">Source: <code>{escape(str(source))}</code></p>',
            _controls(datasets, trajectories),
            "</header>",
            "<main>",
            _overview(manifest, summary, trajectories),
            _trajectory_index(trajectories),
            "\n".join(_trajectory_section(row, index) for index, row in enumerate(trajectories, start=1))
            or '<section><p class="empty">No trajectories found.</p></section>',
            "</main>",
            f"<script>{_js()}</script>",
            "</body>",
            "</html>",
        ]
    )


def _overview(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    trajectories: list[dict[str, Any]],
) -> str:
    aggregate = summary.get("aggregate") if isinstance(summary.get("aggregate"), dict) else {}
    llm = manifest.get("llm") if isinstance(manifest.get("llm"), dict) else {}
    stats = _computed_stats(trajectories)
    cards = [
        _metric("Episodes", str(len(trajectories))),
        _metric("Success", _percent(aggregate.get("mean_task_success", stats["task_success"]))),
        _metric("Informed", _percent(aggregate.get("mean_informed_success", stats["informed_success"]))),
        _metric("RIU Recall", _percent(aggregate.get("mean_public_reference_information_unit_recall", stats["rui_recall"]))),
        _metric("Mean Turns", _number(aggregate.get("mean_turn_count", stats["turn_count"]))),
        _metric("Model", str(llm.get("model") or "unknown")),
    ]
    label_match = aggregate.get("mean_decision_label_matches", stats["decision_label_matches"])
    if label_match is not None:
        cards.append(_metric("Decision label matches", _percent(label_match)))
    manifest_rows = [
        ("Actor runtime", manifest.get("actor_runtime") or "llm"),
        ("Input root", manifest.get("input_root") or ""),
        ("Turn budget", (manifest.get("budgets") or {}).get("turn_budget") if isinstance(manifest.get("budgets"), dict) else ""),
        ("Prompt tracing", manifest.get("prompt_tracing_enabled")),
    ]
    return (
        "<section>"
        "<h2>Run Overview</h2>"
        f'<div class="metrics">{"".join(cards)}</div>'
        '<div class="table-wrap"><table><tbody>'
        + "".join(
            f"<tr><th>{escape(str(label))}</th><td>{escape(str(value))}</td></tr>"
            for label, value in manifest_rows
            if value not in {None, ""}
        )
        + "</tbody></table></div>"
        "</section>"
    )


def _computed_stats(trajectories: list[dict[str, Any]]) -> dict[str, float | None]:
    evaluations = [
        row.get("evaluation")
        for row in trajectories
        if isinstance(row.get("evaluation"), dict)
    ]
    return {
        "task_success": _mean_path(evaluations, ["primary_eval", "task_success"]),
        "informed_success": _mean_path(evaluations, ["primary_eval", "informed_success"]),
        "decision_label_matches": _mean_path(evaluations, ["primary_eval", "decision_label_matches"]),
        "rui_recall": _mean_path(evaluations, ["information_flow_eval", "public_reference_information_unit_recall"]),
        "turn_count": _mean_path(evaluations, ["communication_eval", "turn_count"]),
    }


def _trajectory_index(trajectories: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
        datapoint = episode.get("datapoint") if isinstance(episode.get("datapoint"), dict) else {}
        grouped[str(datapoint.get("dataset") or "unknown")].append(row)
    rows = []
    for dataset, items in sorted(grouped.items()):
        stats = _computed_stats(items)
        rows.append(
            "<tr>"
            f"<td><span class=\"chip dataset\">{escape(dataset)}</span></td>"
            f"<td>{len(items)}</td>"
            f"<td>{_number(stats['task_success'])}</td>"
            f"<td>{_number(stats['informed_success'])}</td>"
            f"<td>{_number(stats['decision_label_matches'])}</td>"
            f"<td>{_number(stats['rui_recall'])}</td>"
            "</tr>"
        )
    return (
        "<section>"
        "<h2>Dataset Index</h2>"
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Dataset</th><th>Episodes</th><th>Success</th><th>Informed</th>"
        "<th>Decision label matches</th><th>RIU Recall</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table></div>"
        "</section>"
    )


def _trajectory_section(row: dict[str, Any], index: int) -> str:
    episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    datapoint = episode.get("datapoint") if isinstance(episode.get("datapoint"), dict) else {}
    structure = episode.get("communication_structure") if isinstance(episode.get("communication_structure"), dict) else {}
    instance_id = str(datapoint.get("instance_id") or "unknown_instance")
    dataset = str(datapoint.get("dataset") or "unknown")
    heading = f"{dataset} / {instance_id}"
    trajectory_id = _trajectory_id(index)
    searchable = " ".join(
        [
            dataset,
            instance_id,
            str(structure.get("name") or ""),
            _plain_text(datapoint.get("task_description")),
            _plain_text(episode.get("messages")),
            _plain_text(episode.get("final_decision")),
        ]
    )
    task_success = _path(evaluation, ["primary_eval", "task_success"])
    informed_success = _path(evaluation, ["primary_eval", "informed_success"])
    label_match = _path(evaluation, ["primary_eval", "decision_label_matches"])
    badges = (
        _status_badge("decision label matches", label_match)
        if label_match is not None
        else _status_badge("task", task_success) + _status_badge("informed", informed_success)
    )
    return (
        f'<section class="trajectory" data-trajectory-id="{escape(trajectory_id)}" data-dataset="{escape(dataset)}" data-search="{escape(searchable.casefold())}">'
        '<div class="section-head">'
        f"<h2>{escape(heading)}</h2>"
        '<div class="badges">'
        f'{badges}'
        f'<span class="chip">{escape(str(structure.get("name") or "unknown_structure"))}</span>'
        "</div>"
        "</div>"
        f'<p class="subtle">File: <code>{escape(str(row.get("_path") or ""))}</code></p>'
        f"{_task_block(datapoint)}"
        f"{_turn_timeline(episode)}"
        f"{_final_decision_block(episode, evaluation)}"
        f"{_rui_matching_block(evaluation)}"
        f"{_prompt_records_block(episode)}"
        "</section>"
    )


def _task_block(datapoint: dict[str, Any]) -> str:
    decisions = datapoint.get("decision_option_definitions")
    if not isinstance(decisions, dict):
        decisions = {}
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(key))}</code></td>"
        f"<td>{escape(_plain_text(value))}</td>"
        "</tr>"
        for key, value in sorted(decisions.items())
    )
    participant_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row.get('participant_id') or ''))}</code></td>"
        f"<td>{escape(str(row.get('role_name') or row.get('public_role') or ''))}</td>"
        "</tr>"
        for row in datapoint.get("participant_directory", [])
        if isinstance(row, dict)
    )
    return (
        '<div class="split">'
        '<div class="panel">'
        "<h3>Task</h3>"
        f"<p>{escape(_plain_text(datapoint.get('task_description')))}</p>"
        '<div class="table-wrap"><table><thead><tr><th>Decision</th><th>Definition</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        "</div>"
        '<div class="panel">'
        "<h3>Participants</h3>"
        '<div class="table-wrap"><table><thead><tr><th>ID</th><th>Role</th></tr></thead>'
        f"<tbody>{participant_rows}</tbody></table></div>"
        "</div>"
        "</div>"
    )


def _turn_timeline(episode: dict[str, Any]) -> str:
    turns = episode.get("turns") if isinstance(episode.get("turns"), list) else []
    parts = ["<h3>Transcript</h3>", '<div class="timeline">']
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if turn.get("turn_type") == "final_decision":
            continue
        messages = [msg for msg in turn.get("messages", []) if isinstance(msg, dict)]
        parts.append(
            '<div class="turn">'
            f'<div class="turn-label">Turn {escape(str(turn.get("turn")))}</div>'
            '<div class="messages">'
            + "".join(_message_bubble(message) for message in messages)
            + "</div></div>"
        )
    parts.append("</div>")
    return '<div class="panel transcript">' + "".join(parts) + "</div>"


def _message_bubble(message: dict[str, Any]) -> str:
    role = str(message.get("speaker_role") or "unknown")
    speaker = str(message.get("speaker_id") or "")
    role_name = str(message.get("role_name") or role)
    target_ids = message.get("target_ids") if isinstance(message.get("target_ids"), list) else []
    targets = ", ".join(str(item) for item in target_ids)
    meta = f"{speaker} / {role_name}"
    if targets:
        meta += f" -> {targets}"
    analysis = message.get("utterance_analysis")
    if isinstance(analysis, dict) and analysis.get("kind"):
        meta += f" / utterance: {analysis.get('kind')}"
    return (
        f'<article class="message {escape(role)}">'
        f'<div class="message-meta">{escape(meta)}</div>'
        f'<p>{escape(str(message.get("content") or ""))}</p>'
        "</article>"
    )


def _final_decision_block(episode: dict[str, Any], evaluation: dict[str, Any]) -> str:
    final = episode.get("final_decision") if isinstance(episode.get("final_decision"), dict) else {}
    primary = evaluation.get("primary_eval") if isinstance(evaluation.get("primary_eval"), dict) else {}
    return (
        '<div class="panel final">'
        "<h3>Final Decision</h3>"
        '<div class="final-grid">'
        f'<div><span>Answer</span><strong><code>{escape(str(final.get("answer") or ""))}</code></strong></div>'
        f'<div><span>Oracle</span><strong><code>{escape(str(primary.get("oracle_answer") or ""))}</code></strong></div>'
        f'<div><span>Confidence</span><strong>{escape(_number(final.get("confidence")))}</strong></div>'
        f'<div><span>Actor</span><strong>{escape(str(final.get("actor_id") or ""))}</strong></div>'
        "</div>"
        f'<p>{escape(str(final.get("rationale") or ""))}</p>'
        "</div>"
    )


def _rui_matching_block(evaluation: dict[str, Any]) -> str:
    matching = evaluation.get("reference_information_matching")
    if not isinstance(matching, dict):
        return ""
    matches = [row for row in matching.get("lexical_matches", []) if isinstance(row, dict)]
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row.get('rui_id') or ''))}</code></td>"
        f"<td>{escape(str(row.get('matched_by') or ''))}</td>"
        f"<td>{escape(_number(row.get('token_recall')))}</td>"
        f"<td>{escape(', '.join(str(item) for item in row.get('sample_overlap_tokens', [])[:12]))}</td>"
        "</tr>"
        for row in matches
    )
    return (
        '<div class="panel rui">'
        "<h3>Evaluator-Only RIU Matching</h3>"
        '<div class="rui-row">'
        f"<span>matched <strong>{len(matching.get('matched_rui_ids') or [])}</strong></span>"
        f"<span>explicit <strong>{len(matching.get('explicit_disclosed_rui_ids') or [])}</strong></span>"
        f"<span>lexical <strong>{len(matching.get('lexically_matched_rui_ids') or [])}</strong></span>"
        "</div>"
        '<div class="table-wrap"><table><thead><tr><th>RIU</th><th>Method</th><th>Recall</th><th>Overlap Tokens</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        f'<p class="subtle">{escape(str(matching.get("method_note") or ""))}</p>'
        "</div>"
    )


def _prompt_records_block(episode: dict[str, Any]) -> str:
    records = [row for row in episode.get("prompt_records", []) if isinstance(row, dict)]
    audit = _prompt_audit(records)
    status = "clean" if not audit else "check"
    return (
        '<div class="panel prompts">'
        "<h3>Prompt Records</h3>"
        f'<p class="subtle">records: {len(records)}; hidden field-name audit: <span class="{status}">{escape(status)}</span></p>'
        + ("<p class=\"warn\">Found hidden field names in prompt text: " + escape(", ".join(audit)) + "</p>" if audit else "")
        + "".join(_prompt_record(record, index) for index, record in enumerate(records, start=1))
        + "</div>"
    )


def _prompt_record(record: dict[str, Any], index: int) -> str:
    title = " / ".join(
        str(item)
        for item in [
            f"{index:02d}",
            record.get("actor_id"),
            record.get("prompt_kind"),
        ]
        if item
    )
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    meta = record.get("request_meta") if isinstance(record.get("request_meta"), dict) else {}
    return (
        '<details class="prompt-record">'
        f"<summary>{escape(title)}</summary>"
        '<div class="prompt-grid">'
        f'<div><h4>System</h4><pre>{escape(str(record.get("system_prompt") or ""))}</pre></div>'
        f'<div><h4>User Payload</h4><pre>{escape(str(record.get("user_prompt") or ""))}</pre></div>'
        f'<div><h4>Model JSON</h4><pre>{escape(_json_pretty(response))}</pre></div>'
        f'<div><h4>Request Meta</h4><pre>{escape(_json_pretty(meta))}</pre></div>'
        "</div>"
        "</details>"
    )


def _prompt_audit(records: list[dict[str, Any]]) -> list[str]:
    text = "\n".join(
        str(row.get("system_prompt") or "") + "\n" + str(row.get("user_prompt") or "")
        for row in records
    )
    needles = [
        "owned_rui_ids",
        "reference_information_units",
        "smallest_sufficient_rui_sets",
        '"oracle"',
    ]
    return [needle for needle in needles if needle in text]


def _controls(datasets: list[str], trajectories: list[dict[str, Any]]) -> str:
    dataset_options = ['<option value="">all datasets</option>']
    dataset_options.extend(f'<option value="{escape(dataset)}">{escape(dataset)}</option>' for dataset in datasets)
    trajectory_options = ['<option value="">all trajectories</option>']
    trajectory_options.extend(
        f'<option value="{escape(_trajectory_id(index))}">{escape(_trajectory_label(row, index))}</option>'
        for index, row in enumerate(trajectories, start=1)
    )
    return (
        '<div class="controls">'
        '<label>Trajectory <select id="trajectoryFilter">' + "".join(trajectory_options) + "</select></label>"
        '<label>Dataset <select id="datasetFilter">' + "".join(dataset_options) + "</select></label>"
        '<label>Search <input id="searchBox" type="search" placeholder="instance, role, message"></label>'
        '<button id="expandPrompts" type="button">Expand prompts</button>'
        '<button id="collapsePrompts" type="button">Collapse prompts</button>'
        "</div>"
    )


def _trajectory_id(index: int) -> str:
    return f"trajectory-{index}"


def _trajectory_label(row: dict[str, Any], index: int) -> str:
    episode = row.get("episode") if isinstance(row.get("episode"), dict) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    datapoint = episode.get("datapoint") if isinstance(episode.get("datapoint"), dict) else {}
    structure = episode.get("communication_structure") if isinstance(episode.get("communication_structure"), dict) else {}
    structure_name = str(structure.get("name") or "unknown_structure")
    instance_id = str(datapoint.get("instance_id") or "unknown_instance")
    label_match = _path(evaluation, ["primary_eval", "decision_label_matches"])
    if label_match is not None:
        status = "label match" if label_match else "label mismatch"
        return f"{index}. {structure_name} / {instance_id} / {status}"
    task_success = "ok" if bool(_path(evaluation, ["primary_eval", "task_success"])) else "fail"
    informed_success = "informed" if bool(_path(evaluation, ["primary_eval", "informed_success"])) else "not informed"
    return f"{index}. {structure_name} / {instance_id} / {task_success}, {informed_success}"


def _status_badge(label: str, value: Any) -> str:
    class_name = "pass" if bool(value) else "fail"
    text = "yes" if bool(value) else "no"
    return f'<span class="chip {class_name}">{escape(label)}: {text}</span>'


def _metric(label: str, value: str) -> str:
    return f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'


def _path(row: dict[str, Any], path: list[str]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _mean_path(rows: list[Any], path: list[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _path(row, path)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return mean(values) if values else None


def _percent(value: Any) -> str:
    if isinstance(value, bool):
        value = 1.0 if value else 0.0
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _number(value: Any) -> str:
    if isinstance(value, bool):
        return "1.0" if value else "0.0"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value is None:
        return "n/a"
    return str(value)


def _json_pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                parts.append(f"{key}: {_plain_text(item)}")
            else:
                parts.append(f"{key}: {item}")
        return "; ".join(parts)
    if isinstance(value, list):
        return " ".join(_plain_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #182230;
  --muted: #667085;
  --line: #d5dce8;
  --blue: #1f5d8f;
  --green: #27724d;
  --red: #a13b36;
  --amber: #8f631b;
  --violet: #6f4d9b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  background: rgba(246, 247, 249, 0.96);
  border-bottom: 1px solid var(--line);
  padding: 18px 28px 16px;
  backdrop-filter: blur(8px);
}
h1, h2, h3, h4, p { margin-top: 0; }
h1 { font-size: 24px; margin-bottom: 4px; }
h2 { font-size: 18px; margin-bottom: 12px; }
h3 { font-size: 15px; margin-bottom: 10px; }
h4 { font-size: 13px; margin-bottom: 6px; }
main { max-width: 1480px; margin: 0 auto; padding: 22px 28px 48px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 18px;
  padding: 18px;
}
.subtle { color: var(--muted); font-size: 13px; }
code, pre {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
}
pre {
  background: #111827;
  color: #f8fafc;
  border-radius: 6px;
  overflow: auto;
  padding: 12px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 440px;
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-top: 12px;
}
label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-weight: 600;
}
select, input, button {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  min-height: 34px;
  padding: 6px 9px;
}
input { width: min(360px, 72vw); }
button { cursor: pointer; font-weight: 650; }
button:hover { border-color: var(--blue); color: var(--blue); }
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  background: #fbfcfe;
}
.metric span, .final-grid span { display: block; color: var(--muted); font-size: 12px; }
.metric strong, .final-grid strong { font-size: 18px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }
th { color: #475467; font-size: 12px; background: #f8fafc; }
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}
.badges { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.chip {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 2px 9px;
  background: #fff;
  color: #344054;
  font-size: 12px;
  font-weight: 700;
}
.chip.pass { border-color: #b7dbc6; color: var(--green); background: #f0fbf4; }
.chip.fail { border-color: #e7beb8; color: var(--red); background: #fff5f3; }
.chip.dataset { border-color: #c8d8ea; color: var(--blue); background: #f1f7fd; }
.split {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.55fr);
  gap: 14px;
  margin: 14px 0;
}
.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  background: #fff;
}
.timeline { display: grid; gap: 12px; }
.turn {
  display: grid;
  grid-template-columns: 80px minmax(0, 1fr);
  gap: 12px;
}
.turn-label {
  color: var(--muted);
  font-weight: 750;
  padding-top: 9px;
}
.messages { display: grid; gap: 8px; }
.message {
  border-left: 4px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  background: #fbfcfe;
}
.message p { margin: 4px 0 0; }
.message.coordinator { border-left-color: var(--blue); }
.message.participant { border-left-color: var(--green); }
.message-meta { color: var(--muted); font-size: 12px; font-weight: 700; }
.final { border-color: #cdddf0; background: #f8fbff; }
.final-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}
.final-grid div {
  background: #fff;
  border: 1px solid #d8e5f4;
  border-radius: 6px;
  padding: 9px;
}
.rui { border-color: #d8dfca; background: #fbfdf6; }
.rui-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin-bottom: 10px;
  color: var(--muted);
}
.prompts { border-color: #ded6ec; background: #fbfaff; }
.prompt-record {
  border: 1px solid #ddd5ea;
  border-radius: 6px;
  margin-top: 8px;
  background: #fff;
}
.prompt-record summary {
  cursor: pointer;
  padding: 10px 12px;
  color: var(--violet);
  font-weight: 750;
}
.prompt-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  padding: 0 12px 12px;
}
.clean { color: var(--green); font-weight: 750; }
.check, .warn { color: var(--red); font-weight: 750; }
.empty { color: var(--muted); }
.hidden { display: none !important; }
@media (max-width: 860px) {
  header { position: static; padding: 16px; }
  main { padding: 16px; }
  .split, .turn, .prompt-grid { grid-template-columns: 1fr; }
  .section-head { display: block; }
  .badges { justify-content: flex-start; margin-bottom: 8px; }
}
"""


def _js() -> str:
    return """
const trajectoryFilter = document.getElementById('trajectoryFilter');
const datasetFilter = document.getElementById('datasetFilter');
const searchBox = document.getElementById('searchBox');
function applyFilters() {
  const trajectoryId = trajectoryFilter.value;
  const dataset = datasetFilter.value;
  const query = searchBox.value.trim().toLowerCase();
  document.querySelectorAll('.trajectory').forEach((section) => {
    const trajectoryOk = !trajectoryId || section.dataset.trajectoryId === trajectoryId;
    const datasetOk = !dataset || section.dataset.dataset === dataset;
    const queryOk = !query || section.dataset.search.includes(query);
    section.classList.toggle('hidden', !(trajectoryOk && datasetOk && queryOk));
  });
}
trajectoryFilter.addEventListener('change', applyFilters);
datasetFilter.addEventListener('change', applyFilters);
searchBox.addEventListener('input', applyFilters);
document.getElementById('expandPrompts').addEventListener('click', () => {
  document.querySelectorAll('.prompt-record').forEach((node) => { node.open = true; });
});
document.getElementById('collapsePrompts').addEventListener('click', () => {
  document.querySelectorAll('.prompt-record').forEach((node) => { node.open = false; });
});
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Run directory, trajectories directory, or one trajectory JSON.")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults to interaction_report.html in the run directory.")
    args = parser.parse_args()
    out = build_report(input_path=args.input, output_path=args.output)
    print(out)


if __name__ == "__main__":
    main()
