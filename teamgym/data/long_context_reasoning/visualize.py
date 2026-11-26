"""Render long-context PIU ownership and source/context visualization."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import estimate_tokens
from teamgym.core.dataset import iter_common_datapoints


PALETTE = [
    "#fff2a8",
    "#c8f7dc",
    "#cfe8ff",
    "#ffd6d6",
    "#ead7ff",
    "#d8f4f7",
    "#ffe1b8",
    "#e0e6a8",
]


def render_visualization(
    *,
    input_root: str | Path,
    output_html: str | Path,
    max_datapoints: int | None = None,
    source_window_chars: int = 1800,
    context_window_chars: int = 1800,
) -> dict[str, Any]:
    datapoints = list(iter_common_datapoints(input_root))
    if max_datapoints is not None:
        datapoints = datapoints[: max(0, int(max_datapoints))]
    output = Path(output_html).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    body = "\n".join(
        render_datapoint(
            datapoint,
            source_window_chars=source_window_chars,
            context_window_chars=context_window_chars,
            index=index,
        )
        for index, datapoint in enumerate(datapoints, start=1)
    )
    summary = summarize(datapoints)
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Long-Context RUI Visualization</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f7f4; color: #222; }}
    header {{ padding: 28px 34px 20px; background: #263238; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 4px 0; color: #d8e3e7; }}
    main {{ padding: 24px 34px 44px; max-width: 1400px; margin: 0 auto; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-bottom: 24px; }}
    .metric {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px 14px; }}
    .metric .label {{ color: #5f6b70; font-size: 12px; text-transform: uppercase; }}
    .metric .value {{ font-size: 22px; font-weight: 650; margin-top: 4px; }}
    details.datapoint {{ background: white; border: 1px solid #d8d8d2; border-radius: 8px; margin: 18px 0; overflow: hidden; }}
    details.datapoint > summary {{ cursor: pointer; padding: 16px 18px; background: #fbfbf8; font-weight: 650; }}
    .dp-body {{ padding: 18px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fff; }}
    .panel h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .small {{ color: #687276; font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #f1f2ed; text-align: left; }}
    .ownership-matrix td {{ text-align: center; }}
    .ownership-matrix td:first-child, .ownership-matrix th:first-child {{ text-align: left; }}
    .own-cell {{ font-weight: 700; border-radius: 4px; }}
    code {{ background: #f0f1ed; border-radius: 4px; padding: 1px 4px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #fbfbf8; border: 1px solid #ddd; border-radius: 8px; padding: 12px; max-height: 360px; overflow: auto; line-height: 1.45; font-size: 13px; }}
    mark {{ border-radius: 3px; padding: 1px 2px; }}
    .rui-badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; margin: 1px 3px 1px 0; font-size: 12px; border: 1px solid rgba(0,0,0,.15); }}
    .participant {{ margin: 14px 0; border-top: 1px solid #e3e3dc; padding-top: 14px; }}
    .two-col {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
    @media (max-width: 980px) {{ .two-col {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Long-Context RUI Visualization</h1>
    <p>Input: <code>{escape(str(input_root))}</code></p>
    <p>Shows original source snippets, reference information units, ownership, and participant private-context snippets.</p>
  </header>
  <main>
    {render_summary(summary)}
    {body}
  </main>
</body>
</html>
"""
    output.write_text(html_text, encoding="utf-8")
    return {**summary, "output_html": str(output)}


def render_datapoint(
    datapoint: dict[str, Any],
    *,
    source_window_chars: int,
    context_window_chars: int,
    index: int,
) -> str:
    instance_id = str(datapoint.get("instance_id") or "")
    oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
    source_text = load_source_text(datapoint)
    ruis = datapoint.get("reference_information_units", []) or []
    rui_by_id = {str(row.get("rui_id")): row for row in ruis if isinstance(row, dict)}
    required_ids = [str(item) for item in oracle.get("required_rui_ids", []) if str(item)]
    if not required_ids:
        required_ids = sorted(rui_by_id)
    colors = {rui_id: PALETTE[offset % len(PALETTE)] for offset, rui_id in enumerate(required_ids)}
    participant_map = participant_ownership(datapoint)
    source_status = source_presence(source_text, rui_by_id, required_ids)
    rui_rows = "\n".join(
        render_rui_row(
            rui_id,
            rui_by_id.get(rui_id, {}),
            colors=colors,
            owners=participant_map.get(rui_id, []),
            source_state=source_status.get(rui_id, "unknown"),
        )
        for rui_id in required_ids
    )
    ownership_matrix = render_ownership_matrix(
        datapoint=datapoint,
        required_ids=required_ids,
        participant_map=participant_map,
        colors=colors,
    )
    source_panels = "\n".join(
        render_source_window(
            rui_id,
            rui_by_id.get(rui_id, {}),
            source_text=source_text,
            color=colors.get(rui_id, "#fff2a8"),
            window_chars=source_window_chars,
        )
        for rui_id in required_ids
    )
    participant_panels = "\n".join(
        render_participant(
            datapoint,
            participant,
            rui_by_id=rui_by_id,
            colors=colors,
            context_window_chars=context_window_chars,
        )
        for participant in datapoint.get("participants", []) or []
        if isinstance(participant, dict)
    )
    first_open = " open" if index == 1 else ""
    return f"""
<details class="datapoint"{first_open}>
  <summary>{index}. {escape(instance_id)} | answer: <code>{escape(str(oracle.get("answer") or ""))}</code> | RUIs: {len(required_ids)}</summary>
  <div class="dp-body">
    <div class="meta-grid">
      <div class="panel">
        <h3>Task</h3>
        <p>{escape(str((datapoint.get("task_description") or {}).get("question") or ""))}</p>
        <p class="small">Short answer: {escape(str(oracle.get("short_answer") or ""))}</p>
      </div>
      <div class="panel">
        <h3>Source</h3>
        <p><code>{escape(str((datapoint.get("source") or {}).get("source_path") or ""))}</code></p>
        <p class="small">Loaded source tokens: {estimate_tokens(source_text) if source_text else 0}</p>
      </div>
    </div>
    <div class="panel">
      <h3>Required RUIs</h3>
      <table>
        <thead><tr><th>RUI</th><th>Type</th><th>Owners</th><th>Original Source</th><th>Claim</th><th>Text</th></tr></thead>
        <tbody>{rui_rows}</tbody>
      </table>
    </div>
    <div class="panel">
      <h3>Participant RUI Ownership Matrix</h3>
      {ownership_matrix}
    </div>
    <div class="two-col">
      <div class="panel">
        <h3>Original Source Windows</h3>
        {source_panels}
      </div>
      <div class="panel">
        <h3>Participant Private Contexts</h3>
        {participant_panels}
      </div>
    </div>
  </div>
</details>
"""


def render_ownership_matrix(
    *,
    datapoint: dict[str, Any],
    required_ids: list[str],
    participant_map: dict[str, list[str]],
    colors: dict[str, str],
) -> str:
    participants = [
        str(row.get("participant_id") or "")
        for row in datapoint.get("participants", []) or []
        if isinstance(row, dict) and row.get("participant_id")
    ]
    header_cells = "".join(
        f'<th><span class="rui-badge" style="background:{colors.get(rui_id, "#eee")}">{escape(rui_id)}</span></th>'
        for rui_id in required_ids
    )
    body_rows = []
    for participant_id in participants:
        cells = []
        for rui_id in required_ids:
            owns = participant_id in set(participant_map.get(rui_id, []))
            color = colors.get(rui_id, "#eee") if owns else "#fff"
            cells.append(
                f'<td class="{"own-cell" if owns else ""}" style="background:{color}">{"yes" if owns else ""}</td>'
            )
        body_rows.append(f"<tr><th>{escape(participant_id)}</th>{''.join(cells)}</tr>")
    return f"""
<table class="ownership-matrix">
  <thead><tr><th>Participant</th>{header_cells}</tr></thead>
  <tbody>{''.join(body_rows)}</tbody>
</table>
"""


def render_rui_row(
    rui_id: str,
    rui: dict[str, Any],
    *,
    colors: dict[str, str],
    owners: list[str],
    source_state: str,
) -> str:
    color = colors.get(rui_id, "#fff2a8")
    owners_text = ", ".join(owners) if owners else "common/public or unowned"
    return f"""
<tr>
  <td><span class="rui-badge" style="background:{color}">{escape(rui_id)}</span></td>
  <td>{escape(str(rui.get("rui_type") or ""))}</td>
  <td>{escape(owners_text)}</td>
  <td>{escape(source_state)}</td>
  <td>{escape(str(rui.get("summary") or rui.get("information_feature") or ""))}</td>
  <td>{escape(shorten(str(rui.get("text") or ""), 600))}</td>
</tr>
"""


def render_source_window(
    rui_id: str,
    rui: dict[str, Any],
    *,
    source_text: str,
    color: str,
    window_chars: int,
) -> str:
    text = str(rui.get("text") or "")
    if not source_text or not text or text not in source_text:
        return f"""
<details>
  <summary><span class="rui-badge" style="background:{color}">{escape(rui_id)}</span> not found verbatim in original source</summary>
  <pre>{escape(shorten(text, window_chars))}</pre>
</details>
"""
    idx = source_text.find(text)
    start = max(0, idx - window_chars // 2)
    end = min(len(source_text), idx + len(text) + window_chars // 2)
    prefix = "[...]\n" if start > 0 else ""
    suffix = "\n[...]" if end < len(source_text) else ""
    before = source_text[start:idx]
    after = source_text[idx + len(text) : end]
    snippet = (
        escape(prefix + before)
        + f'<mark style="background:{color}">{escape(text)}</mark>'
        + escape(after + suffix)
    )
    return f"""
<details>
  <summary><span class="rui-badge" style="background:{color}">{escape(rui_id)}</span> original source window</summary>
  <pre>{snippet}</pre>
</details>
"""


def render_participant(
    datapoint: dict[str, Any],
    participant: dict[str, Any],
    *,
    rui_by_id: dict[str, dict[str, Any]],
    colors: dict[str, str],
    context_window_chars: int,
) -> str:
    participant_id = str(participant.get("participant_id") or "")
    owned = [str(item) for item in participant.get("owned_rui_ids", []) if str(item)]
    badges = " ".join(
        f'<span class="rui-badge" style="background:{colors.get(rui_id, "#eee")}">{escape(rui_id)}</span>'
        for rui_id in owned
    ) or '<span class="small">No owned required RUI</span>'
    contexts = context_units_for(datapoint, participant_id)
    context_text = "\n\n".join(str(unit.get("text") or "") for unit in contexts)
    windows = []
    for rui_id in owned:
        rui_text = str((rui_by_id.get(rui_id) or {}).get("text") or "")
        windows.append(
            render_context_window(
                rui_id,
                context_text=context_text,
                rui_text=rui_text,
                color=colors.get(rui_id, "#fff2a8"),
                window_chars=context_window_chars,
            )
        )
    if not windows:
        windows.append(f"<pre>{escape(shorten(context_text, context_window_chars))}</pre>")
    return f"""
<div class="participant">
  <h3>{escape(participant_id)}</h3>
  <p>{badges}</p>
  <p class="small">Context tokens: {estimate_tokens(context_text) if context_text else 0}</p>
  {''.join(windows)}
</div>
"""


def render_context_window(
    rui_id: str,
    *,
    context_text: str,
    rui_text: str,
    color: str,
    window_chars: int,
) -> str:
    if not rui_text or rui_text not in context_text:
        return f"""
<details open>
  <summary><span class="rui-badge" style="background:{color}">{escape(rui_id)}</span> not found verbatim in participant context</summary>
  <pre>{escape(shorten(context_text, window_chars))}</pre>
</details>
"""
    idx = context_text.find(rui_text)
    start = max(0, idx - window_chars // 2)
    end = min(len(context_text), idx + len(rui_text) + window_chars // 2)
    prefix = "[...]\n" if start > 0 else ""
    suffix = "\n[...]" if end < len(context_text) else ""
    before = context_text[start:idx]
    after = context_text[idx + len(rui_text) : end]
    snippet = (
        escape(prefix + before)
        + f'<mark style="background:{color}">{escape(rui_text)}</mark>'
        + escape(after + suffix)
    )
    return f"""
<details open>
  <summary><span class="rui-badge" style="background:{color}">{escape(rui_id)}</span> participant context window</summary>
  <pre>{snippet}</pre>
</details>
"""


def context_units_for(datapoint: dict[str, Any], participant_id: str) -> list[dict[str, Any]]:
    units_by_id = {
        str(unit.get("context_unit_id")): unit
        for unit in datapoint.get("private_context_units", []) or []
        if isinstance(unit, dict) and unit.get("context_unit_id")
    }
    participant = next(
        (
            row
            for row in datapoint.get("participants", []) or []
            if isinstance(row, dict) and str(row.get("participant_id") or "") == participant_id
        ),
        {},
    )
    refs = [str(item) for item in participant.get("private_context_unit_ids", []) or []]
    if refs:
        return [units_by_id[ref] for ref in refs if ref in units_by_id]
    return [
        unit
        for unit in datapoint.get("private_context_units", []) or []
        if isinstance(unit, dict) and participant_id in {str(item) for item in unit.get("assigned_to", []) or []}
    ]


def load_source_text(datapoint: dict[str, Any]) -> str:
    source = datapoint.get("source") if isinstance(datapoint.get("source"), dict) else {}
    path_text = str(source.get("source_path") or "")
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def participant_ownership(datapoint: dict[str, Any]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for participant in datapoint.get("participants", []) or []:
        if not isinstance(participant, dict):
            continue
        participant_id = str(participant.get("participant_id") or "")
        for rui_id in participant.get("owned_rui_ids", []) or []:
            owners.setdefault(str(rui_id), []).append(participant_id)
    return {rui_id: sorted(participants) for rui_id, participants in owners.items()}


def source_presence(source_text: str, rui_by_id: dict[str, dict[str, Any]], rui_ids: list[str]) -> dict[str, str]:
    states = {}
    for rui_id in rui_ids:
        text = str((rui_by_id.get(rui_id) or {}).get("text") or "")
        if not text:
            states[rui_id] = "missing text"
        elif source_text and text in source_text:
            states[rui_id] = "verbatim source span"
        else:
            states[rui_id] = "not found verbatim in source file"
    return states


def summarize(datapoints: list[dict[str, Any]]) -> dict[str, Any]:
    answers = Counter(str(((dp.get("oracle") or {}).get("answer")) or "") for dp in datapoints)
    participant_counts = [len(dp.get("participants", []) or []) for dp in datapoints]
    rui_counts = [len(((dp.get("oracle") or {}).get("required_rui_ids")) or []) for dp in datapoints]
    return {
        "datapoints": len(datapoints),
        "answers": dict(answers),
        "participant_count_min": min(participant_counts) if participant_counts else 0,
        "participant_count_max": max(participant_counts) if participant_counts else 0,
        "required_rui_count_min": min(rui_counts) if rui_counts else 0,
        "required_rui_count_max": max(rui_counts) if rui_counts else 0,
    }


def render_summary(summary: dict[str, Any]) -> str:
    metrics = [
        ("Datapoints", summary["datapoints"]),
        ("Answers", json.dumps(summary["answers"], sort_keys=True)),
        ("Participants", f"{summary['participant_count_min']}-{summary['participant_count_max']}"),
        ("Required RUIs", f"{summary['required_rui_count_min']}-{summary['required_rui_count_max']}"),
    ]
    return '<section class="summary">' + "\n".join(
        f'<div class="metric"><div class="label">{escape(label)}</div><div class="value">{escape(str(value))}</div></div>'
        for label, value in metrics
    ) + "</section>"


def shorten(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    half = max(1, max_chars // 2)
    return text[:half].rstrip() + "\n[... omitted ...]\n" + text[-half:].lstrip()


def escape(value: str) -> str:
    return html.escape(str(value), quote=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--max-datapoints", type=int)
    parser.add_argument("--source-window-chars", type=int, default=1800)
    parser.add_argument("--context-window-chars", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_visualization(
        input_root=args.input_root,
        output_html=args.output_html,
        max_datapoints=args.max_datapoints,
        source_window_chars=args.source_window_chars,
        context_window_chars=args.context_window_chars,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
