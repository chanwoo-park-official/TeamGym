import html
import json
import os
import re
import math
from types import SimpleNamespace

import gradio as gr
import torch
from fire import Fire


def _load_json_trajectories(trajectory_file: str) -> list:
    def _load_episode(path: str) -> list:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        task = payload.get("task", {})
        episode_id = payload.get("episode_id")
        solver_seen = False
        trajectories = []
        for traj in payload.get("trajectories", []):
            is_base = False
            if traj.get("name") == "solver" and not solver_seen:
                solver_seen = True
                is_base = True
            steps = []
            for idx, step_data in enumerate(traj.get("steps", [])):
                step_obj = SimpleNamespace(**step_data)
                if not hasattr(step_obj, "step"):
                    step_obj.step = idx
                steps.append(step_obj)
            trajectories.append(
                SimpleNamespace(
                    steps=steps,
                    reward=float(traj.get("reward", 0.0)),
                    task=task,
                    info=traj.get("info", {}),
                    name=traj.get("name"),
                    uid=traj.get("uid"),
                    episode_id=episode_id,
                    is_base=is_base,
                )
            )
        return trajectories

    if os.path.isdir(trajectory_file):
        json_files = [
            os.path.join(trajectory_file, name)
            for name in sorted(os.listdir(trajectory_file))
            if name.endswith(".json")
        ]
        all_trajs = []
        for path in json_files:
            all_trajs.extend(_load_episode(path))
        return all_trajs
    return _load_episode(trajectory_file)


def _combine_assistant_text(message: dict) -> str:
    if not message:
        return ""
    reasoning = message.get("reasoning") or ""
    content = message.get("content") or ""
    if reasoning and content:
        return f"{reasoning}\n\n{content}".strip()
    return (reasoning or content).strip()


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)


def _extract_task_info(trajs: list) -> dict:
    task = {}
    for traj in trajs:
        maybe_task = getattr(traj, "task", None)
        if isinstance(maybe_task, dict) and maybe_task:
            task = maybe_task
            break
    data_source = task.get("data_source") or task.get("dataset") or task.get("task_name") or ""
    question = (
        task.get("problem")
        or task.get("question")
        or task.get("prompt")
        or task.get("input")
        or ""
    )
    ground_truth = (
        task.get("answer")
        or task.get("solution")
        or task.get("ground_truth")
        or task.get("target")
        or ""
    )
    return {"data_source": data_source, "question": question, "ground_truth": ground_truth}


def _normalize_auditor_action(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[`\\s]", "", str(text))
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    return match.group(0).lower() if match else cleaned.lower()


def _build_auditor_response_map(trajs: list) -> dict[str, list[str]]:
    response_map: dict[str, list[str]] = {}
    for traj in trajs:
        if getattr(traj, "name", "") != "auditor":
            continue
        for step in getattr(traj, "steps", []) or []:
            action = getattr(step, "action", None)
            info = getattr(step, "info", None) or {}
            forced_action = str(info.get("forced_action") or "").strip()
            forced_action = forced_action or ""
            forced_match = info.get("forced_match")
            action_key = _normalize_auditor_action(forced_action or action or "")
            if not action_key:
                continue
            response = getattr(step, "model_response", None) or ""
            if not response:
                messages = getattr(step, "chat_completions", None) or []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        response = msg.get("content") or ""
                        break
            response = str(response).strip()
            if not response:
                continue
            header = []
            if forced_action:
                header.append(f"Forced action: {forced_action}")
            if forced_match is not None:
                header.append(f"Forced match: {bool(forced_match)}")
            header_text = "\n".join(header).strip()
            detail = response
            if header_text:
                detail = f"{header_text}\n\n{response}"
            existing = response_map.setdefault(action_key, [])
            if detail not in existing:
                existing.append(detail)
    return response_map


def _format_root_detail(task_info: dict) -> str:
    question = task_info.get("question") or ""
    data_source = task_info.get("data_source") or ""
    ground_truth = task_info.get("ground_truth")
    if not question and not data_source and not ground_truth:
        return ""
    lines = []
    if question:
        lines.append("Question:")
        lines.append(str(question))
    if data_source:
        lines.append("")
        lines.append(f"Data source: {data_source}")
    if ground_truth is not None and ground_truth != "":
        lines.append("")
        lines.append("Ground truth:")
        lines.append(str(ground_truth))
    return "\n".join(lines).strip()


def _format_tool_calls(tool_calls: list) -> str:
    lines = []
    for call in tool_calls or []:
        if isinstance(call, str):
            lines.append(call)
            continue
        if not isinstance(call, dict):
            lines.append(str(call))
            continue
        name = call.get("name")
        args = call.get("arguments")
        if name is None and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
            args = call["function"].get("arguments", args)
        name = name or "tool"
        if isinstance(args, dict):
            args_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
        elif args is None:
            args_text = ""
        else:
            args_text = str(args)
        if args_text:
            lines.append(f"{name}({args_text})")
        else:
            lines.append(f"{name}()")
    return "\n".join(lines).strip()


def _extract_tool_entries(messages: list[dict]) -> list[tuple[str, str, str]]:
    entries = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        assistant_text = _combine_assistant_text(msg)
        tool_text = _format_tool_calls(msg.get("tool_calls"))
        tool_output = ""
        for j in range(idx + 1, len(messages)):
            if messages[j].get("role") == "tool":
                tool_output = str(messages[j].get("content") or "").strip()
                break
        detail = tool_output.strip() if tool_output else ""
        next_msg = None
        for j in range(idx + 1, len(messages)):
            if messages[j].get("role") == "assistant":
                next_msg = messages[j]
                break
        assistant_after = _combine_assistant_text(next_msg or {})
        entries.append((detail, assistant_text, assistant_after))
    if entries:
        return entries
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        prev_msg = None
        for j in range(idx - 1, -1, -1):
            if messages[j].get("role") == "assistant":
                prev_msg = messages[j]
                break
        next_msg = None
        for j in range(idx + 1, len(messages)):
            if messages[j].get("role") == "assistant":
                next_msg = messages[j]
                break
        entries.append(
            (
                str(msg.get("content") or "").strip(),
                _combine_assistant_text(prev_msg or {}),
                _combine_assistant_text(next_msg or {}),
            )
        )
    return entries


def _extract_tool_paths(
    trajectory,
) -> list[tuple[list[str], list[str], list[str], list[str], list[bool | None]]]:
    if not trajectory.steps:
        return []
    def _normalize_forced_match(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "0", "no"}:
                return False
            if lowered in {"true", "1", "yes"}:
                return True
        return None
    tool_contents = []
    assistant_before = []
    assistant_after = []
    forced_actions = []
    forced_matches = []
    for step in trajectory.steps:
        messages = getattr(step, "chat_completions", None) or []
        forced_action = ""
        forced_match = None
        info = getattr(step, "info", None) or {}
        if isinstance(info, dict):
            forced_action = str(info.get("forced_action") or "").strip()
            if "forced_match" in info:
                forced_match = _normalize_forced_match(info.get("forced_match"))
        for detail, before_text, after_text in _extract_tool_entries(messages):
            if detail:
                tool_contents.append(detail)
            else:
                tool_contents.append("(no tool detail)")
            assistant_before.append(before_text)
            assistant_after.append(after_text)
            forced_actions.append(forced_action)
            forced_matches.append(forced_match)
    if not tool_contents:
        return []
    return [(tool_contents, assistant_before, assistant_after, forced_actions, forced_matches)]


def _build_audit_tree(trajs: list) -> tuple[list[dict], list[dict]]:
    node_id = 0
    root_id = "root"
    nodes = {root_id: {"id": root_id, "label": "ROOT", "detail": "Start"}}
    edges = {}
    child_lookup = {}
    leaf_rewards: dict[str, list[float]] = {}
    base_solver = None
    if not any(
        getattr(traj, "is_base", False)
        for traj in trajs
        if getattr(traj, "name", "") == "solver"
    ):
        for traj in trajs:
            if getattr(traj, "name", "") == "solver":
                base_solver = traj
                break

    for traj in trajs:
        if getattr(traj, "name", "") != "solver":
            continue
        is_base = bool(getattr(traj, "is_base", False)) or (traj is base_solver)
        paths = _extract_tool_paths(traj)
        for tool_contents, assistant_before, assistant_after, forced_actions, forced_matches in paths:
            parent_id = root_id
            for idx, tool_content in enumerate(tool_contents):
                solver_text = assistant_after[idx] if idx < len(assistant_after) else ""
                if not solver_text:
                    solver_text = assistant_before[idx] if idx < len(assistant_before) else ""
                solver_text = solver_text.strip()
                if not solver_text:
                    solver_text = "(no solver response)"
                forced_action = (
                    forced_actions[idx] if idx < len(forced_actions) else ""
                ).strip()
                forced_match = (
                    forced_matches[idx] if idx < len(forced_matches) else None
                )
                forced_key = forced_action.lower() if forced_action else ""
                key = (parent_id, solver_text, forced_key)
                if key in child_lookup:
                    child_id = child_lookup[key]
                else:
                    node_id += 1
                    child_id = f"node_{node_id}"
                    detail = solver_text
                    label = solver_text
                    nodes[child_id] = {
                        "id": child_id,
                        "label": label,
                        "detail": detail,
                    }
                    child_lookup[key] = child_id

                edge_detail = forced_action or tool_content.strip() or "(no auditor action)"
                edge_label = forced_action or tool_content.strip() or ""
                label_muted = forced_match is False
                edge_key = (parent_id, child_id)
                if edge_key not in edges:
                    edges[edge_key] = {
                        "id": f"edge_{len(edges) + 1}",
                        "detail": [],
                        "is_base": False,
                        "label": edge_label,
                        "label_muted": False,
                    }
                if is_base:
                    edges[edge_key]["is_base"] = True
                if label_muted:
                    edges[edge_key]["label_muted"] = True
                if edge_detail:
                    edges[edge_key]["detail"].append(edge_detail)
                if idx == len(tool_contents) - 1:
                    reward_value = getattr(traj, "reward", None)
                    if reward_value is None and getattr(traj, "steps", None):
                        reward_value = sum(step.reward for step in traj.steps)
                    if isinstance(reward_value, (int, float)):
                        leaf_rewards.setdefault(child_id, []).append(float(reward_value))
                parent_id = child_id

    node_list = list(nodes.values())
    sources = {src for (src, _dst) in edges.keys()}
    for node in node_list:
        node_id_value = node["id"]
        if node_id_value == root_id:
            continue
        if node_id_value not in sources:
            rewards = leaf_rewards.get(node_id_value, [])
            if rewards and any(val > 0 for val in rewards):
                node["leaf_status"] = "success"
            elif rewards:
                node["leaf_status"] = "failure"
    edge_list = []
    for (src, dst), edge in edges.items():
        detail = "\n\n---\n\n".join(edge["detail"]).strip() or "(no solver response captured)"
        edge_list.append(
            {
                "id": edge["id"],
                "source": src,
                "target": dst,
                "detail": detail,
                "is_base": bool(edge.get("is_base", False)),
                "label": edge.get("label", ""),
                "label_muted": bool(edge.get("label_muted", False)),
            }
        )
    return node_list, edge_list


def _layout_tree(nodes: list[dict], edges: list[dict]) -> tuple[dict, dict]:
    children = {}
    for edge in edges:
        children.setdefault(edge["source"], []).append(edge["target"])

    x_pos = {}
    y_pos = {}
    order = 0

    def _dfs(node_id: str, depth: int) -> float:
        nonlocal order
        kids = children.get(node_id, [])
        if not kids:
            order += 1
            x_pos[node_id] = order
            y_pos[node_id] = depth
            return x_pos[node_id]
        child_positions = []
        for kid in kids:
            child_positions.append(_dfs(kid, depth + 1))
        x_pos[node_id] = sum(child_positions) / len(child_positions)
        y_pos[node_id] = depth
        return x_pos[node_id]

    _dfs("root", 0)
    return x_pos, y_pos


def _build_tree_html(trajs: list) -> str:
    nodes, edges = _build_audit_tree(trajs)
    if len(nodes) <= 1:
        return "<div>No auditor tool responses found in solver trajectories.</div>"

    x_pos, y_pos = _layout_tree(nodes, edges)
    x_vals = list(x_pos.values())
    y_vals = list(y_pos.values())
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    x_gap = 160
    y_gap = 120
    width = int((max_x - min_x + 2) * x_gap)
    height = int((max_y - min_y + 2) * y_gap)

    def _node_xy(node_id: str) -> tuple[int, int]:
        x = int((x_pos[node_id] - min_x + 1) * x_gap)
        y = int((y_pos[node_id] - min_y + 1) * y_gap)
        return x, y

    node_details = {n["id"]: n["detail"] for n in nodes}
    edge_details = {e["id"]: e["detail"] for e in edges}
    auditor_map = _build_auditor_response_map(trajs)
    if auditor_map:
        for edge_id, detail in list(edge_details.items()):
            parts = [p.strip() for p in detail.split("\n\n---\n\n") if p.strip()]
            expanded = []
            for part in parts:
                key = _normalize_auditor_action(part)
                if key and key in auditor_map:
                    expanded.append("\n\n---\n\n".join(auditor_map[key]))
                else:
                    expanded.append(part)
            edge_details[edge_id] = "\n\n---\n\n".join(expanded).strip() or detail
        for edge in edges:
            detail = edge_details.get(edge["id"], "")
            if "forced match: false" in detail.lower():
                edge["label_muted"] = True
    task_info = _extract_task_info(trajs)
    root_detail = _format_root_detail(task_info)
    if "root" in node_details and root_detail:
        node_details["root"] = root_detail
    node_details_with_tokens = {}
    for node_id, detail in node_details.items():
        if node_id == "root":
            node_details_with_tokens[node_id] = detail
            continue
        count = _estimate_token_count(detail)
        node_details_with_tokens[node_id] = f"Token estimate: {count}\n\n{detail}"
    edge_details_with_tokens = {}
    for edge_id, detail in edge_details.items():
        count = _estimate_token_count(detail)
        edge_details_with_tokens[edge_id] = f"Token estimate: {count}\n\n{detail}"
    node_details_json = json.dumps(node_details_with_tokens).replace("</", "<\\/")
    edge_details_json = json.dumps(edge_details_with_tokens).replace("</", "<\\/")

    edge_divs = []
    edge_label_divs = []
    for edge in edges:
        x1, y1 = _node_xy(edge["source"])
        x2, y2 = _node_xy(edge["target"])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        stroke = "#ef4444" if edge.get("is_base") else "#334155"
        thickness = 12 if edge.get("is_base") else 10
        edge_divs.append(
            f'<div class="tree-edge" data-edge-id="{edge["id"]}" '
            f'onclick="showEdge(\'{edge["id"]}\')" '
            f'style="left:{x1}px; top:{y1}px; width:{length}px; '
            f'height:{thickness}px; background:{stroke}; '
            f'transform:rotate({angle}deg);"'
            f'></div>'
        )
        label_text = str(edge.get("label") or "").strip()
        if label_text:
            label_clean = re.sub(r"\s+", " ", label_text)
            if len(label_clean) > 28:
                label_clean = label_clean[:28].rstrip() + "..."
            label_clean = html.escape(label_clean)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label_class = "tree-edge-label"
            if edge.get("label_muted"):
                label_class += " tree-edge-label-muted"
            edge_label_divs.append(
                f'<div class="{label_class}" '
                f'style="left:{mid_x}px; top:{mid_y}px;">{label_clean}</div>'
            )

    node_divs = []
    for node in nodes:
        x, y = _node_xy(node["id"])
        if node["id"] == "root":
            short_label = "ROOT"
        else:
            label = node.get("detail") or node.get("label") or ""
            label_clean = re.sub(r"\s+", " ", label).strip()
            words = label_clean.split()
            if len(words) > 6:
                label_clean = " ".join(words[:6]) + "..."
            short_label = html.escape(label_clean or "(no solver response)")
        node_class = "tree-node"
        if node.get("leaf_status") == "success":
            node_class += " tree-node-success"
        elif node.get("leaf_status") == "failure":
            node_class += " tree-node-failure"
        node_divs.append(
            f'<div class="{node_class}" data-node-id="{node["id"]}" '
            f'onclick="showNode(\'{node["id"]}\')" '
            f'style="left:{x - 24}px; top:{y - 24}px;">'
            f'<div class="tree-node-circle"></div>'
            f'<div class="tree-node-label">{short_label}</div>'
            f'</div>'
        )

    raw_html = f"""
    <style>
      .tree-wrap {{
        display: flex;
        gap: 16px;
      }}
      .tree-debug {{
        font-size: 12px;
        color: #334155;
        margin: 4px 0 8px;
      }}
      .tree-canvas {{
        flex: 2;
        overflow: auto;
        border: 1px solid #94a3b8;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: inset 0 0 0 1px #e2e8f0;
        position: relative;
      }}
      .tree-canvas-inner {{
        position: relative;
        width: {width}px;
        height: {height}px;
      }}
      .tree-edge {{
        position: absolute;
        transform-origin: 0 50%;
        cursor: pointer;
        pointer-events: auto;
        z-index: 1;
      }}
      .tree-edge-label {{
        position: absolute;
        transform: translate(-50%, -50%);
        font-size: 11px;
        color: #0f172a;
        background: rgba(248, 250, 252, 0.95);
        border: 1px solid #cbd5f5;
        border-radius: 999px;
        padding: 2px 8px;
        pointer-events: none;
        white-space: nowrap;
        z-index: 3;
      }}
      .tree-edge-label-muted {{
        color: #cbd5f5;
        background: rgba(254, 240, 138, 0.9);
        border-color: #e2e8f0;
      }}
      .tree-node-success .tree-node-circle {{
        background: #86efac;
        border-color: #16a34a;
      }}
      .tree-node-failure .tree-node-circle {{
        background: #d8b4fe;
        border-color: #7e22ce;
      }}
      .tree-node {{
        position: absolute;
        cursor: pointer;
        width: 48px;
        height: 48px;
        z-index: 2;
      }}
      .tree-node-circle {{
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #fde68a;
        border: 2px solid #b45309;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #0f172a;
      }}
      .tree-node-label {{
        position: absolute;
        left: 50%;
        top: 54px;
        transform: translateX(-50%);
        font-size: 12px;
        color: #0f172a;
        white-space: nowrap;
        pointer-events: none;
      }}
      .tree-detail {{
        flex: 1;
        border: 1px solid #cbd5f5;
        border-radius: 10px;
        padding: 12px;
        background: #0f172a;
        box-shadow: inset 0 0 0 1px #e2e8f0;
      }}
      .tree-detail h3 {{
        margin: 0 0 8px 0;
        font-size: 16px;
        color: #f8fafc;
      }}
      .tree-detail pre {{
        white-space: pre-wrap;
        font-size: 13px;
        line-height: 1.4;
        color: #f8fafc;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px;
        max-height: 460px;
        overflow: auto;
      }}
      .tree-legend {{
        font-size: 12px;
        color: #cbd5f5;
        margin-bottom: 8px;
      }}
    </style>
    <div class="tree-debug">Nodes: {len(nodes) - 1} | Edges: {len(edges)}</div>
    <div class="tree-wrap">
      <div class="tree-canvas">
        <div class="tree-canvas-inner">
          {''.join(edge_divs)}
          {''.join(edge_label_divs)}
          {''.join(node_divs)}
        </div>
      </div>
        <div class="tree-detail">
        <h3>Details</h3>
        <div class="tree-legend">Node = solver response. Edge = auditor action (red = base path, grey = invalid). Leaf color: blue = correct, purple = wrong.</div>
        <pre id="detailBox">Click a node or edge to view its text.</pre>
      </div>
    </div>
    <script>
      const nodeDetails = {node_details_json};
      const edgeDetails = {edge_details_json};
      function showDetail(text) {{
        var box = document.getElementById('detailBox');
        if (!box) return;
        box.textContent = text || '(no detail)';
      }}
      function showNode(nodeId) {{
        showDetail('[Solver response]\\n\\n' + (nodeDetails[nodeId] || '(no detail)'));
      }}
      function showEdge(edgeId) {{
        showDetail('[Auditor action]\\n\\n' + (edgeDetails[edgeId] || '(no detail)'));
      }}
      window.addEventListener('DOMContentLoaded', function() {{
        document.querySelectorAll('[data-node-id]').forEach(function(el) {{
          el.addEventListener('click', function() {{
            var id = el.getAttribute('data-node-id');
            showNode(id);
          }});
        }});
        document.querySelectorAll('[data-edge-id]').forEach(function(el) {{
          el.addEventListener('click', function() {{
            var id = el.getAttribute('data-edge-id');
            showEdge(id);
          }});
        }});
      }});
    </script>
    """
    return raw_html


def _build_tree_iframe(raw_html: str) -> str:
    srcdoc = html.escape(raw_html, quote=True)
    return (
        '<iframe style="width:100%; height:620px; border:none;" '
        'sandbox="allow-scripts allow-same-origin" '
        f'srcdoc="{srcdoc}"></iframe>'
    )


def main(trajectory_file: str = "./trajectories/sample_trajectories/search_trajectories.pt", server_port: int = 12345):
    if trajectory_file.endswith(".json") or os.path.isdir(trajectory_file):
        all_trajs = _load_json_trajectories(trajectory_file)
    else:
        trajs_data = torch.load(trajectory_file, weights_only=False)
        all_trajs = list(filter(lambda x: hasattr(x, "steps") and len(x.steps) > 0, trajs_data))
    tree_html = _build_tree_html(all_trajs)
    tree_iframe = _build_tree_iframe(tree_html)
    def _tree_component(value: str):
        return gr.HTML(
            value=value,
            apply_default_css=False,
            min_height=700,
            container=True,
        )

    def _tree_summary(trajs: list) -> str:
        nodes, edges = _build_audit_tree(trajs)
        return f"Tree summary: {len(nodes) - 1} nodes, {len(edges)} edges."

    def _write_tree_html(trajectory_file: str, html_body: str) -> str:
        if os.path.isdir(trajectory_file):
            out_dir = trajectory_file
        else:
            out_dir = os.path.dirname(os.path.abspath(trajectory_file)) or os.getcwd()
        out_path = os.path.join(out_dir, "audit_tree.html")
        doc = (
            "<!doctype html><html><head><meta charset='utf-8'></head><body>"
            f"{html_body}</body></html>"
        )
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(doc)
        return out_path

    tree_html_path = _write_tree_html(trajectory_file, tree_html)

    def filter_trajectories_by_reward(filter_option: str):
        """Filter trajectories based on reward"""
        if filter_option == "All Trajectories":
            return all_trajs
        elif filter_option == "Zero Reward (Failed)":
            return [t for t in all_trajs if float(t.reward) == 0.0]
        elif filter_option == "Nonzero Reward (Partial/Full Success)":
            return [t for t in all_trajs if float(t.reward) > 0.0]
        elif filter_option == "Perfect Score (Reward = 1)":
            return [t for t in all_trajs if float(t.reward) == 1.0]
        else:
            return all_trajs

    def extract_thinking_and_response(model_response: str) -> tuple[str, str]:
        """Extract thinking and final response from model output"""
        if not model_response:
            return "", ""

        think_match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            response = model_response.split("</think>", 1)[-1].strip()
        else:
            thinking = ""
            response = model_response.strip()

        return thinking, response

    def format_tool_call_detailed(tool_call: dict) -> str:
        """Format tool calls with detailed information"""
        if isinstance(tool_call, str):
            return tool_call

        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        args = function.get("arguments", {})
        tool_id = tool_call.get("id", "no-id")

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw_arguments": args}

        if not isinstance(args, dict):
            args = {"raw_arguments": str(args)}

        if name == "local_search":
            query = args.get("query", "No query")
            return f"🔍 **Search Query:** `{query}`\n*Tool ID: {tool_id}*"
        elif name == "finish":
            response = args.get("response", "No response")
            return f"✅ **Finish Action:**\n```\n{response}\n```\n*Tool ID: {tool_id}*"
        else:
            return f"🛠️ **Tool:** `{name}`\n**Arguments:**\n```json\n{json.dumps(args, indent=2)}\n```\n*Tool ID: {tool_id}*"

    def format_tool_outputs(tool_outputs: dict) -> str:
        """Format tool execution results"""
        if not tool_outputs:
            return "*No tool outputs*"

        formatted_outputs = []
        for tool_id, output in tool_outputs.items():
            display_output = str(output)
            if len(display_output) > 500:
                display_output = display_output[:500] + "... (truncated)"

            formatted_outputs.append(f"**Tool ID `{tool_id}`:**\n```\n{display_output}\n```")

        return "\n\n".join(formatted_outputs)

    def extract_boxed_answer(text: str) -> str:
        """Extract answer from \\boxed{} format"""
        if not text:
            return ""

        match = re.search(r"\\boxed\{([^}]*)\}", text)
        if match:
            return match.group(1)
        return ""

    def get_trajectory_metadata(trajectory) -> dict:
        """Extract metadata from trajectory - supports both observation and info fields"""
        if not trajectory.steps:
            return {}

        first_step = trajectory.steps[0]

        # Try multiple sources for metadata in order of preference
        if isinstance(first_step.observation, dict):
            return first_step.observation
        elif hasattr(first_step, "info") and isinstance(first_step.info, dict):
            return first_step.info
        elif hasattr(trajectory, "task") and isinstance(trajectory.task, dict):
            return trajectory.task
        else:
            return {}

    def detect_response_structure(step):
        """Detect if this step uses thinking structure or direct response structure - updated for new API"""
        # Check for thinking format (separate thought and action fields)
        has_thinking = hasattr(step, "thought") and step.thought and hasattr(step, "action") and step.action

        # Check for direct response format (model_response and action)
        has_direct_response = hasattr(step, "model_response") and step.model_response and hasattr(step, "action") and step.action

        # Updated logic for new API pattern
        uses_thinking_format = has_thinking and step.thought.strip()

        return {"has_thinking": has_thinking, "has_direct_response": has_direct_response, "uses_thinking_format": uses_thinking_format, "response_field": "thought" if uses_thinking_format else "model_response"}

    def get_task_type(metadata):
        """Detect task type from metadata"""
        if "data_source" in metadata:
            if metadata["data_source"] in ["hotpotqa", "bamboogle", "musique"]:
                return "search"
            elif metadata["data_source"] in ["deepcoder", "livecodebench", "code_generation"]:
                return "code"
            elif metadata["data_source"] in ["math", "gsm8k", "math_word_problems"]:
                return "math"

        if "question_type" in metadata or "level" in metadata:
            return "search"
        elif "difficulty" in metadata or "contest_id" in metadata or "platform" in metadata:
            return "code"
        elif "problem_type" in metadata or "solution_type" in metadata:
            return "math"

        return "unknown"

    def advance_step_or_trajectory(current_traj_idx_val, current_step_idx_val, direction, level, filtered_trajs):
        current_traj_idx_val = int(current_traj_idx_val)
        current_step_idx_val = int(current_step_idx_val)
        num_filtered_trajectories = len(filtered_trajs)

        if num_filtered_trajectories == 0:
            return 0, 0

        if level == "step":
            num_steps_current_traj = len(filtered_trajs[current_traj_idx_val].steps)
            if direction == "next":
                next_step_idx = current_step_idx_val + 1
                next_traj_idx = current_traj_idx_val
                if next_step_idx >= num_steps_current_traj:
                    next_step_idx = 0
            else:  # prev
                next_step_idx = current_step_idx_val - 1
                next_traj_idx = current_traj_idx_val
                if next_step_idx < 0:
                    next_step_idx = num_steps_current_traj - 1 if num_steps_current_traj > 0 else 0
        else:  # trajectory
            if direction == "next":
                next_traj_idx = current_traj_idx_val + 1
                if next_traj_idx >= num_filtered_trajectories:
                    next_traj_idx = 0
            else:  # prev
                next_traj_idx = current_traj_idx_val - 1
                if next_traj_idx < 0:
                    next_traj_idx = num_filtered_trajectories - 1 if num_filtered_trajectories > 0 else 0
            next_step_idx = 0

        return next_traj_idx, next_step_idx

    def update_step_view(traj_idx: int, step_idx: int, filter_option: str):
        """Update the step view with detailed information - updated for new API pattern"""
        empty_content = "*No data available*"

        filtered_trajs = filter_trajectories_by_reward(filter_option)

        if traj_idx >= len(filtered_trajs):
            return (empty_content,) * 10

        trajectory = filtered_trajs[traj_idx]
        num_steps = len(trajectory.steps)

        position_text = f"**Trajectory {traj_idx + 1}/{len(filtered_trajs)}**  |  **Step {step_idx + 1}/{num_steps}**"
        metadata = get_trajectory_metadata(trajectory)
        task_type = get_task_type(metadata)

        if task_type == "search":
            question = metadata.get("question", "No question found")
            gt = metadata.get("ground_truth", "N/A")
            ground_truth = str(gt).lower() if isinstance(gt, str | int | float | bool) else str(gt)
        elif task_type == "code":
            question = metadata.get("question_content", metadata.get("question", metadata.get("problem", "No problem statement found")))
            test_cases = metadata.get("test_cases", [])
            if test_cases and len(test_cases) > 0:
                expected_output = test_cases[0].get("expected_output", "N/A")
                ground_truth = str(expected_output)
            else:
                ground_truth = "See test cases"
        elif task_type == "math":
            question = metadata.get("problem", metadata.get("question", "No problem found"))
            ground_truth = str(metadata.get("answer", metadata.get("solution", "N/A")))
        else:
            question = str(trajectory.steps[0].observation) if num_steps > 0 else "No question"
            ground_truth = "Unknown"

        task_icons = {"search": "🔍", "code": "💻", "math": "🧮", "unknown": "❓"}
        task_names = {"search": "Search", "code": "Code", "math": "Math", "unknown": "Unknown"}

        metadata_text = f"**Task Type:** {task_icons[task_type]} {task_names[task_type]}\n"
        metadata_text += f"**Data Source:** {metadata.get('data_source', 'N/A')}\n"

        if task_type == "search":
            metadata_text += f"**Question Type:** {metadata.get('question_type', 'N/A')}\n"
            metadata_text += f"**Level:** {metadata.get('level', 'N/A')}\n"
        elif task_type == "code":
            metadata_text += f"**Difficulty:** {metadata.get('difficulty', 'N/A')}\n"
            metadata_text += f"**Platform:** {metadata.get('platform', 'N/A')}\n"
            metadata_text += f"**Contest ID:** {metadata.get('contest_id', 'N/A')}\n"
        elif task_type == "math":
            metadata_text += f"**Problem Type:** {metadata.get('problem_type', 'N/A')}\n"
            metadata_text += f"**Difficulty:** {metadata.get('difficulty', 'N/A')}\n"

        metadata_text += f"**Split:** {metadata.get('split', 'N/A')}\n"
        metadata_text += f"**UID:** `{metadata.get('uid', metadata.get('question_id', 'N/A'))}`"

        perf_text = f"**Overall Reward:** {trajectory.reward:.3f}\n"
        perf_text += f"**Total Steps:** {num_steps}\n"
        perf_text += f"**Completed:** {'✅ Yes' if (num_steps > 0 and getattr(trajectory.steps[-1], 'done', False)) else '❌ No'}"

        question_text = f"**Question:**\n{question}\n\n**Ground Truth Answer:** `{ground_truth}`"

        if num_steps == 0:
            return (position_text, metadata_text, perf_text, question_text, empty_content, empty_content, empty_content, empty_content, empty_content, empty_content)

        step = trajectory.steps[step_idx]
        structure = detect_response_structure(step)

        if structure["uses_thinking_format"]:
            thinking_text = getattr(step, "thought", "") or "*No thinking recorded*"
            response_text = getattr(step, "action", "") or "*No response recorded*"
        else:
            thinking, response = extract_thinking_and_response(getattr(step, "model_response", ""))
            thinking_text = thinking if thinking else "*No thinking recorded*"
            response_text = response if response else (getattr(step, "action", "") or "*No response recorded*")

        # Safe field access for step performance
        step_reward = getattr(step, "reward", 0.0)
        step_mc_return = getattr(step, "mc_return", 0.0)
        step_done = getattr(step, "done", False)
        step_number = getattr(step, "step", step_idx)  # Fallback to step_idx if step field doesn't exist

        step_perf_text = f"**Step Reward:** {step_reward}\n"
        step_perf_text += f"**MC Return:** {step_mc_return:.3f}\n"
        step_perf_text += f"**Done:** {'✅ Yes' if step_done else '❌ No'}\n"
        step_perf_text += f"**Step Number:** {step_number}\n"
        step_perf_text += f"**Response Structure:** {'🧠 Thinking Format' if structure['uses_thinking_format'] else '📝 Direct Response'}"

        actions_text = empty_content
        step_action = getattr(step, "action", None)
        if step_action:
            if task_type == "search" and isinstance(step_action, list):
                actions_text = "\n\n".join([format_tool_call_detailed(tc) for tc in step_action])
            elif task_type in ["code", "math"] and isinstance(step_action, str):
                actions_text = f"**Generated {'Code' if task_type == 'code' else 'Solution'}:**\n```\n{step_action}\n```"
            else:
                actions_text = format_tool_call_detailed(step_action) if isinstance(step_action, dict) else str(step_action)

        outputs_text = empty_content
        # Handle outputs - be more flexible with field access
        if task_type == "search":
            # Try multiple ways to get tool outputs for search tasks
            current_obs = getattr(step, "observation", None)
            if current_obs and isinstance(current_obs, dict):
                tool_outputs = current_obs.get("tool_outputs", {})
                if tool_outputs:
                    outputs_text = format_tool_outputs(tool_outputs)
            elif hasattr(step, "next_observation"):
                next_obs = getattr(step, "next_observation", None)
                if next_obs and isinstance(next_obs, dict):
                    tool_outputs = next_obs.get("tool_outputs", {})
                    if tool_outputs:
                        outputs_text = format_tool_outputs(tool_outputs)
            # Also check if outputs are in the current step's info
            elif hasattr(step, "info") and isinstance(step.info, dict):
                tool_outputs = step.info.get("tool_outputs", {})
                if tool_outputs:
                    outputs_text = format_tool_outputs(tool_outputs)
        elif task_type in ["code", "math"]:
            step_observation = getattr(step, "observation", None)
            if step_idx > 0 and isinstance(step_observation, dict):
                if "test_results" in step_observation:
                    outputs_text = f"**Test Results:**\n{step_observation['test_results']}"
                elif "error" in step_observation:
                    outputs_text = f"**Error:**\n{step_observation['error']}"
                elif "feedback" in step_observation:
                    outputs_text = f"**Feedback:**\n{step_observation['feedback']}"

        predicted_answer = ""
        has_finish_action = False

        if task_type == "search" and step_action:
            actions_to_check = step_action if isinstance(step_action, list) else [step_action]
            for action in actions_to_check:
                if isinstance(action, dict) and action.get("function", {}).get("name") == "finish":
                    has_finish_action = True
                    try:
                        args_str = action["function"]["arguments"]
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        finish_response = args.get("response", "")
                        predicted_answer = extract_boxed_answer(finish_response)
                        break
                    except Exception:
                        pass
        elif task_type in ["code", "math"]:
            if step_done and step_action:
                predicted_answer = step_action
                has_finish_action = True

        if has_finish_action:
            if predicted_answer:
                if task_type == "search":
                    is_correct = predicted_answer.lower().strip() == ground_truth.lower().strip()
                    final_answer_text = "**🎯 Final Answer Provided:**\n"
                    final_answer_text += f"**Predicted:** `{predicted_answer}`\n"
                    final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                    final_answer_text += f"**Correct:** {'✅ Yes' if is_correct else '❌ No'}"
                else:
                    # Code/Math tasks: show the final solution with test results
                    final_answer_text = f"**{'💻' if task_type == 'code' else '🧮'} Final {'Code' if task_type == 'code' else 'Solution'} Submitted:**\n"
                    final_answer_text += f"**Length:** {len(predicted_answer)} characters\n"
                    final_answer_text += f"**Test Status:** {'✅ Passed' if step_reward > 0 else '❌ Failed'}\n"
                    final_answer_text += f"**Expected Output:** `{ground_truth}`"
            else:
                final_answer_text = "**⚠️ Finish action found but no answer extracted:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`"
        else:
            if step_idx == num_steps - 1:
                final_answer_text = "**❌ No finish action in final step:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += "**Status:** Trajectory ended without explicit final answer"
            else:
                final_answer_text = "**⏳ No final answer yet:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += f"**Status:** Step {step_idx + 1}/{num_steps} - {'No finish action' if task_type == 'search' else 'Continuing...'}"

        return (position_text, metadata_text, perf_text, question_text, thinking_text, response_text, step_perf_text, actions_text, outputs_text, final_answer_text)

    custom_css = """
    .trajectory-container { margin-bottom: 20px !important; }

    .metadata-box { 
        background-color: #f8f9fa !important; 
        color: #2d3748 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #dee2e6 !important; 
    }
    .performance-box { 
        background-color: #e8f5e8 !important; 
        color: #2d5016 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #c3e6c3 !important; 
    }
    .thinking-box { 
        background-color: #fff3cd !important; 
        color: #856404 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #ffeaa7 !important; 
    }
    .actions-box { 
        background-color: #e2e3f5 !important; 
        color: #3c366b !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #b8bdf0 !important; 
    }
    .outputs-box { 
        background-color: #f0f8ff !important; 
        color: #1e3a8a !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #b3d9ff !important; 
    }
    
    .dark .metadata-box { 
        background-color: #2d3748 !important; 
        color: #e2e8f0 !important; 
        border-color: #4a5568 !important; 
    }
    .dark .performance-box { 
        background-color: #2f4f2f !important; 
        color: #c6f6d5 !important; 
        border-color: #48bb78 !important; 
    }
    .dark .thinking-box { 
        background-color: #7c6f47 !important; 
        color: #fef5e7 !important; 
        border-color: #d69e2e !important; 
    }
    .dark .actions-box { 
        background-color: #3c366b !important; 
        color: #e9e7fd !important; 
        border-color: #805ad5 !important; 
    }
    .dark .outputs-box { 
        background-color: #1e3a8a !important; 
        color: #dbeafe !important; 
        border-color: #3b82f6 !important; 
    }
    
    .gr-textbox textarea { 
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important; 
        color: #2d3748 !important;
    }
    .dark .gr-textbox textarea { 
        color: #e2e8f0 !important; 
    }
    
    .step-display-box textarea { 
        text-align: center !important; 
        font-weight: bold !important; 
        font-size: 1.2em !important; 
        background-color: #ebf8ff !important; 
        color: #2b6cb0 !important; 
        border: 2px solid #bee3f8 !important; 
    }
    .dark .step-display-box textarea { 
        background-color: #1e3a8a !important; 
        color: #bfdbfe !important; 
        border-color: #3b82f6 !important; 
    }
    
    .nav-button { min-width: 120px !important; }
    .gr-button { 
        font-size: 1.1em !important; 
        padding: 0.5em 0.8em !important; 
        border-radius: 8px !important;
        background-color: #3182ce !important;
        color: white !important;
    }
    .gr-button:hover { 
        background-color: #2c5aa0 !important; 
    }
    
    .metadata-box p, .metadata-box h1, .metadata-box h2, .metadata-box h3, .metadata-box h4,
    .performance-box p, .performance-box h1, .performance-box h2, .performance-box h3, .performance-box h4,
    .actions-box p, .actions-box h1, .actions-box h2, .actions-box h3, .actions-box h4,
    .outputs-box p, .outputs-box h1, .outputs-box h2, .outputs-box h3, .outputs-box h4 {
        color: inherit !important;
    }
    
    .metadata-box code, .performance-box code, .actions-box code, .outputs-box code {
        background-color: rgba(0,0,0,0.1) !important;
        color: inherit !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    
    .dark .metadata-box code, .dark .performance-box code, .dark .actions-box code, .dark .outputs-box code {
        background-color: rgba(255,255,255,0.1) !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Agent Trajectory Visualizer") as interface:
        gr.Markdown("# 🔍 Agent Trajectory Visualizer")

        current_traj_idx_state = gr.State(0)
        current_step_idx_state = gr.State(0)

        with gr.Tab("Step View"):
            with gr.Row():
                with gr.Column(scale=1):
                    filter_dropdown = gr.Dropdown(choices=["All Trajectories", "Zero Reward (Failed)", "Nonzero Reward (Partial/Full Success)", "Perfect Score (Reward = 1)"], value="All Trajectories", label="🎯 Filter by Reward", interactive=True)
                with gr.Column(scale=1):
                    zero_count = len([t for t in all_trajs if float(t.reward) == 0.0])
                    nonzero_count = len([t for t in all_trajs if float(t.reward) > 0.0])
                    perfect_count = len([t for t in all_trajs if float(t.reward) == 1.0])

                    _ = gr.Markdown(f"**Dataset Stats:**\n- Total: {len(all_trajs)} trajectories\n- Failed (0): {zero_count}\n- Partial/Full Success (>0): {nonzero_count}\n- Perfect Score (=1): {perfect_count}")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        prev_traj_button = gr.Button("⬅️ Previous Trajectory", elem_classes=["nav-button"])
                        next_traj_button = gr.Button("Next Trajectory ➡️", elem_classes=["nav-button"])
                    with gr.Row():
                        prev_step_button = gr.Button("⬅️ Previous Step", elem_classes=["nav-button"])
                        next_step_button = gr.Button("Next Step ➡️", elem_classes=["nav-button"])

                with gr.Column(scale=2):
                    current_pos_display = gr.Textbox(label="Current Position", interactive=False, elem_classes=["step-display-box"])

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("📊 Trajectory Metadata", open=True):
                        metadata_output = gr.Markdown(elem_classes=["metadata-box"])

                    with gr.Accordion("🎯 Performance", open=True):
                        performance_output = gr.Markdown(elem_classes=["performance-box"])

                    with gr.Accordion("❓ Question & Answer", open=True):
                        question_output = gr.Markdown()
                        final_answer_output = gr.Markdown()

                with gr.Column(scale=2):
                    with gr.Accordion("🧠 Agent Thinking", open=True):
                        thinking_output = gr.Textbox(label="Internal Reasoning", lines=6, interactive=False, elem_classes=["thinking-box"])

                    with gr.Accordion("💬 Agent Response", open=True):
                        response_output = gr.Textbox(label="Final Response to User", lines=4, interactive=False)

                    with gr.Accordion("📈 Step Performance", open=True):
                        step_perf_output = gr.Markdown(elem_classes=["performance-box"])

                    with gr.Accordion("🛠️ Actions Taken", open=True):
                        actions_output = gr.Markdown(elem_classes=["actions-box"])

                    with gr.Accordion("📋 Tool Results", open=True):
                        outputs_output = gr.Markdown(elem_classes=["outputs-box"])

            all_outputs = [current_pos_display, metadata_output, performance_output, question_output, thinking_output, response_output, step_perf_output, actions_output, outputs_output, final_answer_output]

            def reset_to_first_trajectory():
                return 0, 0

            prev_traj_button.click(fn=lambda t, s, f: advance_step_or_trajectory(t, s, "prev", "trajectory", filter_trajectories_by_reward(f)), inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=[current_traj_idx_state, current_step_idx_state])
            next_traj_button.click(fn=lambda t, s, f: advance_step_or_trajectory(t, s, "next", "trajectory", filter_trajectories_by_reward(f)), inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=[current_traj_idx_state, current_step_idx_state])
            prev_step_button.click(fn=lambda t, s, f: advance_step_or_trajectory(t, s, "prev", "step", filter_trajectories_by_reward(f)), inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=[current_traj_idx_state, current_step_idx_state])
            next_step_button.click(fn=lambda t, s, f: advance_step_or_trajectory(t, s, "next", "step", filter_trajectories_by_reward(f)), inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=[current_traj_idx_state, current_step_idx_state])

            filter_dropdown.change(fn=reset_to_first_trajectory, outputs=[current_traj_idx_state, current_step_idx_state])

            current_traj_idx_state.change(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)
            current_step_idx_state.change(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)
            filter_dropdown.change(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)

            interface.load(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)

        with gr.Tab("Audit Tree"):
            gr.Markdown("Click a node to see the solver response; click an edge to see the auditor action that led to the next node.")
            gr.Markdown(_tree_summary(all_trajs))
            gr.Markdown(f"Interactive HTML written to `{tree_html_path}`.")
            _tree_component(tree_iframe)
            # Static fallback UI removed per user request.

    interface.launch(share=True, server_port=server_port)


if __name__ == "__main__":
    Fire(main)
