"""
python rllm/trajectory_visualizer.py /home/cpark97/rllm_prev_version/rllm/run_debug/20260125/solver_auditor_training/213906/trajectory/episodes/train_step_1_epoch_0/episode_hash60a84e88_id7b25e959-bd6a-445e-9c65-09c781632ccb_1.json

python rllm/webshop_trajectory_visualizer.py /home/cpark97/rllm_prev_version/rllm/run_debug/20260203/solver_auditor_training_webshop/133541/trajectory/episodes/train_step_1_epoch_0/episode_hash0d1a9977_id365c97a2-4ee4-4234-b534-cd1a1950c07e_0.json

First one is for the "math / qa type" trajectories, second one is for the "webshop / environment side" trajectories.
"""

import html
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from fire import Fire

def _load_json_trajectories(trajectory_file: str) -> list:
    def _load_episode(path: str) -> list:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        task = payload.get("task", {}) or {}
        episode_id = payload.get("episode_id")
        termination_reason = payload.get("termination_reason")
        termination_reason_detail = payload.get("termination_reason_detail")
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
                    termination_reason=termination_reason,
                    termination_reason_detail=termination_reason_detail,
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
        if not json_files:
            json_files = [str(p) for p in sorted(Path(trajectory_file).rglob("*.json"))]
        all_trajs = []
        for path in json_files:
            all_trajs.extend(_load_episode(path))
        return all_trajs
    return _load_episode(trajectory_file)


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)


def _last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _format_webshop_task_meta(task: object) -> list[str]:
    if not isinstance(task, dict):
        return []
    lines = []
    seed = task.get("seed")
    num_products = task.get("num_products")
    if seed is not None:
        lines.append(f"Seed: {seed}")
    if num_products is not None:
        lines.append(f"Products: {num_products}")
    return lines


def _format_webshop_observation(observation: object) -> str:
    if isinstance(observation, dict):
        text = str(observation.get("obs") or "")
    else:
        text = str(observation or "")
    text = text.strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split("[SEP]") if part.strip()]
    if len(parts) <= 1:
        return text
    lines = []
    idx = 0
    while idx < len(parts):
        part = parts[idx]
        lower = part.lower()
        if lower.startswith("instruction"):
            instruction = part.split(":", 1)[1].strip() if ":" in part else ""
            if not instruction and idx + 1 < len(parts):
                idx += 1
                instruction = parts[idx].strip()
            if instruction:
                lines.extend(["Instruction:", instruction])
            else:
                lines.append(part)
        else:
            lines.append(part)
        idx += 1
    return "\n".join(lines).strip()


def _is_help_action(action: str) -> bool:
    if not action:
        return False
    cleaned = action.strip().lower().replace("`", "")
    if re.match(r"^help(\b|[\(\[])", cleaned):
        return True
    return bool(re.search(r"\baction\s*:\s*help\b", cleaned))


def _short_label(text: str) -> str:
    label = re.sub(r"\s+", " ", (text or "").strip())
    if len(label) > 36:
        return label[:36].rstrip() + "..."
    return label or "(empty)"


def _normalize_action_key(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[`\\s]", "", str(text))
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    return match.group(0).lower() if match else cleaned.lower()


def _canonical_action_label(text: str) -> str:
    if not text:
        return ""
    raw = str(text).strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(.*)$", raw)
    if not match:
        return raw
    name, rest = match.groups()
    canonical = {
        "notice": "Notice",
        "filteredaction": "FilteredAction",
        "try": "Try",
        "invalid": "Invalid",
    }.get(_normalize_action_key(name))
    if not canonical:
        return raw
    return f"{canonical}{rest}"


def _extract_thought(text: str) -> str:
    if not text:
        return ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    thought_match = re.search(r"Thought:(.*?)(?:\nAction:|$)", text, re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    reasoning_match = re.search(r"Reasoning:(.*?)(?:\nAction:|$)", text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return text.strip()


def _extract_action_text(step) -> str:
    action = str(getattr(step, "action", "") or "").strip()
    model_response = str(getattr(step, "model_response", "") or "").strip()

    def _parse_action(text: str) -> str:
        if not text:
            return ""
        matches = re.findall(r"```(.*?)```", text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        match = re.search(r"Action:\s*([^\n]+)", text)
        if match:
            return match.group(1).strip()
        return ""

    if action and "Thought:" not in action and "Action:" not in action and "\n" not in action:
        return action
    for text in (action, model_response):
        parsed = _parse_action(text)
        if parsed:
            return parsed
    for text in (action, model_response):
        matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))", text)
        if matches:
            return matches[-1].strip()
    return action


def _step_is_help(step) -> bool:
    action_text = _extract_action_text(step)
    if _is_help_action(action_text):
        return True
    raw_action = str(getattr(step, "action", "") or "")
    return _is_help_action(raw_action)


def _format_interaction_block(step) -> str:
    lines = []
    step_idx = getattr(step, "step", None)
    if step_idx is not None:
        lines.append(f"Step {step_idx}")
    user = _last_user_message(getattr(step, "chat_completions", None) or [])
    if user:
        lines.extend(["User:", user])
    assistant = str(getattr(step, "model_response", "") or "").strip()
    if assistant:
        lines.extend(["Assistant:", assistant])
    action = _extract_action_text(step)
    lines.append(f"Action: {action or '(empty)'}")
    reward = getattr(step, "reward", None)
    done = getattr(step, "done", None)
    if reward is not None:
        lines.append(f"Reward: {reward}")
    if done is not None:
        lines.append(f"Done: {done}")
    return "\n".join(lines).strip()


def _format_segment_detail(
    traj, steps: list, start_index: int, help_detail: str | None = None
) -> str:
    env_id = ""
    task = getattr(traj, "task", None) or {}
    if isinstance(task, dict):
        env_id = task.get("env_id") or task.get("task_name") or ""
    header = [
        f"Episode: {getattr(traj, 'episode_id', '')}",
        f"Trajectory: {getattr(traj, 'uid', '')}",
    ]
    if env_id:
        header.append(f"Env: {env_id}")
    header.extend(_format_webshop_task_meta(task))
    termination_reason = None
    termination_detail = None
    for step in reversed(steps or []):
        term = getattr(step, "termination_reason", None)
        if term:
            termination_reason = term
            termination_detail = getattr(step, "termination_reason_detail", None)
            break
    if termination_reason:
        header.append(f"Termination: {termination_reason}")
        if termination_detail:
            header.append(f"Termination detail: {termination_detail}")
    if not steps:
        if help_detail:
            header.append("No non-help solver actions.")
            return "\n".join(
                [
                    "\n".join(header).strip(),
                    "",
                    "[[SECTION]] Help call:",
                    help_detail.strip(),
                ]
            ).strip()
        header.append("No solver steps.")
        return "\n".join(header).strip()
    action_lines = []
    thought_lines = []
    interaction_blocks = []
    for idx, step in enumerate(steps):
        action_num = idx + 1
        action = _extract_action_text(step)
        action_lines.append(f"[[ACTION]] {action_num}. {action or '(empty)'}")
        model_response = str(getattr(step, "model_response", "") or "").strip()
        thought = _extract_thought(model_response)
        if thought:
            thought_lines.append(f"[[THOUGHT]] {action_num}. {thought}")
        else:
            thought_lines.append(f"[[THOUGHT]] {action_num}. (no reasoning)")
        interaction_blocks.append(_format_interaction_block(step))
    sections = [
        "[[SECTION]] Action sequence:",
        "\n".join(action_lines).strip(),
    ]
    if help_detail:
        sections.extend(
            [
                "",
                "[[SECTION]] Help call:",
                help_detail.strip(),
            ]
        )
    sections.extend(
        [
            "",
            "==========",
            "",
            "[[SECTION]] Action thoughts:",
            "\n".join(thought_lines).strip(),
            "",
            "==========",
            "",
            "[[SECTION]] Interaction:",
            "\n\n---\n\n".join(interaction_blocks).strip(),
        ]
    )
    detail = "\n".join([line for line in sections if line != ""]).strip()
    return "\n".join(header + ["", detail]).strip()

def _format_terminal_detail(traj) -> str:
    lines = [
        f"Episode: {getattr(traj, 'episode_id', '')}",
        f"Trajectory: {getattr(traj, 'uid', '')}",
    ]
    task = getattr(traj, "task", None) or {}
    lines.extend(_format_webshop_task_meta(task))
    lines.append("Terminal: no further help")
    termination_reason = getattr(traj, "termination_reason", None)
    if termination_reason:
        lines.append(f"Termination: {termination_reason}")
        termination_detail = getattr(traj, "termination_reason_detail", None)
        if termination_detail:
            lines.append(f"Termination detail: {termination_detail}")
    steps = getattr(traj, "steps", []) or []
    if steps:
        last_step = steps[-1]
        reward = getattr(last_step, "reward", None)
        done = getattr(last_step, "done", None)
        if reward is not None:
            lines.append(f"Reward: {reward}")
        if done is not None:
            lines.append(f"Done: {done}")
    return "\n".join(lines).strip()

def _format_goal_object(goal: object) -> str:
    if goal is None:
        return ""
    if isinstance(goal, list):
        lines = []
        for item in goal:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    lines.append(f"- {text}")
                else:
                    lines.append(f"- {json.dumps(item, ensure_ascii=True)}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()
    if isinstance(goal, dict):
        text = goal.get("text") if isinstance(goal, dict) else None
        if text:
            return str(text)
        return json.dumps(goal, ensure_ascii=True)
    return str(goal)


def _format_root_detail(trajs: list) -> str:
    for traj in trajs:
        if getattr(traj, "name", "") != "solver":
            continue
        steps = getattr(traj, "steps", []) or []
        if not steps:
            continue
        obs = getattr(steps[0], "observation", None)
        if obs is None:
            continue
        task = getattr(traj, "task", None) or {}
        goal = _format_goal_object(obs.get("goal_object")) if isinstance(obs, dict) else ""
        observation_text = _format_webshop_observation(obs)
        lines = []
        lines.extend(_format_webshop_task_meta(task))
        if goal:
            if lines:
                lines.append("")
            lines.extend(["Goal:", goal])
        if observation_text:
            if lines:
                lines.append("")
            lines.extend(["Observation:", observation_text])
        return "\n".join([line for line in lines if line != ""]).strip() or "Start"
    return "Start"


def _extract_auditor_action_from_observation(observation: object) -> tuple[str, str]:
    if not isinstance(observation, str):
        return "", ""
    raw = observation.strip()
    if not raw:
        return "", ""
    matches = re.findall(r"```(.*?)```", raw, re.DOTALL)
    action_text = matches[-1].strip() if matches else ""
    label = action_text or raw.splitlines()[0].strip()
    label = _short_label(label)
    if action_text:
        detail = f"Auditor action:\n{action_text}\n\nAuditor observation:\n{raw}"
    else:
        detail = f"Auditor observation:\n{raw}"
    return label, detail.strip()

def _infer_auditor_action_from_step(step) -> tuple[str, str]:
    action_text = str(getattr(step, "action", "") or "").strip()
    if not action_text:
        return "", ""
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", action_text)
    if not match:
        return "", ""
    name = match.group(1)
    action_key = _normalize_action_key(name)
    if action_key not in {"notice", "filteredaction", "try", "invalid"}:
        return "", ""
    return action_key, name

def _extract_help_call_text(step) -> str:
    help_response = str(getattr(step, "model_response", "") or "").strip()
    if not help_response:
        messages = getattr(step, "chat_completions", None) or []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                help_response = str(msg.get("content") or "").strip()
                break
    return help_response or ""

def _detail_dedup_key(detail: str) -> str:
    if not detail:
        return ""
    lines = []
    for line in detail.splitlines():
        if line.startswith("Trajectory:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_help_action_info(steps: list, help_idx: int) -> tuple[str, str, str, bool | None]:
    help_step = steps[help_idx]
    info = getattr(help_step, "info", None) or {}
    forced_action = None
    forced_match = None
    if isinstance(info, dict):
        forced_action = info.get("forced_action")
        forced_match = info.get("forced_match")
    if forced_action is None:
        forced_action = getattr(help_step, "forced_action", None)
    if forced_match is None:
        forced_match = getattr(help_step, "forced_match", None)
    next_step = steps[help_idx + 1] if help_idx + 1 < len(steps) else None
    next_info = getattr(next_step, "info", None) or {}
    if forced_action is None and isinstance(next_info, dict):
        forced_action = next_info.get("forced_action")
    if forced_match is None and isinstance(next_info, dict):
        forced_match = next_info.get("forced_match")
    if forced_action is None and next_step is not None:
        forced_action = getattr(next_step, "forced_action", None)
    if forced_match is None and next_step is not None:
        forced_match = getattr(next_step, "forced_match", None)
    audit_obs = getattr(next_step, "observation", None) if next_step is not None else None
    label, detail = _extract_auditor_action_from_observation(audit_obs)
    if not forced_action and not label and next_step is not None:
        inferred_key, inferred_label = _infer_auditor_action_from_step(next_step)
        if inferred_key:
            label = inferred_label
    action_key = _normalize_action_key(forced_action or label)
    edge_label = forced_action or label
    if forced_action:
        if detail:
            detail = f"{detail}\n\nForced action: {forced_action}"
        else:
            detail = f"Forced action: {forced_action}"
    if forced_match is False:
        detail = f"{detail}\nForced match: false" if detail else "Forced match: false"
    return action_key, edge_label, detail, forced_match


def _build_auditor_response_map(trajs: list) -> dict[str, list[str]]:
    response_map: dict[str, list[str]] = {}
    for traj in trajs:
        if getattr(traj, "name", "") != "auditor":
            continue
        for step in getattr(traj, "steps", []) or []:
            info = getattr(step, "info", None) or {}
            action_label = ""
            source_label = ""
            if isinstance(info, dict):
                action_label = str(info.get("auditor_action") or info.get("forced_action") or "")
                source_label = str(info.get("auditor_source") or "")
            if not action_label:
                action_label = str(getattr(step, "action", "") or "")
            action_key = _normalize_action_key(action_label)
            if not action_key:
                continue
            response = str(getattr(step, "model_response", "") or "").strip()
            if not response:
                messages = getattr(step, "chat_completions", None) or []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        response = str(msg.get("content") or "").strip()
                        break
            if not response:
                continue
            header = []
            if action_label:
                header.append(f"Auditor action: {action_label}")
            if isinstance(info, dict) and info.get("forced_action"):
                header.append(f"Forced action: {info.get('forced_action')}")
            if source_label:
                header.append(f"Auditor source: {source_label}")
            detail = response
            if header:
                detail = "\n".join(header + ["", response]).strip()
            existing = response_map.setdefault(action_key, [])
            if detail not in existing:
                existing.append(detail)
    return response_map


def _build_auditor_response_map_by_depth(trajs: list) -> dict[tuple[int, str], list[str]]:
    response_map: dict[tuple[int, str], list[str]] = {}
    for traj in trajs:
        if getattr(traj, "name", "") != "auditor":
            continue
        for step in getattr(traj, "steps", []) or []:
            info = getattr(step, "info", None) or {}
            if not isinstance(info, dict):
                info = {}
            depth = info.get("audit_depth")
            action_label = str(info.get("auditor_action") or info.get("forced_action") or "")
            source_label = str(info.get("auditor_source") or "")
            if not action_label:
                action_label = str(getattr(step, "action", "") or "")
            action_key = _normalize_action_key(action_label)
            if depth is None or not action_key:
                continue
            response = str(getattr(step, "model_response", "") or "").strip()
            if not response:
                messages = getattr(step, "chat_completions", None) or []
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        response = str(msg.get("content") or "").strip()
                        break
            if not response:
                continue
            header = []
            if action_label:
                header.append(f"Auditor action: {action_label}")
            if info.get("forced_action"):
                header.append(f"Forced action: {info.get('forced_action')}")
            if source_label:
                header.append(f"Auditor source: {source_label}")
            detail = response
            if header:
                detail = "\n".join(header + ["", response]).strip()
            key = (int(depth), action_key)
            existing = response_map.setdefault(key, [])
            if detail not in existing:
                existing.append(detail)
    return response_map


def _path_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, str):
        cleaned = value.strip()
        return tuple([cleaned]) if cleaned else ()
    return ()


def _has_tree_metadata(trajs: list) -> bool:
    for traj in trajs:
        for step in getattr(traj, "steps", []) or []:
            info = getattr(step, "info", None) or {}
            if isinstance(info, dict) and (
                "tree_path" in info or "tree_parent_path" in info
            ):
                return True
    return False


def _auditor_response_from_step(step) -> str:
    response = str(getattr(step, "model_response", "") or "").strip()
    if response:
        return response
    messages = getattr(step, "chat_completions", None) or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            response = str(msg.get("content") or "").strip()
            if response:
                return response
    return ""


def _build_webshop_tree_from_paths(trajs: list) -> tuple[list[dict], list[dict]]:
    node_id = 0
    root_id = "root"
    nodes = {root_id: {"id": root_id, "label": "ROOT", "detail": _format_root_detail(trajs)}}
    node_by_path: dict[tuple[str, ...], str] = {}
    node_detail_keys: dict[str, set[str]] = {}
    edges: dict[tuple[str, str], dict] = {}
    leaf_rewards: dict[str, list[float]] = {}
    terminal_by_parent: dict[str, str] = {}

    for traj in trajs:
        if getattr(traj, "name", "") != "solver":
            continue
        steps = getattr(traj, "steps", []) or []
        segments = []
        current_steps = []
        current_start_index = None
        segment_path: tuple[str, ...] | None = None
        action_counter = 0
        for idx, step in enumerate(steps):
            info = getattr(step, "info", None) or {}
            step_path = _path_tuple(info.get("tree_path")) if isinstance(info, dict) else ()
            if segment_path is None and isinstance(info, dict) and "tree_path" in info:
                segment_path = step_path
            if _step_is_help(step):
                help_detail = _extract_help_call_text(step)
                if current_steps or help_detail:
                    segments.append(
                        {
                            "steps": current_steps,
                            "start_index": current_start_index,
                            "help_detail": help_detail,
                            "path": segment_path or step_path or (),
                        }
                    )
                current_steps = []
                current_start_index = None
                segment_path = None
            else:
                if not current_steps:
                    current_start_index = action_counter + 1
                action_counter += 1
                current_steps.append(step)
        if current_steps or segment_path is not None:
            segments.append(
                {
                    "steps": current_steps,
                    "start_index": current_start_index,
                    "help_detail": "",
                    "path": segment_path or (),
                }
            )

        last_child_id = None
        for segment in segments:
            path_key = segment.get("path") or ()
            if path_key not in node_by_path:
                node_id += 1
                node_by_path[path_key] = f"node_{node_id}"
            child_id = node_by_path[path_key]
            segment_steps = segment.get("steps") or []
            detail = _format_segment_detail(
                traj,
                segment_steps,
                segment.get("start_index") or 1,
                help_detail=segment.get("help_detail") or "",
            )
            if child_id not in nodes:
                if segment_steps:
                    label = _short_label(str(getattr(segment_steps[0], "action", "") or "").strip())
                else:
                    label = "help()" if (segment.get("help_detail") or "").strip() else "(no solver action)"
                nodes[child_id] = {
                    "id": child_id,
                    "label": label,
                    "detail": detail,
                }
                detail_key = _detail_dedup_key(detail)
                if detail_key:
                    node_detail_keys[child_id] = {detail_key}
                if segment.get("start_index") is not None:
                    nodes[child_id]["start_index"] = segment.get("start_index")
            else:
                detail_key = _detail_dedup_key(detail)
                detail_keys = node_detail_keys.setdefault(child_id, set())
                if detail_key and detail_key not in detail_keys:
                    existing_detail = nodes[child_id].get("detail", "")
                    if existing_detail:
                        nodes[child_id]["detail"] = f"{existing_detail}\n\n---\n\n{detail}"
                    else:
                        nodes[child_id]["detail"] = detail
                    detail_keys.add(detail_key)
            last_child_id = child_id

        if last_child_id:
            terminal_id = terminal_by_parent.get(last_child_id)
            if terminal_id is None:
                node_id += 1
                terminal_id = f"node_{node_id}"
                nodes[terminal_id] = {
                    "id": terminal_id,
                    "label": "(terminal)",
                    "detail": _format_terminal_detail(traj),
                }
                terminal_by_parent[last_child_id] = terminal_id
                edges[(last_child_id, terminal_id)] = {
                    "id": f"edge_{len(edges) + 1}",
                    "detail": ["Terminal: no further help."],
                    "label": "end",
                    "edge_type": "terminal",
                    "action_key": "terminal",
                    "is_base": False,
                    "invalid": False,
                    "source": last_child_id,
                    "target": terminal_id,
                }
            reward_value = getattr(traj, "reward", None)
            if reward_value is None and getattr(traj, "steps", None):
                reward_value = sum(getattr(s, "reward", 0.0) for s in traj.steps)
            if isinstance(reward_value, (int, float)) and terminal_id:
                leaf_rewards.setdefault(terminal_id, []).append(float(reward_value))

    for traj in trajs:
        if getattr(traj, "name", "") != "auditor":
            continue
        for step in getattr(traj, "steps", []) or []:
            info = getattr(step, "info", None) or {}
            if not isinstance(info, dict):
                continue
            parent_path = _path_tuple(info.get("tree_parent_path"))
            child_path = _path_tuple(info.get("tree_path"))
            if child_path == () and parent_path == ():
                continue
            if parent_path not in node_by_path:
                node_id += 1
                node_by_path[parent_path] = f"node_{node_id}"
                if parent_path == ():
                    nodes[node_by_path[parent_path]] = {
                        "id": node_by_path[parent_path],
                        "label": "(start)",
                        "detail": "No solver steps recorded before help.",
                    }
                else:
                    nodes[node_by_path[parent_path]] = {
                        "id": node_by_path[parent_path],
                        "label": "(missing node)",
                        "detail": "Missing solver node for parent path.",
                    }
            parent_id = node_by_path[parent_path]
            if child_path not in node_by_path:
                node_id += 1
                node_by_path[child_path] = f"node_{node_id}"
                fail_reason = info.get("tree_edge_fail_reason")
                fail_stage = info.get("tree_edge_fail_stage")
                display_label = _canonical_action_label(
                    str(info.get("tree_edge_action") or info.get("forced_action") or "")
                )
                missing_detail = f"Missing branch for action: {display_label}."
                if fail_reason:
                    missing_detail += f"\nFailure reason: {fail_reason}"
                if fail_stage:
                    missing_detail += f"\nFailure stage: {fail_stage}"
                nodes[node_by_path[child_path]] = {
                    "id": node_by_path[child_path],
                    "label": "(missing branch)",
                    "detail": missing_detail,
                }
            child_id = node_by_path[child_path]

            action_label = (
                info.get("tree_edge_action")
                or info.get("forced_action")
                or info.get("auditor_action")
                or getattr(step, "action", "")
            )
            action_label = str(action_label or "")
            action_key = _normalize_action_key(action_label)
            edge_invalid = bool(info.get("forced_match") is False or action_key == "invalid")
            source_label = str(info.get("auditor_source") or "")
            is_base = source_label.startswith("base")

            headers = []
            if action_label:
                headers.append(f"Auditor action: {action_label}")
            if info.get("forced_action"):
                headers.append(f"Forced action: {info.get('forced_action')}")
            if source_label:
                headers.append(f"Auditor source: {source_label}")
            if info.get("tree_edge_failed"):
                headers.append("Edge failed: true")
            if info.get("tree_edge_fail_reason"):
                headers.append(f"Failure reason: {info.get('tree_edge_fail_reason')}")
            if info.get("tree_edge_fail_stage"):
                headers.append(f"Failure stage: {info.get('tree_edge_fail_stage')}")
            response = _auditor_response_from_step(step)
            detail = response
            if headers:
                detail = "\n".join(headers + (["", response] if response else []))

            edge_key = (parent_id, child_id)
            if edge_key not in edges:
                edges[edge_key] = {
                    "id": f"edge_{len(edges) + 1}",
                    "detail": [],
                    "label": _canonical_action_label(action_label),
                    "edge_type": "help",
                    "action_key": action_key,
                    "is_base": is_base,
                    "invalid": edge_invalid,
                    "source": parent_id,
                    "target": child_id,
                }
            edge_entry = edges[edge_key]
            if action_label and not edge_entry.get("label"):
                edge_entry["label"] = _canonical_action_label(action_label)
            if is_base:
                edge_entry["is_base"] = True
            if edge_invalid:
                edge_entry["invalid"] = True
            if detail and detail not in edge_entry["detail"]:
                if not edge_entry["detail"]:
                    edge_entry["detail"].append(detail)

    start_node_id = node_by_path.get(())
    if start_node_id and (root_id, start_node_id) not in edges:
        edges[(root_id, start_node_id)] = {
            "id": f"edge_{len(edges) + 1}",
            "detail": ["(start)"],
            "label": "",
            "edge_type": "root",
            "action_key": "root",
            "is_base": False,
            "invalid": False,
            "source": root_id,
            "target": start_node_id,
        }

    node_list = list(nodes.values())
    sources = {edge["source"] for edge in edges.values() if "source" in edge}
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
    for (_src, _dst), edge in edges.items():
        details = edge.get("detail") or []
        detail = "\n\n---\n\n".join(details).strip() if isinstance(details, list) else str(details)
        if not detail:
            if edge.get("edge_type") == "terminal":
                detail = "(terminal)"
            else:
                detail = "(no auditor action)"
        edge_list.append(
            {
                "id": edge["id"],
                "source": edge.get("source") or _src,
                "target": edge.get("target") or _dst,
                "detail": detail,
                "label": edge.get("label", ""),
                "edge_type": edge.get("edge_type", "help"),
                "is_base": bool(edge.get("is_base", False)),
                "invalid": bool(edge.get("invalid", False)),
            }
        )
    return node_list, edge_list


def _expected_action_keys(trajs: list) -> list[str]:
    actions = []
    for traj in trajs:
        if getattr(traj, "name", "") != "auditor":
            continue
        for step in getattr(traj, "steps", []) or []:
            info = getattr(step, "info", None) or {}
            action_label = ""
            if isinstance(info, dict):
                action_label = str(info.get("auditor_action") or info.get("forced_action") or "")
            if not action_label:
                action_label = str(getattr(step, "action", "") or "")
            action_key = _normalize_action_key(action_label)
            if action_key and action_key != "invalid":
                actions.append(action_key)
    if not actions:
        actions = ["notice", "filteredaction", "try"]
    priority = ["notice", "filteredaction", "try"]
    seen = set()
    ordered = []
    for key in priority:
        if key in actions and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in sorted(set(actions)):
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _forced_action_depths(trajs: list) -> set[int]:
    depths: set[int] = set()
    for traj in trajs:
        if getattr(traj, "name", "") != "auditor":
            continue
        for step in getattr(traj, "steps", []) or []:
            info = getattr(step, "info", None) or {}
            if not isinstance(info, dict):
                continue
            if not info.get("forced_action"):
                continue
            depth = info.get("audit_depth")
            if depth is None:
                continue
            try:
                depths.add(int(depth))
            except (TypeError, ValueError):
                continue
    return depths


def _build_webshop_tree(trajs: list) -> tuple[list[dict], list[dict]]:
    if _has_tree_metadata(trajs):
        return _build_webshop_tree_from_paths(trajs)
    node_id = 0
    root_id = "root"
    nodes = {root_id: {"id": root_id, "label": "ROOT", "detail": _format_root_detail(trajs)}}
    edges = {}
    child_lookup = {}
    leaf_rewards: dict[str, list[float]] = {}
    help_nodes: dict[str, dict] = {}
    solver_count = 0
    expected_actions = _expected_action_keys(trajs)
    auditor_responses = _build_auditor_response_map_by_depth(trajs)
    forced_depths = _forced_action_depths(trajs)
    for traj in trajs:
        if getattr(traj, "name", "") != "solver":
            continue
        is_base = bool(getattr(traj, "is_base", False))
        solver_count += 1
        parent_id = root_id
        steps = getattr(traj, "steps", []) or []
        segments = []
        current_steps = []
        current_start_index = None
        action_counter = 0
        pending_action_key = ""
        pending_label = ""
        pending_detail = ""
        pending_obs = None
        pending_invalid = False
        help_depth = 0
        pending_depth = None
        for idx, step in enumerate(steps):
            if _step_is_help(step):
                help_detail = _extract_help_call_text(step)
                action_key, edge_label, edge_detail, forced_match = _extract_help_action_info(
                    steps, idx
                )
                edge_invalid = (forced_match is False) or (action_key == "invalid")
                help_action_key = action_key
                help_depth_current = help_depth
                if current_steps or pending_action_key or pending_label or pending_detail or pending_obs:
                    segments.append(
                        {
                            "steps": current_steps,
                            "start_index": current_start_index,
                            "action_key": pending_action_key,
                            "edge_label": pending_label,
                            "edge_detail": pending_detail,
                            "audit_obs": pending_obs,
                            "audit_depth": pending_depth,
                            "edge_invalid": pending_invalid,
                            "had_help": True,
                            "help_detail": help_detail,
                            "help_depth": help_depth_current,
                            "help_action_key": help_action_key,
                        }
                    )
                    current_steps = []
                    current_start_index = None
                if idx + 1 < len(steps):
                    pending_action_key = action_key
                    pending_label = edge_label
                    pending_detail = edge_detail
                    pending_obs = getattr(steps[idx + 1], "observation", None)
                    pending_depth = help_depth
                    pending_invalid = edge_invalid
                help_depth += 1
            else:
                if not current_steps:
                    current_start_index = action_counter + 1
                action_counter += 1
                current_steps.append(step)
        if current_steps or pending_action_key or pending_label or pending_detail or pending_obs:
            segments.append(
                {
                    "steps": current_steps,
                    "start_index": current_start_index,
                    "action_key": pending_action_key,
                    "edge_label": pending_label,
                    "edge_detail": pending_detail,
                    "audit_obs": pending_obs,
                    "audit_depth": pending_depth,
                    "edge_invalid": pending_invalid,
                }
            )
        last_child_id = None
        for segment in segments:
            segment_steps = segment.get("steps") or []
            start_index = segment.get("start_index") or 1
            detail = _format_segment_detail(
                traj,
                segment_steps,
                start_index,
                help_detail=segment.get("help_detail") or "",
            )
            if segment_steps:
                label = _short_label(str(getattr(segment_steps[0], "action", "") or "").strip())
            else:
                if segment.get("had_help") or (segment.get("help_detail") or "").strip():
                    label = "help()"
                else:
                    label = "(no solver action)"
            action_key = segment.get("action_key") or ""
            audit_depth = segment.get("audit_depth")
            if parent_id == root_id:
                key = (parent_id, "pre_help")
            else:
                key = (parent_id, action_key or "unknown", audit_depth)
            if key in child_lookup:
                child_id = child_lookup[key]
                existing_detail = nodes[child_id].get("detail", "")
                detail_key = _detail_dedup_key(detail)
                detail_keys = nodes[child_id].setdefault("_detail_keys", set())
                if detail_key and detail_key not in detail_keys:
                    if existing_detail:
                        nodes[child_id]["detail"] = f"{existing_detail}\n\n---\n\n{detail}"
                    else:
                        nodes[child_id]["detail"] = detail
                    detail_keys.add(detail_key)
            else:
                node_id += 1
                child_id = f"node_{node_id}"
                nodes[child_id] = {
                    "id": child_id,
                    "label": label,
                    "detail": detail,
                }
                detail_key = _detail_dedup_key(detail)
                if detail_key:
                    nodes[child_id]["_detail_keys"] = {detail_key}
                child_lookup[key] = child_id
                if segment.get("start_index") is not None:
                    nodes[child_id]["start_index"] = segment.get("start_index")
            if segment.get("had_help"):
                help_nodes[child_id] = {
                    "detail": segment.get("help_detail") or "",
                    "audit_depth": segment.get("help_depth"),
                    "base_action_key": segment.get("help_action_key") or "",
                    "is_base": bool(is_base),
                }
            edge_key = (parent_id, child_id)
            edge_label = _canonical_action_label(segment.get("edge_label") or "")
            edge_detail = segment.get("edge_detail") or ""
            edge_invalid = bool(segment.get("edge_invalid"))
            detail_list = []
            if not edge_detail and isinstance(segment.get("audit_obs"), str):
                edge_detail = str(segment.get("audit_obs"))
            if edge_detail:
                detail_list.append(edge_detail)
            edge_type = "help" if (edge_label or detail_list) else "root"
            if edge_type == "help" and audit_depth is not None and action_key:
                for resp in auditor_responses.get((int(audit_depth), action_key), []):
                    if resp not in detail_list:
                        detail_list.append(resp)
            if edge_key not in edges:
                edges[edge_key] = {
                    "id": f"edge_{len(edges) + 1}",
                    "detail": [],
                    "label": edge_label,
                    "edge_type": edge_type,
                    "action_key": action_key,
                    "is_base": False,
                    "invalid": edge_invalid,
                    "audit_depth": audit_depth,
                }
            edge_entry = edges[edge_key]
            if edge_label and not edge_entry.get("label"):
                edge_entry["label"] = edge_label
            if edge_type == "help":
                edge_entry["edge_type"] = "help"
            if edge_type == "help" and is_base:
                edge_entry["is_base"] = True
            if edge_invalid:
                edge_entry["invalid"] = True
            if edge_entry.get("audit_depth") is None:
                edge_entry["audit_depth"] = audit_depth
            if not edge_entry["detail"]:
                for entry in detail_list:
                    if entry not in edge_entry["detail"]:
                        edge_entry["detail"].append(entry)
            parent_id = child_id
            last_child_id = child_id
        if last_child_id:
            terminal_key = (last_child_id, "__terminal__")
            if terminal_key in child_lookup:
                terminal_id = child_lookup[terminal_key]
            else:
                node_id += 1
                terminal_id = f"node_{node_id}"
                nodes[terminal_id] = {
                    "id": terminal_id,
                    "label": "(terminal)",
                    "detail": _format_terminal_detail(traj),
                }
                child_lookup[terminal_key] = terminal_id
            edge_key = (last_child_id, terminal_id)
            if edge_key not in edges:
                edges[edge_key] = {
                    "id": f"edge_{len(edges) + 1}",
                    "detail": [],
                    "label": "end",
                    "edge_type": "terminal",
                    "action_key": "terminal",
                    "is_base": False,
                }
            edge_entry = edges[edge_key]
            edge_entry["label"] = "end"
            edge_entry["edge_type"] = "terminal"
            if is_base:
                edge_entry["is_base"] = True
            if not edge_entry["detail"]:
                edge_entry["detail"].append("Terminal: no further help.")
            reward_value = getattr(traj, "reward", None)
            if reward_value is None and getattr(traj, "steps", None):
                reward_value = sum(getattr(s, "reward", 0.0) for s in traj.steps)
            if isinstance(reward_value, (int, float)):
                leaf_rewards.setdefault(terminal_id, []).append(float(reward_value))

    parent_actions = {}
    parent_depths = {}
    for (src, _dst), edge in edges.items():
        if edge.get("edge_type") != "help":
            continue
        action_key = edge.get("action_key") or ""
        if not action_key:
            continue
        parent_actions.setdefault(src, set()).add(action_key)
        if src not in parent_depths:
            parent_depths[src] = edge.get("audit_depth")

    for parent_id, info in help_nodes.items():
        parent_actions.setdefault(parent_id, set())
        if parent_id not in parent_depths:
            parent_depths[parent_id] = info.get("audit_depth")

    for parent_id, present_actions in parent_actions.items():
        for action_key in expected_actions:
            if action_key in present_actions:
                continue
            audit_depth = parent_depths.get(parent_id)
            if audit_depth is None:
                continue
            try:
                depth_key = int(audit_depth)
            except (TypeError, ValueError):
                continue
            if depth_key not in forced_depths:
                continue
            help_info = help_nodes.get(parent_id, {}) or {}
            help_detail = help_info.get("detail") or ""
            base_action_key = help_info.get("base_action_key") or ""
            base_edge = bool(help_info.get("is_base")) and base_action_key == action_key
            missing_key = (parent_id, action_key, audit_depth, "missing")
            if missing_key in child_lookup:
                continue
            display_label = _canonical_action_label(action_key)
            node_id += 1
            child_id = f"node_{node_id}"
            nodes[child_id] = {
                "id": child_id,
                "label": "(missing branch)",
                "detail": f"Missing branch for action: {display_label}\n\nNo solver steps after help.",
            }
            child_lookup[missing_key] = child_id
            edge_key = (parent_id, child_id)
            missing_detail = []
            missing_detail.extend(
                auditor_responses.get((depth_key, action_key), [])
            )
            missing_detail.append(f"Missing branch: {display_label}.")
            edges[edge_key] = {
                "id": f"edge_{len(edges) + 1}",
                "detail": missing_detail,
                "label": display_label,
                "edge_type": "help",
                "action_key": action_key,
                "is_base": base_edge,
                "invalid": False,
                "audit_depth": audit_depth,
            }

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
        details = edge.get("detail") or []
        detail = "\n\n---\n\n".join(details).strip() if isinstance(details, list) else str(details)
        if not detail:
            if edge.get("edge_type") == "root":
                detail = "(start)"
            elif edge.get("edge_type") == "terminal":
                detail = "(terminal)"
            else:
                detail = "(no auditor action)"
        edge_list.append(
            {
                "id": edge["id"],
                "source": src,
                "target": dst,
                "detail": detail,
                "label": edge.get("label", ""),
                "edge_type": edge.get("edge_type", "no_help"),
                "is_base": bool(edge.get("is_base", False)),
                "invalid": bool(edge.get("invalid", False)),
            }
        )
    return node_list, edge_list


def _layout_tree(nodes: list[dict], edges: list[dict]) -> tuple[dict, dict]:
    children = {}
    incoming = {}
    for edge in edges:
        children.setdefault(edge["source"], []).append(edge["target"])
        incoming[edge["target"]] = incoming.get(edge["target"], 0) + 1
        incoming.setdefault(edge["source"], incoming.get(edge["source"], 0))

    node_ids = [node["id"] for node in nodes]
    for node_id in node_ids:
        incoming.setdefault(node_id, 0)

    x_pos = {}
    y_pos = {}
    order = 0
    visiting = set()

    def _dfs(node_id: str, depth: int) -> float:
        nonlocal order
        if node_id in x_pos:
            return x_pos[node_id]
        if node_id in visiting:
            order += 1
            x_pos[node_id] = order
            y_pos[node_id] = depth
            return x_pos[node_id]
        visiting.add(node_id)
        kids = children.get(node_id, [])
        if not kids:
            order += 1
            x_pos[node_id] = order
            y_pos[node_id] = depth
            visiting.discard(node_id)
            return x_pos[node_id]
        child_positions = []
        for kid in kids:
            child_positions.append(_dfs(kid, depth + 1))
        if child_positions:
            x_pos[node_id] = sum(child_positions) / len(child_positions)
        else:
            order += 1
            x_pos[node_id] = order
        y_pos[node_id] = depth
        visiting.discard(node_id)
        return x_pos[node_id]

    if "root" in incoming:
        _dfs("root", 0)
    for node_id in node_ids:
        if node_id not in x_pos and incoming.get(node_id, 0) == 0:
            _dfs(node_id, 0)
    for node_id in node_ids:
        if node_id not in x_pos:
            _dfs(node_id, 0)
    return x_pos, y_pos


def _build_tree_html(trajs: list) -> str:
    nodes, edges = _build_webshop_tree(trajs)
    if len(nodes) <= 1:
        return "<div>No solver trajectories found in JSON logs.</div>"

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
    node_details_with_tokens = {}
    node_start_index = {}
    for node in nodes:
        if node.get("start_index") is not None:
            node_start_index[node["id"]] = node.get("start_index")
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
    node_start_index_json = json.dumps(node_start_index).replace("</", "<\\/")
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
        edge_class = "tree-edge"
        thickness = 10
        if edge.get("edge_type") == "help":
            if edge.get("is_base"):
                edge_class += " tree-edge-base"
                thickness = 12
            else:
                edge_class += " tree-edge-alt"
                thickness = 10
        elif edge.get("edge_type") == "terminal":
            edge_class += " tree-edge-terminal"
            thickness = 8
        else:
            edge_class += " tree-edge-root"
            thickness = 8
        if edge.get("invalid"):
            edge_class += " tree-edge-invalid"
        edge_divs.append(
            f'<div class="{edge_class}" data-edge-id="{edge["id"]}" '
            f'onclick="showEdge(\'{edge["id"]}\')" '
            f'style="left:{x1}px; top:{y1}px; width:{length}px; '
            f'height:{thickness}px; transform:rotate({angle}deg);"'
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
            edge_label_divs.append(
                f'<div class="tree-edge-label" '
                f'style="left:{mid_x}px; top:{mid_y}px;">{label_clean}</div>'
            )

    node_divs = []
    for node in nodes:
        x, y = _node_xy(node["id"])
        if node["id"] == "root":
            short_label = "ROOT"
        else:
            label = node.get("label") or ""
            label_clean = re.sub(r"\s+", " ", label).strip()
            words = label_clean.split()
            if len(words) > 6:
                label_clean = " ".join(words[:6]) + "..."
            short_label = html.escape(label_clean or "(no action)")
        node_class = "tree-node"
        if node["id"] == "root":
            node_class += " tree-node-root"
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
      .tree-toolbar {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
      }}
      .tree-toolbar button {{
        border: 1px solid #cbd5f5;
        background: #f8fafc;
        color: #0f172a;
        border-radius: 6px;
        padding: 4px 10px;
        cursor: pointer;
      }}
      .tree-toolbar button:hover {{
        background: #e2e8f0;
      }}
      .tree-toolbar input[type="range"] {{
        width: 140px;
      }}
      .tree-zoom-label {{
        font-size: 12px;
        color: #334155;
        min-width: 48px;
        text-align: right;
      }}
      .tree-canvas {{
        flex: 2;
        overflow: auto;
        border: 1px solid #94a3b8;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: inset 0 0 0 1px #e2e8f0;
        position: relative;
        cursor: grab;
      }}
      .tree-canvas.dragging {{
        cursor: grabbing;
      }}
      .tree-canvas-zoom {{
        position: relative;
        width: {width}px;
        height: {height}px;
      }}
      .tree-canvas-inner {{
        position: relative;
        width: {width}px;
        height: {height}px;
        transform-origin: 0 0;
        user-select: none;
      }}
      .tree-edge {{
        position: absolute;
        transform-origin: 0 50%;
        cursor: pointer;
        pointer-events: auto;
        z-index: 1;
      }}
      .tree-edge-base {{
        background: #ef4444;
      }}
      .tree-edge-alt {{
        background: #334155;
      }}
      .tree-edge-root {{
        background: #94a3b8;
      }}
      .tree-edge-terminal {{
        background: #64748b;
      }}
      .tree-edge-invalid {{
        background-image: repeating-linear-gradient(
          45deg,
          rgba(245, 158, 11, 0.9),
          rgba(245, 158, 11, 0.9) 6px,
          rgba(255, 255, 255, 0) 6px,
          rgba(255, 255, 255, 0) 12px
        );
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
        border: 2px solid #b45309;
        background: #fde68a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #0f172a;
      }}
      .tree-node-success .tree-node-circle {{
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.6);
      }}
      .tree-node-failure .tree-node-circle {{
        box-shadow: 0 0 0 3px rgba(126, 34, 206, 0.6);
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
      .tree-detail pre .action-seq {{
        font-size: 15px;
        font-weight: 700;
        display: block;
      }}
      .tree-detail pre .thought-seq {{
        font-weight: 600;
        display: block;
      }}
      .tree-detail pre .section-title {{
        font-size: 15px;
        font-weight: 700;
        display: block;
        margin: 6px 0 2px;
      }}
      .tree-detail pre .section-sep {{
        display: block;
        height: 1px;
        background: #334155;
        margin: 8px 0;
      }}
      .tree-legend {{
        font-size: 12px;
        color: #cbd5f5;
        margin-bottom: 8px;
      }}
    </style>
    <div class="tree-debug">Nodes: {len(nodes) - 1} | Edges: {len(edges)}</div>
    <div class="tree-toolbar">
      <button id="zoomOut" type="button">-</button>
      <button id="zoomIn" type="button">+</button>
      <button id="zoomReset" type="button">Reset</button>
      <input id="zoomRange" type="range" min="0.2" max="2.5" step="0.1" value="1" />
      <span id="zoomLabel" class="tree-zoom-label">100%</span>
    </div>
    <div class="tree-wrap">
      <div class="tree-canvas">
        <div class="tree-canvas-zoom" data-base-width="{width}" data-base-height="{height}">
          <div class="tree-canvas-inner">
            {''.join(edge_divs)}
            {''.join(edge_label_divs)}
            {''.join(node_divs)}
          </div>
        </div>
      </div>
      <div class="tree-detail">
        <h3>Details</h3>
        <div class="tree-legend">Node = solver responses until next help(). Red edge = base auditor action. Dark edge = alternative auditor action. Striped edge = invalid forced action. Gray edge = terminal. Leaf: blue = correct, purple = wrong.</div>
        <pre id="detailBox">Click a node or edge to view its text.</pre>
      </div>
    </div>
    <script>
      const nodeDetails = {node_details_json};
      const nodeStartIndex = {node_start_index_json};
      const edgeDetails = {edge_details_json};
      function escapeHtml(text) {{
        return text
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/\"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }}
      function renderDetail(text, startIndex) {{
        var safe = escapeHtml(text || '(no detail)');
        safe = safe.replace(/^\\[\\[SECTION\\]\\]\\s*(.*)$/gm, '<span class=\"section-title\">$1</span>');
        safe = safe.replace(/^\\[\\[ACTION\\]\\]\\s*(.*)$/gm, '<span class=\"action-seq\">$1</span>');
        safe = safe.replace(/^\\[\\[THOUGHT\\]\\]\\s*(.*)$/gm, '<span class=\"thought-seq\">$1</span>');
        safe = safe.replace(/^==========$/gm, '<span class=\"section-sep\"></span>');
        if (typeof startIndex === 'number') {{
          var offset = startIndex - 1;
          if (offset !== 0) {{
            safe = safe.replace(/<span class=\"action-seq\">(\\d+)\\.\\s/g, function(match, num) {{
              return '<span class=\"action-seq\">' + (parseInt(num, 10) + offset) + '. ';
            }});
            safe = safe.replace(/<span class=\"thought-seq\">(\\d+)\\.\\s/g, function(match, num) {{
              return '<span class=\"thought-seq\">' + (parseInt(num, 10) + offset) + '. ';
            }});
          }}
        }}
        return safe;
      }}
      function showDetail(text, startIndex) {{
        var box = document.getElementById('detailBox');
        if (!box) return;
        box.innerHTML = renderDetail(text || '(no detail)', startIndex);
      }}
      function showNode(nodeId) {{
        var startIndex = nodeStartIndex[nodeId];
        showDetail('[Solver response]\\n\\n' + (nodeDetails[nodeId] || '(no detail)'), startIndex);
      }}
      function showEdge(edgeId) {{
        showDetail('[Auditor action]\\n\\n' + (edgeDetails[edgeId] || '(no detail)'));
      }}
      window.addEventListener('DOMContentLoaded', function() {{
        var zoomWrap = document.querySelector('.tree-canvas-zoom');
        var zoomInner = document.querySelector('.tree-canvas-inner');
        var zoomRange = document.getElementById('zoomRange');
        var zoomLabel = document.getElementById('zoomLabel');
        var zoomInBtn = document.getElementById('zoomIn');
        var zoomOutBtn = document.getElementById('zoomOut');
        var zoomResetBtn = document.getElementById('zoomReset');
        var baseWidth = zoomWrap ? parseFloat(zoomWrap.getAttribute('data-base-width') || '0') : 0;
        var baseHeight = zoomWrap ? parseFloat(zoomWrap.getAttribute('data-base-height') || '0') : 0;
        function applyZoom(value) {{
          if (!zoomWrap || !zoomInner) return;
          var zoom = Math.min(2.5, Math.max(0.2, value));
          zoomInner.style.transform = 'scale(' + zoom + ')';
          zoomWrap.style.width = (baseWidth * zoom) + 'px';
          zoomWrap.style.height = (baseHeight * zoom) + 'px';
          if (zoomRange) {{
            zoomRange.value = zoom.toFixed(2);
          }}
          if (zoomLabel) {{
            zoomLabel.textContent = Math.round(zoom * 100) + '%';
          }}
        }}
        if (zoomRange) {{
          zoomRange.addEventListener('input', function() {{
            applyZoom(parseFloat(zoomRange.value));
          }});
        }}
        if (zoomInBtn) {{
          zoomInBtn.addEventListener('click', function() {{
            var current = zoomRange ? parseFloat(zoomRange.value || '1') : 1;
            applyZoom(current + 0.1);
          }});
        }}
        if (zoomOutBtn) {{
          zoomOutBtn.addEventListener('click', function() {{
            var current = zoomRange ? parseFloat(zoomRange.value || '1') : 1;
            applyZoom(current - 0.1);
          }});
        }}
        if (zoomResetBtn) {{
          zoomResetBtn.addEventListener('click', function() {{
            applyZoom(1);
          }});
        }}
        applyZoom(1);
        var canvas = document.querySelector('.tree-canvas');
        var isDragging = false;
        var startX = 0;
        var startY = 0;
        var scrollLeft = 0;
        var scrollTop = 0;
        if (canvas) {{
          canvas.addEventListener('mousedown', function(event) {{
            if (event.button !== 0) return;
            isDragging = true;
            canvas.classList.add('dragging');
            startX = event.clientX;
            startY = event.clientY;
            scrollLeft = canvas.scrollLeft;
            scrollTop = canvas.scrollTop;
          }});
          window.addEventListener('mousemove', function(event) {{
            if (!isDragging) return;
            var dx = event.clientX - startX;
            var dy = event.clientY - startY;
            canvas.scrollLeft = scrollLeft - dx;
            canvas.scrollTop = scrollTop - dy;
          }});
          window.addEventListener('mouseup', function() {{
            if (!isDragging) return;
            isDragging = false;
            canvas.classList.remove('dragging');
          }});
          canvas.addEventListener('mouseleave', function() {{
            if (!isDragging) return;
            isDragging = false;
            canvas.classList.remove('dragging');
          }});
        }}
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


def _write_tree_html(trajectory_file: str, html_body: str) -> str:
    if os.path.isdir(trajectory_file):
        out_dir = trajectory_file
    else:
        out_dir = os.path.dirname(os.path.abspath(trajectory_file)) or os.getcwd()
    out_path = os.path.join(out_dir, "webshop_tree.html")
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'></head><body>"
        f"{html_body}</body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return out_path


def main(trajectory_file: str, server_port: int = 12345) -> None:
    trajs = _load_json_trajectories(trajectory_file)
    if not trajs:
        print(f"No trajectories found in {trajectory_file}")
        return
    tree_html = _build_tree_html(trajs)
    tree_iframe = _build_tree_iframe(tree_html)
    tree_html_path = _write_tree_html(trajectory_file, tree_html)

    def _tree_component(value: str):
        return gr.HTML(
            value=value,
            apply_default_css=False,
            min_height=700,
            container=True,
        )

    def _tree_summary(trajs: list) -> str:
        nodes, edges = _build_webshop_tree(trajs)
        return f"Tree summary: {len(nodes) - 1} nodes, {len(edges)} edges."

    with gr.Blocks(theme=gr.themes.Soft(), title="WebShop Trajectory Visualizer") as interface:
        gr.Markdown("# 🧭 WebShop Trajectory Visualizer")
        with gr.Tab("WebShop Tree"):
            gr.Markdown("Click a node to see the solver action and next user message.")
            gr.Markdown(_tree_summary(trajs))
            gr.Markdown(f"Interactive HTML written to `{tree_html_path}`.")
            _tree_component(tree_iframe)

    interface.launch(share=True, server_port=server_port)


if __name__ == "__main__":
    Fire(main)
