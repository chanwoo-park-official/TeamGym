"""
Structured trajectory visualizer for environment tasks (MiniWoB / ALFWorld / WebShop).

Usage:
python rllm/structured_trajectory_visualizer.py /path/to/episode_or_directory
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import gradio as gr
from fire import Fire

SECTION_HEADERS = [
    "[Task]",
    "[Recent Window]",
    "[Memory]",
    "[Current Observation]",
    "[Current Action]",
    "[Turn Instruction]",
]

FALLBACK_SECTION_HEADERS = [
    "[Action History]",
    "[Recent Interactions]",
    "[Recent Observations]",
    "[Recent Actions]",
    "[Current Step Context]",
    "[Admissible Actions]",
]

DISPLAY_SECTION_HEADERS = SECTION_HEADERS + FALLBACK_SECTION_HEADERS + ["[Raw User Prompt]"]

RECENT_OBS_MARKER = "Below are the most recent"
RECENT_ACTION_MARKER = "Below are the actions you've taken from beginning to now:"
CURRENT_STEP_MARKER = "You are now at step"
TURN_INSTRUCTION_MARKER = "Now it's your turn to take an action."
ADMISSIBLE_ACTIONS_MARKER = "Your admissible actions of the current situation are:"

RECENT_WINDOW_ENTRY_RE = re.compile(
    r"(?m)^\s*(Observation|Action)\s+at\s+step\s+(\d+):\s*"
)

EXTRACTED_ACTION_FAILED_TEXT = "extraction failed"
ASSISTANT_REASONING_UNAVAILABLE_TEXT = "assistant reasoning unavailable"

ADMISSIBLE_INLINE_RE = re.compile(
    rf"({re.escape(ADMISSIBLE_ACTIONS_MARKER)}\s*)\[(.*?)\]",
    flags=re.IGNORECASE | re.DOTALL,
)

VALID_ACTIONS_LINE_RE = re.compile(
    r"\bThere are\s+\d+\s+valid actions you can take:\s*$",
    flags=re.IGNORECASE,
)

CLICK_EXAMPLE_RE = re.compile(r"click\['([^']+)'\]")

AUDITOR_ACTION_LABELS = {
    "notice": "Notice",
    "filteredaction": "FilteredAction",
    "try": "Try",
    "invalid": "Invalid",
}

AUDITOR_CALL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$",
    flags=re.DOTALL,
)


def _load_json_payloads(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    files: list[Path]
    if target.is_dir():
        files = sorted(target.rglob("*.json"))
    else:
        files = [target]
    payloads: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            payloads.append(
                {
                    "payload": json.load(handle),
                    "source_file": str(file),
                }
            )
    return payloads


def _last_user_text(chat_completions: list[dict[str, Any]] | None) -> str:
    if not chat_completions:
        return ""
    for msg in reversed(chat_completions):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _extract_sections(text: str) -> dict[str, str]:
    if not text:
        return {}
    sections = _extract_bracket_sections(text)
    if sections:
        return sections
    return _extract_recent_window_sections(text)


def _extract_bracket_sections(text: str) -> dict[str, str]:
    positions: list[tuple[int, str]] = []
    for header in SECTION_HEADERS:
        idx = text.find(header)
        if idx >= 0:
            positions.append((idx, header))
    if not positions:
        return {}
    positions.sort(key=lambda x: x[0])
    sections: dict[str, str] = {}
    for i, (start, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start + len(header) : end].strip()
        sections[header] = content
    return sections


def _extract_recent_window_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    text = str(text or "")

    obs_pos = text.find(RECENT_OBS_MARKER)
    act_pos = text.find(RECENT_ACTION_MARKER)
    step_pos = text.find(CURRENT_STEP_MARKER)
    turn_pos = text.find(TURN_INSTRUCTION_MARKER)

    if act_pos >= 0:
        act_end_candidates = [p for p in (obs_pos, step_pos, turn_pos) if p > act_pos]
        act_end = min(act_end_candidates) if act_end_candidates else len(text)
        sections["[Action History]"] = text[act_pos:act_end].strip()

    if obs_pos >= 0:
        obs_end_candidates = [p for p in (step_pos, turn_pos) if p > obs_pos]
        obs_end = min(obs_end_candidates) if obs_end_candidates else len(text)
        recent_window_text = text[obs_pos:obs_end].strip()
        sections["[Recent Window]"] = recent_window_text
        observations, actions = _extract_recent_window_entries(recent_window_text)
        if observations or actions:
            sections["[Recent Interactions]"] = _format_recent_interactions(
                observations,
                actions,
            )
        if observations:
            sections["[Recent Observations]"] = _format_recent_window_entries(observations)
        if actions:
            sections["[Recent Actions]"] = _format_recent_window_entries(actions)

    if step_pos >= 0:
        step_end_candidates = [p for p in (turn_pos,) if p > step_pos]
        step_end = min(step_end_candidates) if step_end_candidates else len(text)
        step_text = text[step_pos:step_end].strip()
        sections["[Current Step Context]"] = step_text
        observation_text, admissible_actions = _extract_current_context_parts(step_text)
        if observation_text:
            sections["[Current Observation]"] = observation_text
        if admissible_actions:
            sections["[Admissible Actions]"] = admissible_actions

    if turn_pos >= 0:
        sections["[Turn Instruction]"] = text[turn_pos:].strip()

    # Final fallback: split markdown-like headings if present.
    if not sections:
        heading_matches = list(re.finditer(r"(?m)^#\s+.+$", text))
        if heading_matches:
            for i, match in enumerate(heading_matches):
                start = match.start()
                end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
                heading = match.group(0).strip().removeprefix("# ").strip()
                key = f"[{heading}]"
                sections[key] = text[start:end].strip()

    return sections


def _extract_recent_window_entries(
    text: str,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    observations: list[tuple[int, str]] = []
    actions: list[tuple[int, str]] = []
    matches = list(RECENT_WINDOW_ENTRY_RE.finditer(str(text or "")))
    for idx, match in enumerate(matches):
        kind = match.group(1).strip().lower()
        step_num = int(match.group(2))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if kind == "observation":
            observations.append((step_num, content))
        else:
            actions.append((step_num, content))
    return observations, actions


def _format_recent_window_entries(entries: list[tuple[int, str]]) -> str:
    blocks: list[str] = []
    for step_num, content in entries:
        blocks.append(f"Step {step_num}:\n{content or '(empty)'}")
    return "\n\n".join(blocks).strip()


def _format_recent_interactions(
    observations: list[tuple[int, str]],
    actions: list[tuple[int, str]],
) -> str:
    obs_map = {step: content for step, content in observations}
    act_map = {step: content for step, content in actions}
    steps = sorted(set(obs_map.keys()) | set(act_map.keys()))
    blocks: list[str] = []
    for step in steps:
        obs_text = obs_map.get(step, "(missing)")
        act_text = act_map.get(step, "(missing)")
        blocks.append(
            "\n".join(
                [
                    f"Step {step}",
                    f"Observation:\n{obs_text}",
                    f"Action:\n{act_text}",
                ]
            )
        )
    return "\n\n".join(blocks).strip()


def _extract_current_context_parts(step_text: str) -> tuple[str, str]:
    text = str(step_text or "").strip()
    if not text:
        return "", ""
    marker_match = re.search(
        r"You are now at step\s+\d+\s+and your current observation:\s*",
        text,
    )
    if marker_match:
        text = text[marker_match.end() :].strip()
    marker_idx = text.find(ADMISSIBLE_ACTIONS_MARKER)
    if marker_idx < 0:
        return text.strip(), ""
    observation = text[:marker_idx].strip()
    admissible = text[marker_idx + len(ADMISSIBLE_ACTIONS_MARKER) :].strip()
    return observation, admissible


def _extract_tagged_action(text: str) -> str:
    text = str(text or "")
    if not text:
        return ""

    def _extract_last_fenced_payload(value: str) -> str:
        fenced_matches = re.findall(
            r"```(?:[a-zA-Z0-9_-]+\n)?(.*?)```|'''(.*?)'''",
            value,
            flags=re.DOTALL,
        )
        if not fenced_matches:
            return ""
        for backtick_payload, quote_payload in reversed(fenced_matches):
            candidate = (backtick_payload or quote_payload or "").strip()
            if candidate:
                return candidate
        return ""

    tag_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    action_labeled_fenced = re.findall(
        r"(?is)Action\s*:\s*(?:```(?:[a-zA-Z0-9_-]+\n)?(.*?)```|'''(.*?)''')",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if action_labeled_fenced:
        for backtick_payload, quote_payload in reversed(action_labeled_fenced):
            action_text = (backtick_payload or quote_payload or "").strip()
            if action_text:
                return action_text

    guidance_match = re.search(r"<guidance>\s*(.*?)\s*</guidance>", text, flags=re.IGNORECASE | re.DOTALL)
    if guidance_match:
        guidance_text = guidance_match.group(1).strip()
        guidance_action = _normalize_auditor_action(guidance_text)
        if guidance_action:
            return guidance_action
        if guidance_text:
            return guidance_text

    guidance_labeled_fenced = re.findall(
        r"(?is)Guidance\s*:\s*(?:```(?:[a-zA-Z0-9_-]+\n)?(.*?)```|'''(.*?)''')",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if guidance_labeled_fenced:
        for backtick_payload, quote_payload in reversed(guidance_labeled_fenced):
            guidance_text = (backtick_payload or quote_payload or "").strip()
            if not guidance_text:
                continue
            guidance_action = _normalize_auditor_action(guidance_text)
            if guidance_action:
                return guidance_action
            return guidance_text

    generic_fenced_payload = _extract_last_fenced_payload(text)
    if generic_fenced_payload:
        guidance_action = _normalize_auditor_action(generic_fenced_payload)
        if guidance_action:
            return guidance_action

    action_line_match = re.search(r"(?m)^\s*Action:\s*(.+?)\s*$", text)
    if action_line_match:
        return action_line_match.group(1).strip()
    for line in text.splitlines():
        auditor_action = _normalize_auditor_action(line)
        if auditor_action:
            return auditor_action
    return ""


def _normalize_auditor_action(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    call_match = AUDITOR_CALL_RE.match(value)
    if call_match:
        fn = call_match.group(1).strip()
        args = str(call_match.group(2) or "").strip()
        action_name = AUDITOR_ACTION_LABELS.get(fn.lower(), fn)
        return f"{action_name}({args})" if args else f"{action_name}()"
    if value.lower() in AUDITOR_ACTION_LABELS:
        return AUDITOR_ACTION_LABELS[value.lower()]
    return ""


def _extract_tagged_reasoning(text: str) -> str:
    text = str(text or "")
    if not text:
        return ""
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    # Handle responses that begin with <think> but were truncated before </think>.
    think_open_match = re.search(r"<think>\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if think_open_match:
        return think_open_match.group(1).strip()
    thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", text, flags=re.DOTALL)
    if thought_match:
        return thought_match.group(1).strip()
    reasoning_match = re.search(r"Reasoning:\s*(.*?)(?:\nAction:|$)", text, flags=re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return ""


def _resolve_assistant_response(step: dict[str, Any]) -> str:
    for field in ("model_response", "action"):
        value = str(step.get(field) or "").strip()
        if value:
            return value
    return ""


def _resolve_assistant_reasoning(assistant_response: str) -> str:
    response = str(assistant_response or "").strip()
    if not response:
        return ""
    structured_reasoning = _extract_tagged_reasoning(response)
    if structured_reasoning:
        return structured_reasoning
    # Fallback: keep raw assistant text so special tags (e.g. <guidance>) stay visible.
    return response


def _resolve_extracted_action(step: dict[str, Any]) -> str:
    info = step.get("info") or {}
    action = str(step.get("action") or "").strip()
    if info.get("action_parse_success") and action and "\n" not in action:
        return action
    for field in ("model_response", "action"):
        parsed = _extract_tagged_action(str(step.get(field) or ""))
        if parsed:
            return parsed
    # Auditor steps may not set `action_parse_success`; keep concise explicit action labels.
    if info.get("action_parse_success") is None and action and "\n" not in action:
        auditor_action = _normalize_auditor_action(action)
        if auditor_action:
            return auditor_action
    return ""


def _inline_code(value: str) -> str:
    escaped = str(value or "").replace("`", "\\`").strip() or "(empty)"
    return f"`{escaped}`"


def _fenced_text_block(text: str) -> str:
    value = str(text or "")
    max_ticks = 0
    for match in re.finditer(r"`+", value):
        max_ticks = max(max_ticks, len(match.group(0)))
    fence = "`" * max(3, max_ticks + 1)
    return f"{fence}text\n{value}\n{fence}"


def _collect_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload_obj in payloads:
        payload = payload_obj.get("payload", {})
        source_file = str(payload_obj.get("source_file") or "")
        episode_id = payload.get("episode_id", "")
        for trajectory in payload.get("trajectories", []):
            traj_name = trajectory.get("name", "unknown")
            traj_uid = trajectory.get("uid", "")
            for step_idx, step in enumerate(trajectory.get("steps", [])):
                chat = step.get("chat_completions") or []
                info = step.get("info") or {}
                assistant_response = _resolve_assistant_response(step)
                user_text = _last_user_text(chat)
                sections = _extract_sections(user_text)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "trajectory_uid": traj_uid,
                        "trajectory_name": traj_name,
                        "step_idx": step_idx,
                        "action": step.get("action"),
                        "reward": step.get("reward"),
                        "done": step.get("done"),
                        "episode_key": source_file,
                        "source_file": source_file,
                        "structured_sections_ok": info.get("structured_sections_ok"),
                        "action_parse_success": bool(info.get("action_parse_success")),
                        "extracted_action": _resolve_extracted_action(step),
                        "assistant_response": assistant_response,
                        "assistant_reasoning": _resolve_assistant_reasoning(assistant_response),
                        "user_text": user_text,
                        "sections": sections,
                        "step_json": step,
                    }
                )
    return rows


def _canonicalize_admissible_actions(actions_text: str) -> str:
    raw_items = [item.strip() for item in str(actions_text or "").split(",")]
    items = [item for item in raw_items if item]
    if not items:
        return "[]"
    sorted_items = sorted(items, key=lambda item: item.lower())
    return "[" + ", ".join(sorted_items) + "]"


def _normalize_admissible_actions_in_text(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""

    def _replace(match: re.Match[str]) -> str:
        prefix = str(match.group(1) or "")
        actions_text = str(match.group(2) or "")
        return f"{prefix}{_canonicalize_admissible_actions(actions_text)}"

    return ADMISSIBLE_INLINE_RE.sub(_replace, value)


def _normalize_admissible_actions_section(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    bracket_only_match = re.match(r"^\[(.*?)\]$", value, flags=re.DOTALL)
    if bracket_only_match:
        return _canonicalize_admissible_actions(bracket_only_match.group(1))
    return _normalize_admissible_actions_in_text(value)


def _action_block_sort_key(header_line: str) -> tuple[int, str]:
    header = str(header_line or "").strip().lower()
    if header.startswith("search["):
        return (0, header)
    if header.startswith("click["):
        return (1, header)
    if header.startswith("help:"):
        return (2, header)
    return (3, header)


def _normalize_click_examples_in_line(line: str) -> str:
    value = str(line or "")
    if "Executable examples:" not in value:
        return value
    examples = CLICK_EXAMPLE_RE.findall(value)
    if not examples:
        return value
    indent_match = re.match(r"^\s*", value)
    indent = indent_match.group(0) if indent_match else ""
    sorted_examples = sorted(set(examples), key=lambda item: item.lower())
    rendered = ", ".join(f"click['{item}']" for item in sorted_examples)
    return f"{indent}Executable examples: {rendered}"


def _normalize_valid_actions_in_text(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    lines = value.splitlines()
    if not lines:
        return value

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if not VALID_ACTIONS_LINE_RE.search(line):
            line_idx += 1
            continue

        block_start = line_idx + 1
        scan_idx = block_start
        blocks: list[list[str]] = []
        current_block: list[str] = []
        while scan_idx < len(lines):
            candidate = lines[scan_idx]
            stripped = candidate.strip()
            is_non_indented = not candidate.startswith((" ", "\t"))
            is_header = is_non_indented and ":" in candidate

            if is_header:
                if current_block:
                    blocks.append(current_block)
                current_block = [candidate]
                scan_idx += 1
                continue

            if candidate.startswith((" ", "\t")):
                if current_block:
                    current_block.append(candidate)
                    scan_idx += 1
                    continue
                break

            if not stripped:
                if current_block:
                    current_block.append(candidate)
                    scan_idx += 1
                    continue
                break

            break

        if current_block:
            blocks.append(current_block)

        if len(blocks) <= 1:
            line_idx = scan_idx if scan_idx > line_idx else line_idx + 1
            continue

        normalized_blocks: list[list[str]] = []
        for block in blocks:
            normalized_block = [_normalize_click_examples_in_line(item) for item in block]
            normalized_blocks.append(normalized_block)
        normalized_blocks.sort(key=lambda block: _action_block_sort_key(block[0]))

        replacement_lines = [entry for block in normalized_blocks for entry in block]
        lines[block_start:scan_idx] = replacement_lines
        line_idx = block_start + len(replacement_lines)

    return "\n".join(lines)


def _normalized_structured_prompt_key(row: dict[str, Any]) -> str:
    sections = row.get("sections") or {}
    if sections:
        normalized_parts: list[str] = []

        def _normalized_section_value(header: str, value: Any) -> str:
            text = str(value or "").strip()
            if header == "[Admissible Actions]":
                return _normalize_admissible_actions_section(text)
            normalized = _normalize_admissible_actions_in_text(text).strip()
            normalized = _normalize_valid_actions_in_text(normalized).strip()
            return normalized

        for header in DISPLAY_SECTION_HEADERS:
            if header not in sections:
                continue
            normalized_parts.append(
                f"{header}\n{_normalized_section_value(header, sections.get(header))}"
            )
        for header in sorted(sections.keys()):
            if header in DISPLAY_SECTION_HEADERS:
                continue
            normalized_parts.append(
                f"{header}\n{_normalized_section_value(header, sections.get(header))}"
            )
        return "\n\n".join(normalized_parts).strip()

    user_text = str(row.get("user_text") or "").strip()
    normalized_user_text = _normalize_admissible_actions_in_text(user_text).strip()
    normalized_user_text = _normalize_valid_actions_in_text(normalized_user_text).strip()
    return normalized_user_text


def _normalized_assistant_output_key(row: dict[str, Any]) -> str:
    assistant_response = str(row.get("assistant_response") or "").strip()
    if assistant_response:
        # Ignore spacing-only differences when deciding if outputs are the same.
        return re.sub(r"\s+", " ", assistant_response).strip()
    extracted_action = str(row.get("extracted_action") or "").strip()
    assistant_reasoning = str(row.get("assistant_reasoning") or "").strip()
    fallback = f"{extracted_action}\n{assistant_reasoning}".strip()
    return re.sub(r"\s+", " ", fallback).strip()


def _exclude_exact_duplicate_structured_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped_rows: list[dict[str, Any]] = []
    seen_prompt_keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        episode_key = str(row.get("episode_key") or "")
        trajectory_name = str(row.get("trajectory_name") or "")
        prompt_key = _normalized_structured_prompt_key(row)
        assistant_key = _normalized_assistant_output_key(row)
        if not prompt_key:
            deduped_rows.append(row)
            continue
        key = (episode_key, trajectory_name, prompt_key, assistant_key)
        if key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(key)
        deduped_rows.append(row)
    return deduped_rows


def _row_label(row: dict[str, Any], idx: int) -> str:
    extracted_action = str(row.get("extracted_action") or "").strip()
    label_action = extracted_action or EXTRACTED_ACTION_FAILED_TEXT
    return (
        f"#{idx} | {row['trajectory_name']} | step={row['step_idx']} | "
        f"action={label_action[:36]}"
    )


def _render_row(row: dict[str, Any]) -> tuple[str, str]:
    sections = row.get("sections", {})
    summary = [
        f"Episode: {row.get('episode_id')}",
        f"Trajectory UID: {row.get('trajectory_uid')}",
        f"Trajectory: {row.get('trajectory_name')}",
        f"Step: {row.get('step_idx')}",
        f"Source: {Path(str(row.get('source_file') or '')).name}",
        f"Action: {row.get('action')}",
        f"Reward: {row.get('reward')}",
        f"Done: {row.get('done')}",
        f"action_parse_success: {row.get('action_parse_success')}",
        f"structured_sections_ok: {row.get('structured_sections_ok')}",
    ]
    header_blocks: list[str] = []
    extracted_action = str(row.get("extracted_action") or "").strip() or EXTRACTED_ACTION_FAILED_TEXT
    header_blocks.append(f"## [Extracted Action]\n# {_inline_code(extracted_action)}")
    assistant_reasoning = (
        str(row.get("assistant_reasoning") or "").strip() or ASSISTANT_REASONING_UNAVAILABLE_TEXT
    )
    header_blocks.append(f"## [Assistant Reasoning]\n{_fenced_text_block(assistant_reasoning)}")
    assistant_response = str(row.get("assistant_response") or "").strip()
    if assistant_response:
        header_blocks.append(f"## [Response from Assistant]\n{_fenced_text_block(assistant_response)}")
    prompt_blocks: list[str] = []
    for header in DISPLAY_SECTION_HEADERS:
        value = sections.get(header)
        if value is None:
            continue
        prompt_blocks.append(f"## {header}\n{value}")
    for header, value in sections.items():
        if header in DISPLAY_SECTION_HEADERS:
            continue
        prompt_blocks.append(f"## {header}\n{value}")
    if not header_blocks and not prompt_blocks:
        user_text = str(row.get("user_text") or "").strip()
        if user_text:
            prompt_blocks.append(f"## [Raw User Prompt]\n{user_text}")
        else:
            prompt_blocks.append("No user prompt found.")

    body_parts: list[str] = []
    if header_blocks:
        body_parts.extend(header_blocks)
    if header_blocks and prompt_blocks:
        body_parts.append("---")
    if prompt_blocks:
        body_parts.extend(prompt_blocks)

    md = "\n\n".join(summary + ["", *body_parts])
    raw = json.dumps(row.get("step_json", {}), ensure_ascii=True, indent=2)
    return md, raw


def _load_tree_renderer():
    # Prefer local sibling import to avoid importing the top-level `rllm` package.
    try:
        from miniwob_trajectory_visualizer import (
            _build_miniwob_tree,
            _build_tree_html,
            _build_tree_iframe,
            _load_json_trajectories,
        )

        return _load_json_trajectories, _build_tree_html, _build_tree_iframe, _build_miniwob_tree
    except Exception:
        module_path = Path(__file__).with_name("miniwob_trajectory_visualizer.py")
        spec = importlib.util.spec_from_file_location(
            "structured_tree_renderer_miniwob", module_path
        )
        if not spec or not spec.loader:
            raise ImportError(f"Unable to load tree renderer module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module._load_json_trajectories,
            module._build_tree_html,
            module._build_tree_iframe,
            module._build_miniwob_tree,
        )


def _tree_summary(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    help_edges = [e for e in edges if e.get("edge_type") == "help"]
    base_help = [e for e in help_edges if e.get("is_base")]
    alt_help = [e for e in help_edges if not e.get("is_base")]
    invalid_help = [e for e in help_edges if e.get("invalid")]
    terminal_edges = [e for e in edges if e.get("edge_type") == "terminal"]
    root_edges = [e for e in edges if e.get("edge_type") == "root"]
    success_leaves = [n for n in nodes if n.get("leaf_status") == "success"]
    failure_leaves = [n for n in nodes if n.get("leaf_status") == "failure"]
    return (
        f"Nodes: {max(0, len(nodes) - 1)} | Edges: {len(edges)} | "
        f"Help edges: {len(help_edges)} (base={len(base_help)}, alt={len(alt_help)}, invalid={len(invalid_help)}) | "
        f"Terminal edges: {len(terminal_edges)} | Root edges: {len(root_edges)} | "
        f"Leaf status: success={len(success_leaves)}, failure={len(failure_leaves)}"
    )


def _build_tree_artifacts(payloads: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    load_trajectories, build_tree_html, build_tree_iframe, build_tree = _load_tree_renderer()
    for payload_obj in payloads:
        source_file = str(payload_obj.get("source_file") or "")
        if not source_file:
            continue
        try:
            trajs = load_trajectories(source_file)
            tree_html = build_tree_html(trajs)
            tree_iframe = build_tree_iframe(tree_html)
            nodes, edges = build_tree(trajs)
            summary = _tree_summary(nodes, edges)
            artifacts[source_file] = {
                "tree_iframe": tree_iframe,
                "summary": summary,
            }
        except Exception as exc:
            artifacts[source_file] = {
                "tree_iframe": f"<div>Tree render failed for {Path(source_file).name}: {exc}</div>",
                "summary": f"Tree render failed: {exc}",
            }
    return artifacts


def _episode_label(episode_key: str, rows: list[dict[str, Any]]) -> str:
    episode_id = str(rows[0].get("episode_id") or "")
    file_name = Path(episode_key).name
    if episode_id:
        return f"{episode_id} | {file_name} | steps={len(rows)}"
    return f"{file_name} | steps={len(rows)}"


def main(
    trajectory_path: str,
    server_port: int = 12345,
    server_name: str = "127.0.0.1",
    share: bool = False,
    exclude_exact_duplicate_structured_prompt: bool = True,
) -> None:
    payloads = _load_json_payloads(trajectory_path)
    rows = _collect_rows(payloads)
    if exclude_exact_duplicate_structured_prompt:
        rows = _exclude_exact_duplicate_structured_prompts(rows)
    if not rows:
        raise ValueError(f"No trajectory steps found in {trajectory_path}")
    tree_artifacts = _build_tree_artifacts(payloads)

    rows_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_episode.setdefault(str(row.get("episode_key") or ""), []).append(row)

    episode_keys = list(rows_by_episode.keys())
    episode_labels = [_episode_label(key, rows_by_episode[key]) for key in episode_keys]
    episode_label_to_key = dict(zip(episode_labels, episode_keys))

    row_label_to_row: dict[str, dict[str, Any]] = {}

    def _row_choices_for_episode(episode_key: str) -> list[str]:
        episode_rows = rows_by_episode.get(episode_key, [])
        choices: list[str] = []
        for i, row in enumerate(episode_rows):
            label = _row_label(row, i)
            row_label_to_row[label] = row
            choices.append(label)
        return choices

    initial_episode_label = episode_labels[0]
    initial_episode_key = episode_label_to_key[initial_episode_label]
    initial_row_choices = _row_choices_for_episode(initial_episode_key)
    initial_row_label = initial_row_choices[0]
    initial_md, initial_json = _render_row(row_label_to_row[initial_row_label])
    initial_tree = tree_artifacts.get(initial_episode_key, {})
    initial_tree_summary = initial_tree.get("summary", "Tree summary unavailable.")
    initial_tree_iframe = initial_tree.get("tree_iframe", "<div>Tree unavailable.</div>")

    def _select_episode(episode_label: str):
        episode_key = episode_label_to_key.get(episode_label, initial_episode_key)
        choices = _row_choices_for_episode(episode_key)
        selected_label = choices[0] if choices else None
        if selected_label:
            md, raw = _render_row(row_label_to_row[selected_label])
        else:
            md, raw = "No steps for selected episode.", "{}"
        tree = tree_artifacts.get(episode_key, {})
        tree_summary = tree.get("summary", "Tree summary unavailable.")
        tree_iframe = tree.get("tree_iframe", "<div>Tree unavailable.</div>")
        return (
            gr.update(choices=choices, value=selected_label),
            md,
            raw,
            f"### Branch Summary\n{tree_summary}",
            tree_iframe,
        )

    def _select_row(row_label: str):
        row = row_label_to_row.get(row_label)
        if not row:
            return "Step not found.", "{}"
        return _render_row(row)

    with gr.Blocks(title="Structured Trajectory Visualizer") as app:
        gr.Markdown(
            "### Structured Trajectory Visualizer\n"
            "Supports MiniWoB / ALFWorld / WebShop trajectories with structured prompts."
        )
        episode_selector = gr.Dropdown(
            choices=episode_labels,
            value=initial_episode_label,
            label="Episode",
        )
        step_selector = gr.Dropdown(
            choices=initial_row_choices,
            value=initial_row_label,
            label="Step",
        )
        with gr.Tab("Structured Prompt"):
            detail_md = gr.Markdown(value=initial_md)
            raw_json = gr.Code(value=initial_json, language="json", label="Raw Step JSON")
        with gr.Tab("Branch Tree"):
            tree_summary_md = gr.Markdown(value=f"### Branch Summary\n{initial_tree_summary}")
            tree_html = gr.HTML(
                value=initial_tree_iframe,
                apply_default_css=False,
                min_height=700,
                container=True,
            )

        episode_selector.change(
            _select_episode,
            inputs=[episode_selector],
            outputs=[step_selector, detail_md, raw_json, tree_summary_md, tree_html],
        )
        step_selector.change(_select_row, inputs=[step_selector], outputs=[detail_md, raw_json])

    app.launch(server_port=server_port, server_name=server_name, share=share)


if __name__ == "__main__":
    Fire(main)
