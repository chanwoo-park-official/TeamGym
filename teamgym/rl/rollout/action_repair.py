"""Explicit compact-action and participant-format repair helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .action_protocol import _compact_action_has_required_field


def _repair_compact_action(
    raw: str, *, prompt_kind: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Recover labeled fields when a short action omits only JSON punctuation.

    This intentionally does not extract an unlabeled option from arbitrary
    prose: the compact policy must still emit an explicit ``answer`` or
    ``target`` field. The rollout's original tokens remain the trained action.
    """

    text = str(raw or "").strip().strip("`").strip()
    if not text:
        return None, None
    candidates = [text]
    if not text.startswith("{"):
        candidates.append("{" + text.strip().rstrip(",") + "}")
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            normalized = {str(key).casefold(): item for key, item in value.items()}
            if _compact_action_has_required_field(normalized, prompt_kind):
                return normalized, "wrapped_json"

    fields: dict[str, Any] = {}
    for match in re.finditer(
        r'''["']?(answer|why|rationale|target|ask|written_answer)["']?\s*[:=]\s*["']([^"'\r\n]*)["']''',
        text,
        flags=re.IGNORECASE,
    ):
        fields[match.group(1).casefold()] = match.group(2).strip()
    if _compact_action_has_required_field(fields, prompt_kind):
        return fields, "labeled_fields"
    return None, None


_UTTERANCE_FIELD_RE = re.compile(
    r"(?:\"utterance\"|'utterance'|utterance)\s*:\s*",
    flags=re.IGNORECASE,
)


def _nested_utterance(value: Any) -> str | None:
    """Find the first non-empty utterance string in a parsed malformed action."""

    if isinstance(value, dict):
        direct = value.get("utterance")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        for child in value.values():
            found = _nested_utterance(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _nested_utterance(child)
            if found:
                return found
    return None


def _partial_utterance_value(raw: str) -> str | None:
    """Recover an utterance value from JSON truncated inside its string."""

    matches = list(_UTTERANCE_FIELD_RE.finditer(raw))
    if not matches:
        return None
    tail = raw[matches[-1].end() :].lstrip()
    if not tail:
        return None
    if tail[0] not in {'"', "'"}:
        return tail.rstrip(" \t\r\n,}").strip() or None

    quote = tail[0]
    escaped = False
    characters: list[str] = []
    for character in tail[1:]:
        if escaped:
            # Decode the common JSON escapes without inventing missing text.
            characters.append(
                {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "b": "\b",
                    "f": "\f",
                }.get(character, character)
            )
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == quote:
            break
        characters.append(character)
    return "".join(characters).strip() or None


def _participant_counterfactual_utterance(
    raw: str, *, strategy: str
) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if strategy == "wrap_raw":
        return text
    if strategy != "extract_utterance_suffix":
        raise ValueError(f"Unknown participant format repair strategy: {strategy}")

    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    extracted = (
        parsed.strip() or None
        if isinstance(parsed, str)
        else _nested_utterance(parsed)
    )
    if extracted:
        return extracted
    extracted = _partial_utterance_value(text)
    if extracted:
        return extracted
    # A top-level quoted natural-language answer is a common truncation mode.
    return text[1:].strip() if text[:1] in {'"', "'"} else text


def _repair_participant_format_counterfactual(
    raw: str,
    *,
    prompt_kind: str,
    strategy: str = "wrap_raw",
    max_utterance_tokens: int = 0,
    tokenizer: Any | None = None,
) -> dict[str, str] | None:
    """Supply an evaluation-only valid envelope around participant content.

    ``wrap_raw`` preserves the earlier upper-bound intervention exactly.
    ``extract_utterance_suffix`` recovers the largest available utterance
    candidate, then retains its final ``max_utterance_tokens`` tokens.  The
    latter is a content-altering parser sensitivity check and is logged as a
    separate condition.
    """

    if prompt_kind not in {"participant_response", "participant_public_message"}:
        return None
    text = _participant_counterfactual_utterance(raw, strategy=strategy)
    if not text:
        return None
    max_tokens = max(0, int(max_utterance_tokens))
    if max_tokens:
        if tokenizer is None:
            raise ValueError(
                "A tokenizer is required to cap counterfactual utterance tokens"
            )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            text = tokenizer.decode(
                token_ids[-max_tokens:],
                skip_special_tokens=True,
            ).strip()
    return {"utterance": text} if text else None
