"""Pure Prolog parsing, term unification, and clause inspection for SARA."""

from __future__ import annotations

import re
from typing import Any


def parse_prolog_clauses(program: str) -> list[dict[str, Any]]:
    """Parse enough Prolog structure to build a conservative predicate dependency graph."""

    raw_clauses = split_prolog_clauses(strip_prolog_comments(program))
    parsed: list[dict[str, Any]] = []
    for text in raw_clauses:
        if not text or text.lstrip().startswith(":-"):
            continue
        head, body = split_prolog_rule(text)
        head_signature = term_signature(head)
        if head_signature is None:
            continue
        parsed.append(
            {
                "text": text.strip(),
                "head": head,
                "body": body,
                "head_term": parse_prolog_term(head),
                "head_signature": head_signature,
                "body_signatures": extract_body_signatures(body),
                "body_calls": extract_body_calls(body),
            }
        )
    return parsed


def parse_prolog_term(value: str) -> dict[str, Any] | None:
    """Parse the callable subset needed for grounded trace-clause unification."""

    text = value.strip()
    if not text:
        return None
    if ":" in text and "(" in text and text.index(":") < text.index("("):
        text = text.split(":", 1)[1].strip()
    if text.startswith("[") and text.endswith("]"):
        return parse_prolog_list(text)
    opening = text.find("(")
    if opening < 0 or not text.endswith(")"):
        return {"functor": None, "args": [], "atom": text}
    functor = text[:opening].strip()
    if not re.fullmatch(r"[a-z][A-Za-z0-9_]*", functor):
        return None
    return {
        "functor": functor,
        "args": [parse_prolog_term(argument) for argument in split_prolog_args(text[opening + 1 : -1])],
        "atom": None,
    }


def extract_body_calls(body: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    collection_ranges: list[tuple[int, int]] = []
    for outer in re.finditer(r"(?<![A-Za-z0-9_])(?:findall|bagof|setof)\s*\(", body):
        opening = body.find("(", outer.start())
        closing = matching_delimiter(body, opening)
        if closing is not None:
            collection_ranges.append((outer.start(), closing))
    for match in re.finditer(r"(?<![A-Za-z0-9_])([a-z][A-Za-z0-9_]*)\s*\(", body):
        opening = body.find("(", match.start())
        closing = matching_delimiter(body, opening)
        if closing is None:
            continue
        text = body[match.start() : closing + 1]
        signature = term_signature(text)
        prefix = body[max(0, match.start() - 8) : match.start()]
        calls.append(
            {
                "text": text,
                "signature": signature,
                "term": parse_prolog_term(text),
                "negated": bool(re.search(r"\\\+\s*$", prefix)),
                "collection": any(start < match.start() < end for start, end in collection_ranges),
            }
        )
    return calls


def parse_prolog_list(value: str) -> dict[str, Any]:
    content = value.strip()[1:-1].strip()
    if not content:
        return {"functor": None, "args": [], "atom": "[]"}
    left, tail = split_list_tail(content)
    items = [parse_prolog_term(item) for item in split_prolog_args(left)] if left else []
    result = parse_prolog_term(tail) if tail is not None else {
        "functor": None,
        "args": [],
        "atom": "[]",
    }
    for item in reversed(items):
        result = {"functor": ".", "args": [item, result], "atom": None}
    return result


def split_list_tail(value: str) -> tuple[str, str | None]:
    depth = 0
    quote: str | None = None
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "|" and depth == 0:
            return value[:index].strip(), value[index + 1 :].strip()
    return value, None


def unify_prolog_terms(
    pattern: dict[str, Any] | None,
    value: dict[str, Any] | None,
    bindings: dict[str, str],
) -> bool:
    if pattern is None or value is None:
        return False
    pattern_atom = pattern.get("atom")
    value_atom = value.get("atom")
    if pattern_atom is not None:
        if is_prolog_variable(str(pattern_atom)):
            variable = str(pattern_atom)
            serialized = serialize_prolog_term(value)
            if variable == "_" or variable.startswith("_"):
                return True
            previous = bindings.get(variable)
            if previous is None:
                bindings[variable] = serialized
                return True
            return canonical_prolog_atom(previous) == canonical_prolog_atom(serialized)
        if value_atom is None:
            return False
        return canonical_prolog_atom(str(pattern_atom)) == canonical_prolog_atom(str(value_atom))
    if value_atom is not None:
        return False
    if pattern.get("functor") != value.get("functor"):
        return False
    pattern_args = pattern.get("args", [])
    value_args = value.get("args", [])
    return len(pattern_args) == len(value_args) and all(
        unify_prolog_terms(left, right, bindings)
        for left, right in zip(pattern_args, value_args)
    )


def serialize_prolog_term(term: dict[str, Any]) -> str:
    if term.get("atom") is not None:
        return str(term["atom"])
    return f"{term.get('functor')}({','.join(serialize_prolog_term(arg) for arg in term.get('args', []))})"


def instantiate_prolog_term(
    term: dict[str, Any] | None,
    bindings: dict[str, str],
) -> str | None:
    if term is None:
        return None
    atom = term.get("atom")
    if atom is not None:
        text = str(atom)
        return bindings.get(text, text) if is_prolog_variable(text) else text
    arguments = [
        instantiate_prolog_term(argument, bindings)
        for argument in term.get("args", [])
    ]
    if any(argument is None for argument in arguments):
        return None
    return f"{term.get('functor')}({','.join(str(argument) for argument in arguments)})"


def canonical_prolog_atom(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return re.sub(r"(?<=\d)_(?=\d)", "", text)


def is_prolog_variable(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z_][A-Za-z0-9_]*", value.strip()))


def strip_prolog_comments(program: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", " ", program, flags=re.S)
    lines: list[str] = []
    for raw_line in without_blocks.splitlines():
        quote: str | None = None
        escaped = False
        kept: list[str] = []
        for char in raw_line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quote:
                kept.append(char)
                escaped = True
                continue
            if quote:
                kept.append(char)
                if char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
                kept.append(char)
            elif char == "%":
                break
            else:
                kept.append(char)
        lines.append("".join(kept))
    return "\n".join(lines)


def split_prolog_clauses(program: str) -> list[str]:
    clauses: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(program):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "." and depth == 0:
            previous = program[index - 1] if index else ""
            following = program[index + 1] if index + 1 < len(program) else ""
            if previous.isdigit() and following.isdigit():
                continue
            clause = program[start:index].strip()
            if clause:
                clauses.append(clause)
            start = index + 1
    tail = program[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses


def split_prolog_rule(clause: str) -> tuple[str, str]:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(len(clause) - 1):
        char = clause[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and clause[index : index + 2] == ":-":
            return clause[:index].strip(), clause[index + 2 :].strip()
    return clause.strip(), ""


def extract_body_signatures(body: str) -> list[str]:
    signatures: list[str] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])([a-z][A-Za-z0-9_]*)\s*\(", body):
        opening = body.find("(", match.start())
        closing = matching_delimiter(body, opening)
        if closing is None:
            continue
        signature = term_signature(body[match.start() : closing + 1])
        if signature and signature not in signatures:
            signatures.append(signature)
    return signatures


def matching_delimiter(value: str, opening: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(value)):
        char = value[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def term_signature(term: str) -> str | None:
    text = term.strip()
    match = re.match(r"(?:[a-z][A-Za-z0-9_]*:)?([a-z][A-Za-z0-9_]*)\s*(?:\((.*)\))?$", text, re.S)
    if not match:
        return None
    arguments = match.group(2)
    arity = len(split_prolog_args(arguments)) if arguments is not None else 0
    return f"{match.group(1)}/{arity}"


def goal_signature(goal: str) -> str | None:
    parsed = parse_goal(goal)
    return f"{parsed[0]}/{len(parsed[1])}" if parsed else term_signature(goal)


def parse_goal(goal: str) -> tuple[str, list[str]] | None:
    text = goal.strip()
    if ":" in text and text.index(":") < text.find("("):
        text = text.split(":", 1)[1]
    match = re.fullmatch(r"([a-z][a-zA-Z0-9_]*)\((.*)\)", text)
    if not match:
        return None
    return match.group(1), split_prolog_args(match.group(2))


def split_prolog_args(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def clean_prolog_atom(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return text
