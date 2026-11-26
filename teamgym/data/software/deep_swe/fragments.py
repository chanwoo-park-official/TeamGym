"""Extract hash-verified source fragments on language syntax boundaries.

Oversized declarations remain selectable and are marked instead of dropped.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import re
from typing import Callable, Sequence


DEFAULT_MAX_TOKENS = 400


class FragmentBudgetError(ValueError):
    """An exact line-based fragment cannot fit the configured read budget."""

# Fallback estimate; callers enforcing a ceiling should pass a tokenizer.
_CHARS_PER_TOKEN = 4


@dataclass
class Fragment:
    """One atomic source unit, verified against a pinned base commit."""

    path: str
    kind: str
    name: str
    line_start: int
    line_end: int
    text: str
    tokens: int
    sha256: str
    parent: str | None = None
    oversized: bool = False

    def as_dict(self) -> dict:
        row = {
            "path": self.path,
            "kind": self.kind,
            "name": self.name,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
            "tokens": self.tokens,
            "sha256": self.sha256,
        }
        if self.parent:
            row["parent"] = self.parent
        if self.oversized:
            row["oversized"] = True
        return row


@dataclass
class ExtractionReport:
    """Fragments plus the reasons anything was left out."""

    fragments: list[Fragment] = field(default_factory=list)
    oversized: list[Fragment] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def extend(self, other: "ExtractionReport") -> None:
        self.fragments.extend(other.fragments)
        self.oversized.extend(other.oversized)
        self.skipped.extend(other.skipped)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _slice(lines: Sequence[str], start: int, end: int) -> str:
    """1-indexed inclusive line slice, matching how atoms are recorded."""
    return "".join(lines[start - 1:end])


def _make(path: str, kind: str, name: str, start: int, end: int,
          lines: Sequence[str], count_tokens: Callable[[str], int],
          max_tokens: int, parent: str | None = None) -> Fragment:
    text = _slice(lines, start, end)
    tokens = count_tokens(text)
    return Fragment(
        path=path, kind=kind, name=name, line_start=start, line_end=end,
        text=text, tokens=tokens,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        parent=parent, oversized=tokens > max_tokens,
    )


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

def _python_decorated_start(node: ast.AST) -> int:
    """Include decorators, which are part of the declaration's contract."""
    decorators = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decorators])


def _python_class_header_end(node: ast.ClassDef) -> int:
    """Last line before the first method, so the attribute block stays intact."""
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return _python_decorated_start(item) - 1
    return node.end_lineno or node.lineno


def extract_python(path: str, source: str, *,
                   count_tokens: Callable[[str], int] = _estimate_tokens,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> ExtractionReport:
    report = ExtractionReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.skipped.append({"path": path, "reason": "syntax_error", "detail": str(exc)})
        return report
    lines = source.splitlines(keepends=True)

    def emit(fragment: Fragment) -> None:
        (report.oversized if fragment.oversized else report.fragments).append(fragment)

    # Consecutive import statements form one block, the way they read in the file.
    block: list[ast.stmt] = []

    def flush_imports() -> None:
        if not block:
            return
        start = block[0].lineno
        end = block[-1].end_lineno or block[-1].lineno
        emit(_make(path, "imports", f"imports@{start}", start, end,
                   lines, count_tokens, max_tokens))
        block.clear()

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            block.append(node)
            continue
        flush_imports()

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(_make(path, "function", node.name, _python_decorated_start(node),
                       node.end_lineno or node.lineno, lines, count_tokens, max_tokens))

        elif isinstance(node, ast.ClassDef):
            start = _python_decorated_start(node)
            header_end = _python_class_header_end(node)
            if header_end >= start:
                emit(_make(path, "class", node.name, start, header_end,
                           lines, count_tokens, max_tokens))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(_make(path, "method", f"{node.name}.{item.name}",
                               _python_decorated_start(item),
                               item.end_lineno or item.lineno,
                               lines, count_tokens, max_tokens, parent=node.name))

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            kind = "exports" if "__all__" in names else "constants"
            name = names[0] if names else f"assign@{node.lineno}"
            emit(_make(path, kind, name, node.lineno, node.end_lineno or node.lineno,
                       lines, count_tokens, max_tokens))

        else:
            report.skipped.append({"path": path, "reason": "unhandled_node",
                                   "detail": type(node).__name__, "line": node.lineno})
    flush_imports()
    return report


# --------------------------------------------------------------------------- #
# Go
# --------------------------------------------------------------------------- #

_GO_FUNC = re.compile(r"^func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>\w+)")
_GO_DECL = re.compile(r"^(?P<keyword>import|type|const|var)\b")
_GO_KIND = {"import": "imports", "type": "type", "const": "constants", "var": "constants"}


def _go_receiver_type(receiver: str) -> str:
    """`s *Scanner` -> `Scanner`, so methods group under their type."""
    parts = receiver.split()
    return parts[-1].lstrip("*") if parts else ""


def _go_block_end(lines: Sequence[str], start: int, opener: str, closer: str) -> int:
    """Scan forward from a 1-indexed line until the opener/closer balance closes.

    Strings, runes, and comments are skipped so that braces inside them do not
    shift the balance.
    """
    depth = 0
    seen = False
    index = start - 1
    while index < len(lines):
        line = lines[index]
        position = 0
        while position < len(line):
            char = line[position]
            if char in ('"', "'", "`"):
                quote = char
                position += 1
                while position < len(line):
                    if line[position] == "\\" and quote != "`":
                        position += 2
                        continue
                    if line[position] == quote:
                        break
                    position += 1
            elif char == "/" and position + 1 < len(line) and line[position + 1] == "/":
                break
            elif char == opener:
                depth += 1
                seen = True
            elif char == closer:
                depth -= 1
                if seen and depth <= 0:
                    return index + 1
            position += 1
        index += 1
    return len(lines)


def extract_go(path: str, source: str, *,
               count_tokens: Callable[[str], int] = _estimate_tokens,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> ExtractionReport:
    """Extract top-level Go declarations.

    Go top-level declarations start in column zero, so a line beginning with a
    keyword reliably opens one. Doc comments immediately above a declaration are
    included: they carry the contract a caller has to match.
    """
    report = ExtractionReport()
    lines = source.splitlines(keepends=True)

    def emit(fragment: Fragment) -> None:
        (report.oversized if fragment.oversized else report.fragments).append(fragment)

    def doc_start(declaration_line: int) -> int:
        start = declaration_line
        cursor = declaration_line - 1
        while cursor >= 1:
            stripped = lines[cursor - 1].lstrip()
            if stripped.startswith("//"):
                start = cursor
                cursor -= 1
                continue
            break
        return start

    index = 0
    while index < len(lines):
        line = lines[index]
        number = index + 1
        if not line or line[0].isspace() or line.startswith("//"):
            index += 1
            continue

        func = _GO_FUNC.match(line)
        if func:
            end = _go_block_end(lines, number, "{", "}")
            receiver = func.group("recv")
            parent = _go_receiver_type(receiver) if receiver else None
            name = f"{parent}.{func.group('name')}" if parent else func.group("name")
            emit(_make(path, "method" if parent else "function", name,
                       doc_start(number), end, lines, count_tokens, max_tokens,
                       parent=parent))
            index = end
            continue

        decl = _GO_DECL.match(line)
        if decl:
            keyword = decl.group("keyword")
            end = _go_block_end(lines, number, "(", ")") if line.rstrip().endswith("(") else number
            if keyword == "type" and "{" in line:
                end = _go_block_end(lines, number, "{", "}")
            emit(_make(path, _GO_KIND[keyword], f"{keyword}@{number}",
                       doc_start(number), end, lines, count_tokens, max_tokens))
            index = end
            continue

        if line.startswith("package "):
            index += 1
            continue

        report.skipped.append({"path": path, "reason": "unrecognised_top_level",
                               "line": number, "detail": line.rstrip()[:80]})
        index += 1
    return report


# --------------------------------------------------------------------------- #
# TypeScript / JavaScript
# --------------------------------------------------------------------------- #

_TS_DECL = re.compile(
    r"^(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?"
    r"(?P<keyword>import|export|interface|type|class|function|const|let|var|enum|namespace)\b"
    r"(?:\s+(?P<name>[A-Za-z_$][\w$]*))?")
_TS_KIND = {"import": "imports", "export": "exports", "interface": "type", "type": "type",
            "enum": "type", "namespace": "type", "class": "class",
            "function": "function", "const": "constants", "let": "constants",
            "var": "constants"}


def _ts_statement_end(lines: Sequence[str], start: int) -> int:
    """End of a top-level TS declaration: balanced braces, else the first `;`.

    Top-level declarations begin in column zero in the formatted sources these
    repositories use, so a brace-balanced scan from the opening line is exact
    enough, and every fragment is hash-verified against the base afterwards.
    """
    depth = 0
    seen_brace = False
    index = start - 1
    while index < len(lines):
        line = lines[index]
        position = 0
        while position < len(line):
            char = line[position]
            if char in ('"', "'", "`"):
                quote = char
                position += 1
                while position < len(line):
                    if line[position] == "\\":
                        position += 2
                        continue
                    if line[position] == quote:
                        break
                    position += 1
            elif char == "/" and position + 1 < len(line) and line[position + 1] == "/":
                break
            elif char in "{([":
                depth += 1
                seen_brace = seen_brace or char == "{"
                position += 1
                continue
            elif char in "})]":
                depth -= 1
                if seen_brace and depth <= 0:
                    return index + 1
                position += 1
                continue
            elif char == ";" and depth <= 0:
                return index + 1
            position += 1
        if depth <= 0 and not seen_brace and lines[index].rstrip().endswith(("}", ";")):
            return index + 1
        index += 1
    return len(lines)


def extract_typescript(path: str, source: str, *,
                       count_tokens: Callable[[str], int] = _estimate_tokens,
                       max_tokens: int = DEFAULT_MAX_TOKENS) -> ExtractionReport:
    """Extract top-level TypeScript/JavaScript declarations.

    Without this, 40 of the benchmark's 113 tasks yield an empty pool: `.ts`
    files were reported as an unsupported language, and the attribution stage
    then had nothing to choose from while still reporting a clean run.
    """
    report = ExtractionReport()
    lines = source.splitlines(keepends=True)

    def emit(fragment: Fragment) -> None:
        (report.oversized if fragment.oversized else report.fragments).append(fragment)

    def doc_start(declaration_line: int) -> int:
        start = declaration_line
        cursor = declaration_line - 1
        while cursor >= 1:
            stripped = lines[cursor - 1].strip()
            if stripped.startswith(("//", "*", "/*")) or stripped.endswith("*/"):
                start = cursor
                cursor -= 1
                continue
            break
        return start

    index = 0
    while index < len(lines):
        line = lines[index]
        number = index + 1
        if not line.strip() or line[0].isspace():
            index += 1
            continue
        match = _TS_DECL.match(line)
        if not match:
            index += 1
            continue
        keyword = match.group("keyword")
        name = match.group("name") or f"{keyword}@{number}"
        end = _ts_statement_end(lines, number)
        emit(_make(path, _TS_KIND.get(keyword, "constants"), name,
                   doc_start(number), end, lines, count_tokens, max_tokens))
        index = max(end, number)
    return report


EXTRACTORS: dict[str, Callable[..., ExtractionReport]] = {
    ".py": extract_python,
    ".go": extract_go,
    ".ts": extract_typescript,
    ".tsx": extract_typescript,
    ".js": extract_typescript,
    ".jsx": extract_typescript,
    ".mjs": extract_typescript,
}


def extract(path: str, source: str, *,
            count_tokens: Callable[[str], int] = _estimate_tokens,
            max_tokens: int = DEFAULT_MAX_TOKENS) -> ExtractionReport:
    """Dispatch on file suffix. Unsupported languages are reported, not raised."""
    for suffix, extractor in EXTRACTORS.items():
        if path.endswith(suffix):
            return extractor(path, source, count_tokens=count_tokens, max_tokens=max_tokens)
    report = ExtractionReport()
    report.skipped.append({"path": path, "reason": "unsupported_language"})
    return report


def extract_complete(path: str, source: str, *,
                     count_tokens: Callable[[str], int] = _estimate_tokens,
                     max_tokens: int = DEFAULT_MAX_TOKENS,
                     context_tokens: int = 6_000) -> ExtractionReport:
    """Retain syntax units plus exact uncovered spans for constructor selection.

    Source context is explicitly a span, not a claimed atomic declaration. This
    also supports Rust/configuration files while no syntax extractor is present.
    Large declarations are supplied as adjacent spans so no single body is
    permanently unreadable under the constructor's per-read budget.
    """
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    for line_no, line in enumerate(source.splitlines(keepends=True), 1):
        tokens = count_tokens(line)
        if tokens > context_tokens:
            raise FragmentBudgetError(
                f"{path}:{line_no}: single source line requires {tokens} tokens; "
                f"exact line-based read budget is {context_tokens}. Increase the explicit "
                "source/context budget or exclude this file with a recorded reason; "
                "repeating the same read cannot supply it. No source was truncated.")
    original = extract(path, source, count_tokens=count_tokens, max_tokens=max_tokens)
    report = ExtractionReport(skipped=list(original.skipped))
    lines = source.splitlines(keepends=True)
    covered = set()

    def emit_span(start, end, name):
        cursor = start
        while cursor <= end:
            stop = cursor
            while stop < end and count_tokens(_slice(lines, cursor, stop + 1)) <= context_tokens:
                stop += 1
            fragment = _make(path, "source_context", name + f"@{cursor}", cursor,
                             stop, lines, count_tokens, max_tokens)
            (report.oversized if fragment.oversized else report.fragments).append(fragment)
            cursor = stop + 1

    for fragment in original.fragments + original.oversized:
        covered.update(range(fragment.line_start, fragment.line_end + 1))
        if fragment.tokens > context_tokens:
            emit_span(fragment.line_start, fragment.line_end, fragment.name)
        else:
            (report.oversized if fragment.oversized else report.fragments).append(fragment)
    cursor = 1
    while cursor <= len(lines):
        if cursor in covered:
            cursor += 1
            continue
        end = cursor
        while end < len(lines) and end + 1 not in covered:
            end += 1
        # Retain comments/config as well: they can carry contracts and wiring.
        if _slice(lines, cursor, end).strip():
            emit_span(cursor, end, "unparsed_context")
        cursor = end + 1
    return report
