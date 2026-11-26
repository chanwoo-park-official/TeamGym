"""Parse verifier output and complete model-selected fragment attribution."""
from __future__ import annotations

import json
from pathlib import Path
import re


_TEST_LINE = re.compile(r"^\[verifier\] ✗ \[(?P<bucket>f2p|p2p)\] (?P<name>\S+)\s*$")
_SCAFFOLD = ("=== RUN", "=== PAUSE", "=== CONT", "--- FAIL", "--- PASS", "--- SKIP")
_NAME_NOISE = re.compile(r"^\[(?:f2p|p2p)\]\s*|\s+")


def _normalized(name: str) -> str:
    """A comparable test identity across report formats.

    CTRF writes `[f2p] path/to/file.ts: suite > case` where the text markers
    write `pkg.TestName/subtest`; stripping the bucket tag, whitespace, and
    separators leaves enough to match the same test from either source.
    """
    stripped = _NAME_NOISE.sub("", str(name)).lower()
    return re.sub(r"[^a-z0-9]+", "", stripped)


def _clean_detail(text: str) -> str:
    """Drop test-runner scaffolding so the cause survives, not the banner.

    Applies to both report kinds: a CTRF `trace` often opens with the same
    `=== RUN` / `--- FAIL` lines the text logs use.
    """
    kept: list[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(_SCAFFOLD):
            continue
        kept.append(stripped)
        if len(kept) >= 3:
            break
    return " | ".join(kept)[:400]


def _ctrf_failures(run_dir: Path) -> dict[str, dict]:
    """Failing tests from CTRF reports, for verifiers that emit no text markers.

    Not every task image prints ``[verifier] x [f2p] <name>``. The JavaScript and
    TypeScript images write a CTRF JSON report instead, and reading only the text
    markers there yields zero failures from a run that failed every test — which
    stage 4b then reports as "no headroom" and selects nothing. That is exactly
    what happened to query-persist: baseline 0/8 on every run, attribution
    convinced there was nothing to attribute.
    """
    rows: dict[str, dict] = {}
    for report in sorted(run_dir.glob("*/verifier/ctrf.json")):
        episode = report.parent.parent.name
        if ".invalid_" in episode or ".interrupted_" in episode:
            continue
        result_path = report.parent.parent / "result.json"
        if result_path.exists() and json.loads(result_path.read_text()).get("episode_valid") is False:
            continue
        try:
            payload = json.loads(report.read_text())
        except ValueError:
            continue
        for test in (payload.get("results") or {}).get("tests") or []:
            if test.get("status") == "passed":
                continue
            name = str(test.get("name") or "")
            bucket = "f2p" if "[f2p]" in name else "p2p"
            detail = _clean_detail(
                " ".join(str(test[key]) for key in ("message", "trace") if test.get(key)))
            row = rows.setdefault(name, {"test": name, "bucket": bucket,
                                         "episodes": [], "detail": detail})
            row["episodes"].append(episode)
            if detail and not row["detail"]:
                row["detail"] = detail
    return rows


def failing_tests(run_dir: Path) -> list[dict]:
    """Merge normalized failing-test identities across verifier report formats."""
    rows: dict[str, dict] = {}
    for row in _ctrf_failures(run_dir).values():
        rows[_normalized(row["test"])] = row
    for log in sorted(run_dir.glob("*/verifier/test-stdout.txt")):
        episode = log.parent.parent.name
        if ".invalid_" in episode or ".interrupted_" in episode:
            continue
        result_path = log.parent.parent / "result.json"
        if result_path.exists() and json.loads(result_path.read_text()).get("episode_valid") is False:
            continue
        lines = log.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            match = _TEST_LINE.match(line)
            if not match:
                continue
            name = match.group("name")
            collected: list[str] = []
            for follow in lines[index + 1:index + 20]:
                stripped = follow.strip()
                if stripped.startswith("[verifier]"):
                    break
                collected.append(stripped)
                if len(collected) >= 8:
                    break
            detail = _clean_detail("\n".join(collected))
            key = _normalized(name)
            row = rows.setdefault(key, {"test": name, "bucket": match.group("bucket"),
                                        "episodes": [], "detail": detail})
            if episode not in row["episodes"]:
                row["episodes"].append(episode)
            if detail and not row["detail"]:
                row["detail"] = detail
    return sorted(rows.values(), key=lambda r: (-len(r["episodes"]), r["test"]))


_HUNK_RANGE = re.compile(r"^@@ -(?P<start>\d+)(?:,(?P<count>\d+))? \+")


def reference_touched_ranges(reference_patch: str) -> dict[str, list[tuple[int, int]]]:
    """Return exact base-file line ranges modified by a reference patch."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    path: str | None = None
    for line in reference_patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
        elif line.startswith("@@") and path:
            match = _HUNK_RANGE.match(line)
            if match:
                start = int(match.group("start"))
                count = int(match.group("count") or 1)
                ranges.setdefault(path, []).append((start, start + max(count, 1) - 1))
    return ranges


def touches_fragment(fragment: dict, ranges: dict[str, list[tuple[int, int]]]) -> bool:
    """True when a reference hunk overlaps this fragment's base line span."""
    for start, end in ranges.get(fragment["path"], ()):
        if start <= fragment["line_end"] and end >= fragment["line_start"]:
            return True
    return False


def fragment_inventory(pool: dict, reference_patch: str | None = None) -> list[dict]:
    """Return signatures, prioritizing reference-touched fragments when known."""
    from .candidate import selectable

    ranges = reference_touched_ranges(reference_patch) if reference_patch else {}
    rows = []
    for fragment in selectable(pool):
        head = fragment["text"].strip().splitlines()
        rows.append({
            "id": fragment["id"], "path": fragment["path"], "kind": fragment["kind"],
            "name": fragment["name"], "tokens": fragment["tokens"],
            "oversized": fragment.get("oversized", False),
            "reference_touches_this": touches_fragment(fragment, ranges),
            "signature": head[0][:200] if head else "",
        })
    rows.sort(key=lambda r: (not r["reference_touches_this"], r["id"]))
    return rows
