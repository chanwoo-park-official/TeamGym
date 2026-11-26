"""LLM judge for long-context short-answer reasoning tasks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any

from teamgym.core.io import read_jsonl, write_json
from teamgym.core.dataset import datapoint_variant, iter_common_datapoints
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


from teamgym.scoring.short_answers import judge_one_answer, strict_judgment_passed


def judge_short_answers(
    *,
    input_root: str | Path,
    output_root: str | Path,
    answers_jsonl: str | Path | None = None,
    trajectories_root: str | Path | None = None,
    use_oracle_as_answer: bool = False,
    limit: int | None = None,
    client: OpenAIJsonClient,
    continue_on_error: bool = False,
    trace_prompts: bool = True,
    render_reports: bool = True,
) -> dict[str, Any]:
    """Judge short-answer rows against datapoint oracle rubrics."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    datapoints = list(iter_common_datapoints(input_root))
    datapoints_by_id = defaultdict(list)
    for row in datapoints:
        if row.get("instance_id"):
            datapoints_by_id[str(row["instance_id"])].append(row)
    answer_rows = load_answer_rows(
        datapoints=datapoints,
        answers_jsonl=answers_jsonl,
        trajectories_root=trajectories_root,
        use_oracle_as_answer=use_oracle_as_answer,
    )
    if limit is not None:
        answer_rows = answer_rows[: max(0, int(limit))]
    if not answer_rows:
        raise ValueError("No candidate answers found. Provide --answers-jsonl, --trajectories-root, or --use-oracle-as-answer.")

    judgments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    trace_dir = output / "judge_prompts" if trace_prompts else None
    for index, answer_row in enumerate(answer_rows, start=1):
        instance_id = str(answer_row.get("instance_id") or "")
        variant = datapoint_variant(answer_row)
        candidates = [
            row for row in datapoints_by_id.get(instance_id, [])
            if all(row.get(key) == value for key, value in variant.items())
        ]
        if answer_row.get("dataset") and (variant or len(candidates) > 1):
            candidates = [row for row in candidates if row.get("dataset") == answer_row["dataset"]]
        if len(candidates) != 1:
            error = {
                "instance_id": instance_id,
                **variant,
                "error_type": "ambiguous_datapoint" if candidates else "missing_datapoint",
                "error_message": (
                    "Multiple input rows match this answer. Supply its evidence_condition and distribution_key."
                    if candidates else "No input row matches the candidate answer's task and variant."
                ),
                "answer_row": answer_row,
            }
            errors.append(error)
            append_jsonl(output / "judge_errors.jsonl", error)
            continue
        datapoint = candidates[0]
        try:
            judgment = judge_one_answer(
                client=client,
                datapoint=datapoint,
                answer_row=answer_row,
                trace_dir=trace_dir,
                index=index,
            )
            judgments.append(judgment)
            append_jsonl(output / "judgments.jsonl", judgment)
        except Exception as exc:  # noqa: BLE001
            error = {
                "instance_id": instance_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "answer_row": answer_row,
            }
            errors.append(error)
            append_jsonl(output / "judge_errors.jsonl", error)
            if not continue_on_error:
                raise

    report = build_report(
        input_root=input_root,
        output_root=output,
        judgments=judgments,
        errors=errors,
        answer_count=len(answer_rows),
    )
    if render_reports:
        write_json(output / "judge_report.json", report)
        write_markdown_report(output / "judge_report.md", report, judgments)
        write_html_report(output / "judge_report.html", report, judgments)
    return report


def load_answer_rows(
    *,
    datapoints: list[dict[str, Any]],
    answers_jsonl: str | Path | None,
    trajectories_root: str | Path | None,
    use_oracle_as_answer: bool,
) -> list[dict[str, Any]]:
    sources = int(bool(answers_jsonl)) + int(bool(trajectories_root)) + int(bool(use_oracle_as_answer))
    if sources != 1:
        raise ValueError("Choose exactly one answer source: --answers-jsonl, --trajectories-root, or --use-oracle-as-answer.")
    if answers_jsonl:
        return normalize_answer_rows(read_jsonl(answers_jsonl), source="answers_jsonl")
    if trajectories_root:
        return load_answers_from_trajectories(trajectories_root)
    rows = []
    for datapoint in datapoints:
        oracle = datapoint.get("oracle") if isinstance(datapoint.get("oracle"), dict) else {}
        rows.append(
            {
                "instance_id": datapoint.get("instance_id"),
                "dataset": datapoint.get("dataset"),
                **datapoint_variant(datapoint),
                "answer_text": oracle.get("canonical_response") or oracle.get("short_answer") or "",
                "answer_source": "oracle_self_check",
            }
        )
    return rows


def normalize_answer_rows(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        answer_text = row.get("answer_text") or row.get("written_answer") or row.get("rationale") or row.get("answer")
        normalized.append({**row, "answer_text": str(answer_text or ""), "answer_source": row.get("answer_source") or source})
    return normalized


def load_answers_from_trajectories(root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = Path(root).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Trajectories root not found: {base}")
    for path in sorted(base.rglob("*.json")):
        if any(str(part).endswith(".prompts") for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        episode = payload.get("episode") if isinstance(payload.get("episode"), dict) else payload
        if not isinstance(episode, dict):
            continue
        final_decision = episode.get("final_decision") if isinstance(episode.get("final_decision"), dict) else {}
        if not final_decision:
            continue
        evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {}
        public_datapoint = episode.get("datapoint") if isinstance(episode.get("datapoint"), dict) else {}
        participant_baseline = (
            episode.get("participant_baseline")
            if isinstance(episode.get("participant_baseline"), dict)
            else {}
        )
        instance_id = evaluation.get("instance_id") or public_datapoint.get("instance_id")
        structure = evaluation.get("communication_structure")
        if not structure and isinstance(episode.get("communication_structure"), dict):
            structure = episode["communication_structure"].get("name")
        participant_id = participant_baseline.get("participant_id") or final_decision.get("actor_id")
        owned_rui_ids = [
            str(item)
            for item in participant_baseline.get("participant_owned_rui_ids", [])
            if str(item)
        ]
        answer_text = (
            final_decision.get("written_answer")
            or final_decision.get("rationale")
            or final_decision.get("answer")
            or ""
        )
        rows.append(
            {
                "instance_id": instance_id,
                "dataset": evaluation.get("dataset") or public_datapoint.get("dataset"),
                **datapoint_variant(public_datapoint),
                **datapoint_variant(evaluation),
                "answer_text": str(answer_text),
                "answer_source": "trajectory_final_decision",
                "communication_structure": structure,
                "participant_id": participant_id,
                "participant_owned_rui_ids": owned_rui_ids,
                "participant_owned_rui_count": len(owned_rui_ids),
                "private_sufficient_set_covered": bool(
                    ((evaluation.get("single_participant_eval") or {}).get("private_sufficient_set_covered"))
                    if isinstance(evaluation.get("single_participant_eval"), dict)
                    else False
                ),
                "trajectory_path": str(path),
                "final_answer": final_decision.get("answer"),
                "confidence": final_decision.get("confidence"),
            }
        )
    return [row for row in rows if row.get("instance_id")]


def build_report(
    *,
    input_root: str | Path,
    output_root: Path,
    judgments: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    answer_count: int,
) -> dict[str, Any]:
    scores = [float(row["judgment"]["score"]) for row in judgments]
    passed = [bool(row["judgment"]["passed"]) for row in judgments]
    strict_passed = [strict_all_step_passed(row) for row in judgments]
    by_structure: dict[str, list[dict[str, Any]]] = {}
    for row in judgments:
        by_structure.setdefault(str(row.get("communication_structure") or "unknown"), []).append(row)
    return {
        "schema": "long_context_short_answer_judge_report_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "aggregate": {
            "candidate_answer_count": answer_count,
            "judged_count": len(judgments),
            "error_count": len(errors),
            "passed_count": sum(1 for item in passed if item),
            "pass_rate": _safe_ratio(sum(1 for item in passed if item), len(passed)),
            "strict_all_step_passed_count": sum(1 for item in strict_passed if item),
            "strict_all_step_pass_rate": _safe_ratio(sum(1 for item in strict_passed if item), len(strict_passed)),
            "mean_score": round(mean(scores), 6) if scores else None,
            "mean_covered_step_rate": mean_covered_step_rate(judgments),
        },
        "by_communication_structure": {
            name: {
                "judged_count": len(rows),
                "pass_rate": _safe_ratio(sum(1 for row in rows if row["judgment"]["passed"]), len(rows)),
                "strict_all_step_pass_rate": _safe_ratio(
                    sum(1 for row in rows if strict_all_step_passed(row)),
                    len(rows),
                ),
                "mean_score": round(mean(float(row["judgment"]["score"]) for row in rows), 6) if rows else None,
            }
            for name, rows in sorted(by_structure.items())
        },
        "outputs": {
            "judgments": "judgments.jsonl",
            "errors": "judge_errors.jsonl",
            "markdown_report": "judge_report.md",
            "html_report": "judge_report.html",
        },
        "errors": errors[:20],
    }


def strict_all_step_passed(row: dict[str, Any]) -> bool:
    judgment = row.get("judgment") if isinstance(row.get("judgment"), dict) else {}
    return strict_judgment_passed(judgment)


def mean_covered_step_rate(judgments: list[dict[str, Any]]) -> float | None:
    rates = []
    for row in judgments:
        judgment = row.get("judgment") if isinstance(row.get("judgment"), dict) else {}
        total = int(judgment.get("total_step_count") or 0)
        covered = int(judgment.get("covered_step_count") or 0)
        if total > 0:
            rates.append(covered / total)
    return round(mean(rates), 6) if rates else None


def write_markdown_report(path: str | Path, report: dict[str, Any], judgments: list[dict[str, Any]]) -> None:
    lines = ["# Long-Context Short-Answer Judge Report", ""]
    for key, value in report.get("aggregate", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Examples", ""])
    for row in judgments[:20]:
        judgment = row["judgment"]
        lines.append(
            f"- `{row.get('instance_id')}` score={judgment.get('score')} "
            f"passed={judgment.get('passed')} reason={judgment.get('reason')}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html_report(path: str | Path, report: dict[str, Any], judgments: list[dict[str, Any]]) -> None:
    aggregate_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in report.get("aggregate", {}).items()
    )
    judgment_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('instance_id')))}</td>"
        f"<td>{html.escape(str(row.get('communication_structure') or ''))}</td>"
        f"<td>{html.escape(str((row.get('answer_row') or {}).get('participant_id') or ''))}</td>"
        f"<td>{html.escape(', '.join((row.get('answer_row') or {}).get('participant_owned_rui_ids') or []))}</td>"
        f"<td>{html.escape(str(row['judgment'].get('score')))}</td>"
        f"<td>{html.escape(str(row['judgment'].get('passed')))}</td>"
        f"<td>{html.escape(str(row['judgment'].get('covered_step_count')))} / {html.escape(str(row['judgment'].get('total_step_count')))}</td>"
        f"<td>{html.escape(str(row['judgment'].get('reason')))}</td>"
        "</tr>"
        for row in judgments
    )
    text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Short-Answer Judge Report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:14px 0}}td,th{{border:1px solid #ddd;padding:7px 8px;text-align:left;vertical-align:top}}th{{background:#f6f6f6}}</style>
</head><body>
<h1>Short-Answer Judge Report</h1>
<h2>Aggregate</h2><table>{aggregate_rows}</table>
<h2>Judgments</h2>
<table><thead><tr><th>Instance</th><th>Structure</th><th>Participant</th><th>Owned RUIs</th><th>Score</th><th>Passed</th><th>Steps</th><th>Reason</th></tr></thead><tbody>{judgment_rows}</tbody></table>
</body></html>
"""
    Path(path).write_text(text, encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Common datapoint JSONL or directory.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--answers-jsonl", help="JSONL rows with instance_id and answer_text.")
    parser.add_argument("--trajectories-root", help="Interaction output directory containing trajectory JSON files.")
    parser.add_argument("--use-oracle-as-answer", action="store_true", help="Judge canonical answers against themselves for smoke checks.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT, choices=["/responses", "/chat/completions"])
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--request-log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_root).expanduser().resolve()
    client = OpenAIJsonClient(
        OpenAIJsonClientConfig(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            credentials_file=args.credentials_file,
            endpoint=args.endpoint,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_sec=args.timeout_sec,
            request_retry_attempts=args.request_retry_attempts,
            request_log_path=str(output / "llm_requests.jsonl") if args.request_log else None,
            log_prompts=bool(args.trace_prompts),
            log_responses=bool(args.trace_prompts),
        )
    )
    report = judge_short_answers(
        input_root=args.input_root,
        output_root=output,
        answers_jsonl=args.answers_jsonl,
        trajectories_root=args.trajectories_root,
        use_oracle_as_answer=bool(args.use_oracle_as_answer),
        limit=args.limit,
        client=client,
        continue_on_error=bool(args.continue_on_error),
        trace_prompts=bool(args.trace_prompts),
        render_reports=bool(args.render_reports),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
