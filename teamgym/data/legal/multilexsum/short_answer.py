#!/usr/bin/env python
"""Convert original-source Multi-LexSum casebooks into neutral short-answer tasks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from teamgym.core.io import read_jsonl, write_json, write_jsonl
from teamgym.core.schema import validate_common_datapoint
from teamgym.llm.openai_client import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_ENDPOINT,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)


SYSTEM_PROMPT = """Rewrite a source-grounded legal task as a neutral short-answer casebook prompt.

The prompt must not disclose, imply, or contrast the correct outcome. Remove party names, case names, courts, judges,
locations, docket numbers, distinctive organizations, and facts that identify the historical case. Do not list answer
choices. Ask the solver to determine 2 to 4 interdependent legal or procedural conclusions from the distributed case
file and explain the reasoning connecting them. The requested conclusions should be necessary to distinguish the
canonical answer from the prior counterfactual alternatives, but the question must not state their values.

Return JSON only."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--endpoint", default=DEFAULT_LLM_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--credentials-file", default="api_key/api_key.py")
    args = parser.parse_args()

    rows = read_jsonl(args.input_root)
    selected = [(i, row) for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAIJsonClient(OpenAIJsonClientConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        credentials_file=args.credentials_file,
        temperature=0.0,
        max_output_tokens=3000,
        timeout_sec=300,
        request_retry_attempts=4,
        request_log_path=str(output / "requests.jsonl"),
        log_prompts=False,
        log_responses=False,
    ))
    accepted, errors = [], []
    for index, row in selected:
        try:
            accepted.append(convert(row, client, index))
            print(f"accepted {row['instance_id']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"instance_id": row.get("instance_id"), "error": f"{type(exc).__name__}: {exc}"})
    write_jsonl(output / "common_datapoints.jsonl", accepted)
    write_jsonl(output / "errors.jsonl", errors)
    write_json(output / "report.json", {"input": len(selected), "accepted": len(accepted), "errors": len(errors)})


def convert(row: dict[str, Any], client: OpenAIJsonClient, index: int) -> dict[str, Any]:
    oracle = row["oracle"]
    payload = {
        "current_question": row["task_description"]["question"],
        "prior_outcomes": list(row["decision_option_definitions"].values()),
        "canonical_answer": oracle["canonical_response"],
        "source_grounded_reasoning_steps": [
            {"step": step.get("step"), "claim": step.get("claim")}
            for step in oracle.get("reasoning_steps", [])
        ],
        "response_format": {
            "question": "neutral short-answer prompt",
            "required_conclusions": [
                {"conclusion": "one essential legal conclusion", "why_required": "decision-changing role"}
            ],
        },
    }
    response = client.generate_json(
        prompt_id=f"{row['instance_id']}.neutral_short_answer.{index}",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, indent=2, ensure_ascii=False),
    )
    question = normalize(response.get("question"))
    conclusions = []
    for item in response.get("required_conclusions") or []:
        if not isinstance(item, dict):
            continue
        conclusion = normalize(item.get("conclusion"))
        why = normalize(item.get("why_required"))
        if len(conclusion.split()) >= 8 and len(why.split()) >= 5:
            conclusions.append({"conclusion": conclusion, "why_required": why})
    if len(question.split()) < 20 or not 2 <= len(conclusions) <= 4:
        raise ValueError("invalid neutral question or required conclusions")
    if any(term in question.casefold() for term in ["decision_a", "decision_b", "decision_c", "decision_d", "correct outcome"]):
        raise ValueError("question leaks prior choices")

    out = copy.deepcopy(row)
    out["instance_id"] = f"{row['instance_id']}_short_answer"
    out["task_description"] = {
        "title": "Anonymized original-source procedural casebook analysis",
        "objective": question,
        "question": question,
        "answer_type": "short_answer_judged",
        "answer_format": {
            "type": "free_response",
            "instruction": "State the legal or procedural conclusion and explain the source-grounded reasoning.",
        },
    }
    out["decision_set"] = ["free_response"]
    out["decision_option_definitions"] = {"free_response": "Submit a concise source-grounded legal analysis."}
    out["initial_common_information"] = {
        "task": question,
        "decision_options": {"free_response": "Submit a concise source-grounded legal analysis."},
        "common_reference_units": [],
        "collaboration_note": "Participants hold different portions of the anonymized case file.",
    }
    required_steps = [
        {
            "step": step_index,
            "uses": oracle["required_rui_ids"],
            "claim": item["conclusion"],
            "judge_focus": item["why_required"],
        }
        for step_index, item in enumerate(conclusions, start=1)
    ]
    out["oracle"]["answer"] = "free_response"
    out["oracle"]["required_decision_steps"] = required_steps
    out["oracle"]["reasoning_steps"] = required_steps
    out["oracle"]["judge_rubric"] = {
        "require_all_steps_for_pass": True,
        "correct": "The response reaches the canonical conclusion and covers every required legal conclusion without a material contradiction.",
        "incorrect": "The response misses a required conclusion, reaches the wrong disposition, or contains a material contradiction.",
    }
    out["oracle"]["pass_policy"] = "all_required_decision_steps"
    out["dataset_specific"] = {
        **out.get("dataset_specific", {}),
        "task_variant": "original_multilexsum_distributed_short_answer",
        "answer_type": "short_answer_judged",
        "required_conclusions": conclusions,
    }
    out["generation_metadata"] = {
        **out.get("generation_metadata", {}),
        "short_answer_builder": "teamgym.data.legal.multilexsum.short_answer",
    }
    validation = validate_common_datapoint(out)
    out["format_validation"] = validation
    if not validation["valid"]:
        raise ValueError(f"schema validation failed: {validation}")
    return out


def normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


if __name__ == "__main__":
    main()
