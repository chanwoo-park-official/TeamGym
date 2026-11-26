#!/usr/bin/env python3
"""Convert no-redundancy CUAD/MAUD TeamGym tasks to hard written answers.

The source dataset is a prepared TeamGym dataset whose rows contain a serialized
common datapoint in ``teamgym_datapoint_json``.  GPT-5.5 drafts one option-free
question, its source-grounded canonical answer, an all-step rubric, and
disqualifying errors.  A separate GPT-5.5 critic call and deterministic checks
must accept the draft before it is materialized.

Only train and validation are accepted.  This intentionally leaves a locked
paper test split unopened.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable

import pandas as pd

from teamgym.core.schema import validate_common_datapoint
from teamgym.llm.openai_client import (
    OpenAIJsonClient,
    OpenAIJsonClientConfig,
)
from teamgym.rl.data import stable_json_hash


SCHEMA_VERSION = "teamgym_legal_written_answer_gpt55_v1"
COMPATIBILITY_ANSWER = "written_response"
ALLOWED_SPLITS = ("train", "validation")
FORBIDDEN_QUESTION_PATTERNS = (
    r"\bwhich\s+(?:contract[- ]review\s+)?profile\b",
    r"\bbest\s+matches?\b",
    r"\bchoose\b",
    r"\bselect\b",
    r"\bdecision_[a-z0-9]+\b",
    r"\boption\s+[a-z0-9]+\b",
    r"\bmultiple[- ]choice\b",
)

GENERATOR_SYSTEM_PROMPT = """You design adversarially difficult, option-free written-answer tasks
for a distributed legal reasoning benchmark.

The source task was multiple choice. Rewrite it as one natural, standalone contract-review
question. The question must require synthesizing every REQUIRED_RUI, which is privately split
across participants. A response based on only one participant's evidence must be incomplete.

Difficulty requirements:
1. Do not expose choices, candidate values, the original answer label, or a menu of alternatives.
2. Do not ask which profile/option best matches. Ask for a concrete legal characterization.
3. State the dimensions that a complete answer must address neutrally, without supplying their
   values. Preserve material parties, objects, timeframes, conditions, exceptions, directions,
   and quantitative bases in the answer. If the question names a party or relationship merely to
   identify the contract, do not also score identification of that already supplied value.
4. Make the answer require an integrated conclusion, not merely unrelated quotations. Difficulty
   comes from combining the decision-critical feature in every RIU, not from exhaustively
   transcribing every subclause. Do not ask for a "full", "complete", or "all-conditions"
   restatement of a long definition.
5. Ground the canonical response and every required step only in the literal REQUIRED_RUIS. Do
   not complete truncated clauses, expand [***] redactions, or import general legal doctrine or
   facts from distractors.
6. Create one atomic evidence step per necessary source proposition and a final synthesis step
   that uses all required RIUs. Every REQUIRED_RUI id must appear in at least one evidence step
   and in the final synthesis step.
7. Disqualifying errors must be material affirmative contradictions, not mere omissions.
8. Bare labels, vague answers, and answers missing any required step must fail.
9. Treat each RIU summary and the original rationale as the intended decision boundary. Preserve
   raw-text qualifiers when they change that boundary or are explicitly requested, but do not
   turn unmentioned subordinate clause details into extra answer requirements. Prefer a
   40-140-word canonical response when the evidence permits.

Return exactly one JSON object with no prose outside it."""

CRITIC_SYSTEM_PROMPT = """You are an independent, highly skeptical auditor for an option-free
distributed legal reasoning benchmark. Determine whether the proposed question and oracle form
a difficult, source-grounded task that genuinely requires every required RIU.

Fail the proposal if:
- the question reveals an answer value, gives alternatives, or remains a disguised multiple-choice prompt;
- any required RIU is not necessary to answer an explicitly requested dimension;
- the question is answerable from one RIU or general contract priors;
- the canonical response or a required step adds a material unsupported fact;
- a material party, direction, number, condition, exception, scope, or temporal qualifier is lost;
- the required steps do not include atomic evidence claims plus a genuine cross-RIU synthesis;
- the rubric would permit a partial, vague, or bare-label answer to pass;
- a listed disqualifying error is merely an omission rather than an affirmative contradiction.

Completeness is defined by the proposed question, the original rationale, and the
decision-critical RIU summaries. Raw RIU text bounds what may be claimed, but do not demand every
raw-text subclause or qualifier unless it changes an intended decision boundary or the question
explicitly requests it. A hard task should require cross-RIU synthesis, not exhaustive copying.

Return exactly one JSON object with no prose outside it."""


@dataclass(frozen=True)
class SourceTask:
    split: str
    row: dict[str, Any]
    datapoint: dict[str, Any]
    required_rui_ids: tuple[str, ...]
    source_hash: str
    task_uid: str
    source_instance_id: str


class ThreadLocalClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.local = threading.local()

    def get(self) -> OpenAIJsonClient:
        client = getattr(self.local, "client", None)
        if client is None:
            client = OpenAIJsonClient(
                OpenAIJsonClientConfig(
                    provider=self.args.provider,
                    model=self.args.model,
                    base_url=self.args.base_url,
                    api_key_env=self.args.api_key_env,
                    credentials_file=str(self.args.credentials_file),
                    endpoint=self.args.endpoint,
                    temperature=0,
                    max_output_tokens=self.args.max_output_tokens,
                    timeout_sec=self.args.timeout_sec,
                    request_retry_attempts=self.args.request_retry_attempts,
                    request_log_path=str(self.args.request_log),
                    log_prompts=True,
                    log_responses=True,
                )
            )
            self.local.client = client
        return client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=ALLOWED_SPLITS,
        dest="splits",
        help="Repeat to select splits; default: train and validation.",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Replay accepted generation records and fail if any record is missing or invalid.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--provider", default="bedrock_openai")
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-east-2.api.aws/openai/v1",
    )
    parser.add_argument("--endpoint", default="/responses")
    parser.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--credentials-file", type=Path, default=Path("api_key/api_key.py"))
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=5000,
        help=(
            "Per-call output cap. The critic can exceed 2600 tokens on long MAUD "
            "rubrics, so 5000 is the validated default."
        ),
    )
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--request-retry-attempts", type=int, default=4)
    args = parser.parse_args()
    args.splits = tuple(args.splits or ALLOWED_SPLITS)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_revisions < 0:
        parser.error("--max-revisions must be nonnegative")
    args.output_root = args.output_root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.credentials_file = args.credentials_file.expanduser().resolve()
    args.request_log = args.output_root / "gpt55_requests.jsonl"
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_tasks(args: argparse.Namespace) -> list[SourceTask]:
    tasks: list[SourceTask] = []
    for split in args.splits:
        path = args.source_root / "full_context" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row, task_uid, source_instance_id in rows_with_task_uids(
            read_jsonl(path)
        ):
            datapoint = json.loads(row["teamgym_datapoint_json"])
            sufficient_sets = [
                tuple(str(item) for item in candidate.get("rui_ids", []) if str(item))
                for candidate in (datapoint.get("oracle") or {}).get(
                    "smallest_sufficient_rui_sets", []
                )
                if isinstance(candidate, dict)
            ]
            if len(sufficient_sets) != 1 or len(sufficient_sets[0]) < 2:
                raise ValueError(
                    f"{datapoint.get('instance_id')}: expected exactly one multi-RIU sufficient set"
                )
            available_ids = {
                str(item.get("rui_id"))
                for item in datapoint.get("reference_information_units", [])
                if isinstance(item, dict)
            }
            if set(sufficient_sets[0]) != available_ids:
                raise ValueError(
                    f"{datapoint.get('instance_id')}: required RIUs must equal the retained RIU set"
                )
            tasks.append(
                SourceTask(
                    split=split,
                    row=row,
                    datapoint=datapoint,
                    required_rui_ids=sufficient_sets[0],
                    source_hash=stable_json_hash(datapoint),
                    task_uid=task_uid,
                    source_instance_id=source_instance_id,
                )
            )
    tasks.sort(key=lambda item: (item.split, item.task_uid))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No source tasks selected")
    return tasks


def rows_with_task_uids(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str, str]]:
    """Assign stable unique IDs while preserving matched-view row order.

    A few MAUD validation exports repeat the same source instance id for
    distinct allocation rows.  Full-context and RIU-only exports retain the
    same row order, so an allocation-and-occurrence suffix is stable across
    views without depending on context text.
    """

    counts: dict[str, int] = {}
    for row in rows:
        source_instance_id = str(row.get("instance_id") or "")
        counts[source_instance_id] = counts.get(source_instance_id, 0) + 1
    seen: dict[str, int] = {}
    output: list[tuple[dict[str, Any], str, str]] = []
    for row in rows:
        source_instance_id = str(row.get("instance_id") or "")
        occurrence = seen.get(source_instance_id, 0) + 1
        seen[source_instance_id] = occurrence
        if counts[source_instance_id] == 1:
            task_uid = source_instance_id
        else:
            allocation = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(row.get("allocation") or "unknown").casefold(),
            ).strip("_")
            task_uid = (
                f"{source_instance_id}__written_{allocation}_{occurrence:02d}"
            )
        output.append((row, task_uid, source_instance_id))
    return output


def task_payload(task: SourceTask) -> dict[str, Any]:
    datapoint = task.datapoint
    oracle = datapoint.get("oracle") or {}
    participants = {
        str(participant.get("participant_id")): [
            str(item) for item in participant.get("owned_rui_ids", [])
        ]
        for participant in datapoint.get("participants", [])
        if isinstance(participant, dict)
    }
    required = [
        {
            "rui_id": str(rui.get("rui_id")),
            "information_feature": str(rui.get("information_feature") or ""),
            "summary": str(rui.get("summary") or ""),
            "text": compact_rui_text(
                str(rui.get("text") or rui.get("raw_text") or "")
            ),
            "owned_by": sorted(
                participant_id
                for participant_id, ids in participants.items()
                if str(rui.get("rui_id")) in ids
            ),
        }
        for rui in datapoint.get("reference_information_units", [])
        if isinstance(rui, dict) and str(rui.get("rui_id")) in set(task.required_rui_ids)
    ]
    return {
        "instance_id": task.task_uid,
        "source_instance_id": task.source_instance_id,
        "dataset": datapoint.get("dataset"),
        "scenario_family": datapoint.get("scenario_family"),
        "contract_title": ((datapoint.get("initial_common_information") or {}).get("source_context") or {}).get(
            "contract_title"
        ),
        "original_question": (datapoint.get("task_description") or {}).get("objective")
        or (datapoint.get("task_description") or {}).get("question"),
        "original_options": datapoint.get("decision_option_definitions"),
        "original_gold_option": oracle.get("answer"),
        "original_rationale": oracle.get("rationale"),
        "required_rius": required,
        "required_rui_ids": list(task.required_rui_ids),
        "participant_ownership": participants,
        "required_output_schema": {
            "question": "one option-free standalone question",
            "canonical_response": "complete source-grounded integrated answer",
            "required_decision_steps": [
                {
                    "step_id": "stable short id",
                    "claim": "atomic claim or final synthesis claim",
                    "uses": ["one or more required RIU ids"],
                    "step_type": "evidence or synthesis",
                }
            ],
            "disqualifying_errors": [
                "material affirmative contradiction that makes an answer fail"
            ],
            "judge_rubric": {
                "require_all_steps_for_pass": True,
                "passing_score": 1.0,
                "evaluation_criteria": ["strict criteria"],
                "bare_label_fails": True,
                "vague_answer_fails": True,
                "unsupported_material_claim_fails": True,
            },
            "difficulty_audit": {
                "option_free": True,
                "question_does_not_reveal_answer": True,
                "every_required_rui_is_necessary": True,
                "single_rui_is_insufficient": True,
                "cross_participant_synthesis_required": True,
                "canonical_response_fully_grounded": True,
            },
        },
    }


def generation_prompt(
    task: SourceTask,
    *,
    previous: dict[str, Any] | None,
    feedback: list[str],
) -> str:
    payload = task_payload(task)
    payload["revision"] = {
        "previous_proposal": previous,
        "critic_or_validator_failures": feedback,
        "instruction": (
            "Produce a corrected proposal, not commentary."
            if previous is not None
            else "Produce the first proposal."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def compact_rui_text(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated; do not infer omitted text]"


def critic_prompt(task: SourceTask, proposal: dict[str, Any]) -> str:
    payload = task_payload(task)
    payload.pop("original_gold_option", None)
    payload["proposal"] = proposal
    payload["required_output_schema"] = {
        "passed": "boolean",
        "question_is_option_free": "boolean",
        "question_does_not_reveal_answer": "boolean",
        "every_required_rui_is_necessary": "boolean",
        "single_rui_is_insufficient": "boolean",
        "canonical_response_fully_grounded": "boolean",
        "canonical_response_complete": "boolean",
        "steps_cover_all_rius": "boolean",
        "synthesis_is_genuine": "boolean",
        "rubric_is_strict": "boolean",
        "failures": ["specific actionable failure"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def normalize_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def longest_shared_phrase(left: str, right: str) -> int:
    a = normalize_words(left)
    b = normalize_words(right)
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    longest = 0
    for token in a:
        current = [0]
        for index, other in enumerate(b, start=1):
            value = previous[index - 1] + 1 if token == other else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


def deterministic_failures(task: SourceTask, proposal: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    question = proposal.get("question")
    canonical = proposal.get("canonical_response")
    steps = proposal.get("required_decision_steps")
    disqualifying = proposal.get("disqualifying_errors")
    rubric = proposal.get("judge_rubric")
    audit = proposal.get("difficulty_audit")
    if not isinstance(question, str) or not 12 <= len(normalize_words(question)) <= 140:
        failures.append("question must contain 12-140 words")
        question = str(question or "")
    for pattern in FORBIDDEN_QUESTION_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            failures.append(f"question contains forbidden multiple-choice language: {pattern}")
    if not isinstance(canonical, str) or not 20 <= len(normalize_words(canonical)) <= 220:
        failures.append("canonical_response must contain 20-220 words")
        canonical = str(canonical or "")
    if longest_shared_phrase(question, canonical) >= 12:
        failures.append("question copies a 12+ word answer phrase and may leak the answer")
    required_ids = set(task.required_rui_ids)
    if not isinstance(steps, list) or len(steps) < len(required_ids) + 1:
        failures.append("required_decision_steps need atomic evidence steps plus synthesis")
        steps = []
    used_ids: set[str] = set()
    synthesis_steps = 0
    step_ids: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            failures.append(f"step {index} is not an object")
            continue
        step_id = str(step.get("step_id") or "")
        if not step_id:
            failures.append(f"step {index} lacks step_id")
        step_ids.append(step_id)
        claim = str(step.get("claim") or "")
        if len(normalize_words(claim)) < 4:
            failures.append(f"step {step_id or index} claim is too vague")
        uses = {str(item) for item in step.get("uses", []) if str(item)}
        if not uses or not uses <= required_ids:
            failures.append(f"step {step_id or index} has invalid RIU uses")
        used_ids.update(uses)
        step_type = str(step.get("step_type") or "")
        if step_type == "synthesis":
            synthesis_steps += 1
            if uses != required_ids:
                failures.append("synthesis step must use every required RIU")
        elif step_type != "evidence":
            failures.append(f"step {step_id or index} has invalid step_type")
    if len(step_ids) != len(set(step_ids)):
        failures.append("step_id values must be unique")
    if used_ids != required_ids:
        failures.append(
            f"step RIU coverage mismatch: used={sorted(used_ids)}, required={sorted(required_ids)}"
        )
    if synthesis_steps != 1:
        failures.append("exactly one synthesis step is required")
    if not isinstance(disqualifying, list) or len(disqualifying) < 2:
        failures.append("at least two material disqualifying errors are required")
    if not isinstance(rubric, dict):
        failures.append("judge_rubric must be an object")
    else:
        required_true = (
            "require_all_steps_for_pass",
            "bare_label_fails",
            "vague_answer_fails",
            "unsupported_material_claim_fails",
        )
        for key in required_true:
            if rubric.get(key) is not True:
                failures.append(f"judge_rubric.{key} must be true")
        if float(rubric.get("passing_score") or 0) != 1.0:
            failures.append("judge_rubric.passing_score must be 1.0")
        criteria = rubric.get("evaluation_criteria")
        if not isinstance(criteria, list) or len(criteria) < len(required_ids):
            failures.append("judge_rubric needs explicit evaluation criteria")
    if not isinstance(audit, dict) or not all(
        audit.get(key) is True
        for key in (
            "option_free",
            "question_does_not_reveal_answer",
            "every_required_rui_is_necessary",
            "single_rui_is_insufficient",
            "cross_participant_synthesis_required",
            "canonical_response_fully_grounded",
        )
    ):
        failures.append("all difficulty_audit checks must be true")
    return failures


def critic_failures(critic: dict[str, Any]) -> list[str]:
    required_true = (
        "passed",
        "question_is_option_free",
        "question_does_not_reveal_answer",
        "every_required_rui_is_necessary",
        "single_rui_is_insufficient",
        "canonical_response_fully_grounded",
        "canonical_response_complete",
        "steps_cover_all_rius",
        "synthesis_is_genuine",
        "rubric_is_strict",
    )
    failures = [
        f"critic marked {key}=false"
        for key in required_true
        if critic.get(key) is not True
    ]
    failures.extend(
        str(item) for item in critic.get("failures", []) if str(item).strip()
    )
    return list(dict.fromkeys(failures))


def cache_path(output_root: Path, task_uid: str) -> Path:
    digest = hashlib.sha256(task_uid.encode("utf-8")).hexdigest()[:16]
    return output_root / "generation_records" / f"{digest}.json"


def valid_cached_record(
    path: Path,
    task: SourceTask,
    *,
    allow_source_instance_id: bool = False,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("source_hash") != task.source_hash
        or record.get("accepted") is not True
    ):
        return None
    valid_ids = {task.task_uid}
    if allow_source_instance_id:
        valid_ids.add(task.source_instance_id)
    if record.get("instance_id") not in valid_ids:
        return None
    if deterministic_failures(task, record.get("proposal") or {}):
        return None
    return record


def generate_task(
    task: SourceTask,
    *,
    args: argparse.Namespace,
    clients: ThreadLocalClient,
) -> dict[str, Any]:
    path = cache_path(args.output_root, task.task_uid)
    cached = valid_cached_record(path, task)
    if cached is not None:
        return cached
    legacy_path = cache_path(args.output_root, task.source_instance_id)
    if legacy_path != path:
        legacy = valid_cached_record(
            legacy_path,
            task,
            allow_source_instance_id=True,
        )
        if legacy is not None:
            migrated = {
                **legacy,
                "instance_id": task.task_uid,
                "source_instance_id": task.source_instance_id,
                "migrated_from_legacy_cache": str(legacy_path),
            }
            write_json(path, migrated)
            return migrated
    if args.cached_only:
        raise ValueError(
            f"{task.task_uid}: no matching accepted generation record at {path}"
        )
    client = clients.get()
    previous: dict[str, Any] | None = None
    feedback: list[str] = []
    attempts: list[dict[str, Any]] = []
    for revision in range(args.max_revisions + 1):
        proposal = client.generate_json(
            prompt_id=f"{task.task_uid}.written_generator.r{revision}",
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=generation_prompt(task, previous=previous, feedback=feedback),
            trace_dir=args.output_root / "traces",
        )
        generator_meta = dict(client.last_request_meta)
        deterministic = deterministic_failures(task, proposal)
        critic: dict[str, Any] | None = None
        critic_meta: dict[str, Any] | None = None
        critic_issues: list[str] = []
        if not deterministic:
            critic = client.generate_json(
                prompt_id=f"{task.task_uid}.written_critic.r{revision}",
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=critic_prompt(task, proposal),
                trace_dir=args.output_root / "traces",
            )
            critic_meta = dict(client.last_request_meta)
            critic_issues = critic_failures(critic)
        attempts.append(
            {
                "revision": revision,
                "proposal": proposal,
                "generator_request_meta": generator_meta,
                "deterministic_failures": deterministic,
                "critic": critic,
                "critic_request_meta": critic_meta,
                "critic_failures": critic_issues,
            }
        )
        issues = deterministic + critic_issues
        if not issues:
            record = {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "instance_id": task.task_uid,
                "source_instance_id": task.source_instance_id,
                "dataset": task.datapoint.get("dataset"),
                "split": task.split,
                "source_hash": task.source_hash,
                "required_rui_ids": list(task.required_rui_ids),
                "accepted": True,
                "accepted_revision": revision,
                "proposal": proposal,
                "critic": critic,
                "generator_request_meta": generator_meta,
                "critic_request_meta": critic_meta,
                "attempts": attempts,
            }
            write_json(path, record)
            return record
        previous = proposal
        feedback = issues
    failure_record = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_id": task.task_uid,
        "source_instance_id": task.source_instance_id,
        "dataset": task.datapoint.get("dataset"),
        "split": task.split,
        "source_hash": task.source_hash,
        "required_rui_ids": list(task.required_rui_ids),
        "accepted": False,
        "attempts": attempts,
    }
    write_json(path, failure_record)
    raise RuntimeError(
        f"{task.task_uid}: no accepted proposal after "
        f"{args.max_revisions + 1} attempts"
    )


def scrub_option_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: scrub_option_metadata(item)
            for key, item in value.items()
            if "decision_option" not in str(key).casefold()
            and str(key).casefold()
            not in {
                "old_answer",
                "new_answer",
                "old_decision_set",
                "new_decision_set",
                "old_to_new",
                "new_to_old",
            }
        }
    if isinstance(value, list):
        return [scrub_option_metadata(item) for item in value]
    return value


def transform_datapoint(
    datapoint: dict[str, Any],
    *,
    record: dict[str, Any],
) -> dict[str, Any]:
    proposal = record["proposal"]
    original_oracle = datapoint.get("oracle") or {}
    required_ids = list(record["required_rui_ids"])
    initial = datapoint.get("initial_common_information")
    initial = dict(initial) if isinstance(initial, dict) else {}
    initial.pop("decision_options", None)
    initial["task"] = proposal["question"]
    initial["collaboration_note"] = (
        "The necessary contract evidence is privately distributed. A complete "
        "written answer must synthesize all requested dimensions."
    )
    task_description = {
        "title": (datapoint.get("task_description") or {}).get("title")
        or f"{datapoint.get('scenario_family')} written analysis",
        "question": proposal["question"],
        "objective": proposal["question"],
        "answer_type": "short_answer_judged",
        "response_format_note": (
            "Provide a concise written legal characterization. Address every "
            "requested dimension and preserve material conditions and exceptions."
        ),
    }
    smallest_sets = [
        {
            "rui_ids": required_ids,
            "answer": COMPATIBILITY_ANSWER,
            "why_sufficient": (
                "These distributed RIUs jointly support every required written-answer step."
            ),
        }
    ]
    oracle = {
        "answer": COMPATIBILITY_ANSWER,
        "short_answer": proposal["canonical_response"],
        "canonical_response": proposal["canonical_response"],
        "required_rui_ids": required_ids,
        "required_decision_steps": proposal["required_decision_steps"],
        "reasoning_steps": proposal["required_decision_steps"],
        "judge_rubric": proposal["judge_rubric"],
        "disqualifying_errors": proposal["disqualifying_errors"],
        "pass_policy": "all_required_decision_steps",
        "semantic_judge": {
            "enabled": True,
            "primary_answer_field": "written_answer",
            "fallback_answer_fields": ["rationale"],
        },
        "smallest_sufficient_rui_sets": smallest_sets,
        "minimum_sufficient_rui_set_size": len(required_ids),
        "necessity_audit_ref": original_oracle.get("necessity_audit_ref"),
    }
    dataset_specific = scrub_option_metadata(datapoint.get("dataset_specific") or {})
    dataset_specific.update(
        {
            "answer_type": "short_answer_judged",
            "canonical_response": proposal["canonical_response"],
            "required_decision_steps": proposal["required_decision_steps"],
            "reasoning_steps": proposal["required_decision_steps"],
            "judge_rubric": proposal["judge_rubric"],
            "disqualifying_errors": proposal["disqualifying_errors"],
            "written_answer_generation_schema": SCHEMA_VERSION,
        }
    )
    generation = scrub_option_metadata(datapoint.get("generation_metadata") or {})
    generation["written_answer_generation"] = {
        "schema_version": SCHEMA_VERSION,
        "model": record["generator_request_meta"].get("model"),
        "critic_model": (record.get("critic_request_meta") or {}).get("model"),
        "source_datapoint_sha256": record["source_hash"],
        "accepted_revision": record["accepted_revision"],
        "option_free": True,
        "locked_test_split_opened": False,
    }
    result = dict(datapoint)
    result.update(
        {
            "instance_id": record["instance_id"],
            "task_description": task_description,
            "decision_set": [COMPATIBILITY_ANSWER],
            "decision_option_definitions": {
                COMPATIBILITY_ANSWER: (
                    "Compatibility id only; semantic correctness is determined "
                    "from the written_answer field by the strict judge."
                )
            },
            "initial_common_information": initial,
            "oracle": oracle,
            "dataset_specific": dataset_specific,
            "generation_metadata": generation,
        }
    )
    result["format_validation"] = validate_common_datapoint(result)
    if not result["format_validation"].get("valid"):
        raise ValueError(
            f"{result.get('instance_id')}: transformed datapoint failed common validation"
        )
    return result


def materialize_row(source_row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    datapoint = json.loads(source_row["teamgym_datapoint_json"])
    transformed = transform_datapoint(datapoint, record=record)
    row = dict(source_row)
    row["instance_id"] = record["instance_id"]
    row["answer_type"] = "short_answer_judged"
    row["datapoint_sha256"] = stable_json_hash(transformed)
    row["teamgym_datapoint_json"] = json.dumps(
        transformed, sort_keys=True, ensure_ascii=False
    )
    return row


def verify_matched_views(
    full_datapoint: dict[str, Any],
    riu_datapoint: dict[str, Any],
) -> None:
    for key in ("instance_id", "dataset"):
        if full_datapoint.get(key) != riu_datapoint.get(key):
            raise ValueError(f"matched view mismatch for {key}")
    full_ruis = {
        str(item.get("rui_id")): (
            str(item.get("text") or item.get("raw_text") or ""),
            str(item.get("summary") or ""),
        )
        for item in full_datapoint.get("reference_information_units", [])
        if isinstance(item, dict)
    }
    riu_ruis = {
        str(item.get("rui_id")): (
            str(item.get("text") or item.get("raw_text") or ""),
            str(item.get("summary") or ""),
        )
        for item in riu_datapoint.get("reference_information_units", [])
        if isinstance(item, dict)
    }
    if full_ruis != riu_ruis:
        raise ValueError(f"{full_datapoint.get('instance_id')}: RIU views are not matched")


def write_prepared_files(root: Path, split: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(root / f"{split}.jsonl", rows)
    common_rows = [
        json.loads(row["teamgym_datapoint_json"])
        for row in rows
    ]
    write_jsonl(root / f"{split}_common.jsonl", common_rows)
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / f"{split}.parquet", index=False)
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "TeamGym workflow task"}],
                "reward_model": {"style": "llm_judge", "ground_truth": None},
                "extra_info": row,
            }
            for row in rows
        ]
    ).to_parquet(root / f"{split}_verl.parquet", index=False)
    if split == "validation":
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            subset = [row for row in rows if str(row["dataset"]) == dataset]
            write_jsonl(root / f"validation_{dataset}.jsonl", subset)
            write_jsonl(
                root / f"validation_{dataset}_common.jsonl",
                [
                    json.loads(row["teamgym_datapoint_json"])
                    for row in subset
                ],
            )
            pd.DataFrame(subset).to_parquet(
                root / f"validation_{dataset}.parquet", index=False
            )
            pd.DataFrame(
                [
                    {
                        "prompt": [{"role": "user", "content": "TeamGym workflow task"}],
                        "reward_model": {"style": "llm_judge", "ground_truth": None},
                        "extra_info": row,
                    }
                    for row in subset
                ]
            ).to_parquet(root / f"validation_{dataset}_verl.parquet", index=False)


def materialize(
    args: argparse.Namespace,
    tasks: list[SourceTask],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_uids = {task.task_uid for task in tasks}
    task_by_uid = {task.task_uid: task for task in tasks}
    counts: dict[str, dict[str, int]] = {}
    split_ids: dict[str, set[str]] = {}
    partition_hash_match_count = 0
    transformed_row_count = 0
    public_option_leaks: list[dict[str, str]] = []
    view_semantic_hashes: dict[str, dict[str, str]] = {
        "full_context": {},
        "riu_only": {},
    }
    for variant in ("full_context", "riu_only"):
        counts[variant] = {}
        for split in args.splits:
            source_path = args.source_root / variant / f"{split}.jsonl"
            source_rows = [
                (row, task_uid, source_instance_id)
                for row, task_uid, source_instance_id in rows_with_task_uids(
                    read_jsonl(source_path)
                )
                if task_uid in selected_uids
            ]
            output_rows: list[dict[str, Any]] = []
            for row, task_uid, source_instance_id in source_rows:
                record = records[task_uid]
                if variant == "riu_only":
                    full_task = task_by_uid[task_uid]
                    verify_matched_views(
                        full_task.datapoint,
                        json.loads(row["teamgym_datapoint_json"]),
                    )
                output_row = materialize_row(row, record)
                source_datapoint = json.loads(row["teamgym_datapoint_json"])
                output_datapoint = json.loads(output_row["teamgym_datapoint_json"])
                if stable_json_hash(
                    private_partition_payload(source_datapoint)
                ) != stable_json_hash(private_partition_payload(output_datapoint)):
                    raise RuntimeError(
                        f"{task_uid}: private partition changed in {variant}"
                    )
                partition_hash_match_count += 1
                transformed_row_count += 1
                public_text = json.dumps(
                    public_task_payload(output_datapoint),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                for option_id, option_text in (
                    source_datapoint.get("decision_option_definitions") or {}
                ).items():
                    if str(option_id) in public_text or (
                        str(option_text).strip()
                        and str(option_text).strip() in public_text
                    ):
                        public_option_leaks.append(
                            {
                                "instance_id": task_uid,
                                "variant": variant,
                                "option_id": str(option_id),
                            }
                        )
                view_semantic_hashes[variant][task_uid] = stable_json_hash(
                    {
                        "task_description": output_datapoint.get(
                            "task_description"
                        ),
                        "decision_set": output_datapoint.get("decision_set"),
                        "decision_option_definitions": output_datapoint.get(
                            "decision_option_definitions"
                        ),
                        "oracle": output_datapoint.get("oracle"),
                    }
                )
                output_rows.append(output_row)
            output_rows.sort(key=lambda row: str(row["instance_id"]))
            output_ids = [str(row["instance_id"]) for row in output_rows]
            if len(output_ids) != len(set(output_ids)):
                raise RuntimeError(f"{variant}/{split}: duplicate output ids")
            write_prepared_files(args.output_root / variant, split, output_rows)
            counts[variant][split] = len(output_rows)
            if variant == "full_context":
                split_ids[split] = {str(row["instance_id"]) for row in output_rows}
    if len(split_ids) > 1:
        overlap = set.intersection(*split_ids.values())
        if overlap:
            raise RuntimeError(f"split leakage detected: {sorted(overlap)[:5]}")
    if public_option_leaks:
        raise RuntimeError(
            f"source option leakage detected: {public_option_leaks[:5]}"
        )
    if view_semantic_hashes["full_context"] != view_semantic_hashes["riu_only"]:
        raise RuntimeError("full-context and RIU-only written tasks are not matched")
    manifest = [
        {
            "instance_id": instance_id,
            "dataset": record.get("dataset"),
            "split": record.get("split"),
            "source_hash": record.get("source_hash"),
            "question": record["proposal"]["question"],
            "canonical_response": record["proposal"]["canonical_response"],
            "required_rui_ids": record["required_rui_ids"],
            "accepted_revision": record["accepted_revision"],
            "generator_model": record["generator_request_meta"].get("model"),
            "critic_model": (record.get("critic_request_meta") or {}).get("model"),
        }
        for instance_id, record in sorted(records.items())
    ]
    write_jsonl(args.output_root / "written_answer_manifest.jsonl", manifest)
    calibration_answers: list[dict[str, Any]] = []
    for task in tasks:
        instance_id = task.task_uid
        first_required = next(
            rui
            for rui in task.datapoint.get("reference_information_units", [])
            if isinstance(rui, dict)
            and str(rui.get("rui_id")) == task.required_rui_ids[0]
        )
        calibration_answers.append(
            {
                "instance_id": instance_id,
                "answer_text": str(
                    first_required.get("summary")
                    or first_required.get("text")
                    or first_required.get("raw_text")
                    or ""
                ),
                "answer_source": "single_riu_partial_control",
                "communication_structure": "single_riu_partial_control",
                "control_rui_id": task.required_rui_ids[0],
            }
        )
        original_gold = str((task.datapoint.get("oracle") or {}).get("answer") or "")
        options = task.datapoint.get("decision_option_definitions") or {}
        distractor = next(
            (
                str(text)
                for option_id, text in options.items()
                if str(option_id) != original_gold and str(text).strip()
            ),
            "",
        )
        if distractor:
            calibration_answers.append(
                {
                    "instance_id": instance_id,
                    "answer_text": distractor,
                    "answer_source": "original_non_gold_distractor_control",
                    "communication_structure": "original_non_gold_distractor_control",
                }
            )
    write_jsonl(
        args.output_root / "calibration_negative_answers.jsonl",
        calibration_answers,
    )
    dataset_audit = {
        "schema_version": "teamgym_legal_written_answer_dataset_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "locked_test_split_opened": False,
        "output_contains_test_split": any(
            path.name.startswith("test")
            for path in args.output_root.rglob("*")
            if path.is_file()
        ),
        "selected_task_count": len(tasks),
        "unique_task_uid_count": len(selected_uids),
        "transformed_row_count": transformed_row_count,
        "private_partition_hash_match_count": partition_hash_match_count,
        "private_partition_hash_mismatch_count": (
            transformed_row_count - partition_hash_match_count
        ),
        "public_source_option_leak_count": len(public_option_leaks),
        "matched_view_task_count": len(view_semantic_hashes["full_context"]),
        "matched_view_semantic_hashes_equal": True,
        "common_schema_valid_count": transformed_row_count,
        "accepted_generator_critic_record_count": len(records),
    }
    if dataset_audit["output_contains_test_split"]:
        raise RuntimeError("output unexpectedly contains a test split")
    write_json(args.output_root / "dataset_audit.json", dataset_audit)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "splits": list(args.splits),
        "locked_test_split_opened": False,
        "selected_task_count": len(tasks),
        "counts": counts,
        "datasets": dict(
            sorted(
                (
                    dataset,
                    sum(record.get("dataset") == dataset for record in records.values()),
                )
                for dataset in {str(record.get("dataset")) for record in records.values()}
            )
        ),
        "accepted_revision_counts": {
            str(revision): sum(
                int(record["accepted_revision"]) == revision for record in records.values()
            )
            for revision in sorted(
                {int(record["accepted_revision"]) for record in records.values()}
            )
        },
        "model": args.model,
        "provider": args.provider,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
    }
    write_json(args.output_root / "preparation_report.json", report)
    return report


def private_partition_payload(datapoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "participant_directory": datapoint.get("participant_directory"),
        "participants": datapoint.get("participants"),
        "private_context_units": datapoint.get("private_context_units"),
        "reference_information_units": datapoint.get(
            "reference_information_units"
        ),
    }


def public_task_payload(datapoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_description": datapoint.get("task_description"),
        "decision_set": datapoint.get("decision_set"),
        "decision_option_definitions": datapoint.get(
            "decision_option_definitions"
        ),
        "initial_common_information": datapoint.get(
            "initial_common_information"
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = source_tasks(args)
    clients = ThreadLocalClient(args)
    records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_task, task, args=args, clients=clients): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            instance_id = task.task_uid
            try:
                records[instance_id] = future.result()
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(tasks),
                            "instance_id": instance_id,
                            "accepted_revision": records[instance_id][
                                "accepted_revision"
                            ],
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "instance_id": instance_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:1000],
                    }
                )
                print(json.dumps(failures[-1]), flush=True)
    if failures:
        write_json(args.output_root / "generation_failures.json", failures)
        raise RuntimeError(f"{len(failures)} task generations failed")
    # A resumed run may have repaired every failure while leaving the previous
    # non-empty manifest behind. Keep this file authoritative for the latest run.
    write_json(args.output_root / "generation_failures.json", [])
    report = materialize(args, tasks, records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
