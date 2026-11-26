"""Build one TeamGym MLCR instance from an external case and its source questions."""

import argparse
import json
from pathlib import Path

from teamgym.data.medical.mlcr.evaluation import write_evaluation_jsonl
from teamgym.data.medical.mlcr.pipeline import run_case
from teamgym.data.medical.mlcr.source import build_case_indexes, load_inspectable
from teamgym.data.medical.mlcr.source import sha256_file
from teamgym.llm.construction_client import OpenAIJsonClient, OpenAIJsonClientConfig


def main() -> None:
    from teamgym.core.information_allocation import PAPER_ALLOCATION_MODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument(
        "--min-source-questions", type=int, default=2,
        help="Minimum number of source QAs used in the generated task (default: 2).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--audit-model", "--verification-model", dest="verification_model",
                        help="Model for solver/judge, grounding and leakage checks; defaults to --model.")
    parser.add_argument("--participant-token-budget", type=int, default=40_000)
    parser.add_argument("--allocation-modes", nargs="+", choices=(*PAPER_ALLOCATION_MODES, "all"),
                        help="Audit additional distributions of the fixed task; 'all' requests all five modes.")
    parser.add_argument("--construction-only", action="store_true",
                        help="Freeze task/PIUs after grounding and boundary checks, before distribution.")
    parser.add_argument("--max-input-tokens", type=int, default=180_000)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--source-revision")
    parser.add_argument("--allow-ownership-merges", action="store_true",
                        help="Use the admissible merge distribution; default is strict one PIU per participant.")
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--leakage-rounds", type=int, default=3)
    parser.add_argument("--mapping-source-tokens", type=int, default=120000)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()
    if args.min_source_questions < 1:
        parser.error("--min-source-questions must be at least 1")
    if min(args.participant_token_budget, args.max_input_tokens) < 1:
        parser.error("--participant-token-budget and --max-input-tokens must be positive")
    if args.prompt_id and len(set(args.prompt_id)) < args.min_source_questions:
        parser.error("--prompt-id must supply at least --min-source-questions distinct IDs")

    summaries, questions = load_inspectable(args.source_dir)
    case = build_case_indexes(
        (row for row in summaries if str(row["case_uuid"]) == args.case_id),
        (row for row in questions if str(row["case_uuid"]) == args.case_id),
    )
    if not case:
        parser.error(f"unknown case ID: {args.case_id}")
    case[0]["source_lineage"] = {"source_revision": args.source_revision,
        "source_files": {name: {"path": str((args.source_dir / name).resolve()),
                                "sha256": sha256_file(args.source_dir / name)}
                         for name in ("cases_summaries.jsonl", "questions.jsonl")
                         if (args.source_dir / name).is_file()}}

    def client(model: str | None) -> OpenAIJsonClient:
        options = {"model": model} if model is not None else {}
        options.update({k: getattr(args, k) for k in ("provider", "base_url", "api_key_env") if getattr(args, k)})
        config = OpenAIJsonClientConfig(env_file=args.env_file, temperature=None,
            reasoning_effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens,
            request_log_path=str(args.output_root / "llm_requests.jsonl"), **options)
        return OpenAIJsonClient(config)

    result = run_case(
        case_index=case[0],
        output_root=args.output_root,
        generation_client=client(args.model),
        verification_client=client(args.verification_model or args.model),
        prompt_ids=args.prompt_id,
        min_source_questions=args.min_source_questions,
        strict_one_each=not args.allow_ownership_merges,
        generation_attempts=args.generation_attempts, attempts=args.validation_attempts,
        leakage_rounds=args.leakage_rounds, mapping_source_token_budget=args.mapping_source_tokens,
        participant_token_budget=args.participant_token_budget,
        max_input_tokens=args.max_input_tokens,
        construction_only=args.construction_only or bool(args.allocation_modes),
    )
    if result["status"] == "accepted":
        write_evaluation_jsonl(
            [args.output_root / "run_report.json"], args.output_root / "common_datapoints.jsonl"
        )
    response = {"case_id": args.case_id, "status": result["status"]}
    if args.allocation_modes:
        from teamgym.data.medical.allocations import run_allocations
        response["allocations"] = run_allocations(dataset="mlcr", source_run=args.output_root,
            source_dir=args.source_dir, output_root=args.output_root / "allocations",
            client=client(args.verification_model or args.model),
            modes=PAPER_ALLOCATION_MODES if "all" in args.allocation_modes else args.allocation_modes,
            token_budget=args.participant_token_budget, max_input_tokens=args.max_input_tokens,
            attempts=args.validation_attempts, repair_rounds=args.leakage_rounds)
    print(json.dumps(response))


if __name__ == "__main__":
    main()
