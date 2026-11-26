#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${INPUT_ROOT:-}" ]]; then
  echo "Set INPUT_ROOT to a common_datapoints.jsonl file." >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs/long_context_reasoning_leakage_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-gpt-5.5}"
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-}"
PROVIDER="${PROVIDER:-bedrock_openai}"
BASE_URL="${BASE_URL:-https://bedrock-mantle.us-east-2.api.aws/openai/v1}"
ENDPOINT="${ENDPOINT:-/responses}"
API_KEY_ENV="${API_KEY_ENV:-AWS_BEARER_TOKEN_BEDROCK}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-api_key/api_key.py}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-3000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.7}"
MAX_UNIT_CHARS="${MAX_UNIT_CHARS:-2500}"
MAX_PARTICIPANT_CONTEXT_CHARS="${MAX_PARTICIPANT_CONTEXT_CHARS:-20000}"
TRACE_PROMPTS="${TRACE_PROMPTS:-0}"
ALLOW_EMPTY="${ALLOW_EMPTY:-0}"

leakage_root="$OUTPUT_ROOT/01_leakage_deletion"
args=(
  --input-root "$INPUT_ROOT"
  --output-root "$leakage_root"
  --workers "$WORKERS"
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --max-unit-chars "$MAX_UNIT_CHARS"
  --max-participant-context-chars "$MAX_PARTICIPANT_CONTEXT_CHARS"
  --model "$MODEL"
  --provider "$PROVIDER"
  --base-url "$BASE_URL"
  --endpoint "$ENDPOINT"
  --api-key-env "$API_KEY_ENV"
  --credentials-file "$CREDENTIALS_FILE"
  --temperature "$TEMPERATURE"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --timeout-sec "$TIMEOUT_SEC"
  --max-rounds "$MAX_ROUNDS"
  --exclude-unconverged
)
if [[ -n "$LIMIT" ]]; then args+=(--limit "$LIMIT"); fi
if [[ "$TRACE_PROMPTS" == "1" ]]; then args+=(--trace-prompts); fi

"${TEAMGYM_PYTHON:-python}" -m teamgym.data.legal.leakage_deletion_funnel "${args[@]}"

leakage_output="$leakage_root/leakage_deleted_common_datapoints.jsonl"
if [[ ! -s "$leakage_output" ]]; then
  if [[ "$ALLOW_EMPTY" != "1" ]]; then
    echo "Leakage filtering produced no datapoints." >&2
    exit 1
  fi
  printf '{"status":"rejected_after_leakage"}\n' > "$OUTPUT_ROOT/candidate_status.json"
  echo "Candidate rejected by leakage filtering."
  exit 0
fi
echo "Leakage survivor: $leakage_output"
