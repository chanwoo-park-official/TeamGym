#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"

: "${INPUT_ROOT:?Set INPUT_ROOT to a normalized common-datapoint file or directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_outputs/legal_option_exclusion_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-gpt-5.5}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-20260618}"
REQUIRE_LEAKAGE_SAFE="${REQUIRE_LEAKAGE_SAFE:-0}"
PROVIDER="${PROVIDER:-bedrock_openai}"
BASE_URL="${BASE_URL:-https://bedrock-mantle.us-east-2.api.aws/openai/v1}"
ENDPOINT="${ENDPOINT:-/responses}"
API_KEY_ENV="${API_KEY_ENV:-AWS_BEARER_TOKEN_BEDROCK}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-api_key/api_key.py}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-3000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-240}"
REQUEST_RETRY_ATTEMPTS="${REQUEST_RETRY_ATTEMPTS:-4}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.7}"
MAX_UNIT_CHARS="${MAX_UNIT_CHARS:-2500}"
MAX_PARTICIPANT_CONTEXT_CHARS="${MAX_PARTICIPANT_CONTEXT_CHARS:-20000}"
TRACE_PROMPTS="${TRACE_PROMPTS:-0}"

args=(
  --input-root "$INPUT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --seed "$SEED"
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
  --request-retry-attempts "$REQUEST_RETRY_ATTEMPTS"
  --continue-on-error
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ "$SHUFFLE" == "1" ]]; then
  args+=(--shuffle)
fi

if [[ "$REQUIRE_LEAKAGE_SAFE" == "1" ]]; then
  args+=(--require-leakage-safe)
fi

if [[ "$TRACE_PROMPTS" == "1" ]]; then
  args+=(--trace-prompts)
fi

for dataset in $DATASETS; do
  args+=(--dataset "$dataset")
done

"${TEAMGYM_PYTHON:-python}" -m teamgym.data.legal.option_exclusion_verifier "${args[@]}"

cat <<EOF

Option-exclusion funnel complete.

Report:
  $OUTPUT_ROOT/option_exclusion_report.html

Evaluation input after dropping solo-oracle-reducible datapoints:
  $OUTPUT_ROOT/solo_oracle_kept_common_datapoints.jsonl

Dropped datapoints:
  $OUTPUT_ROOT/solo_oracle_dropped_datapoints.jsonl

Kept misleading-local-view datapoints:
  $OUTPUT_ROOT/misleading_single_option_common_datapoints.jsonl
EOF
