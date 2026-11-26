#!/usr/bin/env bash
set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"

input_jsonl=${INPUT_JSONL:?INPUT_JSONL is required}
shard_root=${SHARD_ROOT:?SHARD_ROOT is required}

"${TEAMGYM_PYTHON}" -m teamgym.interaction.run_single_agent_rui_evaluation \
  --input-root "$input_jsonl" \
  --output-dir "$shard_root/eval" \
  --limit 1 \
  --context-token-budget 40000 \
  --final-rationale-token-budget 1800 \
  --final-reasoning-token-budget 3000 \
  --model gpt-5.5 \
  --provider bedrock_openai \
  --base-url https://bedrock-mantle.us-east-2.api.aws/openai/v1 \
  --endpoint /responses \
  --api-key-env AWS_BEARER_TOKEN_BEDROCK \
  --credentials-file api_key/api_key.py \
  --temperature 0 \
  --max-output-tokens 5000 \
  --timeout-sec 300 \
  --request-retry-attempts 4 \
  --no-continue-on-error \
  --no-trace-prompts \
  --no-render-html

"${TEAMGYM_PYTHON}" -m teamgym.data.long_context_reasoning.judge \
  --input-root "$input_jsonl" \
  --trajectories-root "$shard_root/eval" \
  --output-root "$shard_root/judge" \
  --model gpt-5.5 \
  --provider bedrock_openai \
  --base-url https://bedrock-mantle.us-east-2.api.aws/openai/v1 \
  --endpoint /responses \
  --api-key-env AWS_BEARER_TOKEN_BEDROCK \
  --credentials-file api_key/api_key.py \
  --temperature 0 \
  --max-output-tokens 3500 \
  --timeout-sec 300 \
  --request-retry-attempts 4 \
  --no-continue-on-error \
  --no-trace-prompts
