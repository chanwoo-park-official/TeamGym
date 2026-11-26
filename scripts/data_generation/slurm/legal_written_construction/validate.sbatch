#!/usr/bin/env bash
#SBATCH --job-name=0712-gpt55-val
#SBATCH --array=0-399%400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-gpt55-val-%A_%a.out
#SBATCH --error=slurm_logs/0712-gpt55-val-%A_%a.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
manifest="${VALIDATION_MANIFEST:-$root/validation_manifest.tsv}"
validation_dir="${VALIDATION_DIR_NAME:-leakage_validation}"
task=${SLURM_ARRAY_TASK_ID:?}
manifest_row=$(sed -n "$((task + 1))p" "$manifest")
if [[ -z "$manifest_row" ]]; then
  echo "No manifest row for task $task in $manifest" >&2
  exit 1
fi
IFS=$'\t' read -r dataset mode row_index source_data <<< "$manifest_row"
shard_root="$root/$validation_dir/$dataset/$mode/row_$(printf '%04d' "$row_index")"
if [[ -d "$shard_root" ]] && find "$shard_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Validation shard already contains output: $shard_root" >&2
  exit 1
fi
mkdir -p "$shard_root"
"${TEAMGYM_PYTHON}" scripts/data_generation/common/select_jsonl_row.py --input "$source_data" --output "$shard_root/input.jsonl" --index "$row_index"

INPUT_ROOT="$shard_root/input.jsonl" \
OUTPUT_ROOT="$shard_root" \
MODEL=gpt-5.5 \
WORKERS=1 \
PROVIDER=bedrock_openai \
BASE_URL=https://bedrock-mantle.us-east-2.api.aws/openai/v1 \
ENDPOINT=/responses \
API_KEY_ENV=AWS_BEARER_TOKEN_BEDROCK \
CREDENTIALS_FILE=api_key/api_key.py \
TEMPERATURE=0 \
MAX_OUTPUT_TOKENS=3000 \
TIMEOUT_SEC=300 \
TRACE_PROMPTS=0 \
ALLOW_EMPTY=1 \
bash scripts/data_generation/common/run_leakage_filter.sh
