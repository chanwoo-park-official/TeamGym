#!/usr/bin/env bash
#SBATCH --job-name=0712-allriu
#SBATCH --array=0-399%400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-allriu-%A_%a.out
#SBATCH --error=slurm_logs/0712-allriu-%A_%a.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
survivor_root="$root/${LEAKAGE_SURVIVOR_DIR_NAME:-leakage_survivors}"
qualification_root="$root/${QUALIFICATION_DIR_NAME:-all_piu_qualification}"
manifest="$survivor_root/qualification_manifest.tsv"
source_data="$survivor_root/common_datapoints.jsonl"
task=${SLURM_ARRAY_TASK_ID:?}
manifest_row=$(sed -n "$((task + 1))p" "$manifest")
if [[ -z "$manifest_row" ]]; then
  echo "No qualification manifest row for task $task" >&2
  exit 1
fi
IFS=$'\t' read -r qualification_index dataset mode source_row_index instance_id <<< "$manifest_row"
if (( qualification_index != task )); then
  echo "Manifest index $qualification_index does not match task $task" >&2
  exit 1
fi

shard_root="$qualification_root/$dataset/$mode/row_$(printf '%04d' "$source_row_index")"
if [[ -d "$shard_root" ]] && find "$shard_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Qualification shard already contains output: $shard_root" >&2
  exit 1
fi
mkdir -p "$shard_root"
"${TEAMGYM_PYTHON}" -m teamgym.data.long_context_reasoning.selection \
  --input "$source_data" --output "$shard_root/input.jsonl" --index "$qualification_index"

INPUT_JSONL="$shard_root/input.jsonl" SHARD_ROOT="$shard_root" \
  bash scripts/data_generation/slurm/legal_written_construction/run_semantic_qualification.sh
