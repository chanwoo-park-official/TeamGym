#!/usr/bin/env bash
#SBATCH --job-name=0712-necessity
#SBATCH --array=0-399%400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-necessity-%A_%a.out
#SBATCH --error=slurm_logs/0712-necessity-%A_%a.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
variants_root="$root/${NECESSITY_VARIANTS_DIR_NAME:-necessity_variants}"
qualification_root="$root/${NECESSITY_DIR_NAME:-leave_one_piu_out}"
manifest="$variants_root/necessity_manifest.tsv"
source_data="$variants_root/common_datapoints.jsonl"
task=${SLURM_ARRAY_TASK_ID:?}
manifest_row=$(sed -n "$((task + 1))p" "$manifest")
if [[ -z "$manifest_row" ]]; then
  echo "No necessity manifest row for task $task" >&2
  exit 1
fi
IFS=$'\t' read -r variant_index source_row_index dataset mode source_instance_id ablated_instance_id missing_rui_id <<< "$manifest_row"
if (( variant_index != task )); then
  echo "Manifest index $variant_index does not match task $task" >&2
  exit 1
fi

shard_root="$qualification_root/$dataset/$mode/variant_$(printf '%05d' "$variant_index")"
if [[ -d "$shard_root" ]] && find "$shard_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Necessity shard already contains output: $shard_root" >&2
  exit 1
fi
mkdir -p "$shard_root"
"${TEAMGYM_PYTHON}" -m teamgym.data.long_context_reasoning.selection \
  --input "$source_data" --output "$shard_root/input.jsonl" --index "$variant_index"
INPUT_JSONL="$shard_root/input.jsonl" SHARD_ROOT="$shard_root" \
  bash scripts/data_generation/slurm/legal_written_construction/run_semantic_qualification.sh
