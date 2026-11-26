#!/usr/bin/env bash
#SBATCH --job-name=0712-partitions
#SBATCH --array=0-11%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-partitions-%A_%a.out
#SBATCH --error=slurm_logs/0712-partitions-%A_%a.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
modes=(one_each paired overlap redundant_player common_anchor bottleneck)
datasets=(mslr casebook)
task=${SLURM_ARRAY_TASK_ID:?}
dataset=${datasets[$((task / 6))]}
mode=${modes[$((task % 6))]}

case "$dataset" in
  mslr)
    seed_spec="$root/seeds/mslr100/seed/mslr_source_excerpt_casebook50_seed.jsonl"
    min_participants=3
    ;;
  casebook)
    seed_spec="$root/seeds/casebook20/seed/ecfr_casebook20_short_answer_seed.jsonl"
    min_participants=3
    ;;
esac

"${TEAMGYM_PYTHON}" -m teamgym.data.long_context_reasoning.pipeline \
  --seed-spec "$seed_spec" \
  --output-root "$root/build/$dataset/$mode" \
  --allocation-mode "$mode" \
  --target-context-tokens 12000 \
  --min-participants "$min_participants" \
  --seed 20260712
