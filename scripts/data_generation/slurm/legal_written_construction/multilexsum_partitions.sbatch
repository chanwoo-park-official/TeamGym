#!/usr/bin/env bash
#SBATCH --job-name=0712-mls-part
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-mls-part-%A_%a.out
#SBATCH --error=slurm_logs/0712-mls-part-%A_%a.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
: "${MULTILEXSUM_INPUT_ROOT:?Set MULTILEXSUM_INPUT_ROOT to the constructed MultiLexSum task JSONL}"
cd "$repo"
if [[ ! -r "$MULTILEXSUM_INPUT_ROOT" || ! -f "$MULTILEXSUM_INPUT_ROOT" || ! -s "$MULTILEXSUM_INPUT_ROOT" ]]; then
  echo "MULTILEXSUM_INPUT_ROOT must name a nonempty readable task file: $MULTILEXSUM_INPUT_ROOT" >&2
  exit 2
fi

root="${TEAMGYM_DATAGEN_ROOT}"
source_data=$MULTILEXSUM_INPUT_ROOT
modes=(one_each paired overlap redundant_player common_anchor bottleneck)
mode=${modes[${SLURM_ARRAY_TASK_ID:?}]}
"${TEAMGYM_PYTHON}" scripts/data_generation/common/repartition.py \
  --input "$source_data" \
  --output-root "$root/build/multilexsum/$mode" \
  --allocation-mode "$mode" \
  --limit 30 \
  --seed 20260712
