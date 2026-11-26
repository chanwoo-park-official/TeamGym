#!/usr/bin/env bash
#SBATCH --job-name=sara-subproof-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/sara-subproof-gen-%A_%a.out
#SBATCH --error=slurm_logs/sara-subproof-gen-%A_%a.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

"${TEAMGYM_PYTHON}" -m teamgym.data.legal.sara.worker \
  --audited-cases "$SARA_SUBPROOF_OUTPUT/metadata/audited_cases.json" \
  --output-root "$SARA_SUBPROOF_OUTPUT" --index "$SLURM_ARRAY_TASK_ID" --model gpt-5.4 \
  --policy "${SARA_SUBPROOF_POLICY:-target_sliced}"
