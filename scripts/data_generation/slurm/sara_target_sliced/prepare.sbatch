#!/usr/bin/env bash
#SBATCH --job-name=sara-subproof-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/sara-subproof-prep-%j.out
#SBATCH --error=slurm_logs/sara-subproof-prep-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

"${TEAMGYM_PYTHON}" -m teamgym.data.legal.sara.preparation \
  --source "$SARA_SUBPROOF_SOURCE" --output-root "$SARA_SUBPROOF_OUTPUT" \
  --swipl-image "$SARA_SWIPL_IMAGE"
