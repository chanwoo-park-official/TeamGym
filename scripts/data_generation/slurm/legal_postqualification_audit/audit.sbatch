#!/usr/bin/env bash
#SBATCH --job-name=0712-post-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-post-audit-%j.out
#SBATCH --error=slurm_logs/0712-post-audit-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"

run_root="${TEAMGYM_DATAGEN_ROOT}"

cd "$repo"
"${TEAMGYM_PYTHON}" scripts/data_generation/slurm/legal_postqualification_audit/audit.py \
  --run-root "$run_root" \
  --output-dir postqualification_audit
