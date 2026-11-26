#!/usr/bin/env bash
#SBATCH --job-name=0712-nec-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-nec-merge-%j.out
#SBATCH --error=slurm_logs/0712-nec-merge-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"
"${TEAMGYM_PYTHON}" scripts/data_generation/slurm/legal_written_construction/merge_necessity.py \
  --run-root "${TEAMGYM_DATAGEN_ROOT}" \
  --all-riu-dir "${ALL_RIU_OUTPUT_DIR_NAME:-all_riu_qualified}" \
  --variants-dir "${NECESSITY_VARIANTS_DIR_NAME:-necessity_variants}" \
  --necessity-dir "${NECESSITY_DIR_NAME:-leave_one_piu_out}" \
  --output-dir "${FINAL_OUTPUT_DIR_NAME:-final_qualified}"
