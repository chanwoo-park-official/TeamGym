#!/usr/bin/env bash
#SBATCH --job-name=0712-qual-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-qual-merge-%j.out
#SBATCH --error=slurm_logs/0712-qual-merge-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"
"${TEAMGYM_PYTHON}" scripts/data_generation/slurm/legal_written_construction/merge_qualification.py \
  --run-root "${TEAMGYM_DATAGEN_ROOT}" \
  --survivor-dir "${LEAKAGE_SURVIVOR_DIR_NAME:-leakage_survivors}" \
  --qualification-dir "${QUALIFICATION_DIR_NAME:-all_piu_qualification}" \
  --output-dir "${ALL_RIU_OUTPUT_DIR_NAME:-all_riu_qualified}"

case "${RUN_NECESSITY:-0}" in
  0)
    echo "All-RIU qualification is complete. Leave-one-RIU-out necessity is skipped by default."
    echo "Set RUN_NECESSITY=1 only when running the optional minimality diagnostic."
    ;;
  1)
    necessity_submit_job=$(sbatch --parsable scripts/data_generation/slurm/legal_written_construction/submit_necessity.sbatch)
    printf 'necessity_submit_job=%s\n' "$necessity_submit_job"
    ;;
  *)
    echo "RUN_NECESSITY must be 0 or 1, received: ${RUN_NECESSITY}" >&2
    exit 2
    ;;
esac
