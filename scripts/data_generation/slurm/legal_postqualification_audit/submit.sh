#!/usr/bin/env bash
set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"

submission="$repo/scripts/data_generation/slurm/legal_postqualification_audit"
run_root="${TEAMGYM_DATAGEN_ROOT}"

if [[ ! -s "$run_root/final_qualified/common_datapoints.jsonl" ]]; then
  echo "Final necessity merge is missing or empty: $run_root/final_qualified/common_datapoints.jsonl" >&2
  exit 1
fi

job_id=$(sbatch --parsable "$submission/audit.sbatch")
echo "audit_job=$job_id"
echo "output=$run_root/postqualification_audit"
