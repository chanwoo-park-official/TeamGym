#!/usr/bin/env bash
#SBATCH --job-name=0712-continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-continue-%j.out
#SBATCH --error=slurm_logs/0712-continue-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"
start=${NEXT_START:?NEXT_START is required}
root="${TEAMGYM_DATAGEN_ROOT}"
submission=scripts/data_generation/slurm/legal_written_construction
manifest="${VALIDATION_MANIFEST:-$root/validation_manifest.tsv}"
total=$(wc -l < "$manifest")
if (( start >= total )); then
  echo "NEXT_START=$start is outside manifest with $total tasks" >&2
  exit 1
fi
end=$((start + 399))
if (( end >= total )); then end=$((total - 1)); fi
validation_job=$(sbatch --parsable --array="$start-$end%400" "$submission/validate.sbatch")
if (( end < total - 1 )); then
  next_start=$((end + 1))
  continuation_job=$(sbatch --parsable --dependency="afterany:$validation_job" --export="ALL,NEXT_START=$next_start" "$submission/continue.sbatch")
  printf 'validation_job=%s range=%s-%s\ncontinuation_job=%s next_start=%s\n' "$validation_job" "$start" "$end" "$continuation_job" "$next_start"
else
  qualification_submit_job=$(sbatch --parsable --dependency="afterany:$validation_job" "$submission/submit_qualification.sbatch")
  printf 'validation_job=%s range=%s-%s\nqualification_submit_job=%s\n' \
    "$validation_job" "$start" "$end" "$qualification_submit_job"
fi
