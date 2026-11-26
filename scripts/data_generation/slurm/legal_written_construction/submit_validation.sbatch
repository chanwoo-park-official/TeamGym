#!/usr/bin/env bash
#SBATCH --job-name=0712-val-submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-val-submit-%j.out
#SBATCH --error=slurm_logs/0712-val-submit-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
submission=scripts/data_generation/slurm/legal_written_construction
manifest="${VALIDATION_MANIFEST:-$root/validation_manifest.tsv}"
validation_root="$root/${VALIDATION_DIR_NAME:-leakage_validation}"
if [[ -d "$validation_root" ]] && find "$validation_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Validation output directory is not empty: $validation_root" >&2
  echo "Set VALIDATION_DIR_NAME to a fresh directory for a new run." >&2
  exit 1
fi
"${TEAMGYM_PYTHON}" "$submission/build_manifest.py" --run-root "$root" --output "$manifest"

total=$(wc -l < "$manifest")
if (( total == 0 )); then
  echo "Validation manifest is empty: $manifest" >&2
  exit 1
fi
end=$((total - 1))
if (( end > 399 )); then end=399; fi

validation_job=$(sbatch --parsable --array="0-$end%400" "$submission/validate.sbatch")
if (( end < total - 1 )); then
  continuation_job=$(sbatch --parsable --dependency="afterany:$validation_job" --export=ALL,NEXT_START=$((end + 1)) "$submission/continue.sbatch")
  printf 'manifest=%s\ntotal=%s\nvalidation_job=%s range=0-%s\ncontinuation_job=%s next_start=%s\n' \
    "$manifest" "$total" "$validation_job" "$end" "$continuation_job" "$((end + 1))"
else
  qualification_submit_job=$(sbatch --parsable --dependency="afterany:$validation_job" "$submission/submit_qualification.sbatch")
  printf 'manifest=%s\ntotal=%s\nvalidation_job=%s range=0-%s\nqualification_submit_job=%s\n' \
    "$manifest" "$total" "$validation_job" "$end" "$qualification_submit_job"
fi
