#!/usr/bin/env bash
#SBATCH --job-name=0712-qual-submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-qual-submit-%j.out
#SBATCH --error=slurm_logs/0712-qual-submit-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
submission=scripts/data_generation/slurm/legal_written_construction
survivor_dir="${LEAKAGE_SURVIVOR_DIR_NAME:-leakage_survivors}"
qualification_dir="${QUALIFICATION_DIR_NAME:-all_piu_qualification}"
qualification_root="$root/$qualification_dir"
if [[ -d "$qualification_root" ]] && find "$qualification_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Qualification output directory is not empty: $qualification_root" >&2
  echo "Set QUALIFICATION_DIR_NAME to a fresh directory for a new run." >&2
  exit 1
fi

"${TEAMGYM_PYTHON}" "$submission/merge_leakage.py" \
  --run-root "$root" \
  --validation-dir "${VALIDATION_DIR_NAME:-leakage_validation}" \
  --output-dir "$survivor_dir"

manifest="$root/$survivor_dir/qualification_manifest.tsv"
total=$(wc -l < "$manifest")
if (( total == 0 )); then
  echo "No leakage survivors to qualify: $manifest" >&2
  exit 1
fi
end=$((total - 1))
if (( end > 399 )); then end=399; fi

qualification_job=$(sbatch --parsable --array="0-$end%400" "$submission/qualify_all_riu.sbatch")
if (( end < total - 1 )); then
  continuation_job=$(sbatch --parsable --dependency="afterany:$qualification_job" --export=ALL,NEXT_START=$((end + 1)) "$submission/continue_qualification.sbatch")
  printf 'total=%s\nqualification_job=%s range=0-%s\ncontinuation_job=%s next_start=%s\n' \
    "$total" "$qualification_job" "$end" "$continuation_job" "$((end + 1))"
else
  merge_job=$(sbatch --parsable --dependency="afterany:$qualification_job" "$submission/merge_qualification.sbatch")
  printf 'total=%s\nqualification_job=%s range=0-%s\nmerge_job=%s\n' \
    "$total" "$qualification_job" "$end" "$merge_job"
fi
