#!/usr/bin/env bash
#SBATCH --job-name=0712-nec-cont
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-nec-cont-%j.out
#SBATCH --error=slurm_logs/0712-nec-cont-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
submission=scripts/data_generation/slurm/legal_written_construction
manifest="$root/${NECESSITY_VARIANTS_DIR_NAME:-necessity_variants}/necessity_manifest.tsv"
total=$(wc -l < "$manifest")
start=${NEXT_START:?NEXT_START is required}
if (( start >= total )); then
  echo "NEXT_START=$start is outside manifest with $total tasks" >&2
  exit 1
fi
end=$((start + 399))
if (( end >= total )); then end=$((total - 1)); fi

necessity_job=$(sbatch --parsable --array="$start-$end%400" "$submission/qualify_necessity.sbatch")
if (( end < total - 1 )); then
  next_start=$((end + 1))
  continuation_job=$(sbatch --parsable --dependency="afterany:$necessity_job" --export="ALL,NEXT_START=$next_start" "$submission/continue_necessity.sbatch")
  printf 'necessity_job=%s range=%s-%s\ncontinuation_job=%s next_start=%s\n' \
    "$necessity_job" "$start" "$end" "$continuation_job" "$next_start"
else
  merge_job=$(sbatch --parsable --dependency="afterany:$necessity_job" "$submission/merge_necessity.sbatch")
  printf 'necessity_job=%s range=%s-%s\nmerge_job=%s\n' "$necessity_job" "$start" "$end" "$merge_job"
fi
