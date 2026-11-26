#!/usr/bin/env bash
#SBATCH --job-name=0712-nec-submit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-nec-submit-%j.out
#SBATCH --error=slurm_logs/0712-nec-submit-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

root="${TEAMGYM_DATAGEN_ROOT}"
submission=scripts/data_generation/slurm/legal_written_construction
all_riu_root="$root/${ALL_RIU_OUTPUT_DIR_NAME:-all_riu_qualified}"
variants_root="$root/${NECESSITY_VARIANTS_DIR_NAME:-necessity_variants}"
necessity_root="$root/${NECESSITY_DIR_NAME:-leave_one_piu_out}"
if [[ -d "$necessity_root" ]] && find "$necessity_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Necessity output directory is not empty: $necessity_root" >&2
  echo "Set NECESSITY_DIR_NAME to a fresh directory for a new run." >&2
  exit 1
fi
if [[ -d "$variants_root" ]] && find "$variants_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Necessity variants directory is not empty: $variants_root" >&2
  echo "Set NECESSITY_VARIANTS_DIR_NAME to a fresh directory for a new run." >&2
  exit 1
fi

"${TEAMGYM_PYTHON}" -m teamgym.data.long_context_reasoning.ablation \
  --input-jsonl "$all_riu_root/common_datapoints.jsonl" \
  --output-root "$variants_root"
"${TEAMGYM_PYTHON}" "$submission/build_necessity_manifest.py" --variants-root "$variants_root"

manifest="$variants_root/necessity_manifest.tsv"
total=$(wc -l < "$manifest")
if (( total == 0 )); then
  echo "No leave-one-RIU-out variants to qualify" >&2
  exit 1
fi
end=$((total - 1))
if (( end > 399 )); then end=399; fi

necessity_job=$(sbatch --parsable --array="0-$end%400" "$submission/qualify_necessity.sbatch")
if (( end < total - 1 )); then
  continuation_job=$(sbatch --parsable --dependency="afterany:$necessity_job" --export=ALL,NEXT_START=$((end + 1)) "$submission/continue_necessity.sbatch")
  printf 'total=%s\nnecessity_job=%s range=0-%s\ncontinuation_job=%s next_start=%s\n' \
    "$total" "$necessity_job" "$end" "$continuation_job" "$((end + 1))"
else
  merge_job=$(sbatch --parsable --dependency="afterany:$necessity_job" "$submission/merge_necessity.sbatch")
  printf 'total=%s\nnecessity_job=%s range=0-%s\nmerge_job=%s\n' "$total" "$necessity_job" "$end" "$merge_job"
fi
