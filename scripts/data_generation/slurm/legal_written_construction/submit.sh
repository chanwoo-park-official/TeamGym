#!/usr/bin/env bash
set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"

cd "$repo"
: "${CASEBOOK_BASE_SOURCE:?Set CASEBOOK_BASE_SOURCE to the frozen prepared text of 12 CFR Parts 1 and 3}"
: "${MSLR_INPUT_ROOT:?Set MSLR_INPUT_ROOT to the anonymized MSLR entry_*.json source directory}"
: "${MULTILEXSUM_INPUT_ROOT:?Set MULTILEXSUM_INPUT_ROOT to the constructed MultiLexSum task JSONL}"
if [[ ! -r "$CASEBOOK_BASE_SOURCE" || ! -f "$CASEBOOK_BASE_SOURCE" ]]; then
  echo "CASEBOOK_BASE_SOURCE must name a readable file: $CASEBOOK_BASE_SOURCE" >&2
  exit 2
fi
if [[ ! -d "$MSLR_INPUT_ROOT" ]]; then
  echo "MSLR_INPUT_ROOT must name an existing directory: $MSLR_INPUT_ROOT" >&2
  exit 2
fi
if [[ ! -r "$MULTILEXSUM_INPUT_ROOT" || ! -f "$MULTILEXSUM_INPUT_ROOT" || ! -s "$MULTILEXSUM_INPUT_ROOT" ]]; then
  echo "MULTILEXSUM_INPUT_ROOT must name a nonempty readable task file: $MULTILEXSUM_INPUT_ROOT" >&2
  exit 2
fi
export CASEBOOK_BASE_SOURCE MSLR_INPUT_ROOT MULTILEXSUM_INPUT_ROOT
mkdir -p slurm_logs
submission=scripts/data_generation/slurm/legal_written_construction
export RUN_NECESSITY="${RUN_NECESSITY:-0}"
if [[ "$RUN_NECESSITY" != 0 && "$RUN_NECESSITY" != 1 ]]; then
  echo "RUN_NECESSITY must be 0 or 1, received: $RUN_NECESSITY" >&2
  exit 2
fi

seed_job=$(sbatch --parsable "$submission/seeds.sbatch")
partition_job=$(sbatch --parsable --dependency="afterok:$seed_job" "$submission/partitions.sbatch")
multilexsum_job=$(sbatch --parsable "$submission/multilexsum_partitions.sbatch")
validation_submit_job=$(sbatch --parsable --dependency="afterok:$partition_job:$multilexsum_job" "$submission/submit_validation.sbatch")

printf 'seed_job=%s\npartition_job=%s\nmultilexsum_job=%s\nvalidation_submit_job=%s\n' \
  "$seed_job" "$partition_job" "$multilexsum_job" "$validation_submit_job"
printf 'run_necessity=%s\n' "$RUN_NECESSITY"
