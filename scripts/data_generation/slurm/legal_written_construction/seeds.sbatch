#!/usr/bin/env bash
#SBATCH --job-name=0712-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/0712-seeds-%j.out
#SBATCH --error=slurm_logs/0712-seeds-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
: "${CASEBOOK_BASE_SOURCE:?Set CASEBOOK_BASE_SOURCE to the frozen prepared text of 12 CFR Parts 1 and 3}"
: "${MSLR_INPUT_ROOT:?Set MSLR_INPUT_ROOT to the anonymized MSLR entry_*.json source directory}"
cd "$repo"
if [[ ! -r "$CASEBOOK_BASE_SOURCE" || ! -f "$CASEBOOK_BASE_SOURCE" ]]; then
  echo "CASEBOOK_BASE_SOURCE must name a readable file: $CASEBOOK_BASE_SOURCE" >&2
  exit 2
fi
if [[ ! -d "$MSLR_INPUT_ROOT" ]]; then
  echo "MSLR_INPUT_ROOT must name an existing directory: $MSLR_INPUT_ROOT" >&2
  exit 2
fi

root="${TEAMGYM_DATAGEN_ROOT}"
"${TEAMGYM_PYTHON}" -m teamgym.data.cli legal/casebook \
  --base-source "$CASEBOOK_BASE_SOURCE" \
  --output-root "$root/seeds/casebook20"
"${TEAMGYM_PYTHON}" -m teamgym.data.cli legal/mslr \
  --input-root "$MSLR_INPUT_ROOT" \
  --output-root "$root/seeds/mslr100" \
  --limit 100 \
  --candidate-scan-limit 1389
