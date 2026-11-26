#!/usr/bin/env bash
#SBATCH --job-name=sara-target-post
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/sara-target-post-%j.out
#SBATCH --error=slurm_logs/sara-target-post-%j.err

set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"
cd "$repo"

"${TEAMGYM_PYTHON}" -m teamgym.data.legal.sara.refresh \
  --output-root "$SARA_SUBPROOF_OUTPUT"
"${TEAMGYM_PYTHON}" -m teamgym.data.legal.sara.merge \
  --output-root "$SARA_SUBPROOF_OUTPUT" --limit "$SARA_SUBPROOF_LIMIT"

pids=()
for ((worker=0; worker<SARA_SUBPROOF_QUAL_WORKERS; worker++)); do
  "${TEAMGYM_PYTHON}" -m teamgym.data.legal.sara.qualification \
    --output-root "$SARA_SUBPROOF_OUTPUT" --worker "$worker" \
    --worker-count "$SARA_SUBPROOF_QUAL_WORKERS" --model gpt-5.4 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

"${TEAMGYM_PYTHON}" -m teamgym.data.legal.sara.finalize \
  --output-root "$SARA_SUBPROOF_OUTPUT" --minimum-passes 4 \
  --target-count "${SARA_SUBPROOF_TARGET_COUNT:-$SARA_SUBPROOF_LIMIT}"
