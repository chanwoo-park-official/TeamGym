#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SOURCE_CHECKPOINT:-}" ]]; then
  echo "SOURCE_CHECKPOINT=/path/to/global_step_N is required" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}/_SUCCESS" ]]; then
  echo "Joint source checkpoint lacks its completion marker: ${SOURCE_CHECKPOINT}/_SUCCESS" >&2
  exit 2
fi
CHECKPOINT_BASENAME="$(basename "${SOURCE_CHECKPOINT}")"
if [[ ! "${CHECKPOINT_BASENAME}" =~ ^global_step_([0-9]+)$ ]]; then
  echo "SOURCE_CHECKPOINT must end in global_step_N" >&2
  exit 2
fi

SOURCE_STEP="${BASH_REMATCH[1]}"
SOURCE_RUN="$(basename "$(dirname "${SOURCE_CHECKPOINT}")")"
MERGED_ROOT="${TEAMGYM_MERGED_ROOT:-outputs/teamgym/merged_models}"

for role in coordinator participant; do
  target="${MERGED_ROOT}/${SOURCE_RUN}-step${SOURCE_STEP}-${role}-hf"
  SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
  SOURCE_ACTOR_DIR="${SOURCE_CHECKPOINT}/actor/${role}" \
  TARGET_DIR="${target}" \
  bash "${SCRIPT_DIR}/merge_full_checkpoint_hf.sh"
done

printf 'joint merge complete: %s step %s\n' "${SOURCE_RUN}" "${SOURCE_STEP}"
