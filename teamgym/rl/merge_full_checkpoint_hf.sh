#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SOURCE_CHECKPOINT:-}" || ! -d "${SOURCE_CHECKPOINT}/actor" ]]; then
  echo "SOURCE_CHECKPOINT=/path/to/global_step_N is required" >&2
  exit 2
fi
CHECKPOINT_BASENAME="$(basename "${SOURCE_CHECKPOINT}")"
if [[ ! "${CHECKPOINT_BASENAME}" =~ ^global_step_([0-9]+)$ ]]; then
  echo "SOURCE_CHECKPOINT must end in global_step_N" >&2
  exit 2
fi

SOURCE_STEP="${BASH_REMATCH[1]}"
SOURCE_RUN="$(basename "$(dirname "${SOURCE_CHECKPOINT}")")"
TARGET_DIR="${TARGET_DIR:-${TEAMGYM_MERGED_ROOT:-outputs/teamgym/merged_models}/${SOURCE_RUN}-step${SOURCE_STEP}-hf}"
TEAMGYM_PYTHON="${TEAMGYM_PYTHON:-python}"
SOURCE_ACTOR_DIR="${SOURCE_ACTOR_DIR:-${SOURCE_CHECKPOINT}/actor}"

if [[ -f "${TARGET_DIR}/config.json" ]] &&
   compgen -G "${TARGET_DIR}/model*.safetensors" >/dev/null; then
  echo "Reusing merged checkpoint: ${TARGET_DIR}"
  exit 0
fi
if [[ -e "${TARGET_DIR}" ]]; then
  echo "Refusing to overwrite incomplete merge target: ${TARGET_DIR}" >&2
  exit 2
fi
if [[ ! -d "${SOURCE_ACTOR_DIR}" ]]; then
  echo "Missing source actor directory: ${SOURCE_ACTOR_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${TARGET_DIR}")"
STAGING_DIR="${TARGET_DIR}.partial-${SLURM_JOB_ID:-$$}"
if [[ -e "${STAGING_DIR}" ]]; then
  echo "Refusing to overwrite incomplete merge staging directory: ${STAGING_DIR}" >&2
  exit 2
fi
cleanup_staging() {
  if [[ -d "${STAGING_DIR}" ]]; then
    find "${STAGING_DIR}" -depth -delete
  fi
}
trap cleanup_staging EXIT

"${TEAMGYM_PYTHON}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${SOURCE_ACTOR_DIR}" \
  --target_dir "${STAGING_DIR}" \
  --use_cpu_initialization

if [[ ! -f "${STAGING_DIR}/config.json" ]] ||
   ! compgen -G "${STAGING_DIR}/model*.safetensors" >/dev/null; then
  echo "Merged checkpoint validation failed: ${STAGING_DIR}" >&2
  exit 1
fi
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
SOURCE_ACTOR_DIR="${SOURCE_ACTOR_DIR}" \
STAGING_DIR="${STAGING_DIR}" \
"${TEAMGYM_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

target = Path(os.environ["STAGING_DIR"])
weights = sorted(target.glob("model*.safetensors"))
if not weights or any(path.stat().st_size <= 0 for path in weights):
    raise SystemExit("Merged safetensors are missing or empty")
payload = {
    "source_checkpoint": os.environ["SOURCE_CHECKPOINT"],
    "source_actor_dir": os.environ["SOURCE_ACTOR_DIR"],
    "weight_files": {path.name: path.stat().st_size for path in weights},
}
with (target / "_MERGE_SUCCESS.json").open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
mv "${STAGING_DIR}" "${TARGET_DIR}"
trap - EXIT
echo "Merged checkpoint: ${TARGET_DIR}"
