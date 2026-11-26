#!/usr/bin/env bash
# Shared path setup for source-checkout construction submissions.

export TEAMGYM_INVOKE_DIR="${TEAMGYM_INVOKE_DIR:-${PWD}}"
if [[ -n "${TEAMGYM_ENV_FILE:-}" ]]; then
  if [[ ${TEAMGYM_ENV_FILE} != /* ]]; then
    TEAMGYM_ENV_FILE="${TEAMGYM_INVOKE_DIR}/${TEAMGYM_ENV_FILE}"
  fi
  if [[ ! -r ${TEAMGYM_ENV_FILE} ]]; then
    printf 'Cannot read TEAMGYM_ENV_FILE: %s\n' "${TEAMGYM_ENV_FILE}" >&2
    exit 2
  fi
  export TEAMGYM_ENV_FILE
  source "${TEAMGYM_ENV_FILE}"
fi

export TEAMGYM_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export TEAMGYM_PYTHON="${TEAMGYM_PYTHON:-python}"
if [[ ${TEAMGYM_PYTHON} == */* && ${TEAMGYM_PYTHON} != /* ]]; then
  TEAMGYM_PYTHON="${TEAMGYM_INVOKE_DIR}/${TEAMGYM_PYTHON}"
fi

export TEAMGYM_GENERATED_ROOT="${TEAMGYM_GENERATED_ROOT:-${TEAMGYM_INVOKE_DIR}/generated}"
for _teamgym_path_name in \
  TEAMGYM_GENERATED_ROOT TEAMGYM_DATAGEN_ROOT \
  CASEBOOK_BASE_SOURCE MSLR_INPUT_ROOT MULTILEXSUM_INPUT_ROOT \
  SARA_SUBPROOF_SOURCE SARA_SUBPROOF_OUTPUT SARA_SWIPL_IMAGE \
  SARA_TARGET_SLICE_AUDIT_SOURCE SARA_TARGET_SLICE_OUTPUT INPUT_JSONL SHARD_ROOT; do
  _teamgym_path_value=${!_teamgym_path_name:-}
  if [[ -n ${_teamgym_path_value} ]]; then
    if [[ ${_teamgym_path_value} != /* ]]; then
      _teamgym_path_value="${TEAMGYM_INVOKE_DIR}/${_teamgym_path_value}"
    fi
    printf -v "${_teamgym_path_name}" '%s' "${_teamgym_path_value}"
    export "${_teamgym_path_name}"
  fi
done
unset _teamgym_path_name _teamgym_path_value
export TEAMGYM_DATAGEN_ROOT="${TEAMGYM_DATAGEN_ROOT:-${TEAMGYM_GENERATED_ROOT}/legal_construction}"
export PYTHONPATH="${TEAMGYM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TEAMGYM_REPO_ROOT}/slurm_logs"
cd "${TEAMGYM_REPO_ROOT}"
