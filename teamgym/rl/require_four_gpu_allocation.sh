#!/usr/bin/env bash

# Keep TeamGym replication runs on one stable four-GPU topology.  If Slurm
# exposes fewer GPUs than the job requested, preserve the job ID and its
# dependencies by requeueing that same job.  Allocation attempts are recorded
# in pool storage even when all retries are exhausted.

teamgym_four_gpu_attempt_action() {
  local detected_count="$1"
  local attempt="$2"
  local max_attempts="$3"
  if ((detected_count >= 4)); then
    printf 'run\n'
  elif ((attempt < max_attempts)); then
    printf 'requeue\n'
  else
    printf 'fail\n'
  fi
}

teamgym_require_four_gpus_or_requeue() {
  local max_attempts="${TEAMGYM_GPU_ALLOCATION_MAX_ATTEMPTS:-5}"
  if [[ ! "${max_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEAMGYM_GPU_ALLOCATION_MAX_ATTEMPTS must be a positive integer" >&2
    exit 2
  fi
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "The four-GPU allocation guard must run inside a Slurm job" >&2
    exit 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; cannot verify the Slurm GPU allocation" >&2
    exit 2
  fi

  local -a detected_ids=()
  mapfile -t detected_ids < <(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
      | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
      | sed '/^$/d'
  )
  local detected_count="${#detected_ids[@]}"
  local detected_csv=""
  local used_csv=""
  local gpu_id
  local gpu_index
  for gpu_index in "${!detected_ids[@]}"; do
    gpu_id="${detected_ids[gpu_index]}"
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
      echo "Unexpected GPU index from nvidia-smi: ${gpu_id}" >&2
      exit 2
    fi
    if [[ -n "${detected_csv}" ]]; then
      detected_csv+=","
    fi
    detected_csv+="${gpu_id}"
    if ((gpu_index < 4)); then
      if [[ -n "${used_csv}" ]]; then
        used_csv+=","
      fi
      used_csv+="${gpu_id}"
    fi
  done

  local allocation_root="${TEAMGYM_STORAGE_ROOT}/allocation_attempts/${TEAMGYM_CAMPAIGN_TAG:-unlabeled}"
  local job_root="${allocation_root}/job_${SLURM_JOB_ID}"
  local count_path="${job_root}/attempt_count"
  local attempts_path="${job_root}/attempts.tsv"
  mkdir -p "${job_root}"

  local previous_attempts=0
  if [[ -f "${count_path}" ]]; then
    read -r previous_attempts < "${count_path}"
  fi
  if [[ ! "${previous_attempts}" =~ ^[0-9]+$ ]]; then
    echo "Invalid allocation attempt counter: ${count_path}" >&2
    exit 2
  fi
  local attempt=$((previous_attempts + 1))
  local count_tmp="${count_path}.tmp-$$"
  printf '%s\n' "${attempt}" > "${count_tmp}"
  mv "${count_tmp}" "${count_path}"

  local action
  action="$(teamgym_four_gpu_attempt_action "${detected_count}" "${attempt}" "${max_attempts}")"
  if [[ ! -f "${attempts_path}" ]]; then
    printf '%b\n' 'timestamp\tattempt\tmax_attempts\taction\tjob_id\trestart_count\tnode\tslurm_gpus_on_node\tdetected_count\tdetected_ids' \
      > "${attempts_path}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" \
    "${attempt}" \
    "${max_attempts}" \
    "${action}" \
    "${SLURM_JOB_ID}" \
    "${SLURM_RESTART_COUNT:-0}" \
    "$(hostname)" \
    "${SLURM_GPUS_ON_NODE:-unknown}" \
    "${detected_count}" \
    "${detected_csv:-none}" \
    >> "${attempts_path}"

  printf '%s\n' \
    "gpu_policy=require4_requeue_up_to_${max_attempts}_attempts" \
    "gpu_allocation_attempt=${attempt}/${max_attempts}" \
    "gpu_slurm_on_node=${SLURM_GPUS_ON_NODE:-unknown}" \
    "gpu_detected=${detected_count}" \
    "gpu_ids=${detected_csv:-none}" \
    "gpu_allocation_log=${attempts_path}"

  case "${action}" in
    run)
      export TEAMGYM_REQUESTED_GPUS=4
      export TEAMGYM_DETECTED_GPUS="${detected_count}"
      export TEAMGYM_USED_GPUS=4
      export TEAMGYM_USED_GPU_IDS="${used_csv}"
      export TEAMGYM_GPU_ALLOCATION_POLICY="require4_requeue_up_to_${max_attempts}_attempts"
      export TEAMGYM_CUDA_VISIBLE_DEVICES="${used_csv}"
      export N_GPUS_PER_NODE=4
      export FROZEN_COORDINATOR_GPU="${detected_ids[3]}"
      export FROZEN_PARTICIPANT_GPU="${detected_ids[3]}"
      ;;
    requeue)
      echo "Only ${detected_count} GPUs are visible; requeueing Slurm job ${SLURM_JOB_ID} without releasing its dependencies." >&2
      if ! scontrol requeue "${SLURM_JOB_ID}"; then
        echo "Failed to requeue Slurm job ${SLURM_JOB_ID}; allocation log: ${attempts_path}" >&2
        exit 1
      fi
      # scontrol normally terminates this allocation.  Exit defensively if it
      # returns before Slurm sends the termination signal.
      exit 0
      ;;
    fail)
      local exhausted_path="${job_root}/EXHAUSTED.txt"
      printf '%s\n' \
        "Four-GPU allocation was not provided after ${max_attempts} attempts." \
        "job_id=${SLURM_JOB_ID}" \
        "job_name=${SLURM_JOB_NAME:-unknown}" \
        "last_node=$(hostname)" \
        "last_detected_gpu_count=${detected_count}" \
        "attempt_log=${attempts_path}" \
        > "${exhausted_path}"
      echo "Four-GPU allocation exhausted after ${max_attempts} attempts; diagnostic: ${exhausted_path}" >&2
      exit 1
      ;;
  esac
}
