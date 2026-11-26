#!/usr/bin/env bash
set -u

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT [NUM_SHARDS]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "${script_dir}/../../.." && pwd)
python_bin=${TEAMGYM_PYTHON:-python}
output_root=$1
num_shards=${2:-16}
input_path=${TEAMGYM_FINANCE_SOURCE:?Set TEAMGYM_FINANCE_SOURCE to the accepted finance source JSONL}
if [[ ! -f "${input_path}" || ! -r "${input_path}" ]]; then
  printf 'Cannot read TEAMGYM_FINANCE_SOURCE: %s\n' "${input_path}" >&2
  exit 2
fi
expected_problem_instances=434
shard_root=${output_root}/shards
log_root=${output_root}/logs

mkdir -p "${shard_root}" "${log_root}"
printf 'RUNNING\n' > "${output_root}/status.txt"
: > "${output_root}/workers.tsv"

worker_pids=()
for ((shard_index = 0; shard_index < num_shards; shard_index++)); do
  shard_name=$(printf 'shard_%03d' "${shard_index}")
  shard_output=${shard_root}/${shard_name}
  shard_log=${log_root}/${shard_name}.log
  mkdir -p "${shard_output}"
  PYTHONUNBUFFERED=1 "${python_bin}" \
    "${script_dir}/generate_finance_participant_roles.py" \
    --input "${input_path}" \
    --output-root "${shard_output}" \
    --all-problem-instances \
    --num-shards "${num_shards}" \
    --shard-index "${shard_index}" \
    --model gpt-5.5 \
    --provider bedrock_openai \
    --credentials-file "${repository_root}/api_key/api_key.py" \
    --continue-on-error \
    > "${shard_log}" 2>&1 &
  worker_pid=$!
  worker_pids+=("${worker_pid}")
  printf '%s\t%s\t%s\n' "${shard_index}" "${worker_pid}" "${shard_log}" \
    >> "${output_root}/workers.tsv"
done

worker_status=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    worker_status=1
  fi
done

merge_status=0
if ! "${python_bin}" "${script_dir}/merge_finance_participant_role_shards.py" \
  --shard-root "${shard_root}" \
  --output-root "${output_root}" \
  --expected-shards "${num_shards}" \
  --expected-problem-instances "${expected_problem_instances}" \
  > "${log_root}/merge.log" 2>&1; then
  merge_status=1
fi

if [[ ${worker_status} -eq 0 && ${merge_status} -eq 0 ]]; then
  printf 'COMPLETE\n' > "${output_root}/status.txt"
  exit 0
fi

printf 'FAILED worker_status=%s merge_status=%s\n' \
  "${worker_status}" "${merge_status}" > "${output_root}/status.txt"
exit 1
