#!/usr/bin/env bash
set -u

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT [NUM_SHARDS] [MAX_PASSES]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output_root=$1
num_shards=${2:-16}
max_passes=${3:-3}

for ((pass = 1; pass <= max_passes; pass++)); do
  if [[ -f "${output_root}/status.txt" ]] \
    && [[ $(< "${output_root}/status.txt") == "COMPLETE" ]]; then
    exit 0
  fi
  printf '%s\n' "${pass}" > "${output_root}/current_pass.txt"
  bash "${script_dir}/run_finance_participant_roles_parallel.sh" \
    "${output_root}" "${num_shards}" || true
done

if [[ -f "${output_root}/status.txt" ]] \
  && [[ $(< "${output_root}/status.txt") == "COMPLETE" ]]; then
  exit 0
fi
exit 1
