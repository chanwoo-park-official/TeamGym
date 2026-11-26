#!/usr/bin/env bash
set -euo pipefail

repo="${TEAMGYM_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${repo}/scripts/data_generation/slurm/_environment.sh"
repo="${TEAMGYM_REPO_ROOT}"

submission=$repo/scripts/data_generation/slurm/sara_target_sliced
audit_source=${SARA_TARGET_SLICE_AUDIT_SOURCE:?Set SARA_TARGET_SLICE_AUDIT_SOURCE to a prepared SARA audit directory}
output=${SARA_TARGET_SLICE_OUTPUT:-${TEAMGYM_GENERATED_ROOT}/sara_target_sliced}
available_cases=$(jq 'length' "$audit_source/metadata/audited_cases.json")
case_count=${SARA_TARGET_SLICE_CASE_COUNT:-$available_cases}
generation_concurrency=${SARA_TARGET_SLICE_GENERATION_CONCURRENCY:-40}
qual_workers=${SARA_TARGET_SLICE_QUAL_WORKERS:-50}
partition=${SARA_TARGET_SLICE_PARTITION:-}
partition_args=()
if [[ -n $partition ]]; then partition_args=(--partition="$partition"); fi
generation_time=${SARA_TARGET_SLICE_GENERATION_TIME:-04:00:00}
postprocess_time=${SARA_TARGET_SLICE_POSTPROCESS_TIME:-04:00:00}

if ((case_count < 1 || case_count > available_cases)); then
  printf 'SARA_TARGET_SLICE_CASE_COUNT must be between 1 and %s\n' "$available_cases" >&2
  exit 2
fi
if ((generation_concurrency < 1 || qual_workers < 1)); then
  printf 'Generation and qualification concurrency must be positive\n' >&2
  exit 2
fi
if find "$output/workers" -name record.json -print -quit 2>/dev/null | grep -q .; then
  printf 'Refusing to overwrite existing target-sliced records in %s\n' "$output" >&2
  exit 1
fi

mkdir -p "$output/metadata" "$repo/slurm_logs"
cp "$audit_source/metadata/audited_cases.json" "$output/metadata/audited_cases.json"
cp "$audit_source/metadata/selection_report.json" "$output/metadata/selection_report.json"

exports=ALL,SARA_SUBPROOF_OUTPUT="$output",SARA_SUBPROOF_POLICY=target_sliced,SARA_SUBPROOF_LIMIT="$case_count",SARA_SUBPROOF_TARGET_COUNT="$case_count",SARA_SUBPROOF_QUAL_WORKERS="$qual_workers"
generate_job=$(sbatch --parsable "${partition_args[@]}" --time="$generation_time" \
  --array="0-$((case_count - 1))%$generation_concurrency" --cpus-per-task=1 --mem=4G \
  --export="$exports" "$submission/generate.sbatch")
postprocess_job=$(sbatch --parsable "${partition_args[@]}" --time="$postprocess_time" \
  --cpus-per-task="$qual_workers" --mem=48G --dependency="afterany:$generate_job" \
  --export="$exports" "$submission/postprocess_target_sliced.sbatch")

{
  printf 'policy=target_sliced\n'
  printf 'proof_envelope=grounded_backward_with_explicit_construction_requirements\n'
  printf 'dataset_variant=sara_target_sliced_subproof_v7\n'
  printf 'available_cases=%s\n' "$available_cases"
  printf 'case_count=%s\n' "$case_count"
  printf 'generation_concurrency=%s\n' "$generation_concurrency"
  printf 'qualification_concurrency=%s\n' "$qual_workers"
  printf 'generation_layout=slurm_array\n'
  printf 'partition=%s\n' "$partition"
  printf 'generate=%s\n' "$generate_job"
  printf 'postprocess=%s\n' "$postprocess_job"
} | tee "$output/metadata/submission_jobs.txt"
