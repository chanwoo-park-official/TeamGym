#!/usr/bin/env bash
# Shared RLLM/VERL backend used by generic training and the paper recipes.
set -euo pipefail

TEAMGYM_PYTHON="${TEAMGYM_PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${REPOSITORY_ROOT}/pyproject.toml" && -d "${REPOSITORY_ROOT}/verl/verl" ]]; then
  export PYTHONPATH="${REPOSITORY_ROOT}/verl:${REPOSITORY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-generated/teamgym_smoke_data}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-validation}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAINING_MODE="${TRAINING_MODE:-coordinator_only}"
RUN_NAME="${RUN_NAME:-teamgym-qwen25-3b-${TRAINING_MODE}-two-step}"
RUN_DIR="${RUN_DIR:-outputs/teamgym/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FULL_FINETUNE="${FULL_FINETUNE:-0}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
LOGICAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${LOGICAL_BATCH_SIZE}}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-${LOGICAL_BATCH_SIZE}}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-2}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-4096}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-512}"
DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
CAMPAIGN_SEED="${CAMPAIGN_SEED:-${DATA_SEED:-17}}"
DATA_SEED="${DATA_SEED:-${CAMPAIGN_SEED}}"
ROLLOUT_SEED="${ROLLOUT_SEED:-${CAMPAIGN_SEED}}"
TRAINER_SEED="${TRAINER_SEED:-${CAMPAIGN_SEED}}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-False}"
ROLLOUT_PROFILE="${ROLLOUT_PROFILE:-short_context_smoke}"
REWARD_CONFIG="${REWARD_CONFIG:-task_success_plus_informed_success}"
WRITTEN_JUDGE_MODEL="${WRITTEN_JUDGE_MODEL:-gpt-5.5}"
WRITTEN_JUDGE_PROVIDER="${WRITTEN_JUDGE_PROVIDER:-bedrock_openai}"
WRITTEN_JUDGE_BASE_URL="${WRITTEN_JUDGE_BASE_URL:-https://bedrock-mantle.us-east-2.api.aws/openai/v1}"
WRITTEN_JUDGE_ENDPOINT="${WRITTEN_JUDGE_ENDPOINT:-/responses}"
WRITTEN_JUDGE_CREDENTIALS_FILE="${WRITTEN_JUDGE_CREDENTIALS_FILE:-api_key/api_key.py}"
WRITTEN_JUDGE_API_KEY_ENV="${WRITTEN_JUDGE_API_KEY_ENV:-AWS_BEARER_TOKEN_BEDROCK}"
WRITTEN_JUDGE_MAX_OUTPUT_TOKENS="${WRITTEN_JUDGE_MAX_OUTPUT_TOKENS:-3000}"
WRITTEN_JUDGE_TIMEOUT_SEC="${WRITTEN_JUDGE_TIMEOUT_SEC:-600}"
WRITTEN_JUDGE_REQUEST_RETRY_ATTEMPTS="${WRITTEN_JUDGE_REQUEST_RETRY_ATTEMPTS:-4}"
WRITTEN_JUDGE_FAIL_ON_ERROR="${WRITTEN_JUDGE_FAIL_ON_ERROR:-False}"
WRITTEN_JUDGE_CACHE_DIR="${WRITTEN_JUDGE_CACHE_DIR:-}"
INVALID_JSON_PENALTY="${INVALID_JSON_PENALTY:--0.1}"
INVALID_SCHEMA_PENALTY="${INVALID_SCHEMA_PENALTY:--0.1}"
FORMAT_CREDIT_ASSIGNMENT="${FORMAT_CREDIT_ASSIGNMENT:-trajectory}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console']}"
MAX_PARSE_RETRIES="${MAX_PARSE_RETRIES:-2}"
PARTICIPANT_FORMAT_REPAIR_FOR_EVAL="${PARTICIPANT_FORMAT_REPAIR_FOR_EVAL:-False}"
PARTICIPANT_FORMAT_REPAIR_STRATEGY="${PARTICIPANT_FORMAT_REPAIR_STRATEGY:-wrap_raw}"
PARTICIPANT_FORMAT_REPAIR_MAX_UTTERANCE_TOKENS="${PARTICIPANT_FORMAT_REPAIR_MAX_UTTERANCE_TOKENS:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
VAL_ONLY="${VAL_ONLY:-False}"
TEST_FREQ="${TEST_FREQ:--1}"
DISABLE_VALIDATION="${DISABLE_VALIDATION:-True}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
FIXED_COORDINATOR_BACKEND="${FIXED_COORDINATOR_BACKEND:-base_model}"
FIXED_COORDINATOR_BASE_URL="${FIXED_COORDINATOR_BASE_URL:-}"
FIXED_COORDINATOR_MODEL="${FIXED_COORDINATOR_MODEL:-}"
FIXED_COORDINATOR_API_KEY="${FIXED_COORDINATOR_API_KEY:-teamgym-local}"
FIXED_COORDINATOR_TEMPERATURE="${FIXED_COORDINATOR_TEMPERATURE:-0.7}"
FIXED_COORDINATOR_VAL_TEMPERATURE="${FIXED_COORDINATOR_VAL_TEMPERATURE:-0.0}"
PARTICIPANT_REPLAY_PATH="${PARTICIPANT_REPLAY_PATH:-}"
FIXED_PARTICIPANT_BASE_URL="${FIXED_PARTICIPANT_BASE_URL:-}"
FIXED_PARTICIPANT_MODEL="${FIXED_PARTICIPANT_MODEL:-}"
FIXED_PARTICIPANT_API_KEY="${FIXED_PARTICIPANT_API_KEY:-teamgym-local}"
FIXED_PARTICIPANT_TEMPERATURE="${FIXED_PARTICIPANT_TEMPERATURE:-0.7}"
FIXED_PARTICIPANT_VAL_TEMPERATURE="${FIXED_PARTICIPANT_VAL_TEMPERATURE:-0.0}"
case "${FULL_FINETUNE}" in
  0)
    ROLLOUT_LORA_NAME="default"
    ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
    ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
    ;;
  1)
    LORA_RANK=0
    ROLLOUT_LORA_NAME="null"
    ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
    ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
    ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
    ;;
  *)
    echo "FULL_FINETUNE must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ -z "${PARTICIPANT_BACKEND:-}" ]]; then
  if [[ "${TRAINING_MODE}" == "coordinator_only" ]]; then
    PARTICIPANT_BACKEND="scripted"
  else
    PARTICIPANT_BACKEND="rollout"
  fi
fi
if [[ "${TRAINING_MODE}" != "coordinator_only" && "${TRAINING_MODE}" != "participant_only" && "${TRAINING_MODE}" != "joint" ]]; then
  echo "TRAINING_MODE must be coordinator_only, participant_only, or joint" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VERL_TRAINER_SEED="${TRAINER_SEED}"

IFS=',' read -r -a TEAMGYM_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${#TEAMGYM_GPU_IDS[@]}}"
if ! [[ "${ROLLOUT_TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROLLOUT_TENSOR_PARALLEL_SIZE must be a positive integer" >&2
  exit 2
fi
if (( N_GPUS_PER_NODE % ROLLOUT_TENSOR_PARALLEL_SIZE != 0 )); then
  echo "N_GPUS_PER_NODE (${N_GPUS_PER_NODE}) must be divisible by ROLLOUT_TENSOR_PARALLEL_SIZE (${ROLLOUT_TENSOR_PARALLEL_SIZE})" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"
export HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-${RUN_DIR}/hydra}"
mkdir -p "${HYDRA_RUN_DIR}"

RESUME_ARGS=(trainer.resume_mode="${RESUME_MODE}")
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  RESUME_ARGS+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
fi

TRAIN_LOG_TEE_ARGS=()
if [[ "${RESUME_MODE}" != "disable" && -f "${RUN_DIR}/train.log" ]]; then
  TRAIN_LOG_TEE_ARGS=(-a)
fi

"${TEAMGYM_PYTHON}" -m teamgym.rl.train \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  hydra.output_subdir=.hydra \
  hydra.job.chdir=False \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.shuffle="${DATA_SHUFFLE}" \
  data.seed="${DATA_SEED}" \
  data.validation_shuffle="${VALIDATION_SHUFFLE}" \
  data.max_prompt_length="${DATA_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${DATA_MAX_RESPONSE_LENGTH}" \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.seed="${ROLLOUT_SEED}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  rllm.workflow.use_workflow=True \
  +rllm.workflow.engine_cls=teamgym.rl.training.trainer_compat:TeamGymAgentWorkflowEngine \
  rllm.workflow.n_parallel_tasks="${N_PARALLEL_TASKS}" \
  rllm.workflow.retry_limit=1 \
  rllm.stepwise_advantage.enable=True \
  rllm.stepwise_advantage.mode=broadcast \
  rllm.compact_filtering.enable=False \
  rllm.rejection_sample.enable=False \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=teamgym_rl \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.val_only="${VAL_ONLY}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  +trainer.seed="${TRAINER_SEED}" \
  trainer.log_episodes=True \
  trainer.episode_log_dir="${RUN_DIR}/episode_logs" \
  "${RESUME_ARGS[@]}" \
  +ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_init.num_gpus="${N_GPUS_PER_NODE}" \
  +teamgym.prepared_data_dir="${PREPARED_DATA_DIR}" \
  +teamgym.campaign_seed="${CAMPAIGN_SEED}" \
  +teamgym.train_split="${TRAIN_SPLIT}" \
  +teamgym.validation_split="${VALIDATION_SPLIT}" \
  +teamgym.rollout_profile="${ROLLOUT_PROFILE}" \
  +teamgym.training_mode="${TRAINING_MODE}" \
  +teamgym.rollout_lora_name="${ROLLOUT_LORA_NAME}" \
  +teamgym.participant_backend="${PARTICIPANT_BACKEND}" \
  +teamgym.participant_replay_path="${PARTICIPANT_REPLAY_PATH}" \
  +teamgym.fixed_participant_base_url="${FIXED_PARTICIPANT_BASE_URL}" \
  +teamgym.fixed_participant_model="${FIXED_PARTICIPANT_MODEL}" \
  +teamgym.fixed_participant_api_key="${FIXED_PARTICIPANT_API_KEY}" \
  +teamgym.fixed_participant_temperature="${FIXED_PARTICIPANT_TEMPERATURE}" \
  +teamgym.fixed_participant_val_temperature="${FIXED_PARTICIPANT_VAL_TEMPERATURE}" \
  +teamgym.fixed_coordinator_backend="${FIXED_COORDINATOR_BACKEND}" \
  +teamgym.fixed_coordinator_base_url="${FIXED_COORDINATOR_BASE_URL}" \
  +teamgym.fixed_coordinator_model="${FIXED_COORDINATOR_MODEL}" \
  +teamgym.fixed_coordinator_api_key="${FIXED_COORDINATOR_API_KEY}" \
  +teamgym.fixed_coordinator_temperature="${FIXED_COORDINATOR_TEMPERATURE}" \
  +teamgym.fixed_coordinator_val_temperature="${FIXED_COORDINATOR_VAL_TEMPERATURE}" \
  +teamgym.reward_config="${REWARD_CONFIG}" \
  +teamgym.written_judge_model="${WRITTEN_JUDGE_MODEL}" \
  +teamgym.written_judge_provider="${WRITTEN_JUDGE_PROVIDER}" \
  +teamgym.written_judge_base_url="${WRITTEN_JUDGE_BASE_URL}" \
  +teamgym.written_judge_endpoint="${WRITTEN_JUDGE_ENDPOINT}" \
  +teamgym.written_judge_credentials_file="${WRITTEN_JUDGE_CREDENTIALS_FILE}" \
  +teamgym.written_judge_api_key_env="${WRITTEN_JUDGE_API_KEY_ENV}" \
  +teamgym.written_judge_max_output_tokens="${WRITTEN_JUDGE_MAX_OUTPUT_TOKENS}" \
  +teamgym.written_judge_timeout_sec="${WRITTEN_JUDGE_TIMEOUT_SEC}" \
  +teamgym.written_judge_request_retry_attempts="${WRITTEN_JUDGE_REQUEST_RETRY_ATTEMPTS}" \
  +teamgym.written_judge_fail_on_error="${WRITTEN_JUDGE_FAIL_ON_ERROR}" \
  +teamgym.written_judge_cache_dir="${WRITTEN_JUDGE_CACHE_DIR}" \
  +teamgym.max_parse_retries="${MAX_PARSE_RETRIES}" \
  +teamgym.participant_format_repair_for_eval="${PARTICIPANT_FORMAT_REPAIR_FOR_EVAL}" \
  +teamgym.participant_format_repair_strategy="${PARTICIPANT_FORMAT_REPAIR_STRATEGY}" \
  +teamgym.participant_format_repair_max_utterance_tokens="${PARTICIPANT_FORMAT_REPAIR_MAX_UTTERANCE_TOKENS}" \
  +teamgym.invalid_json_penalty="${INVALID_JSON_PENALTY}" \
  +teamgym.invalid_schema_penalty="${INVALID_SCHEMA_PENALTY}" \
  +teamgym.format_credit_assignment="${FORMAT_CREDIT_ASSIGNMENT}" \
  +teamgym.disable_validation="${DISABLE_VALIDATION}" \
  +teamgym.trace_prompts=True \
  +teamgym.run_dir="${RUN_DIR}" \
  "$@" 2>&1 | tee ${TRAIN_LOG_TEE_ARGS[@]+"${TRAIN_LOG_TEE_ARGS[@]}"} "${RUN_DIR}/train.log"
