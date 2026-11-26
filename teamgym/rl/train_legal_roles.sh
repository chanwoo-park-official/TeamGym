#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAMGYM_PYTHON="${TEAMGYM_PYTHON:-python}"

# Full-parameter Qwen2.5-3B role-gradient comparison on the matched,
# no-empty-participant legal split. Participant-only training launches a
# separate immutable Qwen server for coordinator queries and final synthesis.
TRAINING_MODE="${TRAINING_MODE:-joint}"
CONTEXT_VARIANT="${CONTEXT_VARIANT:-full_context}"
case "${CONTEXT_VARIANT}" in
  full_context|riu_only) ;;
  *)
    echo "CONTEXT_VARIANT must be full_context or riu_only" >&2
    exit 2
    ;;
esac
case "${TRAINING_MODE}" in
  coordinator_only|participant_only|joint) ;;
  *)
    echo "TRAINING_MODE must be coordinator_only, participant_only, or joint" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-teamgym-qwen25-3b-full-full-context-noempty-cuadtrain-${TRAINING_MODE}-64tok-b8-n8-100upd-4h200}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TEAMGYM_RUN_ROOT="${TEAMGYM_RUN_ROOT:-outputs/teamgym}"
RUN_DIR="${RUN_DIR:-${TEAMGYM_RUN_ROOT}/${RUN_NAME}}"
TEAMGYM_CHECKPOINT_ROOT="${TEAMGYM_CHECKPOINT_ROOT:-${TEAMGYM_RUN_ROOT}/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${TEAMGYM_CHECKPOINT_ROOT}/${RUN_NAME}}"
RESUME_MODE="${RESUME_MODE:-disable}"
FIXED_COORDINATOR_BACKEND="${FIXED_COORDINATOR_BACKEND:-base_model}"
FIXED_COORDINATOR_BASE_URL="${FIXED_COORDINATOR_BASE_URL:-}"
FIXED_COORDINATOR_MODEL="${FIXED_COORDINATOR_MODEL:-${MODEL_PATH}}"
PARTICIPANT_BACKEND="${PARTICIPANT_BACKEND:-rollout}"
FIXED_PARTICIPANT_BASE_URL="${FIXED_PARTICIPANT_BASE_URL:-}"
FIXED_PARTICIPANT_MODEL="${FIXED_PARTICIPANT_MODEL:-${MODEL_PATH}}"
FROZEN_COORDINATOR_PROCESS=""
CAMPAIGN_SEED="${CAMPAIGN_SEED:-${DATA_SEED:-17}}"
DATA_SEED="${DATA_SEED:-${CAMPAIGN_SEED}}"
ROLLOUT_SEED="${ROLLOUT_SEED:-${CAMPAIGN_SEED}}"
TRAINER_SEED="${TRAINER_SEED:-${CAMPAIGN_SEED}}"
FROZEN_COORDINATOR_SEED="${FROZEN_COORDINATOR_SEED:-${CAMPAIGN_SEED}}"
FROZEN_PARTICIPANT_SEED="${FROZEN_PARTICIPANT_SEED:-${CAMPAIGN_SEED}}"

if [[ "${RESUME_MODE}" == "disable" ]]; then
  if [[ -e "${RUN_DIR}/run_manifest.json" || -e "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Refusing to overwrite existing run artifacts for ${RUN_NAME}" >&2
    exit 2
  fi
fi

cleanup_frozen_coordinator() {
  if [[ -n "${FROZEN_COORDINATOR_PROCESS}" ]] && kill -0 "${FROZEN_COORDINATOR_PROCESS}" 2>/dev/null; then
    kill -TERM "${FROZEN_COORDINATOR_PROCESS}" 2>/dev/null || true
    wait "${FROZEN_COORDINATOR_PROCESS}" 2>/dev/null || true
  fi
}
trap cleanup_frozen_coordinator EXIT

if [[ "${TRAINING_MODE}" == "participant_only" ]]; then
  # Full-parameter training needs a separate immutable coordinator, but
  # evaluation-only snapshot/probe wrappers deliberately request the
  # deterministic round-robin coordinator. Preserve that explicit backend.
  if [[ "${FIXED_COORDINATOR_BACKEND}" == "base_model" ]]; then
    FIXED_COORDINATOR_BACKEND="external_openai"
  fi
  if [[ "${FIXED_COORDINATOR_BACKEND}" == "external_openai" && -z "${FIXED_COORDINATOR_BASE_URL}" ]]; then
    FROZEN_COORDINATOR_HOST="${FROZEN_COORDINATOR_HOST:-127.0.0.1}"
    FROZEN_COORDINATOR_PORT="${FROZEN_COORDINATOR_PORT:-18080}"
    FROZEN_COORDINATOR_GPU="${FROZEN_COORDINATOR_GPU:-3}"
    FROZEN_COORDINATOR_GPU_MEMORY_UTILIZATION="${FROZEN_COORDINATOR_GPU_MEMORY_UTILIZATION:-0.15}"
    FROZEN_COORDINATOR_MAX_MODEL_LEN="${FROZEN_COORDINATOR_MAX_MODEL_LEN:-32768}"
    FROZEN_COORDINATOR_SERVER_ARGS=()
    if [[ "${FROZEN_COORDINATOR_ENFORCE_EAGER:-False}" == "True" || "${FROZEN_COORDINATOR_ENFORCE_EAGER:-False}" == "true" ]]; then
      FROZEN_COORDINATOR_SERVER_ARGS+=(--enforce-eager)
    fi
    FIXED_COORDINATOR_BASE_URL="http://${FROZEN_COORDINATOR_HOST}:${FROZEN_COORDINATOR_PORT}/v1"
    if curl -fsS "http://${FROZEN_COORDINATOR_HOST}:${FROZEN_COORDINATOR_PORT}/health" >/dev/null 2>&1; then
      echo "Refusing to reuse an unidentified server already listening on ${FROZEN_COORDINATOR_HOST}:${FROZEN_COORDINATOR_PORT}" >&2
      exit 2
    fi
    mkdir -p "${RUN_DIR}"
    CUDA_VISIBLE_DEVICES="${FROZEN_COORDINATOR_GPU}" \
      VLLM_USE_V1="${FROZEN_COORDINATOR_VLLM_USE_V1:-1}" \
      "${TEAMGYM_PYTHON:-python}" \
      -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
      --served-model-name "${FIXED_COORDINATOR_MODEL}" \
      --host "${FROZEN_COORDINATOR_HOST}" \
      --port "${FROZEN_COORDINATOR_PORT}" \
      --gpu-memory-utilization "${FROZEN_COORDINATOR_GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${FROZEN_COORDINATOR_MAX_MODEL_LEN}" \
      --dtype bfloat16 \
      --seed "${FROZEN_COORDINATOR_SEED}" \
      --enable-prefix-caching \
      --disable-log-requests \
      ${FROZEN_COORDINATOR_SERVER_ARGS[@]+"${FROZEN_COORDINATOR_SERVER_ARGS[@]}"} \
      >"${RUN_DIR}/frozen_coordinator_server.log" 2>&1 &
    FROZEN_COORDINATOR_PROCESS=$!
    for _ in $(seq 1 120); do
      if curl -fsS "http://${FROZEN_COORDINATOR_HOST}:${FROZEN_COORDINATOR_PORT}/health" >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "${FROZEN_COORDINATOR_PROCESS}" 2>/dev/null; then
        echo "Frozen coordinator server exited during startup; inspect ${RUN_DIR}/frozen_coordinator_server.log" >&2
        exit 2
      fi
      sleep 2
    done
    if ! curl -fsS "http://${FROZEN_COORDINATOR_HOST}:${FROZEN_COORDINATOR_PORT}/health" >/dev/null 2>&1; then
      echo "Timed out waiting for frozen coordinator server at ${FIXED_COORDINATOR_BASE_URL}" >&2
      exit 2
    fi
  fi
fi

if [[ "${TRAINING_MODE}" == "coordinator_only" && "${PARTICIPANT_BACKEND}" == "external_openai" ]]; then
  if [[ -z "${FIXED_PARTICIPANT_BASE_URL}" ]]; then
    FROZEN_PARTICIPANT_HOST="${FROZEN_PARTICIPANT_HOST:-127.0.0.1}"
    FROZEN_PARTICIPANT_PORT="${FROZEN_PARTICIPANT_PORT:-18080}"
    FROZEN_PARTICIPANT_GPU="${FROZEN_PARTICIPANT_GPU:-3}"
    FROZEN_PARTICIPANT_GPU_MEMORY_UTILIZATION="${FROZEN_PARTICIPANT_GPU_MEMORY_UTILIZATION:-0.15}"
    FROZEN_PARTICIPANT_MAX_MODEL_LEN="${FROZEN_PARTICIPANT_MAX_MODEL_LEN:-32768}"
    FROZEN_PARTICIPANT_SERVER_ARGS=()
    if [[ "${FROZEN_PARTICIPANT_ENFORCE_EAGER:-False}" == "True" || "${FROZEN_PARTICIPANT_ENFORCE_EAGER:-False}" == "true" ]]; then
      FROZEN_PARTICIPANT_SERVER_ARGS+=(--enforce-eager)
    fi
    FIXED_PARTICIPANT_BASE_URL="http://${FROZEN_PARTICIPANT_HOST}:${FROZEN_PARTICIPANT_PORT}/v1"
    if curl -fsS "http://${FROZEN_PARTICIPANT_HOST}:${FROZEN_PARTICIPANT_PORT}/health" >/dev/null 2>&1; then
      echo "Refusing to reuse an unidentified server already listening on ${FROZEN_PARTICIPANT_HOST}:${FROZEN_PARTICIPANT_PORT}" >&2
      exit 2
    fi
    mkdir -p "${RUN_DIR}"
    CUDA_VISIBLE_DEVICES="${FROZEN_PARTICIPANT_GPU}" \
      VLLM_USE_V1="${FROZEN_PARTICIPANT_VLLM_USE_V1:-1}" \
      "${TEAMGYM_PYTHON:-python}" \
      -m vllm.entrypoints.cli.main serve "${FIXED_PARTICIPANT_MODEL}" \
      --served-model-name "${FIXED_PARTICIPANT_MODEL}" \
      --host "${FROZEN_PARTICIPANT_HOST}" \
      --port "${FROZEN_PARTICIPANT_PORT}" \
      --gpu-memory-utilization "${FROZEN_PARTICIPANT_GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${FROZEN_PARTICIPANT_MAX_MODEL_LEN}" \
      --dtype bfloat16 \
      --seed "${FROZEN_PARTICIPANT_SEED}" \
      --enable-prefix-caching \
      --disable-log-requests \
      ${FROZEN_PARTICIPANT_SERVER_ARGS[@]+"${FROZEN_PARTICIPANT_SERVER_ARGS[@]}"} \
      >"${RUN_DIR}/frozen_participant_server.log" 2>&1 &
    FROZEN_COORDINATOR_PROCESS=$!
    for _ in $(seq 1 120); do
      if curl -fsS "http://${FROZEN_PARTICIPANT_HOST}:${FROZEN_PARTICIPANT_PORT}/health" >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "${FROZEN_COORDINATOR_PROCESS}" 2>/dev/null; then
        echo "Frozen participant server exited during startup; inspect ${RUN_DIR}/frozen_participant_server.log" >&2
        exit 2
      fi
      sleep 2
    done
    if ! curl -fsS "http://${FROZEN_PARTICIPANT_HOST}:${FROZEN_PARTICIPANT_PORT}/health" >/dev/null 2>&1; then
      echo "Timed out waiting for frozen participant server at ${FIXED_PARTICIPANT_BASE_URL}" >&2
      exit 2
    fi
  fi
fi

# Query coverage and successful replies are logged diagnostics only.  The
# default task-only reward lets us test whether probing emerges instrumentally.
env \
  RLLM_DISABLE_TRAJECTORY_VISUALIZATION="${RLLM_DISABLE_TRAJECTORY_VISUALIZATION:-1}" \
  RLLM_PTR_DEBUG_MAX_REQUESTS="${RLLM_PTR_DEBUG_MAX_REQUESTS:-0}" \
  TEAMGYM_PYTHON="${TEAMGYM_PYTHON:-python}" \
  LEGAL_DATA_ROOT="${LEGAL_DATA_ROOT:-${TEAMGYM_DATA_ROOT:-data}/rl/legal}" \
  CONTEXT_VARIANT="${CONTEXT_VARIANT}" \
  MODEL_PATH="${MODEL_PATH}" \
  TRAIN_SPLIT="${TRAIN_SPLIT:-train}" \
  VALIDATION_SPLIT="${VALIDATION_SPLIT:-validation_cuad}" \
  TRAINING_MODE="${TRAINING_MODE}" \
  PARTICIPANT_BACKEND="${PARTICIPANT_BACKEND}" \
  PARTICIPANT_REPLAY_PATH="${PARTICIPANT_REPLAY_PATH:-}" \
  FIXED_PARTICIPANT_BASE_URL="${FIXED_PARTICIPANT_BASE_URL}" \
  FIXED_PARTICIPANT_MODEL="${FIXED_PARTICIPANT_MODEL}" \
  FIXED_PARTICIPANT_API_KEY="${FIXED_PARTICIPANT_API_KEY:-teamgym-local}" \
  FIXED_COORDINATOR_BACKEND="${FIXED_COORDINATOR_BACKEND}" \
  FIXED_COORDINATOR_BASE_URL="${FIXED_COORDINATOR_BASE_URL}" \
  FIXED_COORDINATOR_MODEL="${FIXED_COORDINATOR_MODEL}" \
  FIXED_COORDINATOR_API_KEY="${FIXED_COORDINATOR_API_KEY:-teamgym-local}" \
  INVALID_SCHEMA_PENALTY="${INVALID_SCHEMA_PENALTY:--0.1}" \
  PARTICIPANT_FORMAT_REPAIR_FOR_EVAL="${PARTICIPANT_FORMAT_REPAIR_FOR_EVAL:-False}" \
  PARTICIPANT_FORMAT_REPAIR_STRATEGY="${PARTICIPANT_FORMAT_REPAIR_STRATEGY:-wrap_raw}" \
  PARTICIPANT_FORMAT_REPAIR_MAX_UTTERANCE_TOKENS="${PARTICIPANT_FORMAT_REPAIR_MAX_UTTERANCE_TOKENS:-0}" \
  ROLLOUT_PROFILE="${ROLLOUT_PROFILE:-efficient_64_full_context}" \
  REWARD_CONFIG="${REWARD_CONFIG:-task_success}" \
  RUN_NAME="${RUN_NAME}" \
  RUN_DIR="${RUN_DIR}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  TEAMGYM_CUDA_VISIBLE_DEVICES="${TEAMGYM_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}" \
  ROLLOUT_N="${ROLLOUT_N:-8}" \
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}" \
  N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-16}" \
  RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}" \
  PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}" \
  DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-32640}" \
  DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-64}" \
  ACTOR_LR="${ACTOR_LR:-1e-6}" \
  USE_KL_LOSS="${USE_KL_LOSS:-False}" \
  KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}" \
  DATA_SHUFFLE="${DATA_SHUFFLE:-True}" \
  CAMPAIGN_SEED="${CAMPAIGN_SEED}" \
  DATA_SEED="${DATA_SEED}" \
  ROLLOUT_SEED="${ROLLOUT_SEED}" \
  TRAINER_SEED="${TRAINER_SEED}" \
  VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-False}" \
  VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}" \
  VAL_ONLY="${VAL_ONLY:-False}" \
  TEST_FREQ="${TEST_FREQ:-10}" \
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}" \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-101}" \
  SAVE_FREQ="${SAVE_FREQ:-10}" \
  RESUME_MODE="${RESUME_MODE}" \
  RESUME_FROM_PATH="${RESUME_FROM_PATH:-}" \
  WANDB_PROJECT="${WANDB_PROJECT:-teamgym_rl}" \
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-no-redundant-full-context-role-gradient-20260723}" \
  bash "${SCRIPT_DIR}/train_legal.sh" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPTS_TO_KEEP:-12}" \
    "$@"
