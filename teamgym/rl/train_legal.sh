#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEAMGYM_PYTHON="${TEAMGYM_PYTHON:-python}"
CONTEXT_VARIANT="${CONTEXT_VARIANT:-riu_only}"
LEGAL_DATA_ROOT="${LEGAL_DATA_ROOT:-${TEAMGYM_DATA_ROOT:-data}/rl/legal}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-${LEGAL_DATA_ROOT}/${CONTEXT_VARIANT}}"
TRAINING_MODE="${TRAINING_MODE:-coordinator_only}"
PARTICIPANT_BACKEND="${PARTICIPANT_BACKEND:-scripted}"
ROLLOUT_PROFILE="${ROLLOUT_PROFILE:-efficient_64_coordinator}"
REWARD_CONFIG="${REWARD_CONFIG:-task_success}"
RUN_NAME="${RUN_NAME:-teamgym-qwen25-3b-full-${CONTEXT_VARIANT}-64tok-100step}"
TEAMGYM_RUN_ROOT="${TEAMGYM_RUN_ROOT:-outputs/teamgym}"
RUN_DIR="${RUN_DIR:-${TEAMGYM_RUN_ROOT}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${TEAMGYM_CHECKPOINT_ROOT:-${TEAMGYM_RUN_ROOT}/checkpoints}/${RUN_NAME}}"
WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"

if [[ ! -d "${PREPARED_DATA_DIR}" ]]; then
  echo "Missing prepared data: ${PREPARED_DATA_DIR}" >&2
  echo "See teamgym/rl/README.md#prepare-the-paper-datasets." >&2
  echo "Set PREPARED_DATA_DIR to the extracted prepared files or a rebuilt context directory." >&2
  exit 2
fi
if [[ "${TRAINER_LOGGER}" == *wandb* ]] && ! "${TEAMGYM_PYTHON}" -c 'import wandb; assert wandb.api.api_key' >/dev/null 2>&1; then
  echo "W&B is not installed and authenticated in ${TEAMGYM_PYTHON}." >&2
  exit 2
fi

mkdir -p "${WANDB_DIR}"
export WANDB_DIR
export WANDB_PROJECT="${WANDB_PROJECT:-teamgym_rl}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cuad-maud-efficient-64}"

# The local trainer starts at one and stops before its raw limit, so the
# default raw limit of 101 means exactly 100 optimizer updates.
TEAMGYM_PYTHON="${TEAMGYM_PYTHON}" \
PREPARED_DATA_DIR="${PREPARED_DATA_DIR}" \
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}" \
FULL_FINETUNE=1 \
TRAINING_MODE="${TRAINING_MODE}" \
PARTICIPANT_BACKEND="${PARTICIPANT_BACKEND}" \
CUDA_VISIBLE_DEVICES="${TEAMGYM_CUDA_VISIBLE_DEVICES:-0,1}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" \
ROLLOUT_N="${ROLLOUT_N:-4}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}" \
N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-8}" \
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}" \
ROLLOUT_PROFILE="${ROLLOUT_PROFILE}" \
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-8192}" \
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-64}" \
ACTOR_LR="${ACTOR_LR:-1e-6}" \
REWARD_CONFIG="${REWARD_CONFIG}" \
TRAINER_LOGGER="${TRAINER_LOGGER}" \
MAX_PARSE_RETRIES="${MAX_PARSE_RETRIES:-1}" \
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}" \
TEST_FREQ="${TEST_FREQ:-10}" \
DISABLE_VALIDATION="${DISABLE_VALIDATION:-False}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-101}" \
SAVE_FREQ="${SAVE_FREQ:-100}" \
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${RUN_DIR}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
RESUME_MODE="${RESUME_MODE:-disable}" \
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}" \
bash "${SCRIPT_DIR}/train_backend.sh" \
  "actor_rollout_ref.actor.checkpoint.save_contents=['model']" \
  "$@"

if [[ "${VAL_ONLY:-False}" != "True" && "${VAL_ONLY:-False}" != "true" ]]; then
  "${TEAMGYM_PYTHON}" -m teamgym.rl.tools.plot_reward \
    --log "${RUN_DIR}/train.log" \
    --output "${RUN_DIR}/reward_tracking.png"
fi
