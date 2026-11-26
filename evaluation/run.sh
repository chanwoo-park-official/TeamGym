#!/usr/bin/env bash
# Apply the standard evaluation budgets to the existing runtime commands.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash evaluation/run.sh MODE [runner arguments]

Modes:
  single_all_riu  One actor with all pivotal information units (PIUs)
  multi_riu       Team interaction with PIUs only
  multi_full      Team interaction with full participant source packets
  single_full     One actor with all participant source packets
  judge           Judge written answers in existing trajectories

Pass --input-root, --model, and the output path to the selected runner.
Team modes also require --communication-structure or --all-communication-structures.
Use MODE --help for runner options. Later scalar options override preset budgets.
Set TEAMGYM_PYTHON to select the Python executable.
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ $1 == --help || $1 == -h ]]; then
  usage
  exit 0
fi

mode=$1
shift
common=(--temperature 0 --request-retry-attempts 4 --continue-on-error --no-trace-prompts)
actor=(--max-output-tokens 3600 --final-rationale-token-budget 1200 --final-reasoning-token-budget 1800 --no-render-html)
case "$mode" in
  single_all_riu)
    module=teamgym.interaction.run_single_agent_rui_evaluation
    defaults=("${common[@]}" "${actor[@]}" --timeout-sec 600 --context-token-budget 40000)
    ;;
  multi_riu|multi_full)
    if [[ $mode == multi_riu ]]; then
      module=teamgym.interaction.run_rui_context_evaluation
    else
      module=teamgym.interaction.run_evaluation
    fi
    defaults=(
      "${common[@]}" "${actor[@]}" --timeout-sec 600 --turn-budget 16
      --participant-context-token-budget 40000
      --participant-utterance-token-budget 600 --coordinator-utterance-token-budget 600
      --participant-reasoning-token-budget 1200 --coordinator-reasoning-token-budget 1200
    )
    ;;
  single_full)
    module=teamgym.interaction.run_single_agent_full_context_evaluation
    defaults=("${common[@]}" "${actor[@]}" --timeout-sec 900 --context-token-budget 200000)
    ;;
  judge)
    module=teamgym.data.long_context_reasoning.judge
    defaults=("${common[@]}" --max-output-tokens 1800 --timeout-sec 600 --no-render-reports --no-request-log)
    ;;
  *)
    printf 'Unknown evaluation mode: %s\n' "$mode" >&2
    usage >&2
    exit 2
    ;;
esac

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${TEAMGYM_PYTHON:-python}" -m "$module" "${defaults[@]}" "$@"
