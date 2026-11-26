"""Run the Single-Agent Full-Context legal baseline."""

from teamgym.interaction.run_single_agent_rui_evaluation import (
    full_source_context_text,
    full_source_context_units,
    main_full_context as main,
    run_single_agent_full_context_episode,
)

__all__ = [
    "full_source_context_text",
    "full_source_context_units",
    "main",
    "run_single_agent_full_context_episode",
]


if __name__ == "__main__":
    main()
