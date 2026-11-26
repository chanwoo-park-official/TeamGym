"""Run interaction evaluation with participant private context replaced by RIUs only."""

from __future__ import annotations

from teamgym.interaction.rui_context import rui_only_datapoint
from teamgym.interaction.run_evaluation import _parse_args, run_interaction_evaluation


def main() -> None:
    args = _parse_args()
    run_interaction_evaluation(
        args,
        datapoint_transform=rui_only_datapoint,
        run_label="rui_context_interaction_eval",
        manifest_extra={
            "context_variant": "rui_only",
            "context_variant_description": "Participant private context units contain only owned RIU text.",
        },
    )


if __name__ == "__main__":
    main()
