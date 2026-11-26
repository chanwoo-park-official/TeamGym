# TeamGym

This package provides shared TeamGym components for the paper's four domains:
Legal, Finance, Medical, and Software Engineering. Its construction registry
exposes all 12 source constructors. See the
[registry guide](data/README.md) for domain keys and adding construction modules.

## Active Modules

- `core/io.py`: shared JSON/JSONL, packaged prompt, template, and text helpers;
  `data/legal/io.py` preserves existing imports.
- `core/information_allocation.py`: shared private/common evidence assignments
  and allocation metadata; `data/legal/information_allocation.py` preserves existing imports.
- `core/participant_roles.py`: shared evidence hashes, feature validation, and
  participant-role rendering; legal corpus selection and finance-specific checks stay in their domain modules.
- `data/registry.py`: domain labels, qualified construction names, and legacy aliases.
- `data/legal/cuad/pipeline.py`: stable CUAD generation API.
- `data/legal/maud/pipeline.py`: stable MAUD generation API.
- `data/legal/casebook/generation.py`: Casebook20 generation API.
- `data/legal/mslr/generation.py`: source-excerpt MSLR generation API.
- `data/legal/sara/pipeline.py`: final v10 target-sliced SARA construction stages.
- `data/legal/sara/prolog.py`: Prolog parsing and term unification;
  `proof_slicing.py` reconstructs grounded target proofs and premise requirements.
  `subproof_generation.py` handles model-driven generation and candidate audits,
  and preserves the existing helper imports.
- `data/legal/multilexsum/pipeline.py`: staged original-source MultiLexSum generation API.
- `data/long_context_reasoning/`: shared construction, partitioning, filtering, support auditing, and judging.
- `data/legal/archive_validation.py`: offline integrity and reference checks for the frozen six-corpus release.
- `interaction/`: communication-structure runtime and evaluation over normalized common datapoints.

All benchmark-facing rows use the shared schema in `core/schema.py`;
`data/legal/schema.py` preserves its existing import path.
The [dataset interface guide](core/README.md) describes the record fields,
information visibility, scoring conventions, and RL handoff for incoming domains.
CUAD and MAUD use finite decision labels. Casebook, MSLR, MultiLexSum, and SARA
use the compatibility label `free_response` with a written answer evaluated
against the canonical response and required decision steps. Construction and
validation commands are documented in
[`scripts/data_generation/README.md`](../scripts/data_generation/README.md).

## Visualization

Render a static HTML browser for generated datasets:

```bash
python -m teamgym.visualization.legal_dataset_report \
  --input-root /path/to/released_dataset.jsonl \
  --output analysis_outputs/legal_dataset_report.html
```

## Interaction Evaluation

Run the current non-training evaluator:

```bash
python -m teamgym.interaction.run_evaluation \
  --input-root /path/to/released_dataset.jsonl \
  --model gpt-5.4-mini \
  --limit 1 \
  --turn-budget 3 \
  --participant-utterance-token-budget 80 \
  --coordinator-utterance-token-budget 80 \
  --final-rationale-token-budget 120 \
  --participant-reasoning-token-budget 128 \
  --coordinator-reasoning-token-budget 128 \
  --final-reasoning-token-budget 256 \
  --participant-context-token-budget 20000 \
  --max-output-tokens 1200 \
  --communication-structure coordinator_select_one__coordinator_decision
```

Run the current all-structure debug rollout with the stronger model:

```bash
python -m teamgym.interaction.run_evaluation \
  --input-root /path/to/released_dataset.jsonl \
  --output-dir analysis_outputs/legal_debug_all_structures \
  --model gpt-5.5 \
  --limit 1 \
  --turn-budget 3 \
  --participant-utterance-token-budget 80 \
  --coordinator-utterance-token-budget 80 \
  --final-rationale-token-budget 120 \
  --participant-reasoning-token-budget 128 \
  --coordinator-reasoning-token-budget 128 \
  --final-reasoning-token-budget 256 \
  --participant-context-token-budget 20000 \
  --max-output-tokens 1200 \
  --all-communication-structures
```

The runner supports coordinator-mediated and coordinator-free communication
structures over the same task instance. LLM runs save the exact prompts and
responses used for each actor, while hidden PIUs and oracle labels remain
evaluator-only metadata. Actor-visible prompts expose participant IDs but hide
generated role names by default. Participant prompts include common information
under `task.initial_common_information` and all assigned private context-unit
text as one `private_source_material` string, truncated only by
`--participant-context-token-budget` when the assigned material is longer than
the configured context budget. Over-budget private source text is sampled across
the assigned material in original order rather than front-truncated. Assigned
original context units that contain hidden PIU evidence are preserved as source
text. PIU IDs, PIU summaries, oracle labels, and answer keys stay in evaluator
metadata. Participant confidence is not requested by default; pass
`--participant-confidence` only when you want optional confidence values.
Participant prompts receive the public conversation as `chat_history`; the last
message is the current coordinator request, and the participant is instructed to
respond to that last message. Communication structures are explicit CLI choices:
pass `--communication-structure` one or more times, or
`--all-communication-structures` for the prompt-debug sweep. Omitting both is an
error. The coordinator-simultaneous mode is
split into `coordinator_select_multiple__coordinator_decision`, where the
coordinator chooses a subset of participants, and
`coordinator_ask_everybody__coordinator_decision`, where the runtime sends the
coordinator request to every participant. Each run writes
`interaction_report.html`; existing
runs can be rendered again with:

```bash
python -m teamgym.visualization.interaction_trajectory_report \
  --input generated/teamgym_interaction_eval/<run_id>_interaction_eval
```
