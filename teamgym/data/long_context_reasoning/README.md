# Long-Context Reasoning Construction

The shared builder allocates pivotal information units (PIUs) across long
participant contexts and writes the common datapoint schema used by evaluation.
Casebook and MSLR use this builder; other retained datasets also reuse its
schema and auditing helpers.

## Casebook Seed Construction

The Casebook generator retains all 20 authored transaction cases, their rule
spans, calculations, canonical answers, and judging rubrics. Supply the frozen
prepared text of 12 CFR Parts 1 and 3 explicitly:

```bash
teamgym-generate legal/casebook \
  --base-source /path/to/frozen_casebook_rules.txt \
  --output-root generated/casebook
```

This writes `raw/ecfr_title12_parts1_3_casebook20_source.txt`,
`seed/ecfr_casebook20_short_answer_seed.jsonl`, and `seed_build_report.json`.
Each seed combines the regulations with its own transaction memo. An input
containing previously appended Casebook memos is stripped back to the rules
before the current memos are appended.

The original prepared source is absent at the recorded historical paths. A
current download is not evidence of the historical text. The frozen normalized
bundle contains 18 retained tasks and 77 historical allocation records; hashes,
observed settings, and recovery gaps are recorded in
[`provenance/casebook_final_lineage.json`](../../../scripts/data_generation/provenance/casebook_final_lineage.json).
All 20 authored cases remain necessary to reproduce candidate construction.

## Allocation and Structural Checks

```bash
python scripts/data_generation/common/run_pipeline.py \
  --seed-spec generated/casebook/seed/ecfr_casebook20_short_answer_seed.jsonl \
  --output-root generated/casebook/partitions/one_each \
  --allocation-mode one_each \
  --target-context-tokens 12000 \
  --min-participants 3 \
  --seed 20260712
```

The retained batch constructs `one_each`, `paired`, `overlap`,
`redundant_player`, `common_anchor`, and `bottleneck` variants. These six modes
reproduce the historical construction scope; paper analyses select their
reported conditions from the retained variants.

The shared pipeline writes:

- `01_candidates/common_datapoints.jsonl`
- `02_leakage_deleted/leakage_deleted_common_datapoints.jsonl`
- `03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl`
- `pipeline_report.json`, `pipeline_report.md`, and `pipeline_report.html`

These stages check exact PIU visibility and whether a single participant owns a
complete oracle sufficient set. They are deterministic structural checks.
The `03_option_exclusion` directory name also applies to short-answer tasks;
its structural coverage check does not call a finite-choice LLM auditor.
`scripts/data_generation/common/build_seed.py` exposes the allocation stage
alone when structural funnels are not required.

## Model-Based Qualification

The retained [Legal written-answer batch](../../../scripts/data_generation/slurm/legal_written_construction/README.md)
adds participant-context leakage detection and deletion, convergence checks,
an all-PIU single-agent solve, and strict semantic judging with GPT-5.5.
Its default retained pool is `all_riu_qualified/common_datapoints.jsonl`.
Leave-one-PIU-out qualification is an optional necessity diagnostic.

Casebook uses `answer_type: short_answer_judged` and the singleton compatibility
decision `free_response`. Qualification evaluates the written conclusion and
every required reasoning step. Finite-choice LLM option exclusion does not
qualify these tasks. General shared utilities retain finite-choice support for
their other callers.

## Seed Contract

Each JSONL row supplies an `instance_id`, source identity, `source_text`,
`question`, answer, and `rius`. Each PIU has a unique `rui_id`, exact
`source_span`, `normalized_claim`, and `rui_type`; `required_rui_ids` and
`reasoning_steps` reference those IDs. Short-answer seeds additionally carry
`canonical_response`, `required_decision_steps`, and a `judge_rubric`.

`schema.py` preserves these fields in the common datapoint and oracle.
`seed_builder.py` constructs private contexts and information allocations.
`pipeline.py` implements the structural checks above. The semantic judge is
exposed through `scripts/data_generation/common/judge.py` for generated
trajectories and through the retained batch for construction qualification.
