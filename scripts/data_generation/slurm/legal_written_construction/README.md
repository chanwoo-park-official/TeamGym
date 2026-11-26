# Legal Written-Answer Construction

## Scope

This submission constructs every eligible information-partition mode for the current final legal reasoning pools:

| Dataset | Unique base tasks | Partition modes | Candidate variants |
|---|---:|---:|---:|
| MSLR | 100 selected from 613 builder-eligible source cases | 6 | 600 |
| Casebook | all 20 currently authored tasks | 6 | 120 |
| MultiLexSum | 30 fully retained original-source tasks | 6 | 180 |
| **Total** | **150** | | **900** |

The six modes are `one_each`, `paired`, `overlap`, `redundant_player`, `common_anchor`, and `bottleneck`. A mode is retained only when its allocation and subsequent validation pass. Rejection is recorded rather than replaced by a forced allocation.

CUAD and MAUD are not regenerated. Their previously accepted partition attempts remain the final source for those datasets.

The seed job builds MSLR and all 20 authored Casebook candidates. Casebook
requires `CASEBOOK_BASE_SOURCE`, a readable file containing the frozen prepared
text of 12 CFR Parts 1 and 3. Its seed is written to
`seeds/casebook20/seed/ecfr_casebook20_short_answer_seed.jsonl` under the output
root and consumed by the dependent partition job. The original prepared text
has not been recovered at its historical paths; the frozen normalized bundle
alone does not provide a complete source reconstruction recipe.

MSLR requires `MSLR_INPUT_ROOT`, the directory containing the anonymized
`entry_*.json` source files. This directory currently contains 1,389 files in
the local source checkout. The retained invocation explicitly sets
`--limit 100 --candidate-scan-limit 1389`; the standalone CLI defaults are
smaller. Its output remains
`seeds/mslr100/seed/mslr_source_excerpt_casebook50_seed.jsonl`, whose historical
filename is consumed by the dependent partition job. Missing or empty source
directories are rejected by the MSLR CLI before it writes output.

## Validation

The seed plan contains at most 900 task-partition candidates. After structural filtering, an explicit manifest assigns every row that actually exists to its own Slurm array cell. Every cell runs GPT-5.5 through AWS Bedrock for:

1. participant-context leakage detection and deletion;
2. convergence checking after deletion;
3. single-agent all-PIU answer generation;
4. strict semantic judging of the conclusion and every required reasoning step.

The default final pool is the set that passes the all-PIU semantic judgment.
Leave-one-PIU-out evaluation is an optional minimality diagnostic, not a
default retention requirement. It tests whether every individual PIU is
indispensable, which is stronger than the communication and collective
reasoning property targeted by the experiment.

Finite-choice option exclusion is intentionally not used for these datasets.
Their compatibility decision set is the singleton `free_response`, while the
actual outcome is a semantically judged written answer.

The account QOS permits at most 448 submitted jobs. Validation therefore uses 400-cell waves, the largest safe wave alongside prerequisite jobs. Continuation jobs derive each subsequent range from the manifest length as the prior wave leaves the queue. Prompt traces are disabled to limit storage, while reports and request metadata are retained. A candidate removed by leakage filtering is recorded as a successful rejection rather than reported as a crashed Slurm task.

## MultiLexSum Qualification

The six MultiLexSum variants in this submission are explicitly PIU-only repartitions of the 30 retained original-source tasks. They are appropriate for the information-partition experiment. They do not replace the existing complete-original-document condition used to measure retrieval and summarization burden.

The historical funnel reported 48 grounded tasks and selected 30. The exact
29-task evaluation input has been recovered from the paper release archive;
the rejected 30th candidate and original packet-expansion command remain
unrecovered. Set `MULTILEXSUM_INPUT_ROOT` to a readable, constructed task JSONL.
The batch processes up to 30 supplied tasks, so a recovered 29-task input
produces at most 174 initial variants rather than the historical 180.

New source-packet preparation and expansion stages are documented in
`scripts/data_generation/README.md`. Regenerated contexts require fresh
grounding, leakage, and all-PIU qualification. The recovered dataset remains
the reference for paper experiments.

## Outputs

Outputs default to `generated/legal_construction` under the calling directory.
Set `TEAMGYM_DATAGEN_ROOT` to choose an explicit run root. Activate the desired
Python environment before submission, or set `TEAMGYM_PYTHON`; jobs inherit
that selection and do not activate a hardcoded conda environment.

New runs store per-task outputs in the following directories under the run root.

| Stage | Default directory | Directory override |
| --- | --- | --- |
| Participant leakage validation | `leakage_validation` | `VALIDATION_DIR_NAME` |
| All-PIU qualification | `all_piu_qualification` | `QUALIFICATION_DIR_NAME` |
| Leave-one-PIU-out diagnostic | `leave_one_piu_out` | `NECESSITY_DIR_NAME` |

For an older run, set the corresponding overrides to its existing directory
names, such as `validation_v2`, `qualification_v1`, or `necessity_v1`. The
standalone merge commands accept the same names through `--validation-dir`,
`--qualification-dir`, and `--necessity-dir`.

The qualification stages write:

- `leakage_survivors/common_datapoints.jsonl`
- `all_riu_qualified/common_datapoints.jsonl`
- a `summary.json` and detailed judgment records at each stage.

When the optional necessity diagnostic is enabled, it additionally writes
`final_qualified/common_datapoints.jsonl`. That diagnostic output must not
replace the default all-PIU-qualified pool without a separate minimality claim.

Each merge records expected, processed, passed, failed, and missing rows. A
merge fails without writing its accepted dataset when any manifest row is
unaccounted for. Each submission requires fresh validation and qualification
directories; use new `VALIDATION_DIR_NAME`, `QUALIFICATION_DIR_NAME`, and
`NECESSITY_DIR_NAME` values when launching another pass.

Slurm logs remain in the repository `slurm_logs` directory.

## Submission

```bash
CASEBOOK_BASE_SOURCE=/path/to/frozen_casebook_rules.txt \
MSLR_INPUT_ROOT=/path/to/MSLR-Bench/data/processed_anonymized \
MULTILEXSUM_INPUT_ROOT=/path/to/multilexsum/common_datapoints.jsonl \
  bash scripts/data_generation/slurm/legal_written_construction/submit.sh
```

The shell submits seed creation, partition construction, leakage filtering,
and strict all-PIU qualification with explicit Slurm dependencies. It stops
after producing `all_riu_qualified/common_datapoints.jsonl`.

To run the optional leave-one-PIU-out minimality diagnostic as well:

```bash
CASEBOOK_BASE_SOURCE=/path/to/frozen_casebook_rules.txt \
MSLR_INPUT_ROOT=/path/to/MSLR-Bench/data/processed_anonymized \
MULTILEXSUM_INPUT_ROOT=/path/to/multilexsum/common_datapoints.jsonl RUN_NECESSITY=1 \
  bash scripts/data_generation/slurm/legal_written_construction/submit.sh
```
