# Dataset Generation

Reusable generation logic lives under `teamgym.data`. Use `teamgym-generate`
to run a dataset constructor. This directory holds shared construction
commands, source preparation tools, and cluster recipes.

- `provenance/` contains source pins, lineage records, and frozen validation receipts.
- `slurm/` contains the job bundles for Legal construction and qualification.

For construction without downloading the hosted TeamGym dataset, start with the
[reconstruction guide](RECONSTRUCTION.md). It maps the stages, source inputs,
and evaluation files for all 12 datasets. It also includes an offline example
that builds normalized records and shows the RL preparation handoff.

The [release inventory](../../README.md#data-flow-and-release-inventory) lists the
frozen benchmark and prepared RL artifacts, counts, and download status.
Accepted evaluation files are listed in the
[stage map](RECONSTRUCTION.md#stage-map-and-evaluation-inputs). Training uses a separate
[RL preparation stage](../../teamgym/rl/README.md).

From the repository root, install with `python -m pip install ".[data,llm,construction]"`
and use `teamgym-generate DOMAIN/DATASET`. For example,
`teamgym-generate legal/sara finalize --help`. The existing short form
`teamgym-generate sara finalize --help` remains supported.
`teamgym-generate --list` lists the four paper domains and the construction
commands available in this checkout. The
[registry guide](../../teamgym/data/README.md) explains
how construction code from another domain joins the shared command.
For the pinned Python 3.11 reference versions, add
`-c requirements/lightweight-py311.txt` to the installation command. The
[environment guide](../../requirements/README.md) documents CPU-only tests
and the additional Prolog/container tools for SARA source preparation.
`teamgym-judge` and `teamgym-validate-frozen` provide the shared batch judge and
frozen-archive validator. Prompts are included in the installed package; source
datasets and generated datasets remain external inputs.

## Canonical Commands

| Dataset | Command | Package implementation |
|---|---|---|
| CUAD | `teamgym-generate legal/cuad` | `data.legal.cuad.pipeline` |
| MAUD | `teamgym-generate legal/maud` | `data.legal.maud.pipeline` |
| Casebook20 | `teamgym-generate legal/casebook --base-source FILE` | `data.legal.casebook.generation` |
| MSLR | `teamgym-generate legal/mslr --input-root DIRECTORY` | `data.legal.mslr.generation` |
| SARA | `teamgym-generate legal/sara STAGE` | `data.legal.sara.pipeline` |
| MultiLexSum | `teamgym-generate legal/multilexsum STAGE` | `data.legal.multilexsum.pipeline` |

MultiLexSum stages are `prepare-sources`, `prepare-packets`, `generate`,
`short-answer`, `expand-packets`, `audit-grounding`, and `select-grounded`.
The shared long-context modules cover allocation, support audits, task filtering,
and judging. The batch judge runs through `teamgym-judge`. Its grading prompts,
rubric normalization, and numeric checks live in `teamgym.scoring.short_answers`
and are also used by evaluation and RL training.

SARA retains the paper's final v10 target-sliced construction. Its stages are
`prepare`, `generate`, `refresh`, `merge`, `qualify`, and `finalize`. Preparation
requires the 100-case source JSON and a SWI-Prolog Apptainer image; generation
requires the resulting audited cases. The production launcher and stage order
are documented in `slurm/sara_target_sliced/README.md`.

## Finance, Medical, and Software Construction

See the [non-Legal construction guide](NONLEGAL_CONSTRUCTION.md) for the six
dataset commands, upstream inputs, and evaluation exports.

## CUAD and MAUD Final Construction

Both constructors and the MAUD partitioner fetch the source revisions pinned in
the downloader. Use the same `--cache-dir` for generation and partitioning.
Set `HF_HUB_OFFLINE=1` when the required files are already cached. The source
checksums and MAUD contract path are listed in the [Legal source manifest](provenance/legal_source_manifest.json).

The frozen `legal_allocated_v1` bundle identifies the retained construction
inputs. `provenance/cuad_maud_final_lineage.json` records their source hashes, normalized
dataset hashes, observed construction settings, and remaining recovery gaps.

The retained construction path is:

1. Generate finite-choice tasks with the CUAD measured-span or MAUD measured-PIU
   funnel, using the canonical commands above. Both write accepted trials and
   pre-partition task tuples.
2. Convert accepted trials to full-document participant partitions. Pass the
   accepted JSONL as `--accepted-path` to `teamgym-generate legal/cuad partition`
   or `teamgym-generate legal/maud partition`. Use `partition --help` to see
   the available settings.
3. Preserve the opaque decision labels and their recorded permutation metadata.
   The partitioning modules and `data.legal.decision_permutation` implement this
   behavior; `data.legal.permute_generated_decisions` also supports the saved
   post-construction permutation pass.
4. Normalize the retained partition records and match the frozen paper IDs with
   `scripts/storage/build_legal_allocated_archive.py`. The archive builder
   retains source records whose partition validation is valid.

| Dataset | Frozen source | Source rows | Normalized rows | Partition seed |
|---|---|---:|---:|---|
| CUAD | `legal_full/cuad/cuad_full_document_partitions.jsonl` | 539 | 538 | `11` |
| MAUD | `maud_q01_q02_all_partitions_20260615_182615/maud_q01_q02_all_partition_modes.jsonl` | 557 | 557 | q01: `1201`; q02: `1202` |

All preserved records use full-document context, feature-based roles, automatic
allocation, at least three participants, and chunk settings of 500 target
words, 120 minimum words, 900 soft-maximum words, and 1,200 hard-maximum words.
MAUD contains 279 q01 records and 278 q02 records. Each corpus contributes 30
instances to the frozen 163-instance model-comparison selection. The bundle's
618-instance allocation selection is historical and includes the `paired`
condition; current paper analyses select their reported allocation conditions.

These sources confirm the final artifacts and partition settings. The complete
historical generation invocations, MAUD acceptance settings, and q01/q02
question-history and assembly inputs still require recovery. Current CLI
defaults do not establish those historical settings.

CUAD and MAUD share finite-choice decisions, contract chunking, participant
assignment, full-document partition validation, and common record mapping in
`teamgym.data.legal.contracts`.
Both constructors generate participant role names during assignment and export
them as `role_name`. The later role-augmentation command skips these corpora.
[Shared decision helpers](../../teamgym/data/legal/contracts/decisions.py) · [Contract chunking](../../teamgym/data/legal/contracts/chunks.py) · [Participant assignment](../../teamgym/data/legal/contracts/assignment.py) · [Partition validation](../../teamgym/data/legal/contracts/partition_validation.py) · [Common record mapping](../../teamgym/data/legal/contracts/records.py)

## Casebook Final Construction

The included 12 CFR Parts 1 and 3 text builds all 20 authored transaction cases.
It was prepared from the GovInfo Title 12 XML by normalizing whitespace in the
section headings and paragraphs. The evidence spans match all 18 frozen paper
tasks after whitespace normalization. The original paper input is still missing
at its recorded path. [Prepared source](casebook/sources/ecfr_title12_parts_1_and_3.txt) · [Source provenance](provenance/legal_source_manifest.json)

```bash
teamgym-generate legal/casebook \
  --base-source scripts/data_generation/casebook/sources/ecfr_title12_parts_1_and_3.txt \
  --output-root generated/casebook20
```

The construction chain is authored cases → seed JSONL → shared long-context
allocation and structural checks → participant leakage deletion → all-PIU
solve and semantic qualification → paper selection and archive normalization.
The retained `slurm/legal_written_construction` batch now builds its Casebook seed before
partitioning it and requires `CASEBOOK_BASE_SOURCE` at submission time.

The frozen bundle contains 18 retained tasks and 77 allocation records across
six historical modes, including `paired`. All 18 tasks occur in the standard
paper selection. `provenance/casebook_final_lineage.json` records source and artifact
hashes, observed settings, the two authored IDs absent from the bundle, and
remaining recovery gaps. The missing rejection logs do not establish why those
two candidates were removed, so their construction code is retained.

See the [shared builder documentation](../../teamgym/data/long_context_reasoning/README.md)
for commands and the distinction between structural checks and model-based
qualification. Later participant-role augmentation remains shared with MSLR,
MultiLexSum, and SARA; it is separate from this frozen July artifact lineage.

## MSLR Final Construction

The source-excerpt generator retains six exact, nonoverlapping excerpts per
case: the event, timing, knowledge or access, trading conduct, legal basis,
and final handling. The canonical response and required decision steps use
those excerpts. Structured source fields still guide excerpt selection.

Supply the anonymized MSLR source directory explicitly. The retained final
batch uses 100 candidates and scans up to 1,389 source files in numeric entry
order:

```bash
mkdir -p generated/legal_sources
git clone --no-checkout https://github.com/yuwenhan07/MSLR-Bench.git \
  generated/legal_sources/MSLR-Bench
git -C generated/legal_sources/MSLR-Bench checkout --detach \
  50592efa2fe3aa51f14b6dd48172c6d4925847b8

teamgym-generate legal/mslr \
  --input-root generated/legal_sources/MSLR-Bench/data/processed_anonymized \
  --output-root generated/mslr100 \
  --limit 100 \
  --candidate-scan-limit 1389
```

The CLI rejects missing directories and directories without `entry_*.json`
files before creating output. Its standalone defaults remain 50 candidates
and a 1,000-file scan; use the explicit final settings above for the retained
batch recipe. The seed filename
`seed/mslr_source_excerpt_casebook50_seed.jsonl`, report schema, and dataset
variant identifiers retain their historical `casebook50` spelling even when
the requested candidate count is 100.

The retained `slurm/legal_written_construction` batch requires `MSLR_INPUT_ROOT`, builds
the seed, allocates the six historical modes, runs leakage deletion, and
qualifies all-PIU written answers with the semantic judge. Historical paper
task selection and variant assembly remain in
`scripts/data_generation/common/prepare_paper_allocations.py`. This utility
also builds the old experiment matrix and requires its archived result
summaries; standard evaluation can consume the released frozen inputs directly.

The source directory is available locally with 1,389 files. The paper records
613 builder-eligible cases, 100 candidates, and 11 retained tasks. The frozen
bundle confirms the 11 tasks and 24 historical allocation records; all 11
appear in the standard paper selection. `provenance/mslr_final_lineage.json` records
artifact hashes, current source-file hashes for those 11 tasks, observed
metadata, and the distinction between confirmed artifacts and unrecovered
qualification history. Full model qualification is still required when
regenerating a candidate pool.

## MultiLexSum: Frozen Paper Data and New Construction

The frozen paper reference contains 29 tasks and 141 historical allocation
records. Its original-document passages, PIUs, questions, and oracles are
preserved in `legal_allocated_v1/data/multilexsum/instances.jsonl`. The exact
29-task evaluation input also survives inside `legal_paper_release_v1.zip` at
`trajectories/model_evidence.zip` →
`source_components/multilexsum_0712/inputs/multilexsum_full_context_qualified29.jsonl`.
Its SHA256 is recorded in `provenance/multilexsum_final_lineage.json`.

Use those frozen inputs to reproduce paper experiments. The following pipeline
is a documented reconstruction for new examples; it does not establish the
missing historical packet-expansion command or guarantee the same generated
wording, selected cases, or rejection counts.

### Original-document packets

`prepare-packets` reads the upstream `train.json`, `dev.json`, `test.json`, and
`sources.json`, including shard roots produced by the optional
`prepare-sources` command. It streams `sources.json` instead of loading the
entire corpus. Eligibility still uses the corpus summary's length; summaries
are excluded from task-generation evidence and answers.

```bash
hf download allenai/multi_lexsum \
  releases/v20230518/train.json releases/v20230518/dev.json \
  releases/v20230518/test.json releases/v20230518/sources.json \
  --repo-type dataset --revision 055f5fa5300a5fdefb5b691c4d6c1e58f485b379 \
  --local-dir generated/legal_sources/multilexsum

teamgym-generate legal/multilexsum prepare-packets \
  --input-root generated/legal_sources/multilexsum/releases/v20230518 \
  --output-root generated/multilexsum/packets \
  --limit 100 --seed 20260710 \
  --chunk-words 250 --max-chunks-per-document 8 --excerpts-per-document 2
```

The builder selects evenly spaced chunks and excerpts, preserving original
character offsets, document IDs, and document-text hashes. The defaults bound
the context size. They describe the reconstructed recipe, rather than inferred
historical settings. `source_packets.jsonl` is a generation input, not a
qualified task dataset.

### Generate and ground tasks

The model commands require configured credentials. These examples use the
same model/provider settings as the retained qualification scripts:

```bash
model_args=(--model gpt-5.5 --provider bedrock_openai
  --base-url https://bedrock-mantle.us-east-2.api.aws/openai/v1
  --endpoint /responses --api-key-env AWS_BEARER_TOKEN_BEDROCK
  --credentials-file api_key/api_key.py)

teamgym-generate legal/multilexsum generate \
  --input-root generated/multilexsum/packets/source_packets.jsonl \
  --output-root generated/multilexsum/candidates "${model_args[@]}"

teamgym-generate legal/multilexsum short-answer \
  --input-root generated/multilexsum/candidates/common_datapoints.jsonl \
  --output-root generated/multilexsum/short_answer "${model_args[@]}"

teamgym-generate legal/multilexsum expand-packets \
  --input-root generated/multilexsum/short_answer/common_datapoints.jsonl \
  --source-packets generated/multilexsum/packets/source_packets.jsonl \
  --output-root generated/multilexsum/expanded --seed 20260712

teamgym-generate legal/multilexsum audit-grounding \
  --input-root generated/multilexsum/expanded/common_datapoints.jsonl \
  --output-root generated/multilexsum/grounding

teamgym-generate legal/multilexsum select-grounded \
  --input-root generated/multilexsum/expanded/common_datapoints.jsonl \
  --audits generated/multilexsum/grounding/grounding_audits.jsonl \
  --output-root generated/multilexsum/grounded --limit 30
```

The intermediate counterfactual choices establish PIU relevance before the
short-answer rewrite. Keep that generation logic. Expansion preserves the
task, oracle, and PIU ownership while distributing all selected background
chunks. It excludes background copies of the original chunks already used by
PIU source windows; overlapping or semantically redundant PIU windows still
require model-based leakage checks. Previously saved full source packets can
also serve as `--source-packets` when restoring contexts to existing tasks.

`select-grounded` requires one audit per task, exact PIU coverage, Boolean
support for every PIU, and support for the complete canonical answer. It
selects passing rows in input order and fails if none survive.

### Qualify expanded contexts

Expanded contexts need fresh leakage and solvability checks. For a standalone
MultiLexSum run, continue with the retained shared commands:

```bash
INPUT_ROOT=generated/multilexsum/grounded/common_datapoints.jsonl \
OUTPUT_ROOT=generated/multilexsum/leakage \
  bash scripts/data_generation/common/run_leakage_filter.sh

survivors=generated/multilexsum/leakage/01_leakage_deletion/leakage_deleted_common_datapoints.jsonl

python -m teamgym.interaction.run_single_agent_rui_evaluation \
  --input-root "$survivors" --output-dir generated/multilexsum/all_riu_eval \
  --limit 30 --context-token-budget 40000 "${model_args[@]}"

python -m teamgym.data.long_context_reasoning.judge \
  --input-root "$survivors" --trajectories-root generated/multilexsum/all_riu_eval \
  --output-root generated/multilexsum/all_riu_judge "${model_args[@]}"

python -m teamgym.data.long_context_reasoning.filtering \
  --input-jsonl "$survivors" \
  --judgments-jsonl generated/multilexsum/all_riu_judge/judgments.jsonl \
  --output-root generated/multilexsum/qualified \
  --target-count 29 --require-all-steps --skip-structural-funnel
```

The example retention target is explicit; a smaller passing pool stops the
filter and requires more candidates or a revised, disclosed target. New
allocation variants must pass their own schema and qualification checks.
The `slurm/legal_written_construction` batch constructs PIU-only
repartitions and requires `MULTILEXSUM_INPUT_ROOT`. The later full-context
projection lives in
`scripts/data_generation/common/prepare_paper_allocations.py`.
`scripts/data_generation/multilexsum/restore_qualified_context.py` retains the
historical restoration of 29 qualified full-context tasks and their replacement
in the 142-task intermediate evaluation input. Both utilities retain their
original checks and settings; use `--help` for required source artifacts.
Sensitivity campaigns are outside the public construction/evaluation/RL scope.

## Shared Construction and Validation

Run the following modules directly with `python -m`. The short module names in
the table are under `teamgym.data.long_context_reasoning`.

| Module or command | Purpose |
|---|---|
| `seed_builder`, `pipeline` | Casebook/MSLR allocation and deterministic structural checks |
| `repartition` | MultiLexSum/SARA allocation variants |
| `support_audit` | Model-based support checks for authored short-answer seeds |
| `filtering`, `teamgym-judge` | Written-answer judging and strict all-PIU retention |
| `ablation` | Optional leave-one-PIU-out necessity checks |
| `selection` | Select a row for a batch job |
| `teamgym-validate-frozen` | Offline frozen-release integrity, references, and selections |

The scripts in `common/` run Legal participant-role augmentation, leakage
filtering, and finite-choice option exclusion. Written-answer tasks use the
all-PIU qualification and judging commands.

Validate an unpacked Legal allocation bundle offline:

```bash
python -m teamgym.data.legal.archive_validation \
  --archive-root /path/to/legal_allocated_v1 \
  --report analysis_outputs/legal_frozen_validation.json

python -m pytest tests -q
```

The archive check verifies file and record hashes, schemas, unique archive IDs, source
references, context assignments, ownership metadata, and the 163-task standard
and historical 618-allocation selections. It writes reports outside the
immutable archive. `provenance/frozen_validation_reference.json` records the checked
reference bundle and observed visibility diagnostics.

Use `archive_instance_id` for record joins. MAUD intentionally reuses 239
stored instance IDs across its q01/q02 question variants; the index identifies
each record separately. The validator checks stored IDs within each question
variant and verifies the global archive IDs.

Declared PIU ownership can differ from visibility when an original source
chunk contains multiple PIUs. The frozen reference has nine such PIU
differences across eight MAUD records: none are in the standard 163-task set;
one record is in the historical allocation selection. These are reported
separately from file/schema integrity, preserving the original records. Add
`--strict-visibility` to fail on these differences when validating a newly
constructed archive. This offline check does not establish source entailment,
leakage convergence, or model solvability; retain the model-based stages above.

Tests generate their own loader fixtures and require no historical pilot
directories, credentials, model endpoints, or corpus archives.

## Maintenance Rule

Dataset algorithms belong in the package, not in command wrappers or Slurm
files. Submission files select arguments, resources, and dependencies only.
