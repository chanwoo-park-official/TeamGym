<div align="center">

# TeamGym

### A Gym for Team Decision-Making by AI Agents with Distributed Information

Chanwoo Park · Jaywoong Jeong · Zachary Wojtowicz · Yubin Kim · Taekyun Lee · Kaiqing Zhang · Asuman Ozdaglar

<!-- Affiliations: add superscripts after each name and a line such as
<sup>1</sup>University A &nbsp;&nbsp; <sup>2</sup>University B -->

[![arXiv](https://img.shields.io/badge/arXiv-2610.37571-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.37571)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffd21e.svg?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/TeamGym/teamgym-datasets)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-4c8bf5.svg)](CITATION.cff)

[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](requirements/README.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CPU smoke tests](https://github.com/chanwoo-park-official/TeamGym/actions/workflows/cpu-smoke.yml/badge.svg)](https://github.com/chanwoo-park-official/TeamGym/actions/workflows/cpu-smoke.yml)

</div>

<!-- Teaser figure: add e.g. assets/teamgym_overview.png here with a one-line caption.
<p align="center"><img src="assets/teamgym_overview.png" width="85%"></p> -->

## Overview

This repository provides the dataset construction, evaluation, and training code
for *TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information*.

The paper refers to the evidence units needed to solve a task as pivotal
information units (PIUs).

## Contents

- [Quickstart](#quickstart)
- [Citation](#citation) · [License](#license)
- [Main Paths](#main-paths)
- [Paper Domains](#paper-domains)
- [Current Legal Pipelines](#current-legal-pipelines)
- [Data Flow and Release Inventory](#data-flow-and-release-inventory)
- [Installation](#installation)
- [Paths and Local Environments](#paths-and-local-environments)
- [Example Commands](#example-commands)
- [Policy Training](#policy-training)
- [Interaction Evaluation](#interaction-evaluation)

## Quickstart

From the repository root, use Python 3.11 to install the evaluation tools and
list the available datasets:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/lightweight-py311.txt ".[llm]"
teamgym-generate --list
teamgym-evaluate --help
```

The [one-task evaluation example](evaluation/README.md#run-one-task) downloads
the public Legal full-context subset and runs one task with model API
credentials. The [construction recipes](scripts/data_generation/RECONSTRUCTION.md#accepted-task-format)
write accepted standard-evaluation tasks in the same format as Hugging Face.

## Citation

If you use TeamGym in research, cite Park et al., *TeamGym: A Gym for Team
Decision-Making by AI Agents with Distributed Information*, arXiv:2610.37571
(2026). The [citation file](CITATION.cff) includes the full author list and DOI.

## License

TeamGym-original code and documentation are released under the [MIT License](LICENSE).
The bundled [rLLM](rllm/LICENSE) and [VERL](verl/LICENSE) code retain their Apache
2.0 licenses. Source datasets and model weights have their own terms.

## Main Paths

- `teamgym/`: shared core, construction pipelines for all four domains, interaction, and scoring.
- `scripts/data_generation/` documents construction recipes, source provenance, and Slurm jobs.
- [`evaluation/`](evaluation/README.md): portable evaluation commands, reference configurations, reporting, and model tools.
- [`scripts/storage/`](scripts/storage/README.md): packaging and verification for frozen datasets and selected trajectories.
- `generated/`: local construction outputs; released datasets and trajectories are distributed separately.
- [`teamgym/rl/`](teamgym/rl/README.md): legal and finance policy training, endpoint evaluation, and offline checks.

## Paper Domains

TeamGym groups its 12 source corpora into four domains:

| Domain | Registry key | Paper source corpora |
| --- | --- | --- |
| Legal | `legal` | CUAD, MAUD, eCFR Casebook, MSLR, Multi-LexSum, SARA |
| Finance | `finance` | FinDVer, FinLongDocQA |
| Medical | `medical` | MLCR, EvidenceBench |
| Software Engineering | `swe` | SWE Atlas Codebase QnA, DeepSWE |

The construction registry exposes all 12 source constructors under the four
domain prefixes. The [construction guide](scripts/data_generation/README.md#finance-medical-and-software-construction)
describes Finance, Medical, and Software Engineering source preparation.
Finance RL preparation and training are available under `teamgym/rl/`.
See the [registry guide](teamgym/data/README.md) for command names.

The [dataset interface guide](teamgym/core/README.md)
describes construction outputs, actor-visible information, answer formats,
scoring, and the separate RL preparation needed when connecting these domains.

```bash
teamgym-generate --list
teamgym-generate legal/sara --help
teamgym-generate sara --help
```

Qualified names such as `legal/sara` and the existing short aliases invoke the
same constructors with the same arguments. Dataset labels and saved records
retain their existing formats.

## Current Legal Pipelines

The current benchmark-facing outputs use the common datapoint schema:

- CUAD: measured source spans over full contracts.
- MAUD: measured evidence units over full agreements.
- Casebook20: authored short-answer legal reasoning tasks.
- MSLR: source-excerpt administrative legal reasoning tasks.
- SARA: numeric statutory tax reasoning tasks.
- MultiLexSum: original-source distributed litigation reasoning tasks.

These six Legal pipelines use the normalized common datapoint schema. Tasks may use finite
decision labels or judged short answers, with evaluator-only oracle metadata:

```json
{
  "decision_set": ["decision_a", "decision_b"],
  "decision_option_definitions": {"decision_a": "...", "decision_b": "..."},
  "oracle": {"answer": "decision_a"}
}
```

## Data Flow and Release Inventory

The [public TeamGym dataset](https://huggingface.co/datasets/TeamGym/teamgym-datasets)
contains 289 evaluation tasks across nine corpora in Legal, Finance, Medical, and
Software Engineering. Its dataset card documents the per-corpus files, source
terms, and evaluation views. The
[combined Legal full-context file](https://huggingface.co/datasets/TeamGym/teamgym-datasets/resolve/ea0b5210dd9509131bd8f771ed278c74acce2258/legal/multi_full_context_subset.jsonl)
contains 152 tasks at the linked revision.

To build a dataset independently of the hosted TeamGym files, follow the
[reconstruction guide](scripts/data_generation/RECONSTRUCTION.md). It shows the
source and seed formats, a runnable structural construction example, and RL data
preparation. The benchmark inputs are available in the
[public TeamGym dataset](https://huggingface.co/datasets/TeamGym/teamgym-datasets).

Construction produces normalized TeamGym tasks. Those records feed two paths:

- Standard evaluation: frozen benchmark tasks → actor rollouts → task-specific
  scoring and written-answer judging.
- RL: constructed tasks → training-task preparation and split selection → frozen
  prepared splits → policy training → checkpoint evaluation.

Legal RL preparation includes authoring and reviewing written-answer tasks, in
addition to filtering and splitting. Finance preparation builds the PIU-only view
and the intended train/validation/transfer split. Preserve the prepared snapshots
to train on the same examples; model-assisted reconstruction can produce different
records. The [construction guide](scripts/data_generation/README.md) records source
requirements and remaining historical recovery gaps.

The prepared-bundle inventory was checked on **2026-11-17**. Counts refer to records;
allocation variants can share an underlying task.

| Artifact | Files and records | Availability |
| --- | --- | --- |
| Legal paper RL, full-context written tasks | `train`: 334 CUAD; `validation_cuad`: 36 CUAD; `validation_maud`: 26 MAUD | Packaged locally as `teamgym_legal_rl_v1.zip`; release download link pending. |
| Finance paper RL, final deduplicated transfer split | `train`: 538; `validation`: 60; `test`: 40 | Packaged locally as `teamgym_finance_rl_v1.zip`; release download link pending. |
| RL preparation metadata | Legal: `written_answer_manifest.jsonl`, `preparation_report.json`, `dataset_audit.json`. Finance: `split_manifest.jsonl`, `transfer_audit.json` | Included in the corresponding RL bundles with file checksums. |
| Selected trajectories | Six saved GPT-5.4 full-context team examples, one per legal corpus, with matching inputs, evaluation outputs, and applicable judgments | Packaged locally as `teamgym_trajectory_examples_v1.zip`; includes both successful and unsuccessful answers. Release download link pending. |

The [release export command](scripts/storage/README.md#export-the-prepared-rl-data-and-example-trajectories)
recreates these bundles from frozen files using only the standard library. It
preserves source bytes and records counts, selection rules, and SHA-256 checksums.

Each of the three canonical splits in an RL snapshot has `.jsonl`, `.parquet`,
and `_verl.parquet` files: nine data files per domain. Legal's combined
`validation` view has 62 records; paper recipes keep CUAD selection and MAUD
held-out evaluation separate. Finance also has one optional six-participant
qualitative test example, separate from its 40-example main test.

The [construction stage map](scripts/data_generation/RECONSTRUCTION.md#stage-map-and-evaluation-inputs)
specifies the evaluation input produced by each constructor. The
[archived-row receipts](scripts/data_generation/provenance/nonlegal_reconstruction_examples.json)
record earlier reconstruction checks:

| Domain | Saved evaluation artifacts |
| --- | --- |
| Finance | A row was verified in each archived export, `findver_one_each_91.jsonl` and `finlongdocqa_one_each_61.jsonl`. |
| Medical | An EvidenceBench row in `medical/full/accepted.jsonl` was verified. The accepted MLCR export is pending. |
| Software Engineering | The archived SWE Atlas export failed the current evidence-reference check. DeepSWE uses a dedicated verifier, and its accepted `instance.json` is pending. |

See the [training guide](teamgym/rl/README.md) for preparation and the
legal/finance paper regimes, and the [evaluation guide](evaluation/README.md) for
the frozen benchmark path.

## Installation

The Python package is `teamgym`, for example
`python -m teamgym.data.cli --list`. Reinstall the repository after updating a
checkout to refresh its console commands. Existing frozen dataset schema labels
remain supported; newly constructed records use the TeamGym label described in
the [dataset interface guide](teamgym/core/README.md).

The code supports Python 3.10+. Pinned reference environments use Python 3.11
on Linux x86_64. Select dependencies for the work you want to do:

| Use | Installation | Local GPU needed? |
| --- | --- | --- |
| Read provided JSONL | Python's built-in `json` module | No |
| Use the loader and schema | `python -m pip install .` | No |
| Evaluate through model APIs | `python -m pip install ".[llm]"` | No |
| Reconstruct datasets | `python -m pip install ".[data,llm,construction]"` | No |
| Train RL or run local GPU inference | Separate [RL environment](requirements/README.md) and local VERL fork | Yes |

For the pinned construction/evaluation environment, use:

```bash
uv sync --locked --extra construction --extra data --extra test
uv run --no-sync teamgym-generate --list
uv run --no-sync teamgym-generate finance/findver --help
```

The committed lockfile supplies package versions, and `.python-version` selects
Python 3.11. The [environment guide](requirements/README.md#uv-setup) also includes
the pip installation commands. After activating `.venv`, the commands are
available directly.

```bash
source .venv/bin/activate
teamgym-generate --help
teamgym-generate --list
teamgym-generate sara --help
teamgym-evaluate --help
teamgym-judge --help
teamgym-validate-frozen --help
teamgym-paper --help
teamgym-train --help
```

`data` installs the source-table and parquet dependencies; `llm` installs the
model API client and tokenizer. The distribution retains its existing `rllm`
name and includes the TeamGym Python modules, prompts, configurations, and
launchers. Datasets, model weights, and local experiment outputs are distributed
separately. GPU training also requires the environment and local VERL install
described in the [training guide](teamgym/rl/README.md).

The [environment guide](requirements/README.md) covers fresh environments,
CPU-only tests, the observed runtime versions, and the separate CUDA profile.
API generation and judging require credentials and may incur provider costs.
SARA source preparation also needs Apptainer and a SWI-Prolog image; see its
[construction guide](scripts/data_generation/slurm/sara_target_sliced/README.md).

## Paths and Local Environments

Commands use the active Python environment. Shell commands also accept
`TEAMGYM_PYTHON=/absolute/path/to/python`. Activate any required cluster tools
before submitting construction jobs; the submissions inherit that environment.
An explicit `TEAMGYM_ENV_FILE=/path/to/local-setup.sh` can load local setup in
each construction job. Machine-specific presets are optional and stay local.

Explicit command arguments override the following environment-backed defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TEAMGYM_DATA_ROOT` | `data` | Local dataset directory |
| `TEAMGYM_LEGAL_DATA_ROOT` | `data/legal` | Local base directory for the Legal archive utilities |
| `TEAMGYM_GENERATED_ROOT` | `generated` | Construction and archived experiment inputs |
| `TEAMGYM_SECONDARY_GENERATED_ROOT` | Same as `TEAMGYM_GENERATED_ROOT` | Optional second source tree for historical archives |
| `TEAMGYM_ARCHIVE_ROOT` | `artifacts/archives` | Archive and upload staging directories |

Relative paths are interpreted from the calling directory. Construction submit
scripts resolve configured inputs and outputs before changing the job working
directory. `TEAMGYM_DATAGEN_ROOT` overrides the July construction output root.
Use the provided submit scripts, or set `TEAMGYM_REPO_ROOT` and submit with
`--chdir=/absolute/path/to/checkout` when invoking `sbatch` directly.

The [storage guide](scripts/storage/README.md) covers archive commands. The
[training guide](teamgym/rl/README.md) covers explicit prepared-data,
model, checkpoint, and run paths.

## Example Commands

The installed commands work from any working directory. From the repository
root, `python -m teamgym.data.cli` accepts the same arguments as `teamgym-generate`.

```bash
teamgym-generate legal/cuad --help
teamgym-generate legal/maud --help
teamgym-generate legal/casebook --help
teamgym-generate legal/mslr --help
teamgym-generate legal/sara --help
teamgym-generate legal/multilexsum --help
```

Build an installable wheel without resolving runtime dependencies:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

Submit the final July 12 generation run:

```bash
CASEBOOK_BASE_SOURCE=/absolute/path/to/prepared-casebook.txt \
MSLR_INPUT_ROOT=/absolute/path/to/mslr-entries \
MULTILEXSUM_INPUT_ROOT=/absolute/path/to/multilexsum-tasks.jsonl \
bash scripts/data_generation/slurm/legal_written_construction/submit.sh
```

The pipeline has no synthetic fallback. Source rows or PIUs that cannot be
grounded are skipped or recorded as dropped construction metadata.

See [`scripts/data_generation/README.md`](scripts/data_generation/README.md)
for the retained construction paths and shared frozen-dataset validation.

## Policy Training

Use `teamgym-train` for newly prepared datasets, with explicit rollout profile and
reward choices. It supports custom split names and a CPU-only `--dry-run`; local
`--check-data` validates prepared formats and split separation. The
[training guide](teamgym/rl/README.md#train-a-constructed-dataset) documents
supported role configurations and the construction-to-training handoff.

The paper training entry point previews all settings without loading a model:

```bash
python -m teamgym.rl.paper --regime legal-joint \
  --prepared-data-dir /absolute/path/to/prepared/legal/full_context \
  --model /absolute/path/to/qwen-snapshot \
  --run-dir outputs/teamgym/legal-joint-seed29 --seed 29 --dry-run
```

The [training guide](teamgym/rl/README.md) covers the two legal roles,
independent joint policies, finance handoff training, data checks, and matched
endpoint evaluation. RL uses prepared written-answer splits in addition to the
frozen benchmark data.

`teamgym/rl/` contains policy preparation, training, and checkpoint evaluation.
It uses the repository-local `rllm/` and `verl/` runtimes.

## Interaction Evaluation

Run the current fully collaborative evaluator on the generated legal examples:

```bash
python -m teamgym.interaction.run_evaluation \
  --input-root generated/legal_10_each \
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
  --input-root generated/legal_10_each \
  --output-dir generated/legal_10_each_llm_trial_all_structures_gpt55 \
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

The interaction runtime consumes normalized `common_datapoint` rows directly. LLM
trajectories save exact actor prompt records under `trajectories/*.prompts/`.
Hidden PIUs and oracle fields are used only by post-episode evaluation, not by
participant or coordinator prompts.

By default, actor-visible prompts expose participant IDs but not generated role
names. Set `--participant-role-name-mode` to `datapoint`, `participant_id`, or
`generic` only when you explicitly want role labels in prompts. Participant
prompts include common information under `task.initial_common_information` and
assigned private context-unit text as one `private_source_material` string,
truncated only by `--participant-context-token-budget` when the assigned
material is longer than the configured context budget. Over-budget private
source text is sampled across the assigned material in original order rather
than front-truncated. Assigned private context preserves the original source
text. PIU IDs, PIU summaries, oracle labels, and answer keys stay in evaluator
metadata.
Participant confidence is not requested by default; pass
`--participant-confidence` only when you want optional confidence values.
Participant prompts receive the public conversation as `chat_history`; the last
message is the current coordinator request, and the participant is instructed to
respond to that last message.
Communication structures are explicit CLI choices. Pass
`--communication-structure` one or more times, or pass
`--all-communication-structures` for the prompt-debug sweep; omitting both is an
error. Coordinator multi-target modes are
`coordinator_select_multiple__coordinator_decision`, where the coordinator
chooses a subset of participants, and
`coordinator_ask_everybody__coordinator_decision`, where the runtime sends the
coordinator request to every participant.

The utterance/rationale token budgets control public text saved into the
transcript. The reasoning token budgets are prompt instructions for bounded
private deliberation before returning JSON. `--max-output-tokens` is the API
completion cap and should cover reasoning, public output, and JSON overhead.

Each run also writes `interaction_report.html`. To render or refresh a report
for an existing run directory:

```bash
python -m teamgym.visualization.interaction_trajectory_report \
  --input generated/teamgym_interaction_eval/<run_id>_interaction_eval
```
