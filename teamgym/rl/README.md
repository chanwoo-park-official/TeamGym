# TeamGym policy training

`teamgym-train` trains on prepared TeamGym datasets with configurable splits,
roles, rollout profiles, and rewards. `teamgym-paper` retains the fixed legal
and finance paper recipes. Both use the ordinary TeamGym interaction runtime
inside the repository's modified RLLM/VERL trainers.

| Regime | Trainable policy | Fixed policy | Data and selection |
| --- | --- | --- | --- |
| `legal-participant` | Participant replies (Stage B) | Base coordinator | CUAD 334 train / 36 selection; MAUD 26 held out |
| `legal-coordinator` | Coordinator queries and final answer (Stage C) | Selected Stage B participant | Same legal splits |
| `legal-joint` | Separate coordinator and participant policies | None | Same legal splits |
| `finance-handoff` | Shared participant policy for communication, handoff, stopping, and final answer | No coordinator in this communication structure | FinDVer + FinLong 538 train / 60 validation; FinLong four-participant 40 held out |

Legal joint training keeps separate weights, optimizers, rollout servers, and
checkpoints for the two roles. The shared-policy `joint` mode in the small
development launcher is a different experiment.

## Environment and data

Use the separate [RL installation profile](../../requirements/README.md) for
Linux x86_64, Python 3.11, PyTorch 2.8.0/CUDA 12.8, and vLLM 0.11.0. It pins
the full third-party dependency set and installs FlashAttention separately
against the matching CUDA toolchain. Install the repository runtimes in that
environment from the repository root:

```bash
python -m pip install --no-build-isolation --no-deps ./verl .
```

The paper used four H200 GPUs. The command above assumes the profile's GPU
stack and dependencies are already installed. For development, install each
local package separately with `-e`. The root
distribution includes TeamGym, RLLM, prompts, configurations, and launchers;
the modified VERL distribution is installed separately. The entry point uses
the installed packages, or this checkout and its vendored VERL directory when
running from source.

The [observed environment record](../../requirements/reference-environment.json)
preserves the versions used for the CPU checks, including existing dependency
conflicts. The new profile repairs those conflicts and has separate validation
status; a GPU rollout/update/checkpoint smoke test is still required before
claiming that the new environment reproduces training.

Pass a local immutable model snapshot through `--model`. The Qwen experiments
used `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`. The additional finance experiment
used `meta-llama/Llama-3.1-8B-Instruct` revision
`0e9e39f249a16976918f6564b8830bc894c89659` with the same finance recipe.

The prepared training data are separate from the frozen standard-evaluation
data. Each split has `.jsonl`, `.parquet`, and `_verl.parquet` versions;
each row embeds a normalized datapoint in `teamgym_datapoint_json`.

The [release inventory](../../README.md#data-flow-and-release-inventory) lists the
exact split names, counts, required formats, metadata, and current download status.
The data path is construction → domain-specific RL preparation → frozen prepared
splits → training and checkpoint evaluation. Reusing prepared splits preserves
the actual training examples while reconstruction can author new ones.
The [reconstruction guide](../../scripts/data_generation/RECONSTRUCTION.md) includes
an example with embedded source text that builds normalized tasks and prepares
generic training splits without using the hosted TeamGym dataset.

The [dataset interface guide](../../teamgym/core/README.md)
describes that embedded record, actor-visible information, and scoring conventions.
An incoming domain also needs explicit split/identity mapping and a training
reward and recipe; its construction registration alone supplies command routing.

## Prepare the paper datasets

To reuse the paper data, extract the Legal and Finance RL bundles listed in the
[release inventory](../../README.md#data-flow-and-release-inventory). Set
`LEGAL` and `FINANCE` to the corresponding extracted `prepared/` directories,
then run the [offline checks](#verify-prepared-files-offline). To rebuild the
splits, use the RL environment's CPU dependencies and set `RL_BUILD` to a
fresh output directory.

```bash
RL_BUILD=generated/paper-rl-rebuild
```

### Legal

The exact replay uses the frozen CUAD/MAUD splits after filtering redundant
allocations and pruning participants with no owned evidence. Keep `train.jsonl`
and `validation.jsonl` under both `full_context/` and `riu_only/` from that
stage, plus the accepted written-answer `generation_records/` directory.
Copy the records into the new output so the replay can reuse the author and
critic results without model calls.

```bash
LEGAL_FILTERED=/path/to/frozen/legal-filtered-splits
LEGAL_RECORDS=/path/to/frozen/generation_records
LEGAL_WRITTEN="$RL_BUILD/legal/written"

mkdir -p "$LEGAL_WRITTEN"
cp -R "$LEGAL_RECORDS" "$LEGAL_WRITTEN/generation_records"
python -m teamgym.rl.preparation.prepare_written_answer_legal \
  --source-root "$LEGAL_FILTERED" --output-root "$LEGAL_WRITTEN" \
  --split train --split validation --cached-only
LEGAL="$LEGAL_WRITTEN/full_context"
```

`--cached-only` checks that every selected task has a matching accepted
generation record and reports missing or invalid records in
`generation_failures.json`. The output includes both context variants, with
separate CUAD and MAUD validation files in each. The paper uses
`full_context/train` (334 CUAD) for training, `validation_cuad` (36) for
checkpoint selection, and `validation_maud` (26) for held-out evaluation.

To rebuild the filtered input, start with the extracted
`legal_allocated_v1_all.zip` from the release inventory. It contains the
`data/{cuad,maud}/index.jsonl` and `instances.jsonl` files needed to reproduce
the source-group assignments. The archive's standard-evaluation source groups
are reserved during the split.

```bash
LEGAL_ARCHIVE=/path/to/extracted/legal-allocation-archive
LEGAL_FILTERED="$RL_BUILD/legal/no_redundant"

python -m teamgym.rl.preparation.prepare_legal_mixed \
  --archive-root "$LEGAL_ARCHIVE" --output-dir "$RL_BUILD/legal/mixed" \
  --seed 17 --validation-fraction 0.1
python -m teamgym.rl.preparation.prepare_no_redundant_legal \
  --source-root "$RL_BUILD/legal/mixed" --output-root "$LEGAL_FILTERED" \
  --train-dataset cuad --evaluation-dataset cuad --evaluation-dataset maud
```

Use this `LEGAL_FILTERED` in the written-answer command above. With the
recorded generation cache, replay from the allocation archive reproduced
the paper rows with only a different `source_path` provenance field. Keep
the frozen filtered JSONL files to reproduce the release hashes. To author
new questions and rubrics, run the written-answer command without
`--cached-only` in a fresh output directory and configure its model provider
and credentials with the options shown in `--help`.

### Finance

The exact replay uses the frozen deduplicated `finance_riu_only.jsonl` from
`normalized/` in the handoff preparation run.

```bash
FINANCE_NORMALIZED=/path/to/frozen/finance/normalized/finance_riu_only.jsonl
FINANCE="$RL_BUILD/finance/transfer"

python -m teamgym.rl.preparation.prepare_finance_participant_transfer \
  --input "$FINANCE_NORMALIZED" --output-dir "$FINANCE" --seed 17 \
  --findver-validation-fraction 0.1 --finlongdocqa-validation-fraction 0.1
```

`train` combines 350 FinDVer tasks and 188 FinLongDocQA tasks with two or
three participants. `validation` combines 39 FinDVer tasks for checkpoint
selection and 21 FinLongDocQA tasks for monitoring. `test` holds out 40
four-participant FinLongDocQA tasks. `test_finlongdocqa_6p` is a separate
one-task qualitative split. Retain `split_manifest.jsonl` and
`transfer_audit.json` alongside the prepared files.

To build the PIU-only input from upstream tasks, supply the final deduplicated
combined JSONL of normalized, allocated FinDVer and FinLongDocQA tasks with
participant roles.

```bash
FINANCE_TASKS=/path/to/finance-final-dedup/combined.jsonl

python -m teamgym.rl.preparation.prepare_finance_participant_handoff \
  --input "$FINANCE_TASKS" --output-dir "$RL_BUILD/finance/handoff" \
  --seed 17 --validation-fraction 0.1 --test-fraction 0.1
FINANCE_NORMALIZED="$RL_BUILD/finance/handoff/normalized/finance_riu_only.jsonl"
```

Pass that file to the transfer command above. The handoff command also writes
generic splits under `riu_only/`. Use the transfer output for the paper
assignment. Current PIU preparation uses raw source text. Rebuilding from the
recorded combined input preserved the paper split assignments and changed the
stored private-context text. Keep the frozen normalized JSONL to recover the
paper examples exactly.

### Verify prepared files offline

```bash
python -m teamgym.rl.paper \
  --regime legal-joint --prepared-data-dir "$LEGAL" \
  --model unused --run-dir unused --check-data
python -m teamgym.rl.paper \
  --regime finance-handoff --prepared-data-dir "$FINANCE" \
  --model unused --run-dir unused --check-data
```

These commands only read the prepared files. The model and run-directory
arguments are placeholders. The checker verifies JSONL and Parquet consistency,
embedded datapoint hashes, task/source split separation, and dataset and
participant boundaries. Compare `matches_paper_split_counts` and the SHA-256
of each canonical split file against the expected counts and the release
`manifest.json`. Use the pinned pandas and PyArrow versions for byte-level
comparisons. The prepared-data ZIPs contain the final files and reports. Keep
the frozen rebuild inputs described above in a separate archive.

## Train a constructed dataset

After [constructing and preparing data](../../scripts/data_generation/RECONSTRUCTION.md),
select an existing rollout profile and the reward appropriate to its answers:

```bash
custom_args=(
  --prepared-data-dir generated/rebuild_example/prepared
  --train-split train --validation-split validation --test-split test
  --profile short_context_smoke
  --reward-config written_holistic_score
  --training-mode joint
)

teamgym-train "${custom_args[@]}" --check-data
teamgym-train "${custom_args[@]}" \
  --model /absolute/path/to/local-model-snapshot \
  --run-dir outputs/teamgym/custom-preview --dry-run
```

Source-checkout users can invoke `python -m teamgym.rl.launch` with the
same arguments. `--dry-run` uses only the standard library: it prints the command,
profile, and explicit settings without reading data/model files, creating output,
contacting a judge, or checking for GPUs. `--check-data` uses the RL preparation
environment's CPU dependencies. It checks all three prepared formats, embedded
checksums and schemas, identity metadata, selected-split separation, and reward
compatibility. It needs no model or run directory and makes no model calls.
Custom split filenames are supported, including files that retain older generic
split labels inside their records. `--test-split` adds integrity/overlap checks;
training and checkpoint selection use the chosen training and validation splits.

| Training mode | Trainable roles | Fixed role configuration |
| --- | --- | --- |
| `joint` | Coordinator and participants share policy weights | None; choose a profile with a coordinator. |
| `participant_only` | Participants | Profiles with a coordinator require `--fixed-coordinator-base-url` and `--fixed-coordinator-model`. Coordinator-free handoff profiles need neither. |
| `coordinator_only` | Coordinator | Supply `--fixed-participant-base-url` and `--fixed-participant-model`, or explicitly choose `--participant-backend scripted` for source-excerpt participants. |

Fixed-role servers are supplied separately. Their existing `FIXED_COORDINATOR_API_KEY`
and `FIXED_PARTICIPANT_API_KEY` environment variables provide authentication;
the dry-run plan does not print those keys. The generic `joint` mode shares weights;
the paper's `legal-joint` recipe retains separate policies.

Finite-choice data use `task_success` or one of the supported combined
`task_success_plus_*` rewards. Written-answer data require `written_holistic_score`
and reference answers; the data check rejects rewarding the `free_response` label
as task correctness. Set the existing `WRITTEN_JUDGE_*` environment variables for
the hosted judge before actual written-answer training. Retrieval-probe profiles
are excluded from this training command.

The generic defaults are small integration settings: LoRA rank/alpha 16, batch
size 1, four rollouts, at most two optimizer updates, and two epochs. Epochs can
stop training before the update limit. Use `--updates`, `--epochs`, `--batch-size`,
`--rollouts`, `--learning-rate`, and `--finetuning` to configure a new run.
`--devices` accepts one or more distinct CUDA device IDs; their count must be
divisible by `--tensor-parallel-size`. Prompt length defaults to the profile's
private-context allowance plus 4,096 tokens, with a minimum of 4,096; response
length covers the profile's actor completion caps. Both lengths have explicit
overrides. These settings need a model and GPU allocation that can support them.

Omit `--dry-run` to train in the GPU environment after checking the data and
configuring any fixed-role servers and judge. A fresh run directory is required.
The command records `launch_plan.json`, then the trainer writes its usual run
manifest, logs, rollouts, and checkpoints. GPU rollout/update/checkpoint validation
remains a separate check; the CPU checks do not establish training performance.

## Preview, check, train the paper recipes

After installation, set local paths in your working directory:

```bash
MODEL=/absolute/path/to/qwen-snapshot
LEGAL=/absolute/path/to/prepared/legal/full_context
FINANCE=/absolute/path/to/prepared/finance-final-dedup

python -m teamgym.rl.paper \
  --regime legal-joint --model "$MODEL" \
  --prepared-data-dir "$LEGAL" --run-dir outputs/teamgym/legal-joint-seed29 \
  --seed 29 --dry-run
```

`teamgym-paper` is also available as an installed command with the same
arguments. Output paths are resolved against the calling directory. The
canonical paper launcher works outside the checkout.

Shell helpers use the active `python`, or the executable selected through
`TEAMGYM_PYTHON`. Their local output defaults are under `outputs/teamgym`;
the paper entry point supplies explicit run, checkpoint, and judge-cache paths.
`PREPARED_DATA_DIR` and the paper entry point's `--prepared-data-dir` select
prepared training inputs directly.

`--dry-run` prints the command and explicit settings using only the standard
library, with no data reads, file writes, model loading, or network calls.
Replace it with `--check-data` to validate local prepared files offline.
Omit either flag to train; choose a fresh run directory for every invocation.

For finance, use `--regime finance-handoff --prepared-data-dir "$FINANCE"`.
For Stage B, use `--regime legal-participant`. To continue with Stage C, merge
the CUAD-selected participant checkpoint and pass it as the fixed actor:

```bash
TEAMGYM_PYTHON="$(command -v python)" \
SOURCE_CHECKPOINT=/absolute/path/to/stage-b/checkpoints/global_step_60 \
TARGET_DIR=/absolute/path/to/merged-participant \
bash teamgym/rl/merge_full_checkpoint_hf.sh

python -m teamgym.rl.paper \
  --regime legal-coordinator --model "$MODEL" \
  --fixed-participant-model /absolute/path/to/merged-participant \
  --prepared-data-dir "$LEGAL" --run-dir outputs/teamgym/legal-coordinator
```

Select the actual best Stage B update from `checkpoint_selection.json`; step 60
above illustrates the original selected participant, not a universal default.
The immutable actor runs in a separate vLLM process on the last selected GPU.
`--devices` defaults to `0,1,2,3`.

Both families use full finetuning, GRPO, learning rate `1e-6`, weight decay
`0.01`, PPO clipping `0.2`, and dual clipping `3`. Training samples at temperature
`0.7`, top-p `0.95`; validation is greedy. Legal uses 8 tasks × 8 rollouts per
update; finance uses 8 × 16. Defaults are 80 optimizer updates with evaluation
and checkpointing every 10. The underlying trainer's raw step limit is 81.
The public recipe explicitly enables legal KL loss `0.001` as described in the
paper; finance disables it. Use the paper entry point to apply the complete
settings listed here.

The only task reward is the hosted GPT-5.5 holistic score. Invalid trainable
JSON/schema attempts receive a local `-0.1` advantage with one retry. Set
`WRITTEN_JUDGE_*` environment variables for the provider, endpoint, credentials,
and model; their interface is in `train_backend.sh`. The retained default
provider uses Bedrock's OpenAI-compatible Responses endpoint and
`AWS_BEARER_TOKEN_BEDROCK`. Actual training/evaluation can contact that judge;
dry runs, data checks, and unit tests do not. Judge failures abort the public
recipe. Logging defaults to the console.

The final finance generation cap is **750 tokens per actor call**, including
Llama. The Llama completion parser preserves decoded JSON or malformed action
text exactly; it does not repair actions or implement tool calling.

## Evaluate and inspect artifacts

Read the chosen update from `RUN_DIR/checkpoints/checkpoint_selection.json`.
Legal selects on CUAD only. Finance selects on the 39 FinDVer validation tasks;
the 21 FinLong validation tasks are monitoring data. Ties choose the earlier
update. Neither held-out test enters selection.

```bash
python -m teamgym.rl.paper \
  --regime legal-joint --model "$MODEL" \
  --prepared-data-dir "$LEGAL" --run-dir outputs/teamgym/legal-joint-maud-eval \
  --evaluate --eval-checkpoint /absolute/path/to/checkpoints/global_step_80
```

Evaluation defaults to `validation_maud` for legal and `test` for finance.
Use `--eval-split validation_cuad` or `--eval-split validation` for the respective
validation split. Omit `--eval-checkpoint` to evaluate the base policy under
the same budgets. Stage C evaluation also needs `--fixed-participant-model`.
`--dry-run` and `--check-data` work with these evaluation arguments too.

Native four-rank FSDP checkpoints are loaded with model contents only for
evaluation. Legal model-only checkpoints and the selected model backup bundles
do not contain optimizer, RNG, or dataloader state for an exact training resume.
The entry point starts fresh training.

Runs record `run_manifest.json`, `train.log`, `rollouts/`, `episode_logs/`,
`written_judge_cache/`, and `checkpoints/`. Endpoint reports are under
`episode_logs/validation_results/`. `merge_full_checkpoint_hf.sh` and
`merge_joint_checkpoint_hf.sh` export native checkpoints for separately hosted
actors. Keep both actor directories for legal joint checkpoints.

Both checkpoint exporters resolve helper scripts relative to their own
location and can be invoked by absolute path from another working directory.
Set `TARGET_DIR` for a single-policy export, or `TEAMGYM_MERGED_ROOT` for joint
exports. The shared default export directory is `outputs/teamgym/merged_models`
relative to the calling directory.

## Package layout

* `launch.py`, `paper.py`, and `train.py` provide the generic launcher, paper
  recipes, and training process entry point.
* `preparation/` groups source-to-split commands, prepared-split filtering,
  and the offline data checks. `data.py` provides the shared split writer and
  training dataset loader.
* `rollout/` contains the workflow, budgets, rewards, actor clients, policy
  routing, action parsing, and format repair. The rollout client connects
  asynchronous generation to recorded training steps.
  The [written-answer scorer](../scoring/written_answers.py) caches assessments.
  Grading prompts, rubric normalization, and numeric checks are shared through
  [`scoring/short_answers.py`](../scoring/short_answers.py).
* `training/` contains RLLM and VERL integration, local format advantages,
  policy separation, checkpoint selection, and run provenance.
* `tools/` contains standalone Finance evaluation, answer calibration,
  saved-rollout validation, rescoring, and reward plotting.
* `train_backend.sh`: shared RLLM/VERL launch settings used by `teamgym-train`
  and the finance paper recipe. Its direct defaults are small development settings.
* `train_legal_roles.sh` → `train_legal.sh` → `train_backend.sh`: legal role
  setup, legal training defaults, and the shared backend. The paper command supplies
  the paper's complete settings through this chain.
* `merge_full_checkpoint_hf.sh`, `merge_joint_checkpoint_hf.sh`: checkpoint
  exporters for fixed actors and separately hosted models.
* `require_four_gpu_allocation.sh`: an optional Slurm allocation guard.

The current checkout incorporates the final finance caps and the production
Llama parser patch. Training and matched checkpoint evaluation use `paper`.
The package includes the six shell utilities listed above and the development
fixture. The public workflow covers data preparation, training, and matched
endpoint evaluation. Research analyses, counterfactual replay experiments,
campaign job records, and one-off cluster launchers are kept separately.

## Offline checks and development fixture

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest teamgym/rl/tests -q
python -m teamgym.rl.preparation.prepare_data \
  --input teamgym/rl/fixtures/tiny_normalized.jsonl \
  --output-dir generated/teamgym_smoke_data \
  --seed 17 --validation-fraction 0.25 --test-fraction 0.25
```

The tiny fixture and low-level `train_backend.sh` defaults are for plumbing
checks. They do not reproduce the paper's datasets, role policies, or budgets.
