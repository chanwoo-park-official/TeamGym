# Rebuild a dataset without the hosted TeamGym release

The construction code can create new TeamGym records from source documents,
questions, reference evidence, and answers. Most corpora export common-datapoint
JSONL. DeepSWE exports a repository-repair instance for its verifier.
Model-generated questions, wording, and retention counts can differ between
runs. Reproducing the exact paper rows requires the frozen records or their
complete construction artifacts.

For common datapoints, the reusable path is:

```text
source documents → task seeds and evidence → participant allocation
  → structural checks → model-based qualification → normalized TeamGym JSONL
      → standard evaluation
      → RL task preparation and splits → policy training
```

## Reconstruct production datasets

Start from the repository root with the
[construction environment](../../requirements/README.md#dataset-use-construction-and-api-evaluation).
Run `teamgym-generate --list` to find dataset commands and
`teamgym-generate DOMAIN/DATASET --help` to see their options. The commands
below use caller-supplied source paths, run directories, and model credentials.

### Stage map and evaluation inputs

File paths are relative to the run or export directory of the named stage.
For common-datapoint evaluation, pass the specific accepted file as
`--input-root` to the [standard evaluation runner](../../evaluation/README.md).

| Dataset | Construction stages | Check before evaluation | Evaluation input |
| --- | --- | --- | --- |
| CUAD | Generate a finite-choice task from measured contract spans, then `partition` the full contract | Generation funnel report and `partition_report.json` | `common_datapoints.jsonl` in the partition output |
| MAUD | Match PIUs to the agreement, generate a finite-choice task, then `partition` in full or budgeted context mode | Generation funnel report and `partition_report.json` | `common_datapoints.jsonl` in the partition output |
| Casebook | Build authored seeds, allocate evidence across participants, check structure, remove leakage, then qualify with an all-PIU solve and written judge | `seed_build_report.json`, then `all_riu_qualified/summary.json` | `all_riu_qualified/common_datapoints.jsonl` |
| MSLR | Extract grounded case excerpts into seeds, then run the shared allocation, leakage, and written qualification stages | `seed/seed_report.json`, then `all_riu_qualified/summary.json` | `all_riu_qualified/common_datapoints.jsonl` |
| SARA | `prepare`, `generate`, `refresh`, `merge`, `qualify`, `finalize`, then allocate the qualified seeds | `metadata/selection_report.json`, `final/final_report.json`, and the shared structural checks for the allocated records | `03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl` from the shared construction run |
| Multi-LexSum | Optional `prepare-sources`, then `prepare-packets`, `generate`, `short-answer`, `expand-packets`, `audit-grounding`, `select-grounded`, leakage checks, all-PIU solving, and written-answer filtering | `grounding_audits.jsonl` and the final `filter_report.json` | `common_datapoints.jsonl` from the final filter output |
| FinDVer | Group QA by filing, map report evidence, construct the task and rubric, qualify the evidence and participant packets, and optionally redistribute the accepted task | `manifest.json`, `funnel_report.json`, and `candidate_results.jsonl` | `common_datapoints.jsonl`, or `MODE/common_datapoints.jsonl` after redistribution |
| FinLongDocQA | `--stage index` parses report blocks. `--stage generate` constructs and qualifies tasks using that index. `--stage all` runs both. The completed run adds the evaluator rubric | Producer `run_manifest.json` and `generation_report.json` | `evaluation/common_datapoints.jsonl` |
| EvidenceBench | Load annotated paper sentences, generate a task, qualify PIU boundaries with all, no, and leave-one-out evidence, audit participant packets, then write accepted tasks. Optional allocation modes audit each distribution | `run_manifest.json` and `summary.json`, or `allocations/allocation_results.json` | `common_datapoints.jsonl` or `allocations/common_datapoints.jsonl` |
| MLCR | Load clinical summaries and source questions, map exact evidence, generate a task, check evidence boundaries and leakage, test participant packets for solo shortcuts, then export | `run_manifest.json` and `run_report.json`, or `allocations/allocation_results.json` | `common_datapoints.jsonl`, or `allocations/common_datapoints.jsonl` for accepted distribution variants |
| SWE Atlas | Load the upstream QA task, collect verified source and runtime evidence, form and qualify PIUs, run controls, allocate and audit participant packets, then write both evidence views | `pipeline_report.json` and the ready receipt checked by the exporter | `evaluation/full/accepted.jsonl` or `evaluation/riu_only/accepted.jsonl` |
| DeepSWE | Resolve the task and images, prepare public checks, screen instruction-only repair, collect repository evidence, qualify repairs with all PIUs, audit participant packets, run leave-one-out and participant diagnostics, then freeze the export | `TASK_ID/report.json` with an accepted decision and the bound validation receipt | `TASK_ID/export/instance.json` for the DeepSWE verifier |

### Accepted task format

The final standard-evaluation JSONL files in the table use
`teamgym_release_datapoint_v1`, the same per-task schema as the
[TeamGym dataset on Hugging Face](https://huggingface.co/datasets/TeamGym/teamgym-datasets).
Rows include participant IDs and roles, private context units, pivotal
information units (PIUs), the task and oracle, evidence condition, allocation
metadata, and construction provenance. Pass the listed file directly to the
[standard evaluator](../../evaluation/README.md). The evaluator derives PIU-only
and single-agent observations from a full-context task when requested. Use the
separately qualified PIU-only file when the recipe produces one, as SWE Atlas does.

### Legal recipes and source pins

The [Legal source manifest](provenance/legal_source_manifest.json) records the pinned CUAD
and MAUD revisions, the checksummed Casebook text, the MSLR commit, the pinned
Multi-LexSum release, and the SARA source and SWI-Prolog image. Use the same
cache for contract generation and partitioning. The documented historical
settings and recovery gaps are in the [Legal construction guide](README.md).

For CUAD and MAUD, these commands use an acceptance minimum of two, which
determines the accepted JSONL filenames:

```bash
run=generated/reconstruction
cache=data/legal_raw

teamgym-generate legal/cuad --cache-dir "$cache" --min-accept-size 2 \
  --output-root "$run/cuad/generated"
teamgym-generate legal/cuad partition --cache-dir "$cache" --seed 11 \
  --accepted-path "$run/cuad/generated/accepted_measured_span_minaccept2.jsonl" \
  --output-root "$run/cuad/partitioned"

teamgym-generate legal/maud --cache-dir "$cache" --min-accept-size 2 \
  --output-root "$run/maud/generated"
teamgym-generate legal/maud partition --cache-dir "$cache" --seed 1201 \
  --accepted-path "$run/maud/generated/accepted_maud_measured_rui_minaccept2.jsonl" \
  --output-root "$run/maud/partitioned"
```

For the written-answer corpora, use these recipes:

- **Casebook and MSLR.** `teamgym-generate legal/casebook --base-source FILE`
  reads the prepared 12 CFR text. `teamgym-generate legal/mslr --input-root DIRECTORY`
  reads the pinned checkout's `data/processed_anonymized` directory. The
  [written-answer Slurm recipe](slurm/legal_written_construction/README.md#submission)
  builds these seeds, allocates them, and qualifies the resulting tasks. The
  batch also expects a constructed Multi-LexSum JSONL. Its default qualified
  output contains rows from the included corpora.
- **SARA.** [Prepare the audited source](slurm/sara_target_sliced/README.md#prepare-the-audited-source)
  from the ordered 100-case split and pinned SWI-Prolog image. `prepare`
  executes the reference queries and enumerates grounded proof targets before
  model generation. Run the remaining stages through the
  [target-sliced launcher](slurm/sara_target_sliced/README.md#production-construction).
  Pass its `final/qualified_seed.jsonl` as `--seed-spec` to
  `python -m teamgym.data.long_context_reasoning.pipeline`, as in the
  [structural example](#a-structural-example-with-embedded-sources).
- **Multi-LexSum.** Use the pinned `releases/v20230518` split and source JSON
  files. Follow the [packet and generation commands](README.md#original-document-packets)
  through `select-grounded`, then the
  [expanded-context qualification commands](README.md#qualify-expanded-contexts).
  The PIU-only repartitions produced by the shared Slurm batch cover a separate
  evaluation condition.

### Finance, Medical, and Software recipes and source pins

The [non-Legal source preparation guide](NONLEGAL_CONSTRUCTION.md#source-preparation)
locates the inputs. It records the checked MLCR revision
`1198206e2df96febf9fd13c110981e5b40515285`. For the other upstream inputs,
record the exact source revision and file hashes used for your run. For software
tasks, also record the selected task ID, base commit, and resolved Docker
image IDs. Use the frozen evaluation inputs for exact paper reruns.

The [frozen example receipts](provenance/nonlegal_reconstruction_examples.json) record
verified FinDVer, FinLongDocQA, and EvidenceBench rows and their archive
checksums. The archived SWE Atlas exports load, but their sufficient sets
refer to PIU IDs missing from the exported rows. Use a freshly validated
export for current-schema checks. MLCR and DeepSWE still need accepted producer
artifacts for this record.

- **FinDVer.** Run `teamgym-generate finance/findver --qa-path FILE --reports-root DIRECTORY --output-root RUN`
  with the source split and model settings in the
  [FinDVer recipe](../../teamgym/data/finance/findver/README.md).
  For additional distributions, pass the accepted `scenarios.jsonl` to the
  [redistribution command](../../teamgym/data/finance/findver/README.md#redistribute-one-frozen-task).
- **FinLongDocQA.** Run
  `teamgym-generate finance/finlongdocqa --stage all --qa-path FILE --reports-zip ZIP --output-root RUN`.
  For separate stages, reuse the same output root or give `--block-index-jsonl`
  to `--stage generate`. The run writes accepted tasks with the standard grader
  rubric to `RUN/evaluation/common_datapoints.jsonl`. The
  [producer-bundle recipe](../../teamgym/data/finance/finlongdocqa/README.md#export-a-completed-producer-run-for-evaluation)
  can verify the source, code, and model inputs of a completed single-report
  run and save a checksum receipt.
- **EvidenceBench.** Run
  `teamgym-generate medical/evidencebench --source-path DIRECTORY --source-revision REV --instance-id ID --output-root RUN`.
  An accepted default run writes `RUN/common_datapoints.jsonl`.
  The [EvidenceBench recipe](../../teamgym/data/medical/evidencebench/README.md)
  gives the model and audit options.
- **MLCR.** Run
  `teamgym-generate medical/mlcr --source-dir DIRECTORY --source-revision REV --case-id ID --output-root RUN`.
  The [MLCR recipe](../../teamgym/data/medical/mlcr/README.md) shows source-question
  selection and model settings. Both Medical builders support
  `--construction-only` to save a qualified task and PIUs before distribution.
  Use `--allocation-modes all` or the separate
  [Medical allocation command](../../teamgym/data/medical/README.md#distribution-specific-audits-and-exports)
  to certify and export additional distributions.
- **SWE Atlas.** Run
  `teamgym-generate swe/swe-atlas --task-dir TASK --output-root RUN` with the
  selected task's Docker image available. `RUN` is the task's own build
  directory. A benchmark-ready run writes both evidence views under
  `RUN/evaluation/`. The
  [SWE Atlas recipe](../../teamgym/data/software/swe_atlas/README.md#inputs-and-execution)
  describes the task files, readiness checks, and how to gather several runs.
- **DeepSWE.** Run
  `teamgym-generate swe/deep-swe --tasks-root TASKS --task-id ID --verifier-image IMAGE --output-root RUN`.
  The [DeepSWE recipe](../../teamgym/data/software/deep_swe/README.md)
  describes image preparation, screening, and the acceptance gates.
  Use its `export.load_export` and `render_model_input` to reopen the accepted
  instance and recover the input for each evaluation condition.

## A structural example with embedded sources

Run from the repository root with Python 3.10+ and the base package installed
(`python -m pip install .`). This example embeds four small purchase scenarios
and requires no dataset download, model API, or GPU. It demonstrates the format
and construction stages; it is a development example, not a qualified benchmark.

Create a source/seed JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path

output = Path("generated/rebuild_example")
output.mkdir(parents=True, exist_ok=True)
rows = []
for index, (quantity, price, cap) in enumerate([(3, 10, 40), (6, 8, 40), (5, 5, 30), (4, 9, 30)]):
    total = quantity * price
    decision = "approve" if total <= cap else "reject"
    spans = [
        f"The request is for {quantity} units.",
        f"Each unit costs {price} credits.",
        f"The spending cap is {cap} credits.",
    ]
    steps = [
        {"step": "total", "claim": f"The total cost is {quantity} * {price} = {total} credits.",
         "uses": ["quantity", "price"]},
        {"step": "decision", "claim": f"Compare {total} with the cap of {cap} and {decision}.",
         "uses": ["quantity", "price", "cap"]},
    ]
    rows.append({
        "instance_id": f"rebuild_example_{index}",
        "source_id": f"rebuild_source_{index}",
        "source_family": "construction_example",
        "source_text": " ".join(spans),
        "question": "Should the purchase be approved under the spending cap? Show the total cost and comparison.",
        "answer_type": "short_answer_judged",
        "answer": "free_response",
        "decision_set": ["free_response"],
        "decision_option_definitions": {"free_response": "Write the decision and its reasoning."},
        "short_answer": decision,
        "canonical_response": f"The cost is {total} credits against a cap of {cap}; {decision} the purchase.",
        "rius": [{"rui_id": identity, "source_span": text,
                  "normalized_claim": text, "rui_type": "case_fact"}
                 for identity, text in zip(["quantity", "price", "cap"], spans)],
        "required_rui_ids": ["quantity", "price", "cap"],
        "reasoning_steps": steps,
        "required_decision_steps": steps,
        "judge_rubric": {"require_all_steps_for_pass": True},
    })
(output / "seeds.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
)
PY
```

Build participant packets and run the existing structural funnels:

```bash
python -m teamgym.data.long_context_reasoning.pipeline \
  --seed-spec generated/rebuild_example/seeds.jsonl \
  --output-root generated/rebuild_example/constructed \
  --allocation-mode one_each --min-participants 3 \
  --target-context-tokens 512 --seed 17
```

The output root contains the candidates, structural leakage results, and
`03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl`. This example
retains four rows, with three participants per row. The directory's historical
`option_exclusion` name denotes a structural evidence-ownership check here.

Each normalized row has `participants`, `private_context_units`,
`reference_information_units`, an evaluator-only `oracle`, and a
`format_validation` report. The [dataset interface](../../teamgym/core/README.md)
explains those fields and actor visibility. The shared builder currently labels
these records `long_context_reasoning`; the seed's source family identifies the
example. Production adapters preserve their domain's dataset labels and identity
metadata.

## Connect constructed records to RL preparation

Use the [RL environment](../../requirements/README.md) for this step. The current
preparation module imports the local RLLM dataset class and needs its CPU
dependencies, including PyTorch, pandas, polars, and parquet support. Preparation
itself needs no GPU or model call.

```bash
python -m teamgym.rl.preparation.prepare_data \
  --input generated/rebuild_example/constructed/03_option_exclusion/solo_oracle_kept_common_datapoints.jsonl \
  --output-dir generated/rebuild_example/prepared \
  --seed 17 --validation-fraction 0.25 --test-fraction 0.25 \
  --split-group canonical_task
```

This produces two training rows, one validation row, and one test row. Each split
has `.jsonl`, `.parquet`, and `_verl.parquet` files, alongside
`split_manifest.jsonl` and `leakage_audit.json`. Prepared rows embed the normalized
record in `teamgym_datapoint_json`.

The example uses distinct sources. For a real dataset, preserve allocation-free
task IDs and explicit source-group IDs so variants and questions from the same
document stay in the intended split. Use `--split-group source_group` with
explicit source-group IDs, for example `_cross.source_group_id`. Task-only
splits can be checked for canonical-task overlap. The checker reports source-group
disjointness as unknown when document IDs are missing. [Data preparation](../../teamgym/rl/data.py)

Preview a training plan for the newly prepared data without loading a model:

```bash
teamgym-train \
  --prepared-data-dir generated/rebuild_example/prepared \
  --profile short_context_smoke --reward-config written_holistic_score \
  --training-mode joint --model /path/to/local-model-snapshot \
  --run-dir outputs/teamgym/rebuild-example --dry-run
```

The generic command supports custom split names and role/profile/reward choices.
Its `--check-data` action validates the prepared files locally, without a model
or GPU. The paper launcher retains its specific legal and finance recipes;
follow the [paper preparation recipes](../../teamgym/rl/README.md#prepare-the-paper-datasets)
for their source inputs, exact replay commands, and split checks. Actual RL
training needs the GPU environment and, for written rewards, a configured hosted
judge. This example covers the handoff from construction through split preparation.

## What to keep if the hosted release is removed

Keep source snapshots or retrievable, versioned source locations; task/seed
builders and prompts; model/provider settings; allocation and split seeds; and
qualification/selection settings. Keep a separately downloadable copy of
`legal_allocated_v1_all.zip` for exact paper reruns if the hosted copy is removed.
Its checksum is listed in the source manifest and the current download is listed
in the release inventory. [Legal source manifest](provenance/legal_source_manifest.json) · [Release inventory](../../README.md#data-flow-and-release-inventory)

That is enough to describe and reconstruct the dataset structure and method once
the required inputs are supplied. Exact paper-row reproduction also needs the
frozen generated tasks, references, and allocations. Exact training-example
reproduction needs the frozen prepared splits or the inputs to the
[paper replay recipes](../../teamgym/rl/README.md#prepare-the-paper-datasets).
Keep a copy of those artifacts independently of the hosted release.
