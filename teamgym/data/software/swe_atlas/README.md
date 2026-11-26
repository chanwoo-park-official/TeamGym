# SWE Atlas Codebase QnA construction

This constructor keeps the upstream question and hidden grading rubrics. It
collects repository evidence, verifies its contribution to the answer, and
distributes code files and execution logs into collaborative private packets.

## Files

| File | Responsibility |
| --- | --- |
| `source.py`, `schema.py` | Load upstream task files and separate public task fields from hidden evaluation material. |
| `candidate.py` | Plan source inspections and runtime observations, execute them in Docker, and validate exact evidence quotations. |
| `pipeline.py` | Consolidate candidates, run ALL/NONE/LOO audits, allocate contexts, repair leakage, and write benchmark variants. |
| `export_common.py` | Export accepted builds into the shared TeamGym schema. |
| `riu_context.py` | Produce the evidence-only context variant. |
| `../../../prompts/software/swe_atlas/` | Construction and audit prompts. |

The constructor uses the repository's shared LLM client, schema, and tokenizer.
It does not import Finance, Medical, Legal, or DeepSWE constructors.

## Inputs and execution

Prepare an upstream SWE Atlas checkout and the Docker image declared by the
selected task's `task.toml`. The input directory contains `task.toml`,
`instruction.md`, `solution/answer.txt`, and the files under `tests/` required by
`source.py`. Both `qa-*` and historical `task-*` task directories are supported.
Source acquisition is currently a separate step; see
`scripts/data_generation/README.md` in the repository root.

```bash
uv run --no-sync teamgym-generate swe/swe-atlas \
  --task-dir SOURCE_CHECKOUT/data/qa/TASK_ID \
  --output-root artifacts/swe-atlas/TASK_ID

uv run --no-sync python -m teamgym.data.software.swe_atlas.export_common \
  --build-dir artifacts/swe-atlas/TASK_ID \
  --task-root SOURCE_CHECKOUT/data/qa \
  --output-root artifacts/swe-atlas-export
```

The output root is the run directory for the selected task. The
`--output-dir` spelling also works. Use a fresh output directory when
changing the task revision, model, or
construction settings. Checkpoint reuse requires a matching task/source/image/
model/config/code/prompt identity and hashes of the completed artifacts. Legacy,
missing, changed, or partial receipts are rejected. Export additionally requires
a matching ready receipt over the final selected artifacts and partition.
The standalone `--reuse-plan` option likewise requires a matching plan receipt.
Inputs are checked again before either successful construction return.

## Stages and current policies

1. Read the upstream task. Repository search planning sees only the public
   question and repository inventory. It receives neither hidden rubrics nor the
   reference answer, including during repair. A separate constructor maps hidden
   rubrics to frozen collected artifacts without further repository search or
   execution. The audit judge sees the reference answer.
2. Review public-question evidence coverage after each collection. The searcher
   can use the collected public source/runtime to repair missing observations,
   sharing the existing bounded collection-revision budget with execution repair.
   It never receives private criteria or judge feedback. Only after collection
   freezes does the hidden constructor receive audit feedback. Malformed public
   coverage-review responses receive bounded format retries with the same public
   inputs and validation errors; model-proposed shell commands are not executed.
   Classify hidden
   rubric evidence as source-only or hybrid using text-pattern rules.
   Hybrid evidence additionally requires observed execution output. At most one
   integrated runtime request is allowed for candidate evidence collection.
3. Collect source and runtime artifacts in the declared Docker image and check
   that candidate quotations occur exactly in verified artifacts. Runtime status
   uses Docker's actual return code. Command output is encoded separately from
   collector metadata and retained as decoded `observed_content`, so printed
   control-like lines cannot override status or checkout metadata.
4. Consolidate candidates into PIUs. The hidden constructor can reselect exact
   contiguous source/output spans from already verified frozen artifacts to
   repair missing support, without further search or execution. Every changed
   eligible grouping reruns ALL/NONE/LOO; no PIU count is prescribed. Regrouping
   receives the failed ALL rubrics and their reasons as private feedback.
5. Audit ALL, NONE, and each leave-one-out condition. Accepted collaborative
   material must contain at least two PIUs. A satisfied positive rubric must cite
   at least one valid displayed evidence item; negative or unmet criteria may
   have empty citations. Citation repair receives the actual evidence and answer.
   Failed selections can be retried.
6. Run additional control/distractor experiments, then allocate whole source
   files and logs. The defaults are one participant per PIU, a 40,000-token cap
   per participant, and at least two verified distractor logs. These defaults
   have CLI options. The full-context variant contains the allocated packet,
   which can omit files because of file filters, capacity, or leakage repair.
7. Audit complete packets for reconstruction of foreign PIUs. Remove leaking
   noncanonical contexts and re-audit. Shared canonical files/logs or semantic
   overlap trigger evidence regrouping, with merging as a fallback. Regrouped
   evidence must pass ALL/NONE/LOO again; fewer than two PIUs is a rejection.
8. Audit visible formatting for construction cues and write three benchmark
   variants under `09_benchmark/`. `export_common.py` separately produces the
   shared-schema full-context and evidence-only JSONL exports.

The runtime command validator rejects network/install commands, and evidence
collector Docker invocations use `--network none`. Execution pins the resolved
image content ID. Runtime collection uses `linux/amd64` images and requires a
working compatible Docker runtime. Candidate request IDs are local to a plan;
final artifact identity binds content and provenance, and failed selection
history cannot shadow the authoritative selected bundle.
Internal `riu`/`rui` identifiers retain compatibility with existing artifacts.

## Checks

The source tests default to `.external_datasets/SWE-Atlas`. Set
`SWE_ATLAS_DATASET_ROOT` to check another upstream checkout. They validate task
identity, rubric structure, and the public/hidden field boundary without fixing
an upstream inventory count or Docker image tag.

```bash
SWE_ATLAS_DATASET_ROOT=SOURCE_CHECKOUT uv run --no-sync python -m pytest -q \
  tests/software/swe_atlas/test_source.py \
  tests/software/swe_atlas/test_candidate.py \
  tests/software/swe_atlas/test_pipeline.py \
  tests/software/swe_atlas/test_reliability.py
```

Recovery tests replace Docker and model calls while exercising real allocation,
regrouping, checkpoint persistence, materialization, and shared-schema export.
Passing them does not establish acceptance of a newly generated real task.

## Actual reader capacity and post-PIU release

The default partition builder fits the actual shared model reader before packet
audits, including separately encoded segment boundaries and join separators.
It drops only the smallest sufficient neutral whole units, preserves every
canonical context, and rejects canonical overflow. The fixed source-read budget
is unchanged. This avoids distributed truncation that can otherwise occur even
when encoding the already-joined text appears to fit.

An existing audited PIU set can undergo an explicit post-PIU packet release:

```bash
uv run --no-sync python -m teamgym.data.software.swe_atlas.post_piu_release \
  --parent-build OLD_IMMUTABLE_BUILD \
  --producer-snapshot ORIGINAL_MATCHING_SOURCE_SNAPSHOT \
  --task-dir SOURCE_CHECKOUT/data/qa/TASK_ID \
  --output-dir NEW_EMPTY_RELEASE \
  --env-file APPROVED_ENV_FILE --model gpt-5.5 --reasoning-effort medium
```

This requires the original ready receipt and matching producer source snapshot.
It keeps question, canonical PIUs and ALL/NONE/LOO artifacts byte-identical, then
runs new mechanical, complete-packet leakage and salience checks after neutral
capacity repair. Export the new release directory only after it is ready. The
release has a separate code/model/environment identity; it does not claim that
new source code generated the upstream frozen PIUs. Legacy outputs without
provenance must be rebuilt rather than certified by file presence.
