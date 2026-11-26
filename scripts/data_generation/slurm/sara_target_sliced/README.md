# SARA target-sliced construction

This directory launches the paper's final v10 construction workflow. Reusable
logic lives in `teamgym.data.legal.sara`; the Python files
here are thin wrappers for the package stages. The supported generation policy
is `target_sliced`.

The reusable implementation separates [Prolog parsing and unification](../../../../teamgym/data/legal/sara/prolog.py),
[grounded proof slicing](../../../../teamgym/data/legal/sara/proof_slicing.py), and
[model-driven construction and candidate audits](../../../../teamgym/data/legal/sara/subproof_generation.py).
Existing imports from `subproof_generation` remain available. The extraction
preserves the v10 proof rules, prompts, and acceptance policy.

## Prepare the audited source

Export the 100-case `sara_numeric` whole split from DeonticBench to a JSON array.
The order is preserved because generation selects cases by index. Run from the
TeamGym checkout with the `data` extra installed and Apptainer available.
[Pinned DeonticBench split](https://huggingface.co/datasets/gydou/DeonticBench/blob/805fb0529d7ae295679c5c1730765ec07d2c20ac/sara_numeric/whole-00000-of-00001.parquet) · [Official SWI-Prolog container](https://hub.docker.com/_/swipl)

```bash
hf download gydou/DeonticBench sara_numeric/whole-00000-of-00001.parquet \
  --repo-type dataset --revision 805fb0529d7ae295679c5c1730765ec07d2c20ac \
  --local-dir generated/legal_sources/sara

python - <<'PY'
import json
from pathlib import Path

from pyarrow.parquet import read_table

root = Path("generated/legal_sources/sara")
cases = read_table(root / "sara_numeric/whole-00000-of-00001.parquet").to_pylist()
if len(cases) != 100:
    raise ValueError(f"Expected 100 SARA cases, found {len(cases)}")
(root / "whole.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

apptainer pull generated/legal_sources/sara/swipl_10.0.2.sif \
  docker://docker.io/library/swipl@sha256:2723fefd35116f92f68da22a487b1b01b584ac0b4b81fcd971c7ab2222f5d3e7

teamgym-generate legal/sara prepare \
  --source generated/legal_sources/sara/whole.json \
  --swipl-image generated/legal_sources/sara/swipl_10.0.2.sif \
  --output-root generated/sara_audit
```

Preparation executes the Prolog references and enumerates intermediate targets.
It writes `metadata/audited_cases.json` and `metadata/selection_report.json`.
The source manifest records the upstream Parquet checksum and the checksum of
the original local SIF. A SIF rebuilt with `apptainer pull` can have different
container metadata. [Legal source manifest](../../provenance/legal_source_manifest.json)

## Production construction

```bash
SARA_TARGET_SLICE_AUDIT_SOURCE=/path/to/sara_audit \
SARA_TARGET_SLICE_OUTPUT=/path/to/sara_run \
bash scripts/data_generation/slurm/sara_target_sliced/submit_target_sliced_full.sh
```

The submission uses the active Python environment, or `TEAMGYM_PYTHON`.
Activate required cluster tools before submission. Set
`SARA_TARGET_SLICE_PARTITION` to select a local Slurm partition; otherwise the
cluster's default is used. `TEAMGYM_ENV_FILE` can name an optional local setup
script for the jobs. Configured relative paths are resolved against the
calling directory. Without `SARA_TARGET_SLICE_OUTPUT`, output defaults to
`generated/sara_target_sliced` there.

The launcher constructs the cases with GPT-5.4 using the fixed target proof
envelope. It then refreshes deterministic acceptance fields, merges candidates,
runs the repeated blind solvability checks, and exports the qualified seeds.
The existing v10 target selection, prompts, repair limits, source checks, and
qualification threshold are retained. A task qualifies with at least four
passes among five blind trials.

## Individual stages

```bash
teamgym-generate legal/sara STAGE --help
```

| Stage | Package module | Purpose |
|---|---|---|
| `prepare` | `sara.preparation` | Execute references and enumerate targets |
| `generate` | `sara.worker` | Construct and validate one audited case |
| `refresh` | `sara.refresh` | Recompute deterministic acceptance fields |
| `merge` | `sara.merge` | Merge records and create the qualification manifest |
| `qualify` | `sara.qualification` | Run the remaining blind solvability trials |
| `finalize` | `sara.finalize` | Apply the retention rule and export seeds |

The production order is `generate`, `refresh`, `merge`, `qualify`, `finalize`
after preparation. The `generate` stage takes a case index; production covers
the source cases through the Slurm array. Qualification workers process the
manifest produced by `merge`.

## Outputs

The final workflow writes `final/qualified_records.jsonl`,
`final/qualified_seed.jsonl`, and `final/final_report.json`. The paper retains
45 qualified tasks. Participant partitioning consumes `qualified_seed.jsonl`
through the shared construction utilities.

The reference run is named `sara_target_grounded100_gpt54_0714_v10`. Its stored
`sara_target_sliced_subproof_v7` dataset-variant tag is retained for schema
compatibility; that tag does not select an earlier construction pipeline.
