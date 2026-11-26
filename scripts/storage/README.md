# Release artifact utilities

These commands package, verify, and stage existing dataset and trajectory
artifacts. They use the active Python environment and can be invoked by
absolute script path from any working directory. Run a command with `--help`
to see its accepted paths. Use [dataset construction](../data_generation/README.md)
to produce new tasks and [standard evaluation](../../evaluation/README.md) to run them.

The [release inventory](../../README.md#data-flow-and-release-inventory) distinguishes
the frozen benchmark, prepared RL splits, and selected trajectory artifacts.
The existing historical results archive also includes campaign artifacts and an
older code snapshot; it is separate from the selected export for this release.

## Export the prepared RL data and example trajectories

`export_release_bundle.py` creates three deterministic ZIP files from existing
frozen inputs. It uses only the standard library and makes no model calls:

```bash
python /path/to/checkout/scripts/storage/export_release_bundle.py \
  --legal-data-dir /path/to/legal/full_context \
  --legal-metadata-dir /path/to/legal \
  --finance-data-dir /path/to/finance/prepared \
  --benchmark-file /path/to/paper_tasks_163.jsonl \
  --trajectory-archive /path/to/model_evidence.zip \
  --output-dir artifacts/releases/teamgym-v1
```

The output directory must be fresh. The exporter builds in a temporary directory
and publishes the completed local directory only after reading back all archive
members and checking their hashes. `--code-revision` can record the runtime code
commit alongside the frozen inputs.

| Bundle | Contents |
| --- | --- |
| `teamgym_legal_rl_v1.zip` | Nine prepared files: 334 CUAD training rows, 36 CUAD selection rows, and 26 MAUD transfer rows; the original written-answer manifest, preparation report, and dataset audit. |
| `teamgym_finance_rl_v1.zip` | Nine prepared files: 538 training rows, 60 validation rows, and 40 held-out rows; the original split manifest and transfer audit. Includes the separate six-participant qualitative example when all three formats are present. |
| `teamgym_trajectory_examples_v1.zip` | One saved example per legal corpus, with matching frozen input records, actor manifests, evaluation results, and the original written judgments where applicable. |

Each bundle includes a README, `manifest.json`, and `FILES.sha256`. The outer
directory also has `release_manifest.json` and checksums. Extract each ZIP into
its own directory; the RL data path inside it is `prepared/`:

```bash
cd artifacts/releases/teamgym-v1
sha256sum -c FILES.sha256
python -m zipfile -e teamgym_legal_rl_v1.zip legal_rl
cd legal_rl
sha256sum -c FILES.sha256
```

Prepared files and preparation metadata retain their exact bytes, including
historical schema labels and provenance paths. Full dataset, parquet decoding,
construction, and GPU training validation remain separate checks.

The trajectory input is the historical `model_evidence.zip` archive, available as
`trajectories/model_evidence.zip` inside the historical results snapshot. Selection
defaults to `gpt54` / `multi_full`: the first complete evaluated cell for each
corpus in the frozen index, with matching input and applicable judgment. It does
not filter by success. `--model-slug` and `--condition` change that selection.
The sample is for inspecting saved episodes, not estimating benchmark scores.
The export includes no raw request logs, HTML reports, or code snapshots.

## Existing archive commands

```bash
python /path/to/checkout/scripts/storage/stage_legal_data.py \
  --source /path/to/legal_allocated_v1_all.zip \
  --destination /path/to/data/legal
```

Defaults use these roots, with relative paths interpreted from the caller:

- `TEAMGYM_ARCHIVE_ROOT`: `artifacts/archives`;
- `TEAMGYM_GENERATED_ROOT`: `generated`;
- `TEAMGYM_SECONDARY_GENERATED_ROOT`: the same generated tree, unless historical
  actor outputs and reports are stored separately;
- `TEAMGYM_LEGAL_DATA_ROOT`: `data/legal`, beneath `TEAMGYM_DATA_ROOT` when set.

Source filenames and experiment subdirectories retain their historical names.
Set the roots to the directories containing those artifacts, or use explicit
input arguments where available. The old manifest-prefix compatibility mapping
is preserved for relocated trajectories. Recorded provenance is unchanged.

`assemble_legal_paper_release.py` discovers this checkout from its script
location unless `--repo` is passed. Its Python version and package metadata
come from the interpreter running the command.
