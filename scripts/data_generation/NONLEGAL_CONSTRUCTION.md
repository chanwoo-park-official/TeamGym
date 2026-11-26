# Finance, Medical, and Software Construction

The shared PIU allocator implements the paper's five distribution conditions
`one_each`, `overlap`, `redundant_player`, `common_anchor`, and `bottleneck`.
Finance and Medical use these conditions where eligible. SWE Atlas and DeepSWE
use the policies and evaluation inputs documented in their construction guides.

Prepare the Python 3.11 environment from the repository root.

```bash
uv sync --locked --python 3.11 --extra construction --extra data --extra test
uv run --no-sync teamgym-generate --list
```

Use `uv run --no-sync` before the commands below, or activate `.venv` to run
them directly. The qualified commands and their short aliases select the same constructor.

| Dataset | Command | Inputs |
|---|---|---|
| FinDVer | `teamgym-generate finance/findver` | `--qa-path`, `--reports-root`, `--output-root` |
| FinLongDocQA | `teamgym-generate finance/finlongdocqa` | `--qa-path`, `--reports-zip`, `--output-root`. Use `--stage index` to index reports without model calls. |
| EvidenceBench | `teamgym-generate medical/evidencebench` | `--source-path` (split JSON directory), `--output-root` |
| MLCR | `teamgym-generate medical/mlcr` | `--source-dir`, `--case-id`, `--output-root` |
| SWE Atlas | `teamgym-generate swe/swe-atlas` | `--task-dir`, `--output-root` |
| DeepSWE | `teamgym-generate swe/deep-swe` | `--tasks-root`, `--task-id`, `--verifier-image`, `--output-root` |

MLCR accepts `--min-source-questions N` with a default of 2. Set N to 1 to
allow a generated task to use multiple evidence spans from one source QA.

### Source preparation

The six constructors read local source paths. Record the upstream revision and
file checksums with each run so later builds can use the same source version.

| Dataset | Public source and preparation |
|---|---|
| FinDVer | Clone [yilunzhao/FinDVer](https://github.com/yilunzhao/FinDVer). Pass a QA JSON file from `data/` and the `financial_reports/` directory. |
| FinLongDocQA | Download `dataset_qa.jsonl` from [Amian/FinLongDocQA](https://huggingface.co/datasets/Amian/FinLongDocQA). Obtain annual reports through the [upstream project](https://github.com/AI-Application-and-Integration-Lab/FinLongDocQA) and provide a ZIP containing `reports/COMPANY/YEAR.md`. The [Markdown report mirror](https://huggingface.co/datasets/timchen0618/finlongdocqa-reports) supplies that directory layout. |
| EvidenceBench | Call the packaged `source.acquire()` helper below, then pass its `original/` directory as `--source-path`. |
| MLCR | Download the `cases_summaries` and `questions` configurations from [Wisedocs/mlcr-dataset](https://huggingface.co/datasets/Wisedocs/mlcr-dataset). Export them to `cases_summaries.jsonl` and `questions.jsonl` in the same directory. Required fields are documented in [the medical guide](../../teamgym/data/medical/README.md). |
| SWE Atlas | Clone [scaleapi/SWE-Atlas](https://github.com/scaleapi/SWE-Atlas). Pass one directory under `data/qa/` as `--task-dir`. Docker must be running and the task's declared image must be available. |
| DeepSWE | Clone [datacurve-ai/deep-swe](https://github.com/datacurve-ai/deep-swe). Pass `tasks/` and a task ID. Prepare a verifier image from that task's `tests/Dockerfile` and supply `--verifier-image`. |

Prepare source checkouts independently of the TeamGym package.

```bash
git clone --depth 1 https://github.com/yilunzhao/FinDVer .external_datasets/FinDVer
git clone --depth 1 https://github.com/scaleapi/SWE-Atlas .external_datasets/SWE-Atlas
git clone --depth 1 https://github.com/datacurve-ai/deep-swe .external_datasets/deep-swe
```

Download EvidenceBench with its existing helper.

```bash
python - <<'PY'
from teamgym.data.medical.evidencebench.source import acquire

result = acquire(".external_datasets/evidencebench")
print(result["raw_root"])
PY
```

To prepare MLCR with the `data` extra installed, download and convert its two
Parquet files. The pinned revision below was checked during source preparation.

```bash
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as parquet

output = Path(".external_datasets/mlcr")
output.mkdir(parents=True, exist_ok=True)
for name in ("cases_summaries", "questions"):
    source = hf_hub_download(
        repo_id="Wisedocs/mlcr-dataset",
        repo_type="dataset",
        revision="1198206e2df96febf9fd13c110981e5b40515285",
        filename=f"data/{name}/train-00000-of-00001.parquet",
    )
    with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for row in parquet.read_table(source).to_pylist():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
```

Generation calls a model and may reject a candidate during grounding, necessity,
sufficiency, or packet-leakage audits. Inspect each run's report before exporting
accepted instances. Medical exports use the packaged `evaluation.py` helpers.
DeepSWE records constructed evidence and patch diagnostic outcomes in its report.

On Apple Silicon, use a Docker runtime that can execute the task's declared
image. For `linux/amd64` images in Colima, start the VM with
`--vm-type vz --vz-rosetta`.

The paper calls evidence units PIUs. Existing internal `riu`/`rui` field names
and command options support saved-artifact compatibility.

## Verification

See the [frozen example receipts](provenance/nonlegal_reconstruction_examples.json) for
archive checksums, example IDs, and the remaining source and acceptance gaps.
For FinLongDocQA, the
[producer bundle recipe](../../teamgym/data/finance/finlongdocqa/README.md#export-a-completed-producer-run-for-evaluation)
assembles a completed single-report run and verifies its evaluator export.

Run the repository's test suite with `uv run --no-sync pytest -q`. On macOS,
put Bash 5 on `PATH` for the shell-launcher tests. Record the
source revision, construction arguments, candidate decision, and export path
for each live run. Finance, Medical, and SWE Atlas exports can be reopened by
`teamgym.core.dataset.load_common_datapoints`. The loader
accepts both version-2 schema identifiers used by the cleanup and public
packages.

DeepSWE's standard complete-evidence qualification requires two official
successes in three attempts. The manifest records any explicit qualification
overrides. Reopen an accepted `export/instance.json` with
`teamgym.data.software.deep_swe.export.load_export` and
compare `render_model_input` for the all-PIU, PIU-only, and Full-Context
conditions against the saved model inputs. See the
[DeepSWE guide](../../teamgym/data/software/deep_swe/README.md)
for source preparation and the acceptance gates.
